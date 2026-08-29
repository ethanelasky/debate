"""Paired judge-only evaluation on frozen step-40 MATH debates.

The expensive policy pass happens once. Every generated debater slot is
recorded as ``sha256(rendered context) -> speech`` even when the disposable
seed judge later fails. Qwen3.5-4B and GPT-5.6 Luna then judge those speeches
through PlaybackModel. An arm is accepted only when its ordered task IDs,
playback hit count, and first-judge context hashes exactly match phase 1.

Pod phase (policy B200-0, stock Qwen judge B200-1)::

    python scripts/math_judge_pair.py --arms qwen --out outputs/math_judge_pair_step40

Local Luna replay (loads OPENAI_API_KEY from .env)::

    python scripts/math_judge_pair.py --reuse-cache outputs/math_judge_pair_step40 \
        --arms luna --out outputs/math_judge_pair_step40_luna

Phase-1 judge output is never scored. All metrics are reconstructed from each
arm's transcript rows. The oracle says Alice wins iff her OPENING solution is
both format-valid and mathematically correct; all other openings target Bob.
Ties and failures stay explicit and are excluded only from the binary common
denominator used to report McNemar's discordant counts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.config import load_experiment  # noqa: E402
from infra.backend.base import Backend, Datum, LossSpec, OptimParams, Sample, SamplingParams  # noqa: E402
from infra.envs.base import Policy  # noqa: E402
from infra.envs.debate.judge import JudgeConfig, SeatVerdict, verdict_from_slot  # noqa: E402
from infra.envs.debate.protocol import Kind, Protocol  # noqa: E402
from infra.envs.debate.round import DebateState, render_context  # noqa: E402
from infra.envs.tasks.math import problem_key  # noqa: E402
from infra.models.playback_model import PlaybackModel, context_key  # noqa: E402
from infra.models.base import Model, SpeechStructure  # noqa: E402
from infra.models.local_model import LocalModel  # noqa: E402
from infra.models.openai_model import OpenAIModel  # noqa: E402
from infra.run_debate import build_env, split_agents  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "infra/jobd/math_judge_pair_step40_config.yaml"
PROMPT = REPO / "infra/prompts/debate/hendrycks_math.yaml"
PROMPT_ENTRY = "math_proposer_critic"
EXPERIMENTS = {
    "qwen": "math_judge_pair_qwen",
    "luna": "math_judge_pair_luna",
}
EXPECTED_DEV_TASKS = 177
MANIFEST_SCHEMA = "math-judge-pair-v2"
REPLAY_MANIFEST_SCHEMA = "math-judge-pair-replay-v3"
CACHE_IDENTITY_KEYS = (
    "comparison_unit",
    "config_path",
    "prompt_path",
    "prompt_entry",
    "prompt_sha256",
    "dataset_protocol_identity",
    "model_checkpoint_identity",
    "debater_generation",
)
BASE_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
CHECKPOINT_SHA256 = {
    "rlvr_step20_rank0": "eccebdc69c251f8a1e9e7b14b07f858a5bb50cb06021876e275bfda49b7fa23e",
    "rlvr_step20_rank1": "229e3b2d95d69e34006f13e8010563b499a86d2792adc7b3a5b3837b4aa9210e",
    "rlvr_step20_lora_train_meta": "349f62eff696826bc2adec7661a8559b44883e6d137cff5a31cd64adf6a9e8ab",
    "debate_step40_rank0": "9c607d11511141742e81bf0baf0ad2cc2d27c34e8878a330770c568bb105088d",
    "debate_step40_rank1": "505abe346f95e15fed7dad02a308094777a8e7240839ac08b4d40bf291c00149",
    "debate_step40_lora_train_meta": "349f62eff696826bc2adec7661a8559b44883e6d137cff5a31cd64adf6a9e8ab",
}
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_BOX_COMMAND = re.compile(r"\\boxed\b", re.IGNORECASE)
_BOX_OPEN = re.compile(r"\\boxed\s*\{", re.IGNORECASE)


class VLLMCompletionsBackend(Backend):
    """Read-only vLLM backend for Policy's exact budget-forced sampler.

    Policy applies the merged model's chat template locally, then
    ``budget_forced_sample`` calls this backend once for the think phase and
    once for each visible-cap bucket. Raw ``/v1/completions`` preserves the
    extended token prefix across those phases. The training-only Backend
    endpoints intentionally raise: this object cannot update weights.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        tokenizer_path: str,
        workers: int = 16,
        tokenizer: Any = None,
        post_fn: Any = None,
        retry_attempts: int = 4,
        sleep_fn: Any = time.sleep,
    ):
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer = tokenizer
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.workers = workers
        self._post_override = post_fn
        self.retry_attempts = retry_attempts
        self._sleep = sleep_fn

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.retry_attempts <= 0:
            raise ValueError("retry_attempts must be positive")
        endpoint = f"{self.base_url}/completions"
        for attempt in range(self.retry_attempts):
            try:
                if self._post_override is not None:
                    return self._post_override(body)
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=7200) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:600]
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt + 1 == self.retry_attempts:
                    suffix = " after retries exhausted" if retryable else ""
                    raise RuntimeError(
                        f"HTTP {exc.code} from {endpoint}{suffix}: {detail}"
                    ) from None
            except urllib.error.URLError as exc:
                if attempt + 1 == self.retry_attempts:
                    raise RuntimeError(
                        f"transport retries exhausted for {endpoint}: {exc.reason}"
                    ) from None
            self._sleep(min(8.0, 2.0**attempt))
        raise AssertionError("unreachable retry loop")

    def _sample_one(
        self, prompt: list[int], params: SamplingParams, n: int
    ) -> list[Sample]:
        if params.max_tokens is None:
            raise ValueError("vLLM completions sampling requires max_tokens")
        body: dict[str, Any] = {
            "model": self.model,
            # Exact token-prefix continuation is the point of this backend.
            # Never decode/re-encode a prefix: tokenizers are not injective.
            "prompt": prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "n": n,
            "return_token_ids": True,
            # budget_forced_sample distinguishes a naturally sampled close
            # from a cap hit by inspecting the returned token suffix.
            "include_stop_str_in_output": True,
        }
        if params.stop:
            body["stop"] = params.stop
        response = self._post(body)
        choices = sorted(response.get("choices") or [], key=lambda choice: choice.get("index", 0))
        if len(choices) != n:
            raise RuntimeError(f"vLLM returned {len(choices)} choices, expected {n}")
        out: list[Sample] = []
        for choice in choices:
            prompt_tokens = choice.get("prompt_token_ids")
            if prompt_tokens != prompt:
                raise RuntimeError(
                    "vLLM returned prompt_token_ids that differ from the exact requested prefix"
                )
            tokens = choice.get("token_ids")
            if not isinstance(tokens, list) or not all(
                isinstance(token, int) and not isinstance(token, bool) for token in tokens
            ):
                raise RuntimeError("vLLM choice is missing valid token_ids")
            tokens = list(tokens)
            text = choice.get("text")
            if not isinstance(text, str):
                text = self.tokenizer.decode(tokens)
            out.append(
                Sample(
                    tokens=tokens,
                    # This is eval-only; logprobs exist solely to satisfy the
                    # Sample fidelity shape consumed by PolicySeat.
                    logprobs=[0.0] * len(tokens),
                    text=text,
                    stop_reason=("length" if choice.get("finish_reason") == "length" else "stop"),
                )
            )
        return out

    def sample(
        self, prompts: list[list[int]], params: SamplingParams, n: int = 1
    ) -> list[list[Sample]]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(self._sample_one, prompt, params, n) for prompt in prompts]
            return [future.result() for future in futures]

    def sync_sampler(self) -> None:
        return None

    def forward(self, data: list[Datum]) -> list[list[float]]:
        raise NotImplementedError("judge-pair backend is inference-only")

    def forward_backward(self, data: list[Datum], loss: LossSpec) -> dict[str, float]:
        raise NotImplementedError("judge-pair backend is inference-only")

    def optim_step(self, params: OptimParams) -> dict[str, float]:
        raise NotImplementedError("judge-pair backend is inference-only")

    def save(self, name: str) -> str:
        raise NotImplementedError("judge-pair backend is inference-only")

    def load(self, path: str) -> None:
        raise NotImplementedError("judge-pair backend is inference-only")


def _response_namespace(value: Any) -> Any:
    """Turn raw JSON into the attribute shape OpenAIModel already parses."""
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: _response_namespace(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return [_response_namespace(child) for child in value]
    return value


class RawVLLMChatModel(LocalModel):
    """LocalModel request/parser semantics over raw vLLM response JSON.

    The installed OpenAI SDK successfully received HTTP 200 from vLLM but
    failed while constructing its Pydantic response object.  This diagnostic
    keeps LocalModel's request body and OpenAIModel's response assembly while
    removing only that SDK deserialization layer.
    """

    def __init__(
        self,
        *,
        alias: str,
        endpoint: str,
        base_url: str,
        is_debater: bool = False,
        reasoning_effort: str | None = None,
        post_fn: Any = None,
        retry_attempts: int = 4,
        sleep_fn: Any = time.sleep,
    ):
        Model.__init__(self, alias=alias, is_debater=is_debater)
        if not base_url:
            raise ValueError("raw vLLM chat transport requires an explicit base_url")
        self.endpoint = endpoint
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self._post_override = post_fn
        self.retry_attempts = retry_attempts
        self._sleep = sleep_fn
        self.logger = logging.getLogger(__name__)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.retry_attempts <= 0:
            raise ValueError("retry_attempts must be positive")
        endpoint = f"{self.base_url}/chat/completions"
        for attempt in range(self.retry_attempts):
            try:
                if self._post_override is not None:
                    response = self._post_override(body)
                else:
                    request = urllib.request.Request(
                        endpoint,
                        data=json.dumps(body).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=7200) as raw:
                        response = json.load(raw)
                if not isinstance(response, dict):
                    raise RuntimeError(
                        f"malformed JSON response from {endpoint}: expected object, "
                        f"got {type(response).__name__}"
                    )
                return response
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:600]
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt + 1 == self.retry_attempts:
                    suffix = " after retries exhausted" if retryable else ""
                    raise RuntimeError(
                        f"HTTP {exc.code} from {endpoint}{suffix}: {detail}"
                    ) from None
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt + 1 == self.retry_attempts:
                    detail = getattr(exc, "reason", exc)
                    raise RuntimeError(
                        f"transport retries exhausted for {endpoint}: {detail}"
                    ) from None
            self._sleep(min(8.0, 2.0**attempt))
        raise AssertionError("unreachable retry loop")

    def call_openai(
        self,
        messages: list[dict[str, Any]],
        speech_structure: SpeechStructure,
        max_new_tokens: int,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> Any:
        body = self.build_request_kwargs(
            messages=messages,
            speech_structure=speech_structure,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            **kwargs,
        )
        # The SDK merges extra_body into the JSON payload. Do the same before
        # posting directly so top_k/min_p/repetition_penalty stay identical.
        extra = body.pop("extra_body", None)
        if extra:
            overlap = set(body) & set(extra)
            if overlap:
                raise ValueError(f"vLLM extra_body collides with request keys: {sorted(overlap)}")
            body.update(extra)
        return _response_namespace(self._post(body))

    def copy(self, is_debater: bool | None = None, **kwargs) -> "RawVLLMChatModel":
        return RawVLLMChatModel(
            alias=kwargs.get("alias", self.alias),
            endpoint=self.endpoint,
            base_url=self.base_url,
            is_debater=self.is_debater if is_debater is None else is_debater,
            reasoning_effort=self.reasoning_effort,
            post_fn=self._post_override,
            retry_attempts=self.retry_attempts,
            sleep_fn=self._sleep,
        )


class _UncappedResponsesResource:
    """Responses resource that omits the caller's synthetic token ceiling."""

    def __init__(self, resource: Any):
        self._resource = resource

    def create(self, **kwargs: Any) -> Any:
        kwargs.pop("max_output_tokens", None)
        return self._resource.create(**kwargs)


class _UncappedOpenAIClient:
    """Transparent client proxy whose Responses calls carry no output cap."""

    def __init__(self, client: Any):
        self._client = client
        self.responses = _UncappedResponsesResource(client.responses)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class UncappedResponsesOpenAIModel(OpenAIModel):
    """OpenAI Responses model with ``max_output_tokens`` genuinely omitted."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self._uses_responses_api():
            raise ValueError(
                "uncapped Responses transport requires a Responses model, "
                f"got {self.endpoint}"
            )
        self.client = _UncappedOpenAIClient(self.client)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_env_file() -> None:
    """Local Luna replay reads credentials without ever printing them."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=REPO / ".env", override=False)


def _protocol(exp: dict) -> Protocol:
    spec = exp["protocol"]
    if isinstance(spec, str):
        spec = exp["_topologies"][spec]
    return Protocol.parse(spec)


def protocol_payload(exp: dict) -> list[dict[str, Any]]:
    return [
        {
            "turn": cs.turn,
            "speaker": cs.speaker,
            "sequence": cs.seq,
            "name": cs.slot.name,
            "kind": cs.slot.kind.value,
            "visibility": cs.slot.visibility.value,
            "max_think_tokens": cs.slot.max_think_tokens,
            "max_visible_tokens": cs.slot.max_visible_tokens,
            "max_total_tokens": cs.slot.max_total_tokens,
            "enable_thinking": cs.slot.enable_thinking,
        }
        for cs in _protocol(exp).compile()
    ]


def load_arm(arm: str) -> dict:
    exp = load_experiment(str(CONFIG), EXPERIMENTS[arm])
    prompt = exp.get("prompt_config") or {}
    if Path(prompt.get("file_path", "")).resolve() != PROMPT.resolve():
        raise ValueError(f"{arm}: prompt path drifted: {prompt.get('file_path')!r}")
    if prompt.get("entry") != PROMPT_ENTRY:
        raise ValueError(f"{arm}: prompt entry drifted: {prompt.get('entry')!r}")
    trained, _ = split_agents(exp)
    if set(trained) != {"alice", "bob"}:
        raise ValueError(
            f"{arm}: seed generation must route alice/bob through PolicySeat, got {sorted(trained)}"
        )
    return exp


def build_arm_env(arm: str):
    exp = load_arm(arm)
    trained, frozen = split_agents(exp)
    env = build_env(exp, trained, frozen)
    if arm == "qwen":
        settings = exp["agents"][env.judge_speaker]["model_settings"]
        env.config.frozen_models[env.judge_speaker] = RawVLLMChatModel(
            alias=settings["alias"],
            endpoint=settings["model_file_path"],
            base_url=settings["base_url"],
            is_debater=False,
        )
    elif arm == "luna":
        settings = exp["agents"][env.judge_speaker]["model_settings"]
        env.config.frozen_models[env.judge_speaker] = UncappedResponsesOpenAIModel(
            alias=settings["alias"],
            endpoint=settings["model_file_path"],
            reasoning_effort=settings["reasoning_effort"],
            is_debater=False,
        )
    return exp, env


def prepare_playback_protocol(env) -> int:
    """Clear only debater template controls that playback cannot execute.

    PlaybackModel returns already-generated speech and never applies a chat
    template. FrozenSeat correctly rejects slot-level template kwargs for a
    live frozen model; carrying them into playback therefore trips a guard on
    metadata with no operation behind it. All context-visible protocol fields
    remain byte-for-byte identical.
    """
    missing = [
        speaker
        for speaker in env.debaters
        if not isinstance(env.config.frozen_models.get(speaker), PlaybackModel)
    ]
    if missing:
        raise ValueError(
            "playback protocol preparation requires PlaybackModel for every "
            f"debater, got live seats {missing}"
        )

    def visible_shape(protocol: Protocol) -> list[tuple[Any, ...]]:
        return [
            (
                cs.index,
                cs.turn,
                cs.speaker,
                cs.seq,
                cs.slot.name,
                cs.slot.kind,
                cs.slot.visibility,
                cs.slot.max_think_tokens,
                cs.slot.max_visible_tokens,
                cs.slot.max_total_tokens,
            )
            for cs in protocol.compile()
        ]

    original = env.protocol
    cleared = 0
    turns: list[dict[str, list[Any]]] = []
    for turn in original.turns:
        replay_turn: dict[str, list[Any]] = {}
        for speaker, slots in turn.items():
            replay_slots = []
            for slot in slots:
                if speaker in env.debaters and slot.enable_thinking is not None:
                    slot = replace(slot, enable_thinking=None)
                    cleared += 1
                replay_slots.append(slot)
            replay_turn[speaker] = replay_slots
        turns.append(replay_turn)
    replay = Protocol(turns=turns)
    replay.validate()
    if visible_shape(replay) != visible_shape(original):
        raise AssertionError("playback protocol changed context-visible slot semantics")
    env.protocol = replay
    env.config.protocol = replay
    return cleared


def task_id_from_meta(meta: dict[str, Any]) -> str:
    question = meta.get("question")
    if not isinstance(question, str) or not question:
        raise ValueError("MATH task has no nonempty question for stable task identity")
    return problem_key(question)


def ordered_task_ids(tasks: Iterable[Any]) -> list[str]:
    ids = [task_id_from_meta(task.meta) for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("MATH dev task IDs are not unique")
    return ids


def ordered_task_labels(tasks: Iterable[Any], task_ids: list[str]) -> list[dict[str, Any]]:
    tasks = list(tasks)
    if len(tasks) != len(task_ids):
        raise ValueError(f"task/ID count mismatch: {len(tasks)} vs {len(task_ids)}")
    return [
        {
            "task_id": task_id,
            "gt": task.meta.get("gt"),
            "level": task.meta.get("level"),
            "split": task.meta.get("split"),
        }
        for task, task_id in zip(tasks, task_ids)
    ]


def validate_opening_extractability(states: list[DebateState], task_ids: list[str]) -> None:
    """Fail the protocol when an existing opening has no relaxed numeric parse."""
    invalid: list[str] = []
    for state, task_id in zip(states, task_ids):
        opening = next(
            (
                record
                for record in state.records
                if record.slot.speaker == "alice" and record.slot.slot.kind == Kind.SOLUTION
            ),
            None,
        )
        if opening is not None and opening.extracted is None:
            invalid.append(task_id)
    if invalid:
        raise RuntimeError(
            "PROTOCOL INVALID: opening solution has no relaxed extractable number for "
            f"{len(invalid)} task(s): {invalid}"
        )


def record_cache(env, states: list[DebateState], task_ids: list[str]) -> list[dict[str, Any]]:
    """Record every generated debater slot, including judge-failed debates."""
    if len(states) != len(task_ids):
        raise ValueError(f"state/task count mismatch: {len(states)} vs {len(task_ids)}")
    entries: list[dict[str, Any]] = []
    for st, task_id in zip(states, task_ids):
        full = st.records
        try:
            for i, rec in enumerate(full):
                if rec.slot.speaker == env.judge_speaker:
                    continue
                st.records = full[:i]
                messages = render_context(st, rec.slot, env.prompts)
                entries.append(
                    {
                        "key": context_key(messages),
                        "speech": rec.text,
                        "thinking": rec.thinking,
                        "slot": f"{rec.slot.speaker}/{rec.slot.slot.name}@{rec.slot.turn}",
                        "task_id": task_id,
                    }
                )
        finally:
            st.records = full
    return entries


def verify_cache_shape(entries: list[dict[str, Any]], expected_records: int) -> None:
    if len(entries) != expected_records:
        raise RuntimeError(
            f"phase 1 did not reach every debater slot: cached {len(entries)}, "
            f"expected {expected_records}"
        )
    keys = [entry.get("key") for entry in entries]
    if len(set(keys)) != len(keys):
        raise RuntimeError(
            f"phase 1 cache has {len(keys) - len(set(keys))} duplicate rendered-context "
            "key(s); slots would not be independently replayable"
        )


def expected_cache_records(env, task_count: int, *, full_run: bool) -> int:
    """Derive cache size from the protocol, then pin the approved full run."""
    debater_slots = sum(
        cs.speaker != env.judge_speaker for cs in env.protocol.compile()
    )
    expected_records = task_count * debater_slots
    if full_run and (debater_slots, task_count, expected_records) != (4, 177, 708):
        raise RuntimeError(
            "full-run cache invariant drifted: expected protocol 4 debater slots x "
            f"177 tasks = 708, got {debater_slots} x {task_count} = {expected_records}"
        )
    return expected_records


def first_judge_context_hashes(
    env, states: list[DebateState], task_ids: list[str]
) -> list[dict[str, str | None]]:
    """Hash the judge's first rendered view, or null if debate failed earlier."""
    first = next(cs for cs in env.protocol.compile() if cs.speaker == env.judge_speaker)
    required = {cs.index for cs in env.protocol.compile() if cs.index < first.index}
    out: list[dict[str, str | None]] = []
    for st, task_id in zip(states, task_ids):
        prior = [rec for rec in st.records if rec.slot.index < first.index]
        if {rec.slot.index for rec in prior} != required:
            digest = None
        else:
            view = replace(st, records=prior)
            digest = context_key(render_context(view, first, env.prompts))
        out.append({"task_id": task_id, "sha256": digest})
    return out


def verify_pairing(
    *,
    expected_ids: list[str],
    actual_ids: list[str],
    expected_contexts: list[dict[str, str | None]],
    actual_contexts: list[dict[str, str | None]],
    expected_playback_hits: int,
    actual_playback_hits: int,
) -> None:
    if actual_ids != expected_ids:
        raise RuntimeError("PAIRING DRIFT: ordered MATH task IDs differ from phase 1")
    if actual_playback_hits != expected_playback_hits:
        raise RuntimeError(
            "PAIRING DRIFT: playback hit count differs from phase 1 cache "
            f"({actual_playback_hits} != {expected_playback_hits})"
        )
    if actual_contexts != expected_contexts:
        for expected, actual in zip(expected_contexts, actual_contexts):
            if expected != actual:
                raise RuntimeError(
                    "PAIRING DRIFT: first judge context differs for "
                    f"{expected.get('task_id')} ({actual.get('sha256')} != {expected.get('sha256')})"
                )
        raise RuntimeError("PAIRING DRIFT: first judge context list length differs")


def state_transcript(env, state: DebateState, task_id: str) -> dict[str, Any]:
    task = state.meta.get("task") or {}
    records = [
        {
            "speaker": rec.slot.speaker,
            "display_name": state.bindings.get(rec.slot.speaker, {}).get("NAME"),
            "turn": rec.slot.turn,
            "slot": rec.slot.slot.name,
            "kind": rec.slot.slot.kind.value,
            "visibility": rec.slot.slot.visibility.value,
            "text": rec.text,
            "thinking": rec.thinking,
            "answer_format_valid": rec.answer_format_valid,
            "retries": rec.retries,
            "truncated": rec.truncated,
            "stop_reason": (
                rec.response.stop_reason
                if rec.response is not None
                else (rec.sample.stop_reason if rec.sample is not None else None)
            ),
        }
        for rec in state.records
    ]
    return {
        "task_id": task_id,
        "task": {
            "gt": task.get("gt"),
            "level": task.get("level"),
            "split": task.get("split"),
        },
        "failed": state.failed,
        "model_failure": state.meta.get("model_failure"),
        "decision_attempts": list(state.meta.get("decision_attempts") or []),
        "records": records,
    }


def _brace_is_escaped(text: str, index: int) -> bool:
    slashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        slashes += 1
        index -= 1
    return slashes % 2 == 1


def strict_final_box(text: str) -> tuple[bool, float | None]:
    """Approved format: exactly one numeric box, closing at end-of-speech."""
    commands = list(_BOX_COMMAND.finditer(text))
    openings = list(_BOX_OPEN.finditer(text))
    if len(commands) != 1 or len(openings) != 1 or commands[0].start() != openings[0].start():
        return False, None

    open_brace = openings[0].end() - 1
    depth = 0
    close_brace = None
    for index in range(open_brace, len(text)):
        char = text[index]
        if char == "{" and not _brace_is_escaped(text, index):
            depth += 1
        elif char == "}" and not _brace_is_escaped(text, index):
            depth -= 1
            if depth == 0:
                close_brace = index
                break
            if depth < 0:
                return False, None
    if close_brace is None or text[close_brace + 1 :].strip():
        return False, None

    # Numeric extraction remains the task family's established grader. The
    # driver owns only the stricter location/cardinality requirement.
    from infra.envs.tasks.math import parse_numeric_answers

    parsed = parse_numeric_answers(text)
    if parsed.strict is None:
        return False, None
    return True, float(parsed.strict)


def score_transcript(row: dict[str, Any]) -> dict[str, Any]:
    """Derive the oracle and judge prediction from one transcript row only."""
    opening = next(
        (
            rec
            for rec in row.get("records", [])
            if rec.get("speaker") == "alice" and rec.get("kind") == Kind.SOLUTION.value
        ),
        None,
    )
    valid, boxed_value = strict_final_box(opening.get("text", "")) if opening else (False, None)
    gt = (row.get("task") or {}).get("gt")
    correct = False
    if valid and boxed_value is not None and gt is not None:
        try:
            correct = abs(float(boxed_value) - float(gt)) < 1e-6
        except (TypeError, ValueError):
            correct = False
    oracle = "Debater_A" if valid and correct else ("Debater_B" if opening else None)

    later_box = any(
        "\\boxed" in str(rec.get("text", "")).lower()
        for rec in row.get("records", [])
        if opening is not None and rec is not opening and rec.get("speaker") != "judge"
    )

    prediction = None
    failed = row.get("failed")
    outcome = (
        "unparseable"
        if failed == "verdict_unparseable"
        else ("failed" if failed else "missing")
    )
    confidence = None
    decision = next(
        (rec for rec in reversed(row.get("records", [])) if rec.get("kind") == Kind.DECISION.value),
        None,
    )
    if not row.get("failed") and decision is not None:
        verdict = verdict_from_slot(
            decision.get("text", ""),
            response=None,
            decode_fn=None,
            cfg=JudgeConfig(schema_name="competitive", retries=4),
            seat_names=["Debater_A", "Debater_B"],
        )
        if not verdict.ok:
            outcome = "unparseable"
        elif verdict.winner is not None:
            prediction = verdict.winner
            outcome = "winner"
            confidence = verdict.confidence[prediction].json
        elif all(v == SeatVerdict.TIE for v in verdict.seats.values()):
            outcome = "tie"
        else:
            outcome = "neither"

    return {
        **row,
        "oracle": {
            "opening_present": opening is not None,
            "answer_format_valid": valid,
            "math_correct": correct,
            "winner": oracle,
        },
        "later_debater_box": later_box,
        "judge": {
            "outcome": outcome,
            "winner": prediction,
            "confidence": confidence,
            "correct": (prediction == oracle) if prediction is not None and oracle is not None else None,
        },
    }


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if native_judge_stack_failures(rows):
        raise RuntimeError("native judge stack failures make arm accuracy invalid")
    outcomes: dict[str, int] = {}
    for row in rows:
        outcome = row["judge"]["outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    scored = [row for row in rows if row["judge"]["correct"] is not None]
    return {
        "comparison_unit": "native_judge_stack",
        "n_attempted": len(rows),
        "n_binary_scored": len(scored),
        "binary_accuracy": (
            sum(bool(row["judge"]["correct"]) for row in scored) / len(scored) if scored else None
        ),
        "outcomes": outcomes,
        "later_debater_box_count": sum(bool(row["later_debater_box"]) for row in rows),
        "invalid_opening_endorsed_count": sum(
            row["oracle"]["opening_present"]
            and not row["oracle"]["answer_format_valid"]
            and row["judge"]["winner"] == "Debater_A"
            for row in rows
        ),
    }


def native_judge_stack_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return failures that invalidate the whole judge-stack arm."""
    failures: list[dict[str, Any]] = []
    for row in rows:
        metadata = row.get("model_failure")
        failed = str(row.get("failed") or "")
        lowered = failed.lower()
        fallback_fatal = any(
            marker in lowered for marker in ("model_failed", "transport", "refusal", "cap")
        )
        judge_cap_slots = [
            {
                "slot": record.get("slot"),
                "turn": record.get("turn"),
                "kind": record.get("kind"),
                "stop_reason": "length",
            }
            for record in row.get("records", [])
            if record.get("speaker") == "judge"
            and record.get("slot") in {"deliberation", "verdict"}
            and record.get("stop_reason") == "length"
        ]
        judge_cap_decision_attempts = [
            {
                "slot": attempt.get("slot"),
                "speaker": attempt.get("speaker"),
                "turn": attempt.get("turn"),
                "kind": attempt.get("kind"),
                "retry_index": attempt.get("retry_index"),
                "stop_reason": "length",
            }
            for attempt in row.get("decision_attempts", [])
            if attempt.get("speaker") == "judge"
            and attempt.get("slot") in {"deliberation", "verdict"}
            and attempt.get("stop_reason") == "length"
        ]
        if (
            metadata is not None
            or fallback_fatal
            or judge_cap_slots
            or judge_cap_decision_attempts
        ):
            failure = {
                "task_id": row.get("task_id"),
                "state_failed": row.get("failed"),
                "model_failure": metadata,
            }
            if judge_cap_slots:
                failure["judge_cap_slots"] = judge_cap_slots
            if judge_cap_decision_attempts:
                failure["judge_cap_decision_attempts"] = judge_cap_decision_attempts
            failures.append(failure)
    return failures


def _paired_label(row: dict[str, Any]) -> dict[str, Any]:
    task = row.get("task") or {}
    oracle = row.get("oracle") or {}
    return {
        "task_id": row.get("task_id"),
        "gt": task.get("gt"),
        "level": task.get("level"),
        "split": task.get("split"),
        "oracle": {
            "opening_present": oracle.get("opening_present"),
            "answer_format_valid": oracle.get("answer_format_valid"),
            "math_correct": oracle.get("math_correct"),
            "winner": oracle.get("winner"),
        },
    }


def paired_summary(qwen: list[dict[str, Any]], luna: list[dict[str, Any]]) -> dict[str, Any]:
    if native_judge_stack_failures(qwen) or native_judge_stack_failures(luna):
        raise RuntimeError("native judge stack failures make McNemar invalid")
    q_ids = [row["task_id"] for row in qwen]
    l_ids = [row["task_id"] for row in luna]
    if q_ids != l_ids:
        raise RuntimeError("PAIRING DRIFT: arm result task order differs")
    if [_paired_label(row) for row in qwen] != [_paired_label(row) for row in luna]:
        raise RuntimeError("PAIRING DRIFT: arm task/GT/oracle labels differ")
    common = [
        (q, l)
        for q, l in zip(qwen, luna)
        if q["judge"]["correct"] is not None and l["judge"]["correct"] is not None
    ]
    both_correct = sum(q["judge"]["correct"] and l["judge"]["correct"] for q, l in common)
    qwen_only = sum(q["judge"]["correct"] and not l["judge"]["correct"] for q, l in common)
    luna_only = sum(not q["judge"]["correct"] and l["judge"]["correct"] for q, l in common)
    both_wrong = sum(not q["judge"]["correct"] and not l["judge"]["correct"] for q, l in common)
    discordant = qwen_only + luna_only
    exact_p = 1.0
    if discordant:
        smaller = min(qwen_only, luna_only)
        exact_p = min(
            1.0,
            2.0
            * sum(math.comb(discordant, k) for k in range(smaller + 1))
            / (2**discordant),
        )
    return {
        "comparison_unit": "native_judge_stack",
        "n_attempted": len(qwen),
        "n_common_binary": len(common),
        "common_task_ids": [q["task_id"] for q, _ in common],
        "mcnemar": {
            "both_correct": both_correct,
            "qwen_correct_luna_wrong": qwen_only,
            "qwen_wrong_luna_correct": luna_only,
            "both_wrong": both_wrong,
            "discordant_n": discordant,
            "exact_two_sided_p": exact_p,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    with open(path, "x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_source_commit(explicit: str | None) -> str:
    commit = explicit
    if commit is None:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
    if not commit or _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError(
            "--source-commit must be the immutable 40-hex commit synced to the pod "
            "(the jobd repo input excludes .git)"
        )
    return commit


def expected_manifest_inputs(
    source_commit: str, dataset_protocol_identity: dict[str, Any]
) -> dict[str, Any]:
    exps = {arm: load_arm(arm) for arm in EXPERIMENTS}
    return {
        "comparison_unit": "native_judge_stack",
        "source_commit": source_commit,
        "config_path": str(CONFIG.relative_to(REPO)),
        "config_sha256": file_sha256(CONFIG),
        "prompt_path": str(PROMPT.relative_to(REPO)),
        "prompt_entry": PROMPT_ENTRY,
        "prompt_sha256": file_sha256(PROMPT),
        "protocol_sha256_by_arm": {
            arm: canonical_sha256(protocol_payload(exp)) for arm, exp in exps.items()
        },
        "dataset_protocol_identity": dataset_protocol_identity,
        "model_checkpoint_identity": {
            "stock_model": "Qwen/Qwen3.5-4B",
            "stock_revision": BASE_MODEL_REVISION,
            "adapter_sha256": CHECKPOINT_SHA256,
        },
        "debater_generation": {
            "backend": "vllm_raw_completions_budget_forced_v1",
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "judge_generation_by_arm": {
            "qwen": {
                "model": "Qwen/Qwen3.5-4B",
                "temperature": 0.0,
                "top_p": 1.0,
                "deliberation_cap": 2048,
                "verdict_cap": 512,
                "chat_template": "server_default",
                "historical_enable_thinking_false_forwarded": False,
                "transport": "raw_vllm_chat_completions_json_v1",
            },
            "luna": {
                "model": "gpt-5.6-luna",
                "reasoning_effort": "low",
                "temperature": "provider_native_not_exposed",
                "deliberation_cap": None,
                "verdict_cap": None,
                "max_output_tokens": "omitted",
                "transport": "openai_responses_uncapped_v1",
            },
        },
    }


def verify_manifest_inputs(manifest: dict[str, Any], current: dict[str, Any]) -> None:
    for key, value in current.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"PAIRING DRIFT: seed manifest {key} differs from current source "
                f"({manifest.get(key)!r} != {value!r})"
            )


def verify_cache_identity(manifest: dict[str, Any], current: dict[str, Any]) -> None:
    """Verify everything that could change the 708 recorded debater speeches.

    Judge settings and the replay implementation may advance after phase 1;
    neither generated the cache. Full protocol/config hashes remain in seed
    and replay provenance, while the later 708-key PlaybackModel gate proves
    every current debater context still maps to the immutable recorded speech.
    """
    for key in CACHE_IDENTITY_KEYS:
        if manifest.get(key) != current.get(key):
            raise RuntimeError(
                f"PAIRING DRIFT: seed cache identity {key} differs from current "
                f"source ({manifest.get(key)!r} != {current.get(key)!r})"
            )


def replay_manifest(
    *,
    seed: dict[str, Any],
    seed_manifest_sha256: str,
    current: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": REPLAY_MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": {
            "source_commit": seed.get("source_commit"),
            "manifest_sha256": seed_manifest_sha256,
            "cache_sha256": seed.get("cache_sha256"),
            "cache_records": seed.get("cache_records"),
            "ordered_task_ids_sha256": seed.get("ordered_task_ids_sha256"),
            "config_sha256": seed.get("config_sha256"),
            "protocol_sha256_by_arm": seed.get("protocol_sha256_by_arm"),
        },
        "replay": {
            "source_commit": current["source_commit"],
            "config_sha256": current["config_sha256"],
            "protocol_sha256_by_arm": current["protocol_sha256_by_arm"],
            "judge_generation_by_arm": current["judge_generation_by_arm"],
        },
        "verified_cache_identity": {
            **{key: current[key] for key in CACHE_IDENTITY_KEYS},
            "debater_context_gate": "708_content_addressed_playback_keys_v1",
        },
    }


def _cache_path(reuse: str | None, out_dir: Path) -> Path:
    if reuse is None:
        return out_dir / "cache.jsonl"
    path = Path(reuse)
    return path if path.is_file() else path / "cache.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", required=True, help="comma-separated subset: qwen,luna")
    ap.add_argument("--out", required=True, help="artifact directory")
    ap.add_argument("--reuse-cache", default=None, help="phase-1 artifact directory or cache.jsonl")
    ap.add_argument("--source-commit", default=None, help="40-hex immutable source commit")
    ap.add_argument("--policy-base-url", default=None, help="override merged-policy vLLM /v1 URL")
    ap.add_argument("--policy-model", default=None, help="override vLLM served model name")
    ap.add_argument("--policy-tokenizer", default=None, help="override merged tokenizer path")
    ap.add_argument("--workers", type=int, default=16, help="raw-completions request fanout")
    ap.add_argument("--limit", type=int, default=None, help="smoke only; full run must omit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    load_env_file()

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = [arm for arm in arms if arm not in EXPERIMENTS]
    if not arms or unknown:
        raise SystemExit(f"unknown/empty arm selection {unknown}; choose from {list(EXPERIMENTS)}")
    if len(set(arms)) != len(arms):
        raise SystemExit("duplicate arm requested")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    source_commit = resolve_source_commit(args.source_commit)
    seed_exp, seed_env = build_arm_env("qwen")
    try:
        pool_tasks = seed_env.tasks(10**9, split="dev")
        if len(pool_tasks) != EXPECTED_DEV_TASKS:
            raise RuntimeError(
                f"expected all {EXPECTED_DEV_TASKS} held-out MATH-L5 dev tasks, "
                f"got {len(pool_tasks)}"
            )
        pool_task_ids = ordered_task_ids(pool_tasks)
        pool_task_labels = ordered_task_labels(pool_tasks, pool_task_ids)
        tasks = pool_tasks
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            tasks = tasks[: args.limit]
        task_ids = ordered_task_ids(tasks)
        task_labels = ordered_task_labels(tasks, task_ids)
        dataset_protocol_identity = seed_env.family.protocol_identity()
        if not isinstance(dataset_protocol_identity, dict):
            raise TypeError("MATH family protocol_identity() must return a mapping")
        source_identity = expected_manifest_inputs(
            source_commit, dataset_protocol_identity
        )
        out_dir = Path(args.out)
        cache_path = _cache_path(args.reuse_cache, out_dir)
        if args.reuse_cache and out_dir.resolve() == cache_path.parent.resolve():
            raise ValueError(
                "--out must differ from the immutable --reuse-cache artifact directory"
            )
        print(
            json.dumps(
                {
                    "arms": arms,
                    "n_tasks": len(tasks),
                    "phase1": "reuse" if args.reuse_cache else "generate",
                    "cache": str(cache_path),
                    "out": str(out_dir),
                    "ordered_task_labels_sha256": canonical_sha256(task_labels),
                    **source_identity,
                },
                indent=2,
            )
        )
        if args.dry_run:
            for arm in arms:
                exp = load_arm(arm)
                # Environment construction above compiled the seed prompt;
                # parsing every arm here proves cap/model overrides resolve.
                _protocol(exp).compile()
            print("dry-run: config, prompt, protocol, and ordered dev pool resolved; no generation")
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        if args.reuse_cache:
            manifest_path = cache_path.parent / "seed_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != MANIFEST_SCHEMA:
                raise RuntimeError(f"unsupported seed manifest schema {manifest.get('schema')!r}")
            verify_cache_identity(manifest, source_identity)
            # A smoke may replay a prefix, but it still verifies the entire
            # immutable seed/cache against the current held-out pool first.
            if manifest.get("ordered_task_ids") != pool_task_ids:
                raise RuntimeError("PAIRING DRIFT: current ordered dev task IDs differ from seed manifest")
            if manifest.get("ordered_task_labels") != pool_task_labels:
                raise RuntimeError("PAIRING DRIFT: current ordered task labels differ from seed manifest")
            if manifest.get("ordered_task_labels_sha256") != canonical_sha256(pool_task_labels):
                raise RuntimeError("PAIRING DRIFT: seed manifest task-label hash is invalid")
            if file_sha256(cache_path) != manifest.get("cache_sha256"):
                raise RuntimeError("PAIRING DRIFT: cache.jsonl hash differs from seed manifest")
            cache_entries = _read_jsonl(cache_path)
            full_cache_records = expected_cache_records(
                seed_env, len(pool_tasks), full_run=True
            )
            verify_cache_shape(cache_entries, full_cache_records)
            if manifest.get("cache_records") != full_cache_records:
                raise RuntimeError("PAIRING DRIFT: seed manifest cache record count is not protocol-derived")
        else:
            print(f"phase 1: generating {len(tasks)} frozen-policy debates; seed judge output discarded", file=sys.stderr)
            alice_settings = seed_exp["agents"]["alice"]["model_settings"]
            policy_model = args.policy_model or alice_settings["model_file_path"]
            backend = VLLMCompletionsBackend(
                base_url=args.policy_base_url or alice_settings["base_url"],
                model=policy_model,
                tokenizer_path=args.policy_tokenizer or policy_model,
                workers=args.workers,
            )
            policy = Policy(
                backend,
                SamplingParams(max_tokens=None, temperature=0.0, top_p=1.0),
                chat_template_kwargs={"enable_thinking": True},
            )
            seed_env.rollout(tasks, policy=policy, group_size=1)
            validate_opening_extractability(seed_env.last_states, task_ids)
            entries = record_cache(seed_env, seed_env.last_states, task_ids)
            expected_records = expected_cache_records(
                seed_env, len(tasks), full_run=args.limit is None
            )
            expected_debater_slots = expected_records // len(tasks)
            verify_cache_shape(entries, expected_records)
            contexts = first_judge_context_hashes(seed_env, seed_env.last_states, task_ids)
            if any(item["sha256"] is None for item in contexts):
                raise RuntimeError("phase 1 did not reach the first judge context for every MATH task")
            _write_jsonl(cache_path, entries)
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                **source_identity,
                "ordered_task_ids": task_ids,
                "ordered_task_ids_sha256": canonical_sha256(task_ids),
                "ordered_task_labels": task_labels,
                "ordered_task_labels_sha256": canonical_sha256(task_labels),
                "first_judge_context_hashes": contexts,
                "debater_slots_per_task": expected_debater_slots,
                "cache_records": len(entries),
                "cache_sha256": file_sha256(cache_path),
                "phase1_workers": args.workers,
                "phase1_judge_output": "discarded",
                "phase1_failures": [
                    {"task_id": task_id, "failed": st.failed}
                    for task_id, st in zip(task_ids, seed_env.last_states)
                    if st.failed is not None
                ],
            }
            manifest_path = out_dir / "seed_manifest.json"
            _write_json(manifest_path, manifest)
    finally:
        seed_env.family.close()

    replay_provenance = replay_manifest(
        seed=manifest,
        seed_manifest_sha256=file_sha256(manifest_path),
        current=source_identity,
    )
    _write_json(out_dir / "replay_manifest.json", replay_provenance)

    for arm in arms:
        print(f"arm {arm}: replaying {len(task_ids)} exact debates", file=sys.stderr)
        _, env = build_arm_env(arm)
        try:
            arm_tasks = env.tasks(10**9, split="dev")
            if args.limit is not None:
                arm_tasks = arm_tasks[: args.limit]
            arm_ids = ordered_task_ids(arm_tasks)
            arm_labels = ordered_task_labels(arm_tasks, arm_ids)
            if env.family.protocol_identity() != manifest["dataset_protocol_identity"]:
                raise RuntimeError("PAIRING DRIFT: arm dataset protocol identity differs")
            if arm_labels != task_labels:
                raise RuntimeError("PAIRING DRIFT: arm ordered task labels differ")
            for speaker in env.debaters:
                env.config.frozen_models[speaker] = PlaybackModel(
                    alias=f"playback-{speaker}", cache_path=cache_path, is_debater=True
                )
            # Config marks the debaters trained only to make seed generation
            # use PolicySeat's two-phase budget forcing. A replay is frozen by
            # definition, so switch ownership before constructing round seats.
            env.config.trained_speakers = []
            env.config.trained_sampling = {}
            env.config.trained_chat_kwargs = {}
            cleared_playback_template_controls = prepare_playback_protocol(env)
            env.rollout(arm_tasks, policy=None, group_size=1)
            contexts = first_judge_context_hashes(env, env.last_states, arm_ids)
            hits = sum(
                model.hits
                for speaker, model in env.config.frozen_models.items()
                if speaker in env.debaters and isinstance(model, PlaybackModel)
            )
            verify_pairing(
                expected_ids=task_ids,
                actual_ids=arm_ids,
                expected_contexts=manifest["first_judge_context_hashes"][: len(task_ids)],
                actual_contexts=contexts,
                expected_playback_hits=expected_cache_records(
                    env, len(arm_tasks), full_run=args.limit is None
                ),
                actual_playback_hits=hits,
            )
            rows = [
                score_transcript(state_transcript(env, state, task_id))
                for state, task_id in zip(env.last_states, arm_ids)
            ]
            _write_jsonl(out_dir / f"{arm}.transcripts.jsonl", rows)
            stack_failures = native_judge_stack_failures(rows)
            if stack_failures:
                _write_json(
                    out_dir / f"{arm}.failure.json",
                    {
                        "comparison_unit": "native_judge_stack",
                        "valid": False,
                        "arm": arm,
                        "seed_source_commit": manifest.get("source_commit"),
                        "replay_source_commit": source_identity["source_commit"],
                        "seed_cache_sha256": manifest.get("cache_sha256"),
                        "failure_count": len(stack_failures),
                        "failures": stack_failures,
                    },
                )
                raise RuntimeError(
                    f"{arm} native judge stack invalid: {len(stack_failures)} "
                    "model/cap/transport/refusal failure(s); no accuracy published"
                )
            summary = summarize_arm(rows)
            summary["judge_generation"] = source_identity["judge_generation_by_arm"][arm]
            summary["seed_source_commit"] = manifest.get("source_commit")
            summary["replay_source_commit"] = source_identity["source_commit"]
            summary["seed_cache_sha256"] = manifest.get("cache_sha256")
            summary["cleared_nonoperative_playback_template_controls"] = (
                cleared_playback_template_controls
            )
            _write_json(out_dir / f"{arm}.summary.json", summary)
        finally:
            env.family.close()

    qwen_path, luna_path = out_dir / "qwen.transcripts.jsonl", out_dir / "luna.transcripts.jsonl"
    if qwen_path.exists() and luna_path.exists():
        pair = paired_summary(_read_jsonl(qwen_path), _read_jsonl(luna_path))
        pair["seed_source_commit"] = manifest.get("source_commit")
        pair["replay_source_commit"] = source_identity["source_commit"]
        pair["seed_cache_sha256"] = manifest.get("cache_sha256")
        _write_json(out_dir / "paired_summary.json", pair)
        print(json.dumps(pair, indent=2))
    print(f"artifacts: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
