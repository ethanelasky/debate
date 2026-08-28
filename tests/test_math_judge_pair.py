from __future__ import annotations

import io
import json
import importlib.util
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from infra.config import load_experiment
from infra.backend.base import SamplingParams
from infra.envs.base import Policy, SlotLimits
from infra.envs.debate.prompts import PromptLibrary, RenderedPrompts
from infra.envs.debate.protocol import Protocol
from infra.envs.debate.round import DebateState, SlotRecord, render_context
from infra.models.base import ModelInput
from infra.models.playback_model import PlaybackModel, context_key
CONFIG = "infra/jobd/math_judge_pair_step40_config.yaml"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/math_judge_pair.py"
JOB_SPEC = Path(__file__).resolve().parents[1] / "infra/jobd/math_judge_pair_step40.yaml"
_SPEC = importlib.util.spec_from_file_location("math_judge_pair", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
pair = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pair)


def _row(*, opening: str, gt: float, verdict: str | None, failed: str | None = None):
    records = [
        {
            "speaker": "alice",
            "kind": "solution",
            "slot": "proposal",
            "text": opening,
        },
        {
            "speaker": "alice",
            "kind": "speech",
            "slot": "alice_rebuttal",
            "text": r"The answer is still \boxed{5}.",
        },
    ]
    if verdict is not None:
        records.append(
            {
                "speaker": "judge",
                "kind": "decision",
                "slot": "verdict",
                "text": verdict,
            }
        )
    return {"task_id": "task", "task": {"gt": gt}, "failed": failed, "records": records}


def test_oracle_requires_correct_box_in_opening_and_ignores_rebuttal_repair() -> None:
    row = pair.score_transcript(
        _row(
            opening="The calculation gives 5.",
            gt=5,
            verdict='{"winner":"Debater_A","confidence":1.0}',
        )
    )
    assert row["oracle"] == {
        "opening_present": True,
        "answer_format_valid": False,
        "math_correct": False,
        "winner": "Debater_B",
    }
    assert row["later_debater_box"] is True
    assert row["judge"]["winner"] == "Debater_A"
    assert row["judge"]["correct"] is False

    valid = pair.score_transcript(
        _row(
            opening=r"Therefore, \boxed{5}",
            gt=5,
            verdict='{"winner":"Debater_A","confidence":0.75}',
        )
    )
    assert valid["oracle"]["answer_format_valid"] is True
    assert valid["oracle"]["math_correct"] is True
    assert valid["oracle"]["winner"] == "Debater_A"


def test_paired_summary_keeps_ties_failures_out_of_common_mcnemar_denominator() -> None:
    def scored(task_id: str, correct):
        return {"task_id": task_id, "judge": {"correct": correct}}

    qwen = [scored("a", True), scored("b", False), scored("c", None), scored("d", True)]
    luna = [scored("a", True), scored("b", True), scored("c", False), scored("d", None)]
    summary = pair.paired_summary(qwen, luna)
    assert summary["n_attempted"] == 4
    assert summary["n_common_binary"] == 2
    assert summary["common_task_ids"] == ["a", "b"]
    assert summary["mcnemar"] == {
        "both_correct": 1,
        "qwen_correct_luna_wrong": 0,
        "qwen_wrong_luna_correct": 1,
        "both_wrong": 0,
        "discordant_n": 1,
        "exact_two_sided_p": 1.0,
    }


def test_mcnemar_reports_exact_two_sided_binomial_p_value() -> None:
    ids = ["a", "b", "c", "d"]
    qwen_correct = [True, False, False, False]
    luna_correct = [False, True, True, True]
    qwen = [
        {"task_id": task_id, "judge": {"correct": correct}}
        for task_id, correct in zip(ids, qwen_correct)
    ]
    luna = [
        {"task_id": task_id, "judge": {"correct": correct}}
        for task_id, correct in zip(ids, luna_correct)
    ]
    assert pair.paired_summary(qwen, luna)["mcnemar"]["exact_two_sided_p"] == 0.625


def test_paired_summary_rejects_gt_or_oracle_label_drift() -> None:
    base = {
        "task_id": "a",
        "task": {"gt": 5.0, "level": 5, "split": "dev"},
        "oracle": {
            "opening_present": True,
            "answer_format_valid": True,
            "math_correct": True,
            "winner": "Debater_A",
        },
        "judge": {"correct": True},
    }
    drifted = json.loads(json.dumps(base))
    drifted["task"]["gt"] = 6.0
    with pytest.raises(RuntimeError, match="task/GT/oracle labels"):
        pair.paired_summary([base], [drifted])


def test_verify_pairing_fails_on_context_or_playback_drift() -> None:
    contexts = [{"task_id": "a", "sha256": "1"}]
    pair.verify_pairing(
        expected_ids=["a"],
        actual_ids=["a"],
        expected_contexts=contexts,
        actual_contexts=contexts,
        expected_playback_hits=1,
        actual_playback_hits=1,
    )
    with pytest.raises(RuntimeError, match="first judge context"):
        pair.verify_pairing(
            expected_ids=["a"],
            actual_ids=["a"],
            expected_contexts=contexts,
            actual_contexts=[{"task_id": "a", "sha256": "2"}],
            expected_playback_hits=1,
            actual_playback_hits=1,
        )
    with pytest.raises(RuntimeError, match="playback hit count"):
        pair.verify_pairing(
            expected_ids=["a"],
            actual_ids=["a"],
            expected_contexts=contexts,
            actual_contexts=contexts,
            expected_playback_hits=1,
            actual_playback_hits=0,
        )


def test_record_cache_keeps_debater_speech_when_seed_judge_failed(tmp_path: Path) -> None:
    protocol = Protocol.parse(
        {
            "turns": [
                {"alice": [{"name": "proposal", "kind": "solution", "max_total_tokens": 8}]},
                {"judge": [{"name": "verdict", "kind": "decision", "max_total_tokens": 8}]},
            ]
        }
    )
    compiled = protocol.compile()
    prompts = PromptLibrary(
        system={"alice": "Solve.", "judge": "Judge."},
        slots={"proposal": "Answer.", "verdict": "Choose."},
    )
    state = DebateState(
        bindings={
            "alice": {"NAME": "Debater_A", "POSITION": "5", "OPPONENT_NAME": "Debater_B", "OPPONENT_POSITION": "not 5", "TOPIC": "p"},
            "judge": {"NAME": "Debater_A", "POSITION": "5", "OPPONENT_NAME": "Debater_B", "OPPONENT_POSITION": "not 5", "TOPIC": "p"},
        },
        records=[SlotRecord(slot=compiled[0], text=r"\boxed{5}", thinking="scratch")],
        failed="verdict_unparseable",
    )
    rendered_prompts = RenderedPrompts(prompts)
    env = SimpleNamespace(judge_speaker="judge", prompts=rendered_prompts)
    entries = pair.record_cache(env, [state], ["task-a"])
    assert len(entries) == 1
    assert entries[0]["task_id"] == "task-a"
    assert entries[0]["speech"] == r"\boxed{5}"

    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(entries[0]) + "\n")
    messages = render_context(DebateState(bindings=state.bindings), compiled[0], rendered_prompts)
    assert entries[0]["key"] == context_key(messages)
    playback = PlaybackModel(alias="playback", cache_path=cache)
    response = playback.predict(
        [[ModelInput(role=m["role"], content=m["content"]) for m in messages]]
    )[0]
    assert response.speech == r"\boxed{5}"
    assert response.thinking == "scratch"


def test_config_pins_models_caps_prompt_and_disjoint_local_endpoints() -> None:
    qwen = load_experiment(CONFIG, "math_judge_pair_qwen")
    luna = load_experiment(CONFIG, "math_judge_pair_luna")
    expected_prompt = {
        "file_path": "infra/prompts/debate/hendrycks_math.yaml",
        "entry": "math_proposer_critic",
    }
    assert qwen["prompt_config"] == expected_prompt
    assert luna["prompt_config"] == expected_prompt
    assert pair.PROMPT == SCRIPT.parents[1] / expected_prompt["file_path"]
    assert pair.PROMPT_ENTRY == expected_prompt["entry"]
    assert qwen["agents"]["alice"]["trained"] is True
    assert qwen["agents"]["bob"]["trained"] is True
    assert qwen["agents"]["alice"]["model_settings"]["sampling"]["eval"] == {
        "temperature": 0.0,
        "top_p": 1.0,
    }
    assert qwen["agents"]["alice"]["model_settings"]["base_url"].endswith(":8790/v1")
    assert qwen["agents"]["judge"]["model_settings"]["base_url"].endswith(":8788/v1")
    assert qwen["agents"]["judge"]["model_settings"]["model_file_path"] == "Qwen/Qwen3.5-4B"
    assert "enable_thinking" not in qwen["agents"]["judge"]["model_settings"]
    assert qwen["agents"]["judge"]["model_settings"]["sampling"]["train"]["temperature"] == 0.0
    assert [slot["max_total_tokens"] for slot in qwen["protocol"]["turns"][4]["judge"]] == [768, 256]
    assert luna["agents"]["judge"]["model_settings"]["model_file_path"] == "gpt-5.6-luna"
    assert luna["agents"]["judge"]["model_settings"]["reasoning_effort"] == "low"
    assert luna["agents"]["judge"]["model_settings"]["base_url"] is None
    assert [slot["max_total_tokens"] for slot in luna["protocol"]["turns"][4]["judge"]] == [1536, 512]


def test_job_spec_requires_two_b200s_and_pins_every_merge_input() -> None:
    spec = yaml.safe_load(JOB_SPEC.read_text(encoding="utf-8"))
    job = spec["jobs"][0]
    assert spec["profile"] == "debate-b200x2"
    assert job["gpus"] == 2
    assert job["gpu_type"] == "NVIDIA B200"
    # This differs from the profile's 32-CPU default on purpose: jobd treats
    # any rebound-field override as an instruction to disable profile
    # alternates, so the job cannot be rebound to debate-h200x2.
    assert job["cpus"] == 31
    assert job["env"] == ["HF_TOKEN"]
    command = job["command"]
    for digest in pair.CHECKPOINT_SHA256.values():
        assert digest in command
    assert "unexpected merger shard set" in command
    assert "expected exactly two GPUs" in command
    assert "approved only for 2x NVIDIA B200" in command


def test_manifest_verification_rejects_source_hash_drift() -> None:
    expected = {
        "source_commit": "a" * 40,
        "config_sha256": "config",
        "prompt_sha256": "prompt",
    }
    pair.verify_manifest_inputs(dict(expected), expected)
    current = dict(expected, prompt_sha256="changed")
    with pytest.raises(RuntimeError, match="prompt_sha256"):
        pair.verify_manifest_inputs(expected, current)


def test_manifest_identity_pins_dataset_model_and_adapter_sources() -> None:
    identity = pair.expected_manifest_inputs("a" * 40, {"split_sha256": "dataset"})
    assert identity["comparison_unit"] == "native_judge_stack"
    assert identity["config_path"] == "infra/jobd/math_judge_pair_step40_config.yaml"
    assert identity["dataset_protocol_identity"] == {"split_sha256": "dataset"}
    assert identity["model_checkpoint_identity"] == {
        "stock_model": "Qwen/Qwen3.5-4B",
        "stock_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "adapter_sha256": pair.CHECKPOINT_SHA256,
    }
    assert pair.CHECKPOINT_SHA256["rlvr_step20_lora_train_meta"] == (
        "349f62eff696826bc2adec7661a8559b44883e6d137cff5a31cd64adf6a9e8ab"
    )
    assert pair.CHECKPOINT_SHA256["debate_step40_lora_train_meta"] == (
        "349f62eff696826bc2adec7661a8559b44883e6d137cff5a31cd64adf6a9e8ab"
    )


def test_native_judge_stack_failure_is_run_invalidating_but_parse_failure_is_not() -> None:
    model_failed = {
        "task_id": "a",
        "failed": "judge/verdict: model_failed",
        "model_failure": {
            "slot": "judge/verdict@0",
            "fail_reason": "model_failed",
            "stop_reason": "length",
        },
    }
    assert pair.native_judge_stack_failures([model_failed]) == [
        {
            "task_id": "a",
            "state_failed": "judge/verdict: model_failed",
            "model_failure": model_failed["model_failure"],
        }
    ]
    assert pair.native_judge_stack_failures(
        [{"task_id": "b", "failed": "verdict_unparseable", "model_failure": None}]
    ) == []
    with pytest.raises(RuntimeError, match="McNemar invalid"):
        pair.paired_summary([model_failed], [model_failed])


def test_transcript_consumes_debate_state_model_failure_provenance() -> None:
    failure = {
        "slot": "judge/verdict@0",
        "fail_reason": "model_failed",
        "stop_reason": "length",
        "raw_response": '{"type":"responses_incomplete"}',
        "served_provider": "openai",
        "generation_id": "resp_123",
    }
    state = DebateState(
        bindings={},
        failed="judge/verdict: model_failed",
        meta={"task": {"gt": 5.0, "level": 5, "split": "dev"}, "model_failure": failure},
    )
    row = pair.state_transcript(SimpleNamespace(), state, "task-a")
    assert row["model_failure"] == failure


def test_opening_without_relaxed_number_is_protocol_invalid() -> None:
    protocol = Protocol.parse(
        {"turns": [{"alice": [{"name": "proposal", "kind": "solution"}]}]}
    )
    record = SlotRecord(slot=protocol.compile()[0], text="No numeric answer", extracted=None)
    with pytest.raises(RuntimeError, match="no relaxed extractable number"):
        pair.validate_opening_extractability(
            [DebateState(bindings={}, records=[record])], ["task-no-number"]
        )


@pytest.mark.parametrize(
    ("text", "valid", "value"),
    [
        (r"Reasoning. \boxed{5}", True, 5.0),
        ("Reasoning. \\boxed {5}\n\t", True, 5.0),
        (r"First \boxed{4}, finally \boxed{5}", False, None),
        (r"Reasoning. \boxed{5} trailing prose", False, None),
        (r"Reasoning. \boxed{5", False, None),
    ],
)
def test_strict_final_box_enforces_cardinality_wellformedness_and_end_position(
    text: str, valid: bool, value: float | None
) -> None:
    assert pair.strict_final_box(text) == (valid, value)


def test_cache_shape_rejects_duplicate_context_keys() -> None:
    with pytest.raises(RuntimeError, match="duplicate rendered-context"):
        pair.verify_cache_shape([{"key": "same"}, {"key": "same"}], expected_records=2)


def test_full_cache_gate_is_protocol_derived_and_pinned_to_708() -> None:
    exp, env = pair.build_arm_env("qwen")
    try:
        assert pair.expected_cache_records(env, 177, full_run=True) == 708
        with pytest.raises(RuntimeError, match="4 debater slots x 177 tasks = 708"):
            pair.expected_cache_records(env, 176, full_run=True)
    finally:
        env.family.close()


def test_vllm_backend_preserves_exact_tokens_through_forced_two_phase() -> None:
    class NonInvertibleTokenizer:
        eos_token_id = 0
        all_special_tokens = ["<think>", "</think>"]

        def encode(self, text, add_special_tokens=False):
            assert text == "</think>\n\n", "generated text must never be re-encoded"
            return [900, 901]

        def decode(self, tokens):
            pieces = {
                11: "<think>",
                **{token: char for token, char in zip(range(101, 111), "abcdefghij")},
                900: "</think>",
                901: "\n\n",
                201: r"answer ",
                202: r"\boxed{5}",
            }
            return "".join(pieces[token] for token in tokens)

        def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
            assert kwargs == {"enable_thinking": True}
            assert add_generation_prompt is True and tokenize is True
            return [11]

    calls = []

    def post(body):
        calls.append(body)
        if len(calls) == 1:
            tokens = list(range(101, 111))
            text = "abcdefghij"
            reason = "length"
        else:
            tokens = [201, 202]
            text = r"answer \boxed{5}"
            reason = "stop"
        return {
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": reason,
                    "prompt_token_ids": body["prompt"],
                    "token_ids": tokens,
                }
            ]
        }

    backend = pair.VLLMCompletionsBackend(
        base_url="http://unused/v1",
        model="policy",
        tokenizer_path="unused",
        workers=1,
        tokenizer=NonInvertibleTokenizer(),
        post_fn=post,
    )
    policy = Policy(
        backend,
        SamplingParams(max_tokens=None, temperature=0.0, top_p=1.0),
        chat_template_kwargs={"enable_thinking": True},
    )
    [[sample]] = policy.predict(
        [[{"role": "user", "content": "problem"}]],
        limits=SlotLimits(max_think_tokens=10, max_total_tokens=40),
    )
    assert calls[0]["stop"] == ["</think>"]
    assert calls[0]["include_stop_str_in_output"] is True
    assert calls[0]["return_token_ids"] is True
    assert calls[0]["prompt"] == [11]
    assert calls[0]["max_tokens"] == 10
    assert calls[1]["prompt"] == [11, *range(101, 111), 900, 901]
    assert calls[1]["temperature"] == 0.0
    assert sample.tokens == [*range(101, 111), 900, 901, 201, 202]
    assert sample.text.endswith(r"answer \boxed{5}")
    assert [region.kind for region in sample.regions] == ["think", "forced_close", "visible"]


def test_vllm_backend_retries_only_retryable_transport_statuses() -> None:
    calls = []
    sleeps = []

    def transient(body):
        calls.append(body)
        if len(calls) < 3:
            raise urllib.error.URLError("temporary")
        return {"ok": True}

    backend = pair.VLLMCompletionsBackend(
        base_url="http://unused/v1",
        model="policy",
        tokenizer_path="unused",
        tokenizer=SimpleNamespace(),
        post_fn=transient,
        sleep_fn=sleeps.append,
    )
    assert backend._post({"request": 1}) == {"ok": True}
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]

    bad_calls = []

    def bad_request(body):
        bad_calls.append(body)
        raise urllib.error.HTTPError(
            "http://unused/v1/completions", 400, "bad", {}, io.BytesIO(b"bad request")
        )

    backend._post_override = bad_request
    with pytest.raises(RuntimeError, match="HTTP 400"):
        backend._post({"request": 2})
    assert len(bad_calls) == 1
    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize("natural_close", [True, False])
def test_vllm_backend_preserves_natural_close_and_eos_tokens(natural_close: bool) -> None:
    class Tokenizer:
        eos_token_id = 0
        all_special_tokens = ["<think>", "</think>", "<eos>"]

        def encode(self, text, add_special_tokens=False):
            assert text == "</think>\n\n"
            return [8, 9]

        def decode(self, tokens):
            pieces = {1: "<think>", 2: "work", 3: "</think>", 0: "<eos>", 4: "answer", 8: "</think>", 9: "\n\n"}
            return "".join(pieces[token] for token in tokens)

        def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
            return [1]

    calls = []

    def post(body):
        calls.append(body)
        if len(calls) == 1:
            tokens = [2, 3] if natural_close else [2, 0]
            text = "work</think>" if natural_close else "work<eos>"
        else:
            tokens = [4]
            text = "answer"
        return {
            "choices": [{
                "index": 0,
                "text": text,
                "finish_reason": "stop",
                "prompt_token_ids": body["prompt"],
                "token_ids": tokens,
            }]
        }

    backend = pair.VLLMCompletionsBackend(
        base_url="http://unused/v1",
        model="policy",
        tokenizer_path="unused",
        workers=1,
        tokenizer=Tokenizer(),
        post_fn=post,
    )
    policy = Policy(
        backend,
        SamplingParams(max_tokens=None, temperature=0.0, top_p=1.0),
        chat_template_kwargs={"enable_thinking": True},
    )
    [[sample]] = policy.predict(
        [[{"role": "user", "content": "problem"}]],
        limits=SlotLimits(max_think_tokens=10, max_total_tokens=40),
    )
    if natural_close:
        assert len(calls) == 2
        assert sample.tokens == [2, 3, 4]
        assert [region.kind for region in sample.regions] == ["think", "visible"]
    else:
        assert len(calls) == 1
        assert sample.tokens == [2, 0]
        assert [region.kind for region in sample.regions] == ["think"]
