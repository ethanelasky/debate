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
from infra.envs.debate.round import (
    DebateRound,
    DebateState,
    FrozenSeat,
    SlotRecord,
    render_context,
)
from infra.models.base import Model, ModelInput, ModelResponse, SpeechStructure
from infra.models.playback_model import PlaybackModel, context_key
CONFIG = "infra/jobd/math_judge_pair_step40_config.yaml"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/math_judge_pair.py"
JOB_SPEC = Path(__file__).resolve().parents[1] / "infra/jobd/math_judge_pair_step40.yaml"
REPLAY_JOB_SPEC = (
    Path(__file__).resolve().parents[1]
    / "infra/jobd/math_judge_pair_qwen_replay.yaml"
)
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
    assert [slot.get("max_total_tokens") for slot in luna["protocol"]["turns"][4]["judge"]] == [None, None]


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
    # hf 1.28+ decorates successful stdout with a status line before the
    # snapshot path. Do not treat presentation output as path identity; the
    # pinned BASE_SNAP/config.json check immediately below is the oracle.
    assert 'hf download Qwen/Qwen3.5-4B --revision "$BASE_REV" >/dev/null' in command
    assert "DOWNLOADED=$(hf download" not in command
    assert 'test -f "$BASE_SNAP/config.json"' in command


def test_qwen_replay_job_uses_verified_cache_smoke_and_raw_transport() -> None:
    spec = yaml.safe_load(REPLAY_JOB_SPEC.read_text(encoding="utf-8"))
    job = spec["jobs"][0]
    assert spec["profile"] == "debate-b200x2"
    assert (job["gpus"], job["gpu_type"], job["cpus"]) == (
        2,
        "NVIDIA B200",
        31,
    )
    assert job["max_attempts"] == 1
    assert job["max_infra_attempts"] == 1
    assert job["env"] == ["HF_TOKEN"]
    assert job["inputs"][1]["exclude"] == []
    assert job["outputs"] == [
        {
            "remote": "/workspace/debate/outputs/math_judge_pair_qwen_replay",
            "local": "~/code/debate/outputs/math_judge_pair_qwen_replay",
        }
    ]
    command = job["command"]
    assert "SOURCE_COMMIT=76773c370616f5f20527b0f112402c02cce68259" in command
    assert "575ab0346d90c22d9715eba7182dea46c08ee01e156075e790e2b5a13988aa8e" in command
    assert "f440837265144a8193623cb05a59ed3b658efdcac5f2ae4318137d6b287a1a4d" in command
    assert "CUDA_VISIBLE_DEVICES=1" in command
    assert "--limit 1" in command
    assert command.index("--limit 1") < command.index("--out \"$OUT_ROOT/full\"")
    assert "QWEN_REPLAY_SMOKE_VERIFIED" in command
    assert "QWEN_REPLAY_FULL_VERIFIED" in command


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


def test_cache_verification_allows_replay_code_and_judge_settings_to_advance() -> None:
    seed = pair.expected_manifest_inputs("a" * 40, {"split_sha256": "dataset"})
    current = json.loads(json.dumps(seed))
    current["source_commit"] = "b" * 40
    current["config_sha256"] = "new-config"
    current["judge_generation_by_arm"]["qwen"]["transport"] = "new-transport"
    current["protocol_sha256_by_arm"]["luna"] = "uncapped-luna-protocol"

    pair.verify_cache_identity(seed, current)
    drifted = json.loads(json.dumps(current))
    drifted["prompt_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="seed cache identity prompt_sha256"):
        pair.verify_cache_identity(seed, drifted)
    drifted = json.loads(json.dumps(current))
    drifted["protocol_sha256_by_arm"]["qwen"] = "changed-seed-protocol"
    with pytest.raises(RuntimeError, match="qwen seed protocol"):
        pair.verify_cache_identity(seed, drifted)

    provenance = pair.replay_manifest(
        seed={
            **seed,
            "cache_sha256": "cache",
            "cache_records": 708,
            "ordered_task_ids_sha256": "tasks",
        },
        seed_manifest_sha256="manifest",
        current=current,
    )
    assert provenance["schema"] == pair.REPLAY_MANIFEST_SCHEMA
    assert provenance["seed"] == {
        "source_commit": "a" * 40,
        "manifest_sha256": "manifest",
        "cache_sha256": "cache",
        "cache_records": 708,
        "ordered_task_ids_sha256": "tasks",
        "config_sha256": seed["config_sha256"],
        "protocol_sha256_by_arm": seed["protocol_sha256_by_arm"],
    }
    assert provenance["replay"]["source_commit"] == "b" * 40
    assert provenance["replay"]["config_sha256"] == "new-config"
    assert provenance["verified_cache_identity"]["qwen_seed_protocol_sha256"] == (
        current["protocol_sha256_by_arm"]["qwen"]
    )


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
    assert identity["judge_generation_by_arm"]["qwen"]["transport"] == (
        "raw_vllm_chat_completions_json_v1"
    )
    assert identity["judge_generation_by_arm"]["luna"] == {
        "model": "gpt-5.6-luna",
        "reasoning_effort": "low",
        "temperature": "provider_native_not_exposed",
        "deliberation_cap": None,
        "verdict_cap": None,
        "max_output_tokens": "omitted",
        "transport": "openai_responses_uncapped_v1",
    }


def test_uncapped_responses_proxy_omits_only_max_output_tokens() -> None:
    calls = []

    class Resource:
        def create(self, **kwargs):
            calls.append(kwargs)
            return "response"

    client = pair._UncappedOpenAIClient(SimpleNamespace(responses=Resource(), marker="kept"))
    result = client.responses.create(
        model="gpt-5.6-luna",
        input=[{"role": "user", "content": "judge"}],
        max_output_tokens=512,
        reasoning={"effort": "low"},
    )

    assert result == "response"
    assert calls == [
        {
            "model": "gpt-5.6-luna",
            "input": [{"role": "user", "content": "judge"}],
            "reasoning": {"effort": "low"},
        }
    ]
    assert client.marker == "kept"


def test_raw_vllm_judge_keeps_local_request_and_response_semantics() -> None:
    calls = []
    schema = {
        "type": "object",
        "properties": {"winner": {"type": "string"}},
        "required": ["winner"],
    }

    def post(body):
        calls.append(body)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"winner":"Debater_A","confidence":1}',
                        "reasoning_content": "checked the opening",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    model = pair.RawVLLMChatModel(
        alias="raw-qwen",
        endpoint="Qwen/Qwen3.5-4B",
        base_url="http://127.0.0.1:8788/v1/",
        post_fn=post,
    )
    [response] = model.predict(
        [[ModelInput(role="user", content="judge this")]],
        max_new_tokens=256,
        speech_structure=SpeechStructure.DECISION,
        temperature=0.0,
        top_p=1.0,
        top_k=20,
        json_schema=schema,
    )
    assert calls == [
        {
            "model": "Qwen/Qwen3.5-4B",
            "messages": [{"role": "user", "content": "judge this"}],
            "max_tokens": 256,
            "logprobs": True,
            "top_logprobs": 5,
            "temperature": 0.0,
            "top_p": 1.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": schema},
            },
            "top_k": 20,
        }
    ]
    assert response.speech == '{"winner":"Debater_A","confidence":1}'
    assert response.thinking == "checked the opening"
    assert response.stop_reason == "stop"
    assert response.failed is False


def test_raw_vllm_judge_retries_only_transport_and_retryable_http_statuses() -> None:
    calls = []
    sleeps = []

    def transient(body):
        calls.append(body)
        if len(calls) < 3:
            raise urllib.error.URLError("temporary")
        return {"choices": []}

    model = pair.RawVLLMChatModel(
        alias="raw-qwen",
        endpoint="qwen",
        base_url="http://unused/v1",
        post_fn=transient,
        sleep_fn=sleeps.append,
    )
    assert model._post({"request": 1}) == {"choices": []}
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]

    bad_calls = []

    def bad_request(body):
        bad_calls.append(body)
        raise urllib.error.HTTPError(
            "http://unused/v1/chat/completions",
            400,
            "bad",
            {},
            io.BytesIO(b"bad request"),
        )

    model._post_override = bad_request
    with pytest.raises(RuntimeError, match="HTTP 400"):
        model._post({"request": 2})
    assert len(bad_calls) == 1

    model._post_override = lambda body: []
    with pytest.raises(RuntimeError, match="expected object"):
        model._post({"request": 3})


def test_qwen_arm_replaces_only_the_judge_transport() -> None:
    _, env = pair.build_arm_env("qwen")
    try:
        assert isinstance(
            env.config.frozen_models[env.judge_speaker], pair.RawVLLMChatModel
        )
        assert env.config.frozen_models[env.judge_speaker].base_url == (
            "http://127.0.0.1:8788/v1"
        )
    finally:
        env.family.close()


def test_luna_arm_omits_output_cap_on_the_real_model_path(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _, env = pair.build_arm_env("luna")
    calls = []

    class Resource:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                status="completed",
                id="response-test",
                output_text='{"winner":"Debater_A","confidence":1.0}',
                output=[],
            )

    try:
        model = env.config.frozen_models[env.judge_speaker]
        assert isinstance(model, pair.UncappedResponsesOpenAIModel)
        model.client.responses._resource = Resource()
        response = model.predict(
            [[ModelInput(role="user", content="Judge this debate.")]],
            max_new_tokens=7,
            speech_structure=SpeechStructure.DECISION,
            json_schema={
                "type": "object",
                "properties": {"winner": {"type": "string"}},
                "required": ["winner"],
                "additionalProperties": False,
            },
        )[0]
    finally:
        env.family.close()

    assert response.failed is False
    assert len(calls) == 1
    assert "max_output_tokens" not in calls[0]
    assert calls[0]["model"] == "gpt-5.6-luna"
    assert calls[0]["reasoning"] == {"effort": "low"}
    assert calls[0]["text"]["format"]["type"] == "json_schema"


def test_playback_clears_only_nonoperative_debater_template_controls(
    tmp_path: Path,
) -> None:
    protocol = Protocol.parse(
        {
            "turns": [
                {"alice": [{"name": "proposal", "kind": "solution"}]},
                {"bob": [{"name": "critique"}]},
                {"alice": [{"name": "alice_rebuttal", "enable_thinking": False}]},
                {"bob": [{"name": "bob_rebuttal", "enable_thinking": False}]},
                {
                    "judge": [
                        {"name": "deliberation", "visibility": "private"},
                        {"name": "verdict", "kind": "decision"},
                    ]
                },
            ]
        }
    )
    prompts = RenderedPrompts(
        PromptLibrary(
            system={"alice": "Alice.", "bob": "Bob.", "judge": "Judge."},
            slots={
                "proposal": "Propose.",
                "critique": "Critique.",
                "alice_rebuttal": "Reply.",
                "bob_rebuttal": "Reply.",
                "deliberation": "Think.",
                "verdict": "Return JSON.",
            },
        )
    )
    bindings = {
        speaker: {
            "NAME": speaker,
            "OPPONENT_NAME": "other",
            "TOPIC": "problem",
            "POSITION": "5",
            "OPPONENT_POSITION": "not 5",
        }
        for speaker in ("alice", "bob", "judge")
    }
    speech = {
        "proposal": r"work \boxed{5}",
        "critique": "critique",
        "alice_rebuttal": "alice reply",
        "bob_rebuttal": "bob reply",
    }
    seed_state = DebateState(bindings=bindings)
    entries = []
    original_compiled = protocol.compile()
    for slot in original_compiled[:4]:
        messages = render_context(seed_state, slot, prompts)
        entries.append({"key": context_key(messages), "speech": speech[slot.slot.name]})
        seed_state.records.append(SlotRecord(slot=slot, text=speech[slot.slot.name]))

    cache = tmp_path / "cache.jsonl"
    cache.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    alice = PlaybackModel(alias="alice", cache_path=cache)
    bob = PlaybackModel(alias="bob", cache_path=cache)
    env = SimpleNamespace(
        protocol=protocol,
        debaters=["alice", "bob"],
        config=SimpleNamespace(
            protocol=protocol,
            frozen_models={"alice": alice, "bob": bob},
        ),
    )
    original_first_judge = original_compiled[4]
    original_context = context_key(
        render_context(seed_state, original_first_judge, prompts)
    )

    assert pair.prepare_playback_protocol(env) == 2
    replay_compiled = env.protocol.compile()
    assert all(slot.slot.enable_thinking is None for slot in replay_compiled[:4])
    assert replay_compiled[4].slot == original_first_judge.slot
    assert context_key(render_context(seed_state, replay_compiled[4], prompts)) == (
        original_context
    )

    judge = _StopSequenceJudge(
        [
            ModelResponse(speech="reasoning", stop_reason="stop"),
            ModelResponse(
                speech='{"winner":"Debater_A","confidence":1}',
                stop_reason="stop",
            ),
        ]
    )
    round_ = DebateRound(
        env.protocol,
        {
            "alice": FrozenSeat(alice),
            "bob": FrozenSeat(bob),
            "judge": FrozenSeat(judge),
        },
        prompts,
        verdict_parser=_parse_test_verdict,
    )
    replay_state = DebateState(bindings=bindings)
    round_.run([replay_state])
    assert replay_state.failed is None
    assert alice.hits + bob.hits == 4


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


def test_transcript_preserves_each_judge_slot_stop_reason() -> None:
    protocol = Protocol.parse(
        {
            "turns": [
                {
                    "judge": [
                        {"name": "deliberation", "visibility": "private"},
                        {"name": "verdict", "kind": "decision"},
                    ]
                }
            ]
        }
    )
    deliberation, verdict = protocol.compile()
    state = DebateState(
        bindings={"judge": {"NAME": "Judge"}},
        records=[
            SlotRecord(
                slot=deliberation,
                text="reasoning",
                response=ModelResponse(speech="reasoning", stop_reason="length"),
            ),
            SlotRecord(
                slot=verdict,
                text='{"winner":"Debater_A","confidence":1}',
                response=ModelResponse(speech="verdict", stop_reason="stop"),
            ),
        ],
    )
    row = pair.state_transcript(SimpleNamespace(), state, "task-a")
    assert [(record["slot"], record["stop_reason"]) for record in row["records"]] == [
        ("deliberation", "length"),
        ("verdict", "stop"),
    ]


class _StopSequenceJudge(Model):
    def __init__(self, responses: list[ModelResponse]):
        super().__init__(alias="sequence-judge", is_debater=False)
        self.responses = list(responses)

    def predict(self, inputs, **kwargs):
        assert len(inputs) == 1
        return [self.responses.pop(0)]


def _parse_test_verdict(text: str):
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) and "winner" in value else None


def _run_real_judge_round(
    responses: list[ModelResponse], *, verdict_retries: int = 1
) -> tuple[DebateState, dict]:
    protocol = Protocol.parse(
        {
            "turns": [
                {
                    "judge": [
                        {"name": "deliberation", "visibility": "private"},
                        {"name": "verdict", "kind": "decision"},
                    ]
                }
            ]
        }
    )
    prompts = RenderedPrompts(
        PromptLibrary(
            system={"judge": "Judge the debate."},
            slots={"deliberation": "Think.", "verdict": "Return the verdict."},
        )
    )
    state = DebateState(
        bindings={
            "judge": {
                "NAME": "Debater_A",
                "OPPONENT_NAME": "Debater_B",
                "TOPIC": "problem",
                "POSITION": "5",
                "OPPONENT_POSITION": "not 5",
            }
        }
    )
    round_ = DebateRound(
        protocol,
        {"judge": FrozenSeat(_StopSequenceJudge(responses))},
        prompts,
        verdict_parser=_parse_test_verdict,
        verdict_retries=verdict_retries,
        judge_schema="competitive",
    )
    round_.run([state])
    return state, pair.state_transcript(SimpleNamespace(), state, "task-a")


def test_real_round_retains_discarded_capped_verdict_before_successful_retry() -> None:
    valid = '{"winner":"Debater_A","confidence":1}'
    state, row = _run_real_judge_round(
        [
            ModelResponse(speech="reasoning", stop_reason="stop"),
            ModelResponse(speech="not valid JSON", stop_reason="length"),
            ModelResponse(speech=valid, stop_reason="stop"),
        ]
    )
    assert state.failed is None
    assert state.records[-1].text == valid
    assert state.records[-1].retries == 1
    assert row["decision_attempts"] == [
        {
            "slot": "verdict",
            "speaker": "judge",
            "turn": 0,
            "kind": "decision",
            "retry_index": 0,
            "stop_reason": "length",
        },
        {
            "slot": "verdict",
            "speaker": "judge",
            "turn": 0,
            "kind": "decision",
            "retry_index": 1,
            "stop_reason": "stop",
        },
    ]
    failure = pair.native_judge_stack_failures([row])[0]
    assert failure["judge_cap_decision_attempts"][0]["retry_index"] == 0
    assert "judge_cap_slots" not in failure


def test_real_round_all_capped_verdict_attempts_survive_failed_final_ingest() -> None:
    state, row = _run_real_judge_round(
        [
            ModelResponse(speech="reasoning", stop_reason="stop"),
            ModelResponse(speech="not valid JSON", stop_reason="length"),
            ModelResponse(
                speech="",
                raw_response='{"type":"responses_incomplete"}',
                failed=True,
                stop_reason="length",
            ),
        ]
    )
    assert state.failed == "judge/verdict: model_failed"
    assert [record.slot.slot.name for record in state.records] == ["deliberation"]
    verdict_attempts = [
        attempt for attempt in row["decision_attempts"] if attempt["slot"] == "verdict"
    ]
    assert [attempt["retry_index"] for attempt in verdict_attempts] == [0, 1]
    assert [attempt["stop_reason"] for attempt in verdict_attempts] == ["length", "length"]
    failure = pair.native_judge_stack_failures([row])[0]
    assert [
        attempt["retry_index"] for attempt in failure["judge_cap_decision_attempts"]
    ] == [0, 1]


def test_real_round_exhausted_uncapped_invalid_verdict_is_capability_outcome() -> None:
    state, raw_row = _run_real_judge_round(
        [
            ModelResponse(speech="reasoning", stop_reason="stop"),
            ModelResponse(speech="not valid JSON", stop_reason="stop"),
            ModelResponse(speech="still not valid JSON", stop_reason="stop"),
        ]
    )
    assert state.failed == "verdict_unparseable"
    assert [record.slot.slot.name for record in state.records] == ["deliberation"]
    assert [attempt["stop_reason"] for attempt in raw_row["decision_attempts"]] == [
        "stop",
        "stop",
    ]
    row = pair.score_transcript(raw_row)
    assert row["judge"]["outcome"] == "unparseable"
    assert pair.native_judge_stack_failures([row]) == []
    summary = pair.summarize_arm([row])
    assert summary["outcomes"] == {"unparseable": 1}
    assert summary["n_binary_scored"] == 0


def _scored_judge_row(
    *, deliberation_stop: str = "stop", verdict_text: str, verdict_stop: str
) -> dict:
    row = _row(opening=r"Therefore, \boxed{5}", gt=5, verdict=verdict_text)
    row["records"].insert(
        -1,
        {
            "speaker": "judge",
            "kind": "speech",
            "slot": "deliberation",
            "turn": 0,
            "text": "reasoning",
            "stop_reason": deliberation_stop,
        },
    )
    row["records"][-1]["turn"] = 0
    row["records"][-1]["stop_reason"] = verdict_stop
    return pair.score_transcript(row)


def test_deliberation_cap_invalidates_native_judge_stack() -> None:
    row = _scored_judge_row(
        deliberation_stop="length",
        verdict_text='{"winner":"Debater_A","confidence":1}',
        verdict_stop="stop",
    )
    failure = pair.native_judge_stack_failures([row])[0]
    assert failure["judge_cap_slots"] == [
        {
            "slot": "deliberation",
            "turn": 0,
            "kind": "speech",
            "stop_reason": "length",
        }
    ]


@pytest.mark.parametrize(
    ("verdict_text", "expected_outcome"),
    [
        ("not valid JSON", "unparseable"),
        ('{"winner":"Debater_A","confidence":1}', "winner"),
    ],
)
def test_verdict_cap_invalidates_even_parseable_json(
    verdict_text: str, expected_outcome: str
) -> None:
    row = _scored_judge_row(verdict_text=verdict_text, verdict_stop="length")
    assert row["judge"]["outcome"] == expected_outcome
    failure = pair.native_judge_stack_failures([row])[0]
    assert [slot["slot"] for slot in failure["judge_cap_slots"]] == ["verdict"]
    with pytest.raises(RuntimeError, match="arm accuracy invalid"):
        pair.summarize_arm([row])


def test_non_length_unparseable_verdict_remains_judge_capability_outcome() -> None:
    row = _scored_judge_row(verdict_text="not valid JSON", verdict_stop="stop")
    assert row["judge"]["outcome"] == "unparseable"
    assert pair.native_judge_stack_failures([row]) == []
    summary = pair.summarize_arm([row])
    assert summary["outcomes"] == {"unparseable": 1}
    assert summary["n_binary_scored"] == 0


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
