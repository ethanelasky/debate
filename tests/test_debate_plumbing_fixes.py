"""Config-plumbing fixes across the debate stack, offline:

- factory LOCAL branch forwards enable_thinking / capture_token_logprobs;
- decision slots generate under SpeechStructure.DECISION (logit channel);
- validate_scoring runs at env build when judge settings are supplied;
- shaping slots/flags are validated against the protocol and family;
- DebateEnvConfig.verdict_retries wins over judge.retries when set, defers
  when None;
- a think cap never becomes a total cap for non-thinking outputs;
- trained-seat sampling overrides carry temperature/top_p (greedy eval mode
  still wins);
- trained seats must share the adapter path and match the training backend;
- trained-seat protocol caps are cross-checked against verl response_length
  (judge-seat caps excluded).
"""

import pytest
import yaml

from infra.backend.base import SamplingParams
from infra.config import load_experiment
from infra.envs.base import Policy, SlotLimits
from infra.envs.debate.env import DebateEnv, DebateEnvConfig
from infra.envs.debate.judge import JudgeConfig
from infra.envs.debate.protocol import Protocol
from infra.envs.debate.rewards import ScoringConfig
from infra.envs.debate.round import FrozenSeat, GenRequest
from infra.envs.tasks.math import MathFamily
from infra.envs.tasks.monitoringbench import MonitoringBenchFamily
from infra.models.base import ModelSettings, SpeechStructure
from infra.models.factory import instantiate_model
from infra.models.local_model import LocalModel
from infra.run_common import build_backend
from infra.run_debate import (
    debate_gen_budgets,
    split_agents,
    validate_experiment,
    validate_trained_seats,
)

from test_budget_sampling import TOK, run as budget_run
from test_debate_env import (
    PROMPT_FILE as MATH_PROMPT_FILE,
    PROTOCOL as MATH_PROTOCOL,
    ScriptedBackend,
    ScriptedModel,
    TaskSource as MathTaskSource,
)
from test_env_extensions import GOOD_VERDICT, MBTaskSource, PROMPTS_YAML, PROTOCOL
from test_frozen_seat_sampling import KwargRecordingModel


@pytest.fixture(scope="module")
def prompt_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("prompts") / "mb_test_prompts.yaml"
    path.write_text(PROMPTS_YAML)
    return str(path)


def _seats(judge_script):
    return (
        ScriptedModel("alice", ["a"] * 4),
        ScriptedModel("bob", ["b"] * 4),
        ScriptedModel("judge", judge_script),
    )


def make_mb_env(prompt_file, alice, bob, judge_model, **cfg_kwargs):
    # the judge MODEL is positional; a `judge=JudgeConfig(...)` kwarg passes
    # through to DebateEnvConfig untouched
    cfg = dict(
        protocol=PROTOCOL,
        prompt_file=prompt_file,
        prompt_entry="mb_test",
        trained_speakers=[],
        frozen_models={"alice": alice, "bob": bob, "judge": judge_model},
        fresh_positions=False,
    )
    cfg.update(cfg_kwargs)
    return DebateEnv(DebateEnvConfig(**cfg), MBTaskSource(), MonitoringBenchFamily())


# 2 ------------------------------------------------------ factory LOCAL branch


def test_local_factory_forwards_thinking_and_logprob_capture(monkeypatch):
    captured = {}

    class FakeLocal:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("infra.models.local_model.LocalModel", FakeLocal)
    settings = ModelSettings(
        alias="j",
        model_type="local",
        model_file_path="qwen3-8b",
        enable_thinking=False,
        capture_token_logprobs=True,
    )
    instantiate_model(settings, is_debater=False, binding="train")
    assert captured["capture_token_logprobs"] is True
    assert captured["enable_thinking"] is False

    captured.clear()
    instantiate_model(
        ModelSettings(alias="j", model_type="local", model_file_path="qwen3-8b"), is_debater=False
    )
    # None = don't send (server default), mirroring the openrouter branch
    assert "enable_thinking" not in captured
    assert captured["capture_token_logprobs"] is False


@pytest.mark.parametrize("enable_thinking", [True, False, None])
def test_local_model_sends_and_copies_thinking_toggle(enable_thinking):
    model = LocalModel(
        alias="judge",
        endpoint="Qwen/Qwen3.5-4B",
        base_url="http://127.0.0.1:8788/v1",
        enable_thinking=enable_thinking,
    )
    request = model.build_request_kwargs(
        messages=[{"role": "user", "content": "problem"}],
        speech_structure=SpeechStructure.OPEN_ENDED,
        max_new_tokens=32,
        top_k=20,
    )

    if enable_thinking is None:
        assert request["extra_body"] == {"top_k": 20}
    else:
        assert request["extra_body"] == {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
    assert model.copy().enable_thinking is enable_thinking


def test_math_pc_qwen35_verl_smoke_config():
    exp = load_experiment("configs/math_pc_debate.yaml", "math_pc_qwen35_verl_smoke")
    validate_experiment(exp)
    trained, frozen = split_agents(exp)
    validate_trained_seats(trained, exp["training"])

    assert exp["prompt_config"] == {
        "file_path": "infra/prompts/debate/hendrycks_math.yaml",
        "entry": "math_proposer_critic",
    }
    assert exp["dataset"] == {
        "type": "math",
        "levels": "3-4",
        "relaxed_extraction": True,
        "seed": 0,
    }
    assert exp["scoring"]["scoring"] == "continuous"
    assert exp["scoring"]["confidence_source"] == "json"
    assert set(trained) == {"alice", "bob"}
    assert set(frozen) == {"judge"}
    assert {seat.model_file_path for seat in trained.values()} == {"Qwen/Qwen3.5-4B"}
    assert all(seat.enable_thinking is True for seat in trained.values())
    assert frozen["judge"].model_file_path == "Qwen/Qwen3.5-4B"
    assert frozen["judge"].base_url == "http://127.0.0.1:8788/v1"
    assert frozen["judge"].enable_thinking is False

    training = exp["training"]
    assert training["backend"] == "verl"
    assert training["lora_rank"] == 32
    assert {
        key: training[key]
        for key in (
            "steps",
            "batch_size",
            "group_size",
            "adv_length_norm",
            "kl_mechanism",
            "eval_every",
            "eval_n",
            "eval_max_tokens",
            "eval_split",
            "final_test_eval",
            "save_every",
        )
    } == {
        "steps": 2,
        "batch_size": 2,
        "group_size": 4,
        "adv_length_norm": "trajectory",
        "kl_mechanism": "loss",
        "eval_every": 1,
        "eval_n": 8,
        "eval_max_tokens": 3072,
        "eval_split": "dev",
        "final_test_eval": False,
        "save_every": 1,
    }
    assert training["verl"] == {
        "n_gpus": 1,
        "strategy": "fsdp2",
        "gpu_memory_utilization": 0.33,
        "prompt_length": 8192,
        "response_length": 3072,
        "max_token_len_per_gpu": 12288,
        "checkpoint_dir": "/workspace/checkpoints",
        "extra_overrides": ["++actor_rollout_ref.model.lora.merge=True"],
    }


# 3 ---------------------------------------------------- decision slot structure


def test_decision_request_generates_under_decision_structure():
    model = KwargRecordingModel("judge", ["ok", "ok"])
    seat = FrozenSeat(model)
    limits = SlotLimits(max_total_tokens=32)
    seat.generate([GenRequest(messages=[{"role": "user", "content": "u"}], limits=limits, decision=True)])
    seat.generate([GenRequest(messages=[{"role": "user", "content": "u"}], limits=limits)])
    decision_kw, speech_kw = model.calls
    assert decision_kw["speech_structure"] is SpeechStructure.DECISION
    assert "speech_structure" not in speech_kw


def test_judge_verdict_slot_requests_decision_structure_end_to_end(prompt_file):
    judge = KwargRecordingModel("judge", ["deliberating", GOOD_VERDICT])
    alice, bob, _ = _seats([])
    env = make_mb_env(prompt_file, alice, bob, judge)
    env.rollout(MBTaskSource(1).tasks(1), policy=None, group_size=1)
    assert env.last_states[0].failed is None
    delib_kw, verdict_kw = judge.calls
    assert "speech_structure" not in delib_kw
    assert verdict_kw["speech_structure"] is SpeechStructure.DECISION


# 4 ---------------------------------------------------- validate_scoring wiring


BIASED_JUDGE = dict(
    alias="judge",
    capture_token_logprobs=True,
    sampling={"eval": {"temperature": 0.7, "top_p": 1.0}},
)


def test_validate_scoring_fires_at_env_build(prompt_file):
    with pytest.raises(ValueError, match="untouched sampling distribution"):
        make_mb_env(
            prompt_file,
            *_seats([GOOD_VERDICT]),
            scoring=ScoringConfig(confidence_source="logit"),
            judge_model_settings=ModelSettings(**BIASED_JUDGE),
        )


def test_validate_scoring_passes_clean_logit_judge(prompt_file):
    make_mb_env(
        prompt_file,
        *_seats([GOOD_VERDICT]),
        scoring=ScoringConfig(confidence_source="logit"),
        judge_model_settings=ModelSettings(
            alias="judge",
            capture_token_logprobs=True,
            sampling={"eval": {"temperature": 1.0, "top_p": 1.0}},
        ),
    )


def test_validate_scoring_nop_for_json_and_absent_settings(prompt_file):
    # json source never constrains the judge's sampling
    make_mb_env(
        prompt_file,
        *_seats([GOOD_VERDICT]),
        scoring=ScoringConfig(confidence_source="json"),
        judge_model_settings=ModelSettings(**BIASED_JUDGE),
    )
    # settings not plumbed (direct config construction): check is skipped
    make_mb_env(
        prompt_file, *_seats([GOOD_VERDICT]), scoring=ScoringConfig(confidence_source="logit")
    )


# 5 --------------------------------------------------------- shaping validation


def test_shaping_slot_typo_rejected_at_build(prompt_file):
    with pytest.raises(ValueError, match=r"openning.*protocol slots"):
        make_mb_env(
            prompt_file,
            *_seats([GOOD_VERDICT]),
            scoring=ScoringConfig(
                shaping=[{"kind": "length_penalty", "coeff": 0.001, "slots": ["openning"]}]
            ),
        )


@pytest.mark.parametrize(
    "flag",
    ["nonexistent_format", "strict_boxed", "has_boxed", "code_fence", "answer_tag"],
)
def test_unknown_and_retired_shaping_flags_rejected_at_build(prompt_file, flag):
    with pytest.raises(ValueError, match=rf"{flag}.*format flags"):
        make_mb_env(
            prompt_file,
            *_seats([GOOD_VERDICT]),
            scoring=ScoringConfig(
                shaping=[
                    {
                        "kind": "format_reward",
                        "coeff": 0.1,
                        "slots": ["opening"],
                        "flag": flag,
                    }
                ]
            ),
        )


def test_valid_shaping_terms_pass(prompt_file):
    make_mb_env(
        prompt_file,
        *_seats([GOOD_VERDICT]),
        scoring=ScoringConfig(
            shaping=[
                {"kind": "length_penalty", "coeff": 0.001, "slots": ["opening", "rebuttal"]},
                {
                    "kind": "format_reward",
                    "coeff": 0.1,
                    "slots": ["opening"],
                    "flag": "answer_format_valid",
                },
                {"kind": "length_penalty", "coeff": 0.001},  # no slots = every slot; no flag
            ]
        ),
    )


# 6 -------------------------------------------------- verdict_retries precedence


def test_explicit_verdict_retries_overrides_judge_retries(prompt_file):
    # judge.retries=4 would recover from one bad attempt; explicit 0 must win
    env = make_mb_env(
        prompt_file,
        *_seats(["delib", "NOT JSON", GOOD_VERDICT]),
        judge=JudgeConfig(retries=4),
        verdict_retries=0,
    )
    env.rollout(MBTaskSource(1).tasks(1), policy=None, group_size=1)
    assert env.last_states[0].failed == "verdict_unparseable"


def test_default_verdict_retries_defers_to_judge_config(prompt_file):
    env = make_mb_env(
        prompt_file, *_seats(["delib", "NOT JSON", GOOD_VERDICT]), judge=JudgeConfig(retries=1)
    )
    env.rollout(MBTaskSource(1).tasks(1), policy=None, group_size=1)
    st = env.last_states[0]
    assert st.failed is None
    assert st.records[-1].retries == 1

    env = make_mb_env(prompt_file, *_seats(["delib", "NOT JSON"]), judge=JudgeConfig(retries=0))
    env.rollout(MBTaskSource(1).tasks(1), policy=None, group_size=1)
    assert env.last_states[0].failed == "verdict_unparseable"


# 7 --------------------------------------------------------- think-cap trap


def test_think_cap_does_not_cap_total_for_non_thinking_output():
    [s], calls = budget_run(
        [[("x" * 30, "length")], [("y" * 100, "stop")]],
        SlotLimits(max_think_tokens=30, max_total_tokens=200),
        prompt="plain prompt",
    )
    assert len(calls) == 2
    assert calls[1]["params"].max_tokens == 170  # remaining total budget
    prefix = TOK.decode(calls[1]["prompts"][0])
    assert prefix.endswith("x" * 30) and "</think>" not in prefix  # no forced close
    assert TOK.decode(s.tokens) == "x" * 30 + "y" * 100
    assert [r.kind for r in s.regions] == ["visible"]
    assert s.stop_reason == "stop"
    assert all(lp == -0.1 for lp in s.logprobs)  # every token sampled, none injected


def test_non_thinking_natural_stop_is_not_continued():
    [s], calls = budget_run(
        [[("short answer", "stop")]],
        SlotLimits(max_think_tokens=30, max_total_tokens=200),
        prompt="plain prompt",
    )
    assert len(calls) == 1
    assert [r.kind for r in s.regions] == ["visible"]


def test_visible_cap_counts_phase1_tokens_for_non_thinking_output():
    [s], calls = budget_run(
        [[("x" * 30, "length")], [("y" * 100, "length")]],
        SlotLimits(max_think_tokens=30, max_visible_tokens=80, max_total_tokens=200),
        prompt="plain prompt",
    )
    assert calls[1]["params"].max_tokens == 50  # 80 visible minus the 30 already emitted
    assert s.stop_reason == "length"
    assert [r.kind for r in s.regions] == ["visible"]


# 8 ------------------------------------------- non-lead trained seat sampling


class ParamRecordingBackend(ScriptedBackend):
    def __init__(self, script):
        super().__init__(script)
        self.params_seen: list[SamplingParams] = []

    def sample(self, prompts, params, n=1):
        self.params_seen.extend([params] * len(prompts))
        return super().sample(prompts, params, n)


def make_math_env(trained_sampling):
    return DebateEnv(
        DebateEnvConfig(
            protocol=MATH_PROTOCOL,
            prompt_file=MATH_PROMPT_FILE,
            prompt_entry="math_proposer_critic",
            trained_speakers=["alice"],
            frozen_models={
                "bob": ScriptedModel("bob", ["critique"] * 8),
                "judge": ScriptedModel("judge", [GOOD_VERDICT] * 8),
            },
            trained_sampling=trained_sampling,
            fresh_positions=True,
        ),
        MathTaskSource(),
        MathFamily(),
    )


def test_trained_seat_override_carries_temperature_and_top_p():
    backend = ParamRecordingBackend(["\\boxed{2}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128, temperature=1.0, top_p=1.0))
    env = make_math_env({"alice": SamplingParams(max_tokens=64, temperature=1.0, top_p=0.9)})
    env.rollout(MathTaskSource().tasks(1), policy, group_size=1)
    assert backend.params_seen
    assert all(p.max_tokens == 64 and p.top_p == 0.9 for p in backend.params_seen)


def test_greedy_eval_mode_wins_over_override_sampling():
    backend = ParamRecordingBackend(["\\boxed{2}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128, temperature=1.0, top_p=1.0)).greedy()
    env = make_math_env({"alice": SamplingParams(max_tokens=64, temperature=1.0, top_p=0.9)})
    env.rollout(MathTaskSource().tasks(1), policy, group_size=1)
    assert backend.params_seen
    assert all(
        p.temperature == 0.0 and p.top_p == 1.0 and p.max_tokens == 64
        for p in backend.params_seen
    )


# 9 ------------------------------------------------ trained seat vs backend


def _trained(model_type, path="Qwen/Base"):
    return ModelSettings(alias="seat", model_type=model_type, model_file_path=path)


def test_backend_model_type_consistency():
    validate_trained_seats({"a": _trained("local"), "b": _trained("local")}, {"backend": "verl"})
    validate_trained_seats({"a": _trained("tinker")}, {})  # tinker is the default backend

    with pytest.raises(ValueError, match=r"model_type 'tinker'.*'verl'"):
        validate_trained_seats({"a": _trained("tinker")}, {"backend": "verl"})
    with pytest.raises(ValueError, match=r"model_type 'local'.*'tinker'"):
        validate_trained_seats({"a": _trained("local")}, {"backend": "tinker"})


def test_trained_seats_must_share_the_adapter_path():
    with pytest.raises(ValueError, match="share one base model"):
        validate_trained_seats(
            {"a": _trained("local"), "b": _trained("local", "Other/Model")}, {"backend": "verl"}
        )


# ---------------------------------------- protocol caps vs verl response_length


CAPPED_PROTOCOL = Protocol.parse(
    yaml.safe_load(
        """
turns:
  - alice: [{name: proposal, kind: solution, max_total_tokens: 4096}]
  - bob:   [{name: critique, max_total_tokens: 3000}]
  - judge: [{name: verdict, kind: decision, max_total_tokens: 32000}]
"""
    )
)


def _verl_tr(ckpt_root, response_length, **extra):
    return {
        "backend": "verl",
        "eval_max_tokens": 800,
        "verl": {"n_gpus": 1, "checkpoint_dir": ckpt_root, "response_length": response_length},
        **extra,
    }


def test_debate_gen_budgets_cover_trained_slots_only():
    budgets = debate_gen_budgets(
        CAPPED_PROTOCOL, {"alice": _trained("local")}, {"eval_max_tokens": 800}
    )
    assert budgets["slot:proposal"] == 4096
    assert budgets["training.eval_max_tokens"] == 800
    # frozen seats (bob, judge) generate on their own servers: not checked
    assert "slot:critique" not in budgets and "slot:verdict" not in budgets


def test_trained_slot_cap_over_response_length_fails_the_build(tmp_path):
    tr = _verl_tr(str(tmp_path), response_length=2000)
    budgets = debate_gen_budgets(CAPPED_PROTOCOL, {"alice": _trained("local")}, tr)
    with pytest.raises(RuntimeError, match=r"slot:proposal=4096"):
        build_backend(tr, "some/model", "run", gen_budgets=budgets)


def test_trained_slot_caps_within_response_length_pass(tmp_path, monkeypatch):
    import infra.backend.verl as verl_mod

    monkeypatch.setattr(verl_mod, "VerlBackend", lambda config: config)
    tr = _verl_tr(str(tmp_path), response_length=4096)
    # the judge's 32000-token verdict cap must NOT trip the check: that slot
    # never touches the rollout engine
    budgets = debate_gen_budgets(CAPPED_PROTOCOL, {"alice": _trained("local")}, tr)
    build_backend(tr, "some/model", "run", gen_budgets=budgets)
