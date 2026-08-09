"""think_tokens: native-<think> RLVR arms (the Think-DPO ablation).

What these pin, and why:

- Routing: an env carrying slot_limits must push EVERY generation — training
  rollouts and train.evaluate's greedy eval alike — through the budget-forced
  two-phase path (phase 1 sampled with stop </think> at the think cap). Both
  splits share one rollout(), so a limits leak in either direction would be
  silent: training under the cap and eval without it measures a policy in a
  mode it is never trained in.
- Default byte-identity: slot_limits None must not even change the
  policy.predict CALL — existing policy stubs accept no `limits` kwarg.
- Config validation: think_tokens + plan_tokens, think_tokens without
  enable_thinking: true, and an eval_max_tokens below think+answer must all
  fail at launch (the eval ceiling would otherwise silently SHRINK the think
  cap — Policy.predict and budget_forced_sample both take mins).
- The overshoot penalty: priced off the Sample's "forced_close" region (only
  budget_forced_sample writes it), applied ONLY when the think phase was
  force-closed at the cap, info key on every branch while active, and 0.0 (the
  default) byte-identical to plain reward().
"""

from __future__ import annotations

import pytest

from test_single_turn_env import ScriptedBackend

from infra.backend.base import Region, Sample, SamplingParams
from infra.config import load_experiment
from infra.envs.base import THINK_CLOSE, Policy, SingleTurnEnv, SlotLimits, Task
from infra.envs.tasks.aime import AimeEnv, AimeFamily
from infra.envs.tasks.math import MathEnv, MathFamily
from infra.run_rlvr import validate_experiment as validate_rlvr
from infra.train import evaluate

FORCED_CLOSE_LEN = len("</think>\n\n")  # FakeTokenizer is one token per char


# ------------------------------------------------------------ scaffolding


class RecordingBackend(ScriptedBackend):
    """ScriptedBackend that records the SamplingParams of each sample() call."""

    def __init__(self, script):
        super().__init__(script)
        self.calls = []  # (params, n)

    def sample(self, prompts, params, n=1):
        self.calls.append((params, n))
        return super().sample(prompts, params, n)


class TinyEnv(SingleTurnEnv):
    def tasks(self, n, split="train"):
        return [Task(messages=[{"role": "user", "content": f"q{i}"}]) for i in range(n)]

    def reward(self, task, text):
        return float(len(text)), {"length": float(len(text))}


def _math_env(coeff: float) -> MathEnv:
    """MathEnv without __init__ (dataset download); reward attrs set by hand,
    same offline recipe as test_tasks.py."""
    env = object.__new__(MathEnv)
    env.correct_reward = 1.0
    env.format_reward = 0.1
    env.relaxed_correct_bonus = 0.1
    env.shaped_reward = 0.0
    env.think_overshoot_penalty = coeff
    return env


def _sample(text: str, regions) -> Sample:
    toks = [ord(c) % 4096 for c in text]
    return Sample(
        tokens=toks, logprobs=[-0.1] * len(toks), text=text, stop_reason="stop", regions=regions
    )


TASK = Task(messages=[{"role": "user", "content": "q"}], meta={"gt": 2.0})
BOXED_AFTER_THINK = "<think>x</think>\n\n\\boxed{2}"
FORCED_REGIONS = (
    Region("think", 0, 8),
    Region("forced_close", 8, 18),
    Region("visible", 18, len(BOXED_AFTER_THINK)),
)
NATURAL_REGIONS = (Region("think", 0, 18), Region("visible", 18, len(BOXED_AFTER_THINK)))


# ---------------------------------------------- routing through SlotLimits


def test_train_rollout_runs_under_the_think_limits():
    env = TinyEnv()
    env.slot_limits = SlotLimits(max_think_tokens=64, max_total_tokens=96)
    backend = RecordingBackend(["plain answer", "another one"])
    groups = env.rollout(
        env.tasks(1), Policy(backend, SamplingParams(max_tokens=96)), group_size=2
    )
    # never-thinking samples end phase 1 with stop_reason "stop": one call
    ((params, n),) = backend.calls
    assert params.stop == [THINK_CLOSE]  # budget-forced phase 1, not plain sampling
    assert params.max_tokens == 64  # the think cap, not the 96 total
    assert n == 2
    assert len(groups[0]) == 2


def test_eval_path_runs_under_the_same_limits():
    env = TinyEnv()
    env.slot_limits = SlotLimits(max_think_tokens=64, max_total_tokens=96)
    backend = RecordingBackend(["plain answer"])
    evaluate(env, Policy(backend, SamplingParams(max_tokens=96, temperature=1.0)), n=1)
    ((params, _),) = backend.calls
    assert params.stop == [THINK_CLOSE] and params.max_tokens == 64
    assert params.temperature == 0.0  # greedy eval, still budget-forced


def test_no_slot_limits_is_the_plain_single_phase_call():
    env = TinyEnv()
    backend = RecordingBackend(["plain answer"])
    env.rollout(env.tasks(1), Policy(backend, SamplingParams(max_tokens=96)), group_size=1)
    ((params, _),) = backend.calls
    assert params.stop is None and params.max_tokens == 96


def test_no_slot_limits_keeps_the_predict_call_limitless():
    """Stub policies without a `limits` kwarg (the shipped codecontests test
    fake) must keep working: the default path may not pass the kwarg at all."""

    class NoLimitsPolicy:
        def predict(self, convos, n):  # no limits parameter, by design
            toks = [1, 2, 3]
            return [
                [
                    Sample(
                        tokens=toks,
                        logprobs=[0.0] * 3,
                        text="abc",
                        stop_reason="stop",
                        prompt_tokens=[0],
                    )
                    for _ in range(n)
                ]
                for _ in convos
            ]

    env = TinyEnv()
    groups = env.rollout(env.tasks(1), NoLimitsPolicy(), group_size=1)
    assert groups[0][0].reward == 3.0


def test_forced_close_flows_through_rollout_into_the_penalty():
    """End to end through the REAL Policy: the model overruns its think cap,
    budget-forced sampling injects </think>, and the forced_close region both
    prices the overshoot in reward_sample and masks the injection's tokens."""
    env = _math_env(0.1)
    env.slot_limits = SlotLimits(max_think_tokens=64, max_total_tokens=96)
    backend = RecordingBackend(["<think>pondering", "\\boxed{2}"])
    groups = env.rollout([TASK], Policy(backend, SamplingParams(max_tokens=96)), group_size=1)
    (p1, _), (p2, _) = backend.calls
    assert p1.stop == [THINK_CLOSE] and p1.max_tokens == 64
    assert p2.max_tokens == 96 - len("<think>pondering") - FORCED_CLOSE_LEN
    (traj,) = groups[0]
    assert traj.reward == pytest.approx(1.1 - 0.1)  # boxed-correct minus overshoot
    assert traj.info["think_overshoot"] == 1.0
    assert traj.info["correct"] == 1.0
    mask = traj.datums[0].mask
    assert mask is not None and mask.count(0.0) == FORCED_CLOSE_LEN  # injected close untrained


# ------------------------------------------------- experiment validation


def _think_exp(**over):
    exp = {
        "model": "allenai/Olmo-3-32B-Think-DPO",
        "enable_thinking": True,
        "think_tokens": 8192,
        "max_completion_tokens": 1000,
        "dataset": {"type": "aime", "seed": 0, "think_overshoot_penalty": 0.1},
        "training": {"steps": 2, "lr": 1e-5, "verl": {"n_gpus": 2}},
    }
    exp.update(over)
    return exp


def test_think_tokens_is_an_accepted_experiment_key():
    validate_rlvr(_think_exp())


def test_think_and_plan_tokens_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_rlvr(_think_exp(plan_tokens=1000))


@pytest.mark.parametrize("enable", ["absent", False])
def test_think_tokens_requires_enable_thinking_true(enable):
    exp = _think_exp()
    if enable == "absent":
        exp.pop("enable_thinking")
    else:
        exp["enable_thinking"] = enable
    with pytest.raises(ValueError, match="enable_thinking"):
        validate_rlvr(exp)


@pytest.mark.parametrize("value", [0, -8192])
def test_nonpositive_think_tokens_rejected(value):
    with pytest.raises(ValueError, match="think_tokens"):
        validate_rlvr(_think_exp(think_tokens=value))


def test_eval_max_tokens_below_think_plus_answer_rejected():
    exp = _think_exp()
    exp["training"]["eval_max_tokens"] = 1000  # would silently shrink the think cap
    with pytest.raises(ValueError, match="eval_max_tokens"):
        validate_rlvr(exp)
    exp["training"]["eval_max_tokens"] = 9192  # exactly think + answer: legal
    validate_rlvr(exp)


def test_shipped_think_arm_is_coherent():
    exp = load_experiment("configs/math_rlvr_olmo.yaml", "aime_rlvr_olmo32think")
    validate_rlvr(exp)
    assert exp["think_tokens"] == 8192 and exp["max_completion_tokens"] == 1000
    assert exp["enable_thinking"] is True
    assert "plan_tokens" not in exp
    assert exp["dataset"]["think_overshoot_penalty"] == 0.1
    # the engine serves think + answer as ONE request
    assert exp["training"]["verl"]["response_length"] >= 8192 + 1000
    assert "eval_max_tokens" not in exp["training"]


# --------------------------------------------------- the overshoot penalty


def test_penalty_applied_only_on_forced_close():
    env = _math_env(0.1)
    base_reward, _ = env.reward(TASK, BOXED_AFTER_THINK)

    reward, info = env.reward_sample(TASK, _sample(BOXED_AFTER_THINK, FORCED_REGIONS))
    assert reward == pytest.approx(base_reward - 0.1)
    assert info["think_overshoot"] == 1.0

    reward, info = env.reward_sample(TASK, _sample(BOXED_AFTER_THINK, NATURAL_REGIONS))
    assert reward == pytest.approx(base_reward)
    assert info["think_overshoot"] == 0.0  # key present even without overshoot

    reward, info = env.reward_sample(TASK, _sample(BOXED_AFTER_THINK, None))
    assert reward == pytest.approx(base_reward)  # single-phase sample: knob inert
    assert info["think_overshoot"] == 0.0


def test_penalty_off_by_default_is_byte_identical():
    env = _math_env(0.0)
    sample = _sample(BOXED_AFTER_THINK, FORCED_REGIONS)
    assert env.reward_sample(TASK, sample) == env.reward(TASK, BOXED_AFTER_THINK)
    _, info = env.reward_sample(TASK, sample)
    assert "think_overshoot" not in info


def test_aime_shares_the_math_reward_sample():
    assert AimeEnv.reward_sample is MathEnv.reward_sample


def test_negative_penalty_rejected_before_dataset_load():
    """Fires in __init__ BEFORE any dataset download — offline by placement."""
    with pytest.raises(ValueError, match="think_overshoot_penalty"):
        MathFamily().source({"think_overshoot_penalty": -0.1})
    with pytest.raises(ValueError, match="think_overshoot_penalty"):
        AimeFamily().source({"think_overshoot_penalty": -0.1})
