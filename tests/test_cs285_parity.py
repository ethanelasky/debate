"""hw4-parity knobs (verify-fixes round-2 audit, 2026-08-11): warmup
schedule, population std, zero-advantage retention, min_tokens plumbing,
tempered re-anchor rebinding, and in-loss KL stamping — each tested through
the real train() loop or the real pack/advantage seams, not mirrors."""

from unittest import mock

import pytest

import infra.train as train_mod
from infra.backend.base import Datum, SamplingParams
from infra.envs.base import Env, Policy, Task, Trajectory
from infra.rl.advantages import compute_grpo_advantages
from infra.run_common import TRAINING_KEYS, training_config_kwargs, runner_parser
from infra.train import Config, train


def _traj(reward: float, lp: float = -0.1) -> Trajectory:
    return Trajectory(
        datums=[
            Datum(
                tokens=[1, 2, 3],
                prompt_len=1,
                sampler_logprobs=[lp, lp],
                advantages=[0.0, 0.0],
            )
        ],
        reward=reward,
    )


class OneStepEnv(Env):
    def tasks(self, n, split="train"):
        return [Task(messages=[{"role": "user", "content": f"t{i}"}]) for i in range(n)]

    def rollout(self, tasks, policy, group_size):
        # `last` records the ACTUAL trajectories the loop saw — assertions on
        # aliasing must use these, not a separately-constructed batch (a
        # fresh batch made the mutation assertion vacuous; round-3 audit).
        self.last = [[_traj(0.0), _traj(1.0)] for _ in tasks]
        return self.last


class RecordingBackend:
    tokenizer = None

    def __init__(self):
        self.optim_lrs: list[float] = []
        self.fb_datums: list[list[Datum]] = []
        self.ref_calls = 0

    def sync_sampler(self):
        pass

    def forward_backward(self, datums, loss):
        self.fb_datums.append(list(datums))

    def optim_step(self, optim):
        self.optim_lrs.append(optim.lr)
        return {}

    def ref_logprobs(self, datums):
        self.ref_calls += 1
        return [[-0.5] * len(d.sampler_logprobs) for d in datums]

    def forward(self, datums):
        return [[-0.25] * len(d.sampler_logprobs) for d in datums]

    def save(self, name):
        pass


def _run(cfg_kwargs, steps=3):
    env = OneStepEnv()
    backend = RecordingBackend()
    cfg = Config(steps=steps, batch_size=2, group_size=2, eval_every=0, save_every=0, **cfg_kwargs)
    logged: list[dict] = []
    with mock.patch.object(
        train_mod, "_make_logger", lambda cfg: lambda step, m: logged.append(m)
    ):
        train(env, backend, cfg)
    backend.logged = logged
    return backend


def test_warmup_scales_lr_linearly_then_holds():
    backend = _run({"lr": 1e-4, "warmup_steps": 2}, steps=3)
    assert backend.optim_lrs == pytest.approx([5e-5, 1e-4, 1e-4])
    # The applied lr is visible per step in wandb (train/lr) — the schedule
    # graph Ethan asked for (2026-08-11).
    assert [m["train/lr"] for m in backend.logged] == pytest.approx([5e-5, 1e-4, 1e-4])


def test_no_warmup_is_constant_lr():
    backend = _run({"lr": 1e-4}, steps=2)
    assert backend.optim_lrs == pytest.approx([1e-4, 1e-4])


def test_cosine_schedule_decays_from_peak_to_floor():
    backend = _run(
        {"lr": 1e-4, "warmup_steps": 2, "lr_schedule": "cosine", "min_lr_ratio": 0.2},
        steps=6,
    )
    lrs = backend.optim_lrs
    assert lrs[0] == pytest.approx(5e-5)          # warmup step 1
    assert lrs[2] == pytest.approx(1e-4)          # cosine start = peak
    assert lrs[-1] > 2e-5                          # floor reached only AT cfg.steps
    assert all(a >= b for a, b in zip(lrs[2:], lrs[3:]))  # monotone decay
    # closed-form check at the midpoint of the decay span (step 4 of [2,6)):
    assert lrs[4] == pytest.approx(1e-4 * (0.2 + 0.8 * 0.5), rel=1e-6)


def test_lr_schedule_rejects_unknown_value():
    with pytest.raises(ValueError, match="lr_schedule"):
        _run({"lr_schedule": "linear"}, steps=1)


def test_population_std_scales_advantages_by_known_ratio():
    sample = compute_grpo_advantages([0.0, 1.0], ["g", "g"])
    population = compute_grpo_advantages([0.0, 1.0], ["g", "g"], population_std=True)
    # n=2: sample std = sqrt(2)*population std -> advantages sqrt(2)x larger
    # rel 1e-4: the +epsilon in the std denominator shifts the two cases by
    # different relative amounts (~1e-6 each), so exact sqrt(2) is off at 1e-6.
    assert population[1] == pytest.approx(sample[1] * (2 ** 0.5), rel=1e-4)


def test_drop_zero_advantage_false_keeps_degenerate_rows():
    from infra.rl.datums import grpo_pack

    groups = [[_traj(1.0), _traj(1.0)]]  # zero variance: dropped by default
    dropped, _ = grpo_pack(groups)
    kept, _ = grpo_pack(groups, drop_zero_advantage=False)
    assert len(dropped) == 0 and len(kept) == 2


def test_tempered_reanchor_rebinds_not_mutates():
    env = OneStepEnv()
    backend = RecordingBackend()
    cfg = Config(
        steps=1, batch_size=2, group_size=2, eval_every=0, save_every=0,
        sampling=SamplingParams(max_tokens=8, temperature=0.8),
    )
    with mock.patch.object(train_mod, "_make_logger", lambda cfg: lambda step, m: None):
        train(env, backend, cfg)
    # The loop's datums were re-anchored (backend.forward values)...
    assert backend.fb_datums and all(
        lp == -0.25 for d in backend.fb_datums[0] for lp in d.sampler_logprobs
    )
    # ...by REBINDING the attribute on the pack-produced Datum: the rollout's
    # own trajectory lists (read by transcripts/docent) keep the sampler's
    # values. env.last is the exact object graph the loop consumed, so a
    # slice-assign mutation (sampler_logprobs[:] = lp) fails here.
    for group in env.last:
        for t in group:
            assert t.datums[0].sampler_logprobs == [-0.1, -0.1]


def test_kl_mechanism_loss_stamps_ref_and_skips_advantage_penalty():
    with mock.patch("infra.rl.kl.apply_kl_penalty") as penalty:
        backend = _run({"kl_coef": 0.05, "kl_mechanism": "loss"}, steps=1)
    assert backend.ref_calls == 1
    assert not penalty.called
    assert all(d.ref_logprobs == [-0.5, -0.5] for d in backend.fb_datums[0])


def test_kl_mechanism_advantage_is_default_path():
    backend = _run({"kl_coef": 0.05}, steps=1)
    assert backend.ref_calls >= 1  # apply_kl_penalty's ref pass
    assert all(d.ref_logprobs is None for d in backend.fb_datums[0])


def test_kl_mechanism_rejects_unknown_value():
    with pytest.raises(ValueError, match="kl_mechanism"):
        _run({"kl_mechanism": "reward"}, steps=1)


def test_parity_knobs_flow_through_training_config():
    parser = runner_parser(None)
    args = parser.parse_args(["--experiment-file", "f.yaml", "--experiment", "e"])
    for key in ("adv_population_std", "drop_zero_advantage", "kl_mechanism"):
        assert key in TRAINING_KEYS
    kw = training_config_kwargs({"adv_population_std": True, "drop_zero_advantage": False}, args)
    assert kw["adv_population_std"] is True and kw["drop_zero_advantage"] is False


def test_min_completion_tokens_reaches_sampling_and_greedy_strips_it():
    # The REAL seam this time: _sampling_params is the function main() calls,
    # so deleting the min_tokens hop in run_rlvr fails here (round-4 audit
    # mutation-proved the previous hand-built version pinned nothing).
    from infra.config import load_experiment
    from infra.run_rlvr import _sampling_params

    exp = load_experiment("configs/cs285_validate.yaml", "cs285_mathhard_grpo")
    params = _sampling_params(exp, int(exp["max_completion_tokens"]))
    assert params.max_tokens == 512  # the ceiling hop shipped the plan-clamp bug once
    assert params.min_tokens == 8 and params.temperature == 0.8 and params.top_p == 0.95
    greedy = Policy(RecordingBackend(), params, None).greedy().params
    assert greedy.temperature == 0.0 and greedy.top_p == 1.0 and greedy.min_tokens is None


def test_sampling_params_default_profile_when_keys_absent():
    # 1.0/1.0 is the debate arms' unbiased-ratio anchor, not an hw4 knob —
    # this pins the default, the only test that does.
    from infra.run_rlvr import _sampling_params

    params = _sampling_params({}, 100)
    assert params.temperature == 1.0 and params.top_p == 1.0 and params.min_tokens is None
    assert params.max_tokens == 100


def test_sampling_ceiling_covers_think_budget():
    """The sampler ceiling must be think_tokens + max_completion_tokens for
    think arms — Policy.predict min()-clamps SlotLimits against it, which is
    exactly how the 8k plan silently shrank to 1k on 2026-08-08. Pin the
    budget arithmetic main() feeds _sampling_params."""
    from infra.config import load_experiment
    from infra.run_rlvr import _sampling_params

    exp = load_experiment("configs/math_rlvr_olmo.yaml", "aime_rlvr_olmo32think_cispo")
    total = int(exp["think_tokens"]) + int(exp["max_completion_tokens"])
    params = _sampling_params(exp, total)
    # the RELATIONSHIP, not a literal: the ceiling must cover think + answer
    # (Policy.predict min()-clamps SlotLimits against it — the plan-clamp bug)
    assert params.max_tokens == total > int(exp["max_completion_tokens"])


def test_kl_mechanism_loss_rejects_non_verl_backend():
    from infra.run_common import build_backend

    with pytest.raises(RuntimeError, match="requires backend 'verl'"):
        build_backend({"backend": "tinker", "kl_mechanism": "loss", "kl_coef": 0.05}, "m", "r")


def test_cs285_debate_cispo_is_a_controlled_recipe_derivation():
    """The debate arm changes the environment, not the validated optimizer."""
    from infra.config import load_experiment
    from infra.run_debate import build_env, split_agents, validate_experiment

    path = "configs/cs285_validate.yaml"
    rlvr = load_experiment(path, "cs285_mathhard_cispo")
    debate = load_experiment(path, "cs285_mathhard_debate_cispo")
    validate_experiment(debate)

    # Choice-style blind opening: the trained proposal renders under the task
    # source's own messages, sharing the RLVR arm's prompt distribution.
    assert debate["first_speech_non_debate_aware"] is True

    same_training = {
        "lora_rank", "loss", "ppo_epochs", "micro_batch", "warmup_steps",
        "adv_length_norm", "adv_population_std", "drop_zero_advantage",
        "kl_mechanism", "steps", "batch_size", "group_size", "lr",
        "kl_coef", "kl_discount_factor", "eval_every", "eval_n",
        "eval_max_tokens", "save_every",
    }
    assert {k: debate["training"][k] for k in same_training} == {
        k: rlvr["training"][k] for k in same_training
    }

    trained, frozen = split_agents(debate)
    assert set(trained) == {"alice", "bob"}
    assert set(frozen) == {"judge"}
    assert {s.model_file_path for s in trained.values()} == {
        "Qwen/Qwen2.5-Math-1.5B-Instruct"
    }
    # Judge-debater parity (2026-08-12): the frozen judge is the same base
    # model the debaters train from.
    assert frozen["judge"].model_file_path == trained["alice"].model_file_path
    assert all(
        s.sampling.train.temperature == 1.0 and s.sampling.train.top_p == 1.0
        for s in trained.values()
    )
    verl = debate["training"]["verl"]
    assert verl["prompt_length"] + verl["response_length"] == 4096
    assert verl["max_token_len_per_gpu"] == 4096

    # Exercise the real prompt-splice/config constructor: math_hw4.yaml is
    # answer-only and cannot satisfy the debate pack; the approved math prompt
    # supplies the inspectable derivation and every required splice.
    env = build_env(debate, trained, frozen)
    trained_slots = [
        (cs.speaker, cs.slot.name, cs.slot.max_total_tokens)
        for cs in env.protocol.compile()
        if cs.speaker in trained
    ]
    assert trained_slots == [
        ("alice", "proposal", 512),
        ("bob", "critique", 300),
        ("alice", "defense", 300),
        ("bob", "rebuttal", 300),
    ]
