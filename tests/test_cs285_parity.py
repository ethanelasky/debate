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
    with mock.patch.object(train_mod, "_make_logger", lambda cfg: lambda step, m: None):
        train(env, backend, cfg)
    return backend


def test_warmup_scales_lr_linearly_then_holds():
    backend = _run({"lr": 1e-4, "warmup_steps": 2}, steps=3)
    assert backend.optim_lrs == pytest.approx([5e-5, 1e-4, 1e-4])


def test_no_warmup_is_constant_lr():
    backend = _run({"lr": 1e-4}, steps=2)
    assert backend.optim_lrs == pytest.approx([1e-4, 1e-4])


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
    assert params.min_tokens == 8 and params.temperature == 0.8 and params.top_p == 0.95
    greedy = Policy(RecordingBackend(), params, None).greedy().params
    assert greedy.temperature == 0.0 and greedy.top_p == 1.0 and greedy.min_tokens is None


def test_sampling_params_default_profile_when_keys_absent():
    from infra.run_rlvr import _sampling_params

    params = _sampling_params({"max_completion_tokens": 100}, 100)
    assert params.temperature == 1.0 and params.top_p == 1.0 and params.min_tokens is None


def test_kl_mechanism_loss_rejects_non_verl_backend():
    from infra.run_common import build_backend

    with pytest.raises(RuntimeError, match="requires backend 'verl'"):
        build_backend({"backend": "tinker", "kl_mechanism": "loss", "kl_coef": 0.05}, "m", "r")
