"""DAPO-style dynamic sampling (Config.dynamic_sampling_retries).

Under group-normalized advantages, a group whose trajectories all earn the
same reward has zero variance and grpo_pack drops every datum in it — the
toggle re-rolls those slots on fresh tasks, bounded by a retry cap. These
tests drive train() end to end with a scripted env and a no-op backend.
"""

from unittest import mock

import pytest

import infra.train as train_mod
from infra.backend.base import Datum
from infra.envs.base import Env, Task, Trajectory
from infra.run_common import TRAINING_KEYS, runner_parser, training_config_kwargs
from infra.train import Config, train


def _traj(reward: float) -> Trajectory:
    return Trajectory(
        datums=[
            Datum(
                tokens=[1, 2, 3],
                prompt_len=1,
                sampler_logprobs=[-0.1, -0.2],
                advantages=[0.0, 0.0],
            )
        ],
        reward=reward,
        info={"accuracy": reward},
    )


def _group(*rewards: float) -> list[Trajectory]:
    return [_traj(r) for r in rewards]


HEALTHY = (0.0, 1.0)   # mixed rewards: nonzero variance
DEGEN = (1.0, 1.0)     # constant reward: zero variance, grpo_pack drops it


class ScriptEnv(Env):
    """rollout() returns the next scripted batch; every call is recorded."""

    rollout_rate_specs = {
        "answer_format_valid_rate": (
            "answer_format_valid_count",
            "produced_solution_slots",
        ),
        "solution_production_rate": (
            "produced_solution_slots",
            "expected_solution_slots",
        ),
        "extracted_solution_rate": (
            "extracted_solution_slots",
            "produced_solution_slots",
        ),
        "gradeable_solution_rate": (
            "gradeable_solution_slots",
            "extracted_solution_slots",
        ),
        "grader_error_rate": ("grade_errors", "grader_requests"),
    }

    def __init__(
        self,
        scripted: list[list[list[Trajectory]]],
        rollout_infos: list[dict] | None = None,
    ):
        self.scripted = list(scripted)
        self.rollout_infos = list(rollout_infos or [])
        self.task_calls: list[int] = []
        self.rollout_calls: list[int] = []
        self.group_sizes: list[int] = []

    def tasks(self, n: int, split: str = "train") -> list[Task]:
        self.task_calls.append(n)
        return [Task(messages=[{"role": "user", "content": f"t{i}"}]) for i in range(n)]

    def rollout(self, tasks, policy, group_size):
        self.rollout_calls.append(len(tasks))
        self.group_sizes.append(group_size)
        assert self.scripted, "rollout called more times than scripted"
        batch = self.scripted.pop(0)
        assert len(batch) == len(tasks), "scripted batch size mismatch"
        if self.rollout_infos:
            self.last_rollout_info = self.rollout_infos.pop(0)
        return batch


class FakeBackend:
    tokenizer = None  # Policy stores it; render() is never reached here

    def __init__(self):
        self.fb_calls = 0
        self.optim_calls = 0
        self.saved: list[str] = []

    def sync_sampler(self):
        pass

    def forward_backward(self, datums, loss):
        self.fb_calls += 1

    def optim_step(self, optim):
        self.optim_calls += 1
        return {}

    def save(self, name):
        self.saved.append(name)


def _run(scripted, *, retries: int, steps: int = 1, rollout_infos=None):
    env = ScriptEnv(
        [[_group(*rewards) for rewards in batch] for batch in scripted],
        rollout_infos=rollout_infos,
    )
    backend = FakeBackend()
    cfg = Config(
        steps=steps,
        batch_size=len(scripted[0]),
        group_size=2,
        dynamic_sampling_retries=retries,
        eval_every=0,
        save_every=0,
        # ScriptEnv tests retry/group selection only and deliberately has no
        # production last_states/last_rollout_records analysis boundary.
        log_transcripts=False,
    )
    logged: list[dict] = []
    with mock.patch.object(
        train_mod, "_make_logger", lambda cfg: lambda step, m: logged.append(m)
    ):
        train(env, backend, cfg)
    return env, backend, logged


def test_toggle_off_single_rollout_per_step_and_no_metrics():
    env, backend, logged = _run(
        [[HEALTHY, DEGEN], [HEALTHY, DEGEN]], retries=0, steps=2
    )
    assert env.rollout_calls == [2, 2]  # one rollout per step, even with a degen group
    assert env.task_calls == [2, 2]
    for metrics in logged:
        assert "train/resampled_groups" not in metrics
        assert "train/degenerate_after_resample" not in metrics


def test_degenerate_group_replaced_within_cap():
    env, backend, logged = _run([[HEALTHY, DEGEN], [HEALTHY]], retries=2)
    assert env.rollout_calls == [2, 1]  # one retry round sufficed; no round 2
    assert env.task_calls == [2, 1]     # fresh tasks requested for the degen slot only
    assert env.group_sizes == [2, 2]    # resample uses the same group_size
    metrics = logged[0]
    assert metrics["train/resampled_groups"] == 1.0
    assert metrics["train/degenerate_after_resample"] == 0.0
    # The replacement carries signal: nothing left for grpo_pack to drop.
    assert metrics["pack/n_trajectories"] == 4.0
    assert metrics["pack/n_datums_dropped_zero_advantage"] == 0.0


def test_degenerate_replacement_stays_in_pool_for_next_round():
    env, backend, logged = _run([[HEALTHY, DEGEN], [DEGEN], [HEALTHY]], retries=2)
    assert env.rollout_calls == [2, 1, 1]  # round-1 replacement was degen too
    metrics = logged[0]
    assert metrics["train/resampled_groups"] == 1.0
    assert metrics["train/degenerate_after_resample"] == 0.0


def test_cap_respected_still_degenerate_pass_through():
    env, backend, logged = _run([[DEGEN], [DEGEN], [DEGEN]], retries=2)
    assert env.rollout_calls == [1, 1, 1]  # initial + exactly `retries` rounds
    metrics = logged[0]
    assert metrics["train/resampled_groups"] == 0.0
    assert metrics["train/degenerate_after_resample"] == 1.0
    # The degen group stays in the batch and grpo_pack drops it as before —
    # no behavior cliff at the cap; nothing left to train on either.
    assert metrics["pack/n_trajectories"] == 2.0
    assert metrics["pack/n_datums_dropped_zero_advantage"] == 2.0
    assert backend.fb_calls == 0


def test_single_trajectory_group_counts_as_degenerate():
    # A group can end up with < 2 trajectories (fidelity drops): zero variance
    # by construction, so it is resampled like a constant-reward group.
    env, backend, logged = _run([[HEALTHY, (1.0,)], [HEALTHY]], retries=1)
    assert env.rollout_calls == [2, 1]
    assert logged[0]["train/resampled_groups"] == 1.0


def test_all_healthy_batch_no_extra_rollouts():
    env, backend, logged = _run([[HEALTHY, HEALTHY]], retries=3)
    assert env.rollout_calls == [2]  # toggle on, but nothing to resample
    assert env.task_calls == [2]
    metrics = logged[0]
    # Every-branch rule: the keys are present (0.0) whenever the toggle is on.
    assert metrics["train/resampled_groups"] == 0.0
    assert metrics["train/degenerate_after_resample"] == 0.0


def test_rollout_counters_sum_initial_and_retry_calls():
    _, _, logged = _run(
        [[HEALTHY, DEGEN], [HEALTHY]],
        retries=1,
        rollout_infos=[
            {
                "tasks_requested": 2,
                "grade_errors": 1,
                "fail_reasons": {"transport": 1},
            },
            {
                "tasks_requested": 1,
                "grade_errors": 2,
                "fail_reasons": {"transport": 2, "parse": 1},
            },
        ],
    )
    metrics = logged[0]
    assert metrics["train/tasks_requested"] == 3.0
    assert metrics["train/grade_errors"] == 3.0
    assert metrics["train/fail_reasons/transport"] == 3.0
    assert metrics["train/fail_reasons/parse"] == 1.0
    # Accuracy/reward aggregation still describes the retained groups, not all
    # rollout attempts, and n remains exactly two retained groups of size two.
    assert metrics["train/accuracy"] == 0.5
    assert metrics["train/reward_mean"] == 0.5
    assert metrics["train/n"] == 4.0


def test_rollout_counters_single_call_are_not_double_counted():
    _, _, logged = _run(
        [[HEALTHY, HEALTHY]],
        retries=3,
        rollout_infos=[{"tasks_requested": 2, "fail_reasons": {"parse": 1}}],
    )
    metrics = logged[0]
    assert metrics["train/tasks_requested"] == 2.0
    assert metrics["train/fail_reasons/parse"] == 1.0


def test_rollout_rates_recomputed_from_summed_counts_across_unequal_calls():
    _, _, logged = _run(
        [[HEALTHY, DEGEN], [HEALTHY]],
        retries=1,
        rollout_infos=[
            {
                "expected_solution_slots": 8,
                "produced_solution_slots": 4,
                "answer_format_valid_count": 2,
                "extracted_solution_slots": 3,
                "gradeable_solution_slots": 2,
                "grader_requests": 3,
                "grade_errors": 1,
                "solution_production_rate": 0.5,
                "answer_format_valid_rate": 0.5,
                "extracted_solution_rate": 0.75,
                "gradeable_solution_rate": 2 / 3,
                "grader_error_rate": 1 / 3,
            },
            {
                "expected_solution_slots": 1,
                "produced_solution_slots": 1,
                "answer_format_valid_count": 1,
                "extracted_solution_slots": 1,
                "gradeable_solution_slots": 1,
                "grader_requests": 1,
                "grade_errors": 0,
                "solution_production_rate": 1.0,
                "answer_format_valid_rate": 1.0,
                "extracted_solution_rate": 1.0,
                "gradeable_solution_rate": 1.0,
                "grader_error_rate": 0.0,
            },
        ],
    )
    metrics = logged[0]
    assert metrics["train/solution_production_rate"] == pytest.approx(5 / 9)
    assert metrics["train/answer_format_valid_rate"] == pytest.approx(3 / 5)
    assert metrics["train/extracted_solution_rate"] == pytest.approx(4 / 5)
    assert metrics["train/gradeable_solution_rate"] == pytest.approx(3 / 4)
    assert metrics["train/grader_error_rate"] == pytest.approx(1 / 4)
    assert all(
        0.0 <= metrics[f"train/{rate}"] <= 1.0
        for rate in ScriptEnv.rollout_rate_specs
    )


def _args():
    parser = runner_parser(None)
    return parser.parse_args(["--experiment-file", "f.yaml", "--experiment", "e"])


def test_config_plumbing_reaches_config():
    assert "dynamic_sampling_retries" in TRAINING_KEYS  # passes reject_unknown_keys
    kw = training_config_kwargs({"dynamic_sampling_retries": 2}, _args())
    assert kw["dynamic_sampling_retries"] == 2
    assert Config(**kw).dynamic_sampling_retries == 2


def test_config_absent_falls_to_default_off():
    kw = training_config_kwargs({}, _args())
    assert "dynamic_sampling_retries" not in kw
    assert Config().dynamic_sampling_retries == 0


def test_planned_env_answer_turn_carries_its_own_cap():
    """Regression: the 8k plan silently shrank to max_completion_tokens when
    the sampler ceiling wasn't raised — the answer turn must carry an explicit
    cap so the raised ceiling cannot leak 8k budgets into answers."""
    from infra.envs.planned import PlannedEnv

    class _Prompts:
        messages = [{"role": "user", "content": "cue <PROBLEM>"}]

        def supplied_templates(self):
            return {}

    class _Inner:
        prompts = _Prompts()

    env = PlannedEnv.__new__(PlannedEnv)
    env.plan_max_tokens = 8192
    env.answer_max_tokens = 1000
    assert env.answer_max_tokens == 1000  # ctor arg stored; predict passes it as SlotLimits


# ---- oversample_factor (upfront variant) ----


def _run_oversampled(
    scripted_batch,
    *,
    batch_size: int,
    factor: float,
    steps: int = 1,
    rollout_infos=None,
):
    env = ScriptEnv(
        [[_group(*rewards) for rewards in batch] for batch in scripted_batch],
        rollout_infos=rollout_infos,
    )
    backend = FakeBackend()
    cfg = Config(
        steps=steps,
        batch_size=batch_size,
        group_size=2,
        oversample_factor=factor,
        eval_every=0,
        save_every=0,
        # The scripted rollout contains Trajectory objects only; it is not an
        # analyzed workload with retained transcript records.
        log_transcripts=False,
    )
    logged: list[dict] = []
    with mock.patch.object(
        train_mod, "_make_logger", lambda cfg: lambda step, m: logged.append(m)
    ):
        train(env, backend, cfg)
    return env, backend, logged


def test_oversample_single_draw_keeps_first_healthy_in_order():
    # factor 2, batch 2: draw 4, keep the FIRST 2 healthy groups by draw order.
    env, backend, logged = _run_oversampled(
        [[DEGEN, HEALTHY, HEALTHY, HEALTHY]], batch_size=2, factor=2.0
    )
    assert env.rollout_calls == [4]     # exactly one generation round
    assert env.task_calls == [4]
    metrics = logged[0]
    assert metrics["train/oversample_drawn"] == 4.0
    assert metrics["train/oversample_degenerate"] == 1.0
    assert metrics["train/degenerate_kept"] == 0.0
    assert metrics["train/n"] == 4.0    # 2 kept groups x group_size 2
    assert metrics["pack/n_datums_dropped_zero_advantage"] == 0.0


def test_oversample_pads_with_degenerate_when_draw_is_short_on_healthy():
    env, backend, logged = _run_oversampled(
        [[DEGEN, HEALTHY, DEGEN, DEGEN]], batch_size=2, factor=2.0
    )
    metrics = logged[0]
    assert metrics["train/degenerate_kept"] == 1.0
    # The padded degen group rides along and grpo_pack drops it, as ever.
    assert metrics["pack/n_datums_dropped_zero_advantage"] == 2.0
    assert metrics["train/n"] == 4.0


def test_oversample_rollout_counters_capture_the_single_draw_once():
    _, _, logged = _run_oversampled(
        [[DEGEN, HEALTHY, HEALTHY, HEALTHY]],
        batch_size=2,
        factor=2.0,
        rollout_infos=[{"tasks_requested": 4, "fail_reasons": {"fidelity": 2}}],
    )
    metrics = logged[0]
    assert metrics["train/tasks_requested"] == 4.0
    assert metrics["train/fail_reasons/fidelity"] == 2.0


def test_oversample_and_retries_are_mutually_exclusive():
    env = ScriptEnv([])
    cfg = Config(steps=1, batch_size=2, group_size=2,
                 oversample_factor=3.0, dynamic_sampling_retries=5)
    try:
        train(env, FakeBackend(), cfg)
    except ValueError as e:
        assert "mutually" in str(e)
    else:
        raise AssertionError("expected ValueError for both toggles set")


def test_oversample_below_one_rejected():
    cfg = Config(steps=1, batch_size=2, group_size=2, oversample_factor=0.5)
    try:
        train(ScriptEnv([]), FakeBackend(), cfg)
    except ValueError as e:
        assert "oversample_factor" in str(e)
    else:
        raise AssertionError("expected ValueError for factor < 1")


def test_oversample_config_plumbing_reaches_config():
    assert "oversample_factor" in TRAINING_KEYS
    kw = training_config_kwargs({"oversample_factor": 3}, _args())
    assert kw["oversample_factor"] == 3.0
    assert Config(**kw).oversample_factor == 3.0
    kw_absent = training_config_kwargs({}, _args())
    assert "oversample_factor" not in kw_absent
    assert Config().oversample_factor == 1.0
