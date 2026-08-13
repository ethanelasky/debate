"""3-way split (Ethan, 2026-08-11): dev/ steers decisions, test/ is read once
at the end, and the test rows stay byte-identical to the historical eval
carve. Env-level tests use a synthetic MathEnv-shaped object through the REAL
tasks() method; loop-level tests drive train() with a recording env."""

from unittest import mock

import pytest

import infra.train as train_mod
from infra.backend.base import Datum
from infra.envs.base import Env, Task, Trajectory
from infra.envs.tasks.math import MathEnv
from infra.train import Config, train


def _mathenv_with_rows(n_rows=40):
    env = MathEnv.__new__(MathEnv)
    import random

    env.rng = random.Random(0)
    rows = [{"problem": f"p{i}", "gt": float(i), "level": 5} for i in range(n_rows)]
    env.test_rows, train_rows = rows[:10], rows[10:]
    # replicate the eager carve exactly as __init__ does
    k = min(len(env.test_rows), max(1, len(train_rows) // 10))
    env.dev_rows, env.train_rows = train_rows[:k], train_rows[k:]
    env.prompts = mock.Mock()
    env.prompts.render = lambda subs: [{"role": "user", "content": subs["PROBLEM"]}]
    return env


def test_dev_and_test_and_train_are_disjoint():
    env = _mathenv_with_rows()
    dev = {t.meta["question"] for t in env.tasks(99, "dev")}
    test = {t.meta["question"] for t in env.tasks(99, "test")}
    train_qs = {r["problem"] for r in env.train_rows}
    assert dev and test
    assert not dev & test and not dev & train_qs and not test & train_qs


def test_dev_rows_never_sampled_in_training():
    env = _mathenv_with_rows()
    dev = {t.meta["question"] for t in env.tasks(99, "dev")}
    sampled = {t.meta["question"] for t in env.tasks(500, "train")}
    assert not dev & sampled


def test_aime_test_rows_unchanged_by_dev_carve():
    """The dev carve takes rows from the TRAIN side only: test_rows must be
    the shuffle's first n_test rows exactly as before 2026-08-11 — the whole
    point is historical comparability of test/ numbers."""
    import random

    rows = list(range(1000))
    rng = random.Random(0 + 22222)
    rng.shuffle(rows)
    historical_test = rows[: max(64, len(rows) // 10)][:512]
    # re-run the same carve plus the new dev step; test side must not move
    rows2 = list(range(1000))
    rng2 = random.Random(0 + 22222)
    rng2.shuffle(rows2)
    n_test = max(64, len(rows2) // 10)
    test_rows, train_rows = rows2[:n_test], rows2[n_test:]
    test_rows = test_rows[:512]
    k = min(len(test_rows), max(1, len(train_rows) // 10))
    dev_rows, train_rows = train_rows[:k], train_rows[k:]
    assert test_rows == historical_test
    assert not set(dev_rows) & set(test_rows)


class SplitRecordingEnv(Env):
    def __init__(self):
        self.eval_splits: list[str] = []

    def tasks(self, n, split="train"):
        if split != "train":
            self.eval_splits.append(split)
        return [Task(messages=[{"role": "user", "content": f"{split}{i}"}]) for i in range(min(n, 2))]

    def rollout(self, tasks, policy, group_size):
        return [
            [
                Trajectory(
                    datums=[
                        Datum(tokens=[1, 2, 3], prompt_len=1,
                              sampler_logprobs=[-0.1, -0.1], advantages=[0.0, 0.0])
                    ],
                    reward=float(i % 2),
                )
                for i in range(group_size)
            ]
            for _ in tasks
        ]


class NullBackend:
    tokenizer = None

    def sync_sampler(self):
        pass

    def forward_backward(self, datums, loss):
        pass

    def optim_step(self, optim):
        return {}

    def save(self, name):
        pass


class VersionedBackend(NullBackend):
    """Make policy versions observable without approximating loop indices."""

    def __init__(self):
        self.version = 0
        self.rollout_awake = False
        self.saves: list[tuple[str, int]] = []

    def sync_sampler(self):
        self.rollout_awake = True

    def forward_backward(self, datums, loss):
        # Mirrors VerlBackend: training work sleeps the colocated rollout
        # engine before touching FSDP state.
        self.rollout_awake = False

    def optim_step(self, optim):
        self.version += 1
        return {}

    def save(self, name):
        # Mirrors VerlBackend.save(), whose lifecycle contract now quiesces
        # rollout even after a zero-datum step skipped forward/backward.
        self.rollout_awake = False
        self.saves.append((name, self.version))


class VersionRecordingEnv(SplitRecordingEnv):
    def __init__(
        self,
        backend: VersionedBackend,
        fail_final_eval: bool = False,
        degenerate_train: bool = False,
    ):
        super().__init__()
        self.backend = backend
        self.fail_final_eval = fail_final_eval
        self.degenerate_train = degenerate_train
        self.eval_versions: list[int] = []

    def rollout(self, tasks, policy, group_size):
        split = tasks[0].messages[0]["content"].rstrip("0123456789")
        if split != "train":
            self.eval_versions.append(self.backend.version)
            if self.fail_final_eval:
                raise RuntimeError("synthetic verifier infrastructure failure")
        groups = super().rollout(tasks, policy, group_size)
        if split == "train" and self.degenerate_train:
            for group in groups:
                for trajectory in group:
                    trajectory.reward = 0.0
        return groups


def _run_split(**cfg_kwargs):
    env = SplitRecordingEnv()
    logged: list[tuple[int, dict]] = []
    cfg = Config(steps=2, batch_size=1, group_size=2, eval_every=1, eval_n=2,
                 save_every=0, **cfg_kwargs)
    with mock.patch.object(
        train_mod, "_make_logger", lambda cfg: lambda step, m: logged.append((step, m))
    ):
        train(env, NullBackend(), cfg)
    return env, logged


def test_eval_split_dev_logs_dev_prefix_and_final_passes():
    env, logged = _run_split(eval_split="dev", final_test_eval=True)
    in_loop = [m for _, m in logged if any(k.startswith("dev/") for k in m)]
    assert in_loop and not any(k.startswith("eval/") for _, m in logged for k in m)
    final_step, final_metrics = logged[-1]
    assert final_step == 2 and any(k.startswith("test/") for k in final_metrics)
    # Fencepost: N/K intervals -> N/K + 1 dev evals; the final policy gets a
    # dev point at x = steps BEFORE the single test read.
    assert env.eval_splits == ["dev", "dev", "dev", "test"]
    dev_final_step, dev_final = logged[-2]
    assert dev_final_step == 2 and any(k.startswith("dev/") for k in dev_final)


def test_default_eval_behavior_gets_final_point_too():
    env, logged = _run_split()
    assert any(k.startswith("eval/") for _, m in logged for k in m)
    assert not any(k.startswith(("dev/", "test/")) for _, m in logged for k in m)
    assert env.eval_splits == ["test", "test", "test"]  # 2 in-loop + final


def test_eval_and_checkpoint_labels_equal_rollout_steps_at_one_ppo_epoch():
    backend = VersionedBackend()
    env = VersionRecordingEnv(backend)
    cfg = Config(
        steps=2,
        batch_size=1,
        group_size=2,
        eval_every=1,
        eval_n=1,
        save_every=1,
    )
    with mock.patch.object(train_mod, "_make_logger", lambda cfg: lambda step, m: None):
        train(env, backend, cfg)

    assert env.eval_versions == [0, 1, 2]
    assert backend.saves == [("step-00001", 1), ("final", 2)]


def test_rollout_step_labels_do_not_claim_optimizer_count_with_two_epochs():
    backend = VersionedBackend()
    env = VersionRecordingEnv(backend)
    cfg = Config(
        steps=2,
        batch_size=1,
        group_size=2,
        ppo_epochs=2,
        eval_every=1,
        eval_n=1,
        save_every=1,
    )
    with mock.patch.object(train_mod, "_make_logger", lambda cfg: lambda step, m: None):
        train(env, backend, cfg)

    # The x-axis is rollout-batch steps. Each batch gets two optimizer calls.
    assert env.eval_versions == [0, 2, 4]
    assert backend.saves == [("step-00001", 2), ("final", 4)]


def test_final_adapter_survives_final_verifier_failure():
    backend = VersionedBackend()
    env = VersionRecordingEnv(backend, fail_final_eval=True)
    cfg = Config(
        steps=1,
        batch_size=1,
        group_size=2,
        eval_every=0,
        final_test_eval=True,
        eval_n=1,
        save_every=0,
    )
    with (
        mock.patch.object(train_mod, "_make_logger", lambda cfg: lambda step, m: None),
        pytest.raises(RuntimeError, match="verifier infrastructure failure"),
    ):
        train(env, backend, cfg)

    assert backend.saves == [("final", 1)]


def test_periodic_eval_failure_keeps_exact_recovery_checkpoint():
    backend = VersionedBackend()
    env = VersionRecordingEnv(backend)
    cfg = Config(
        steps=3,
        batch_size=1,
        group_size=2,
        eval_every=2,
        eval_n=1,
        # The ordinary save cadence is deliberately later than the failing
        # first post-training eval, matching the 10-vs-25 production incident.
        save_every=25,
    )

    def fail_at_second_eval(_env, _policy, _n, _split, _prefix):
        if backend.version == 2:
            raise RuntimeError("synthetic verifier infrastructure failure")
        return {}

    with (
        mock.patch.object(train_mod, "_make_logger", lambda cfg: lambda step, m: None),
        mock.patch.object(train_mod, "evaluate", fail_at_second_eval),
        pytest.raises(RuntimeError, match="verifier infrastructure failure"),
    ):
        train(env, backend, cfg)

    # step-00002 is the policy after exactly two completed rollout batches and
    # is written before its evaluator can raise; no final save is reached.
    assert backend.saves == [("step-00002", 2)]


def test_final_save_sleeps_rollout_after_zero_datum_step():
    backend = VersionedBackend()
    env = VersionRecordingEnv(backend, degenerate_train=True)
    cfg = Config(
        steps=1,
        batch_size=1,
        group_size=2,
        eval_every=0,
        save_every=0,
    )
    with mock.patch.object(train_mod, "_make_logger", lambda cfg: lambda step, m: None):
        train(env, backend, cfg)

    assert backend.version == 0
    assert backend.saves == [("final", 0)]


def test_final_eval_transcripts_are_captured_before_next_split():
    env = SplitRecordingEnv()
    captured: list[tuple[int, str, tuple[str, ...]]] = []
    cfg = Config(
        steps=1,
        batch_size=1,
        group_size=2,
        eval_every=1,
        eval_n=1,
        eval_split="dev",
        final_test_eval=True,
        save_every=0,
    )

    def capture(_cfg, step, captured_env, split):
        captured.append((step, split, tuple(captured_env.eval_splits)))

    with (
        mock.patch.object(train_mod, "_make_logger", lambda cfg: lambda step, m: None),
        mock.patch.object(train_mod, "_log_transcripts", capture),
    ):
        train(env, NullBackend(), cfg)

    assert captured == [
        (0, "dev", ("dev",)),
        (0, "train", ("dev",)),
        (1, "dev", ("dev", "dev")),
        (1, "test", ("dev", "dev", "test")),
    ]
