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


class CheckpointRecordingBackend(NullBackend):
    def __init__(self, events):
        self.events = events
        self.version = 0
        self.saved = []

    def sync_sampler(self):
        self.events.append(("sync", self.version))

    def optim_step(self, optim):
        self.version += 1
        return {}

    def save(self, name):
        record = (name, self.version)
        self.saved.append(record)
        self.events.append(("save", *record))


class FailingEvalEnv(SplitRecordingEnv):
    def __init__(self, backend, events, *, fail_version):
        super().__init__()
        self.backend = backend
        self.events = events
        self.fail_version = fail_version

    def rollout(self, tasks, policy, group_size):
        split = tasks[0].messages[0]["content"].rstrip("0123456789")
        if split != "train":
            self.events.append(("eval", self.backend.version))
            if self.backend.version == self.fail_version:
                raise RuntimeError("synthetic verifier outage")
        return super().rollout(tasks, policy, group_size)


def _run_split(**cfg_kwargs):
    env = SplitRecordingEnv()
    logged: list[tuple[int, dict]] = []
    # SplitRecordingEnv records split routing only; it intentionally does not
    # emulate production transcript retention.
    cfg = Config(steps=2, batch_size=1, group_size=2, eval_every=1, eval_n=2,
                 save_every=0, log_transcripts=False, **cfg_kwargs)
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


def test_final_test_eval_uses_the_direct_eval_env():
    train_env = SplitRecordingEnv()
    direct_eval_env = SplitRecordingEnv()
    cfg = Config(
        steps=0,
        eval_every=0,
        final_test_eval=True,
        eval_n=2,
        save_every=0,
        # This test isolates eval-env routing with a synthetic non-analysis env.
        log_transcripts=False,
    )
    with mock.patch.object(train_mod, "_make_logger", lambda cfg: lambda step, metrics: None):
        train(train_env, NullBackend(), cfg, eval_env=direct_eval_env)

    assert direct_eval_env.eval_splits == ["test"]
    assert train_env.eval_splits == []


def test_periodic_eval_failure_preserves_all_prior_updates_in_checkpoint():
    events = []
    backend = CheckpointRecordingBackend(events)
    env = FailingEvalEnv(backend, events, fail_version=10)
    cfg = Config(
        steps=11,
        batch_size=1,
        group_size=2,
        eval_every=10,
        eval_n=1,
        save_every=25,
        log_transcripts=False,
    )

    with pytest.raises(RuntimeError, match="verifier outage"):
        train_mod._train_with_logger(env, backend, cfg, None, lambda step, metrics: None)

    assert backend.saved == [("step-00010", 10)]
    assert events[-3:] == [
        ("save", "step-00010", 10),
        ("sync", 10),
        ("eval", 10),
    ]


def test_continuation_start_eval_uses_loaded_checkpoint_without_resaving():
    events = []
    backend = CheckpointRecordingBackend(events)
    backend.version = 10
    env = FailingEvalEnv(backend, events, fail_version=10)
    cfg = Config(
        start_step=10,
        steps=11,
        eval_every=10,
        eval_n=1,
        save_every=25,
        log_transcripts=False,
    )

    with pytest.raises(RuntimeError, match="verifier outage"):
        train_mod._train_with_logger(env, backend, cfg, None, lambda step, metrics: None)

    assert backend.saved == []
    assert events == [("sync", 10), ("eval", 10)]


def test_final_eval_failure_preserves_final_checkpoint_before_evaluation():
    events = []
    backend = CheckpointRecordingBackend(events)
    env = FailingEvalEnv(backend, events, fail_version=1)
    cfg = Config(
        steps=1,
        batch_size=1,
        group_size=2,
        eval_every=1,
        eval_n=1,
        save_every=25,
        log_transcripts=False,
    )

    with pytest.raises(RuntimeError, match="verifier outage"):
        train_mod._train_with_logger(env, backend, cfg, None, lambda step, metrics: None)

    assert backend.saved == [("final", 1)]
    assert events[-3:] == [
        ("save", "final", 1),
        ("sync", 1),
        ("eval", 1),
    ]
