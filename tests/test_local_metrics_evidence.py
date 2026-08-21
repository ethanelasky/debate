"""Provider-free evidence for the exact training logger event stream."""

from __future__ import annotations

import fcntl
import json
import os
import stat
from pathlib import Path

import pytest

from infra.local_metrics_evidence import (
    LocalMetricsEvidence,
    scheduler_artifact_attempt_root,
)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_absent_scheduler_artifact_root_leaves_sink_disabled(tmp_path):
    assert (
        LocalMetricsEvidence.from_environment(
            "attempt-001", environ={"UNRELATED": str(tmp_path)}
        )
        is None
    )
    assert list(tmp_path.iterdir()) == []


def test_shared_attempt_root_derivation_is_exact_and_has_no_side_effect(tmp_path):
    root = tmp_path / "retained"
    root.mkdir()

    derived = scheduler_artifact_attempt_root(
        "job_018-attempt-001",
        environ={
            "DEBATE_ARTIFACT_ROOT": str(root),
            "DEBATE_LAUNCH_NAMESPACE": "job_018-attempt-001",
        },
    )

    assert derived == str(root / "job_018-attempt-001")
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("root", ["", "relative/path", "/", "//", "//."])
def test_scheduler_artifact_root_must_be_explicit_safe_absolute_directory(
    tmp_path, root
):
    with pytest.raises(ValueError, match="DEBATE_ARTIFACT_ROOT"):
        LocalMetricsEvidence.from_environment(
            "attempt-001",
            environ={
                "DEBATE_ARTIFACT_ROOT": root,
                "DEBATE_LAUNCH_NAMESPACE": "attempt-001",
            },
        )

    absent = tmp_path / "absent"
    with pytest.raises(ValueError, match="existing scheduler-owned directory"):
        LocalMetricsEvidence.from_environment(
            "attempt-001",
            environ={
                "DEBATE_ARTIFACT_ROOT": str(absent),
                "DEBATE_LAUNCH_NAMESPACE": "attempt-001",
            },
        )
    assert not absent.exists()


def test_sink_uses_exact_namespace_layout_and_fsyncs_each_record(
    tmp_path, monkeypatch
):
    root = tmp_path / "retained"
    root.mkdir()
    real_fsync = os.fsync
    fsynced_modes = []

    def recording_fsync(fd):
        fsynced_modes.append(os.fstat(fd).st_mode)
        return real_fsync(fd)

    monkeypatch.setattr("infra.local_metrics_evidence.os.fsync", recording_fsync)
    sink = LocalMetricsEvidence.from_environment(
        "job_018-attempt-001",
        environ={
            "DEBATE_ARTIFACT_ROOT": str(root),
            "DEBATE_LAUNCH_NAMESPACE": "job_018-attempt-001",
        },
    )
    assert sink is not None
    assert fcntl.fcntl(sink._file.fileno(), fcntl.F_GETFL) & os.O_APPEND
    sink.metrics(7, {"train/reward_mean": 0.25, "train/n": 8})
    sink.finalize(succeeded=True)

    expected = (
        root
        / "job_018-attempt-001"
        / "training-metrics"
        / "events.jsonl"
    )
    assert Path(sink.path) == expected
    records = _records(expected)
    assert [record["event"] for record in records] == [
        "started",
        "metrics",
        "finalized",
    ]
    assert [record["sequence"] for record in records] == [0, 1, 2]
    assert records[1]["step"] == 7
    assert records[1]["metrics"] == {
        "train/n": 8,
        "train/reward_mean": 0.25,
    }
    assert records[2]["status"] == "succeeded"
    assert all(
        record["launch_namespace"] == "job_018-attempt-001"
        for record in records
    )
    assert stat.S_IMODE(expected.stat().st_mode) == 0o600
    assert stat.S_IMODE(expected.parent.stat().st_mode) == 0o700
    assert sum(stat.S_ISREG(mode) for mode in fsynced_modes) == 3
    assert sum(stat.S_ISDIR(mode) for mode in fsynced_modes) == 3


def test_collision_refuses_without_touching_existing_evidence(tmp_path):
    root = tmp_path / "retained"
    existing = root / "attempt-001" / "training-metrics"
    existing.mkdir(parents=True)
    sentinel = existing / "events.jsonl"
    sentinel.write_bytes(b'existing evidence\n')

    with pytest.raises(FileExistsError, match="existing launch destination"):
        LocalMetricsEvidence.from_environment(
            "attempt-001",
            environ={
                "DEBATE_ARTIFACT_ROOT": str(root),
                "DEBATE_LAUNCH_NAMESPACE": "attempt-001",
            },
        )

    assert sentinel.read_bytes() == b'existing evidence\n'
    assert sorted(path.name for path in existing.iterdir()) == ["events.jsonl"]


def test_sink_shares_namespace_with_other_declared_sinks_without_touching_them(
    tmp_path,
):
    root = tmp_path / "retained"
    sibling = root / "attempt-001" / "checkpoints"
    sibling.mkdir(parents=True)
    sentinel = sibling / "manifest.json"
    sentinel.write_bytes(b'{"existing":true}\n')

    sink = LocalMetricsEvidence(str(root), "attempt-001")
    sink.finalize(succeeded=True)

    assert sentinel.read_bytes() == b'{"existing":true}\n'
    assert Path(sink.path) == (
        root / "attempt-001" / "training-metrics" / "events.jsonl"
    )


def test_artifact_root_symlink_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "retained-link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="real directory"):
        LocalMetricsEvidence.from_environment(
            "attempt-001",
            environ={
                "DEBATE_ARTIFACT_ROOT": str(link),
                "DEBATE_LAUNCH_NAMESPACE": "attempt-001",
            },
        )
    assert list(real.iterdir()) == []


def test_non_numeric_values_cannot_put_secrets_in_metric_evidence(tmp_path):
    root = tmp_path / "retained"
    root.mkdir()
    sink = LocalMetricsEvidence(str(root), "attempt-secret-guard")

    with pytest.raises(TypeError, match="only scalar numeric"):
        sink.metrics(1, {"credential": "do-not-serialize-this"})
    sink.finalize(succeeded=False)

    payload = Path(sink.path).read_text()
    assert "do-not-serialize-this" not in payload
    records = _records(Path(sink.path))
    assert [record["event"] for record in records] == ["started", "finalized"]
    assert records[-1] == {
        "event": "finalized",
        "launch_namespace": "attempt-secret-guard",
        "schema": "debate-training-metrics/v1",
        "sequence": 1,
        "status": "failed",
    }


def test_active_artifact_root_requires_exact_scheduler_namespace(tmp_path):
    root = tmp_path / "retained"
    root.mkdir()

    with pytest.raises(ValueError, match="requires scheduler-owned"):
        LocalMetricsEvidence.from_environment(
            "resolved",
            environ={"DEBATE_ARTIFACT_ROOT": str(root)},
        )
    with pytest.raises(ValueError, match="does not match"):
        LocalMetricsEvidence.from_environment(
            "resolved",
            environ={
                "DEBATE_ARTIFACT_ROOT": str(root),
                "DEBATE_LAUNCH_NAMESPACE": "different",
            },
        )
    assert list(root.iterdir()) == []


def test_group_or_world_writable_artifact_root_is_refused(tmp_path):
    root = tmp_path / "retained"
    root.mkdir(mode=0o700)
    root.chmod(0o777)

    with pytest.raises(ValueError, match="owner-controlled"):
        LocalMetricsEvidence.from_environment(
            "attempt-001",
            environ={
                "DEBATE_ARTIFACT_ROOT": str(root),
                "DEBATE_LAUNCH_NAMESPACE": "attempt-001",
            },
        )


def test_partial_append_poisons_stream_before_finalization(tmp_path, monkeypatch):
    root = tmp_path / "retained"
    root.mkdir()
    sink = LocalMetricsEvidence(str(root), "attempt-partial")
    real_write = os.write
    calls = 0

    def partial_then_fail(fd, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            prefix = data[: max(1, len(data) // 2)]
            return real_write(fd, prefix)
        raise OSError("simulated interrupted append")

    monkeypatch.setattr("infra.local_metrics_evidence.os.write", partial_then_fail)
    with pytest.raises(OSError, match="interrupted append"):
        sink.metrics(1, {"train/n": 1.0})
    after_failure = Path(sink.path).read_bytes()
    calls_after_failure = calls
    with pytest.raises(RuntimeError, match="poisoned"):
        sink.metrics(2, {"train/n": 2.0})
    assert calls == calls_after_failure
    assert Path(sink.path).read_bytes() == after_failure
    sink.finalize(succeeded=False)

    payload = Path(sink.path).read_bytes()
    assert payload.count(b'"event":"started"') == 1
    assert b'"event":"finalized"' not in payload
    assert sink._file.closed


def test_train_mirrors_existing_logger_calls_without_new_cadence(
    tmp_path, monkeypatch
):
    import infra.train as train_mod

    root = tmp_path / "retained"
    root.mkdir()
    monkeypatch.setenv("DEBATE_ARTIFACT_ROOT", str(root))
    monkeypatch.setenv("DEBATE_LAUNCH_NAMESPACE", "scheduler-attempt-004")
    seen = []

    def fake_train(env, backend, cfg, eval_env, logger):
        events = [
            (4, {"train/reward_mean": 0.5, "dev/n": 16.0}),
            (5, {"test/reward_mean": 0.625, "test/n": 32.0}),
        ]
        for step, metrics in events:
            logger(step, metrics)
            seen.append((step, metrics))

    monkeypatch.setattr(train_mod, "_train_with_logger", fake_train)
    cfg = train_mod.Config(
        log_transcripts=False,
        launch_namespace="scheduler-attempt-004",
    )
    train_mod.train(object(), object(), cfg)

    records = _records(
        root
        / "scheduler-attempt-004"
        / "training-metrics"
        / "events.jsonl"
    )
    metric_records = [record for record in records if record["event"] == "metrics"]
    assert [(record["step"], record["metrics"]) for record in metric_records] == seen
    assert records[-1]["status"] == "succeeded"


def test_train_failure_finalizes_without_serializing_exception_message(
    tmp_path, monkeypatch
):
    import infra.train as train_mod

    root = tmp_path / "retained"
    root.mkdir()
    monkeypatch.setenv("DEBATE_ARTIFACT_ROOT", str(root))
    monkeypatch.setenv("DEBATE_LAUNCH_NAMESPACE", "scheduler-attempt-failed")

    def fail(*args, **kwargs):
        raise RuntimeError("credential-like-secret-must-not-be-persisted")

    monkeypatch.setattr(train_mod, "_train_with_logger", fail)
    cfg = train_mod.Config(
        log_transcripts=False,
        launch_namespace="scheduler-attempt-failed",
    )
    with pytest.raises(RuntimeError, match="credential-like-secret"):
        train_mod.train(object(), object(), cfg)

    path = (
        root
        / "scheduler-attempt-failed"
        / "training-metrics"
        / "events.jsonl"
    )
    payload = path.read_text()
    assert "credential-like-secret" not in payload
    records = _records(path)
    assert records[-1]["event"] == "finalized"
    assert records[-1]["status"] == "failed"


def test_real_provider_free_train_loop_writes_analysis_metrics(tmp_path, monkeypatch):
    import infra.train as train_mod
    from infra.envs.base import Trajectory

    root = tmp_path / "retained"
    root.mkdir()
    monkeypatch.setenv("DEBATE_ARTIFACT_ROOT", str(root))
    monkeypatch.setenv("DEBATE_LAUNCH_NAMESPACE", "real-local-loop")

    class Env:
        last_rollout_info = {}
        last_phase_seconds = {}

        def tasks(self, n, split="train"):
            return [object()] * n

        def rollout(self, tasks, policy, group_size):
            return [
                [Trajectory(datums=[], reward=0.75, info={"correct": 1.0})]
                for _ in tasks
            ]

    class Backend:
        tokenizer = object()

        def __init__(self):
            self.saved = []

        def sync_sampler(self):
            pass

        def save(self, name):
            self.saved.append(name)
            return name

    backend = Backend()
    train_mod.train(
        Env(),
        backend,
        train_mod.Config(
            steps=1,
            batch_size=1,
            group_size=1,
            eval_every=0,
            save_every=0,
            log_transcripts=False,
            launch_namespace="real-local-loop",
        ),
    )

    records = _records(
        root
        / "real-local-loop"
        / "training-metrics"
        / "events.jsonl"
    )
    (metric_record,) = [
        record for record in records if record["event"] == "metrics"
    ]
    assert metric_record["step"] == 0
    assert metric_record["metrics"]["train/reward_mean"] == 0.75
    assert metric_record["metrics"]["train/correct"] == 1.0
    assert metric_record["metrics"]["train/n"] == 1.0
    assert records[-1]["status"] == "succeeded"
    assert backend.saved == ["final"]
