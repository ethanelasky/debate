from __future__ import annotations

import json
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import runpod_job_supervisor as supervisor


POD_ID = "podabc123"


def _fake_safe(
    tmp_path: Path,
    *,
    owned: bool = True,
    delete_fails: bool = False,
    vanish_after_delete: bool = False,
    expires_at: str = "2099-01-01T00:00:00Z",
    expired: bool = False,
    include_expiry: bool = True,
    include_expired: bool = True,
) -> Path:
    state = tmp_path / "deleted"
    calls = tmp_path / "safe-calls.jsonl"
    path = tmp_path / "runpod-safe"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        f"state = pathlib.Path({str(state)!r})\n"
        f"calls = pathlib.Path({str(calls)!r})\n"
        f"owned = {owned!r}\n"
        f"delete_fails = {delete_fails!r}\n"
        f"vanish_after_delete = {vanish_after_delete!r}\n"
        f"expires_at = {expires_at!r}\n"
        f"expired = {expired!r}\n"
        f"include_expiry = {include_expiry!r}\n"
        f"include_expired = {include_expired!r}\n"
        "tracked_item = {\n"
        "    'id': 'podabc123', 'allocation_exact': True,\n"
        "}\n"
        "if include_expiry:\n"
        "    tracked_item['expires_at'] = expires_at\n"
        "if include_expired:\n"
        "    tracked_item['expired'] = expired\n"
        "with calls.open('a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:] == ['audit']:\n"
        "    live = owned and not state.exists()\n"
        "    print(json.dumps({\n"
        "        'tracked_live': ([tracked_item] if live else []),\n"
        "        'untracked_live_untouched': "
        "([] if owned else [{'id': 'podabc123'}]),\n"
        "        'pending_live': [], 'deletion_pending': [],\n"
        "        'archived_owned_live': [], 'ownership_mismatches': [],\n"
        "        'stale_receipts': [], 'pending_transactions': [],\n"
        "        'malformed_state': [],\n"
        "    }))\n"
        "    raise SystemExit(0)\n"
        "if sys.argv[1:] == ['delete', 'podabc123']:\n"
        "    if delete_fails:\n"
        "        raise SystemExit(1)\n"
        "    state.touch()\n"
        "    if vanish_after_delete:\n"
        "        pathlib.Path(__file__).unlink()\n"
        "    print('{}')\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(9)\n"
    )
    path.chmod(0o700)
    return path


def _artifact_commands(tmp_path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    artifact = tmp_path / "artifact"
    manifest = tmp_path / "SHA256SUMS"
    evacuate = (
        "/bin/sh",
        "-c",
        f"printf artifact > {artifact} && "
        f"/usr/bin/shasum -a 256 {artifact} > {manifest}",
    )
    verify = (
        "/bin/sh",
        "-c",
        f"cd {tmp_path} && /usr/bin/shasum -a 256 -c {manifest}",
    )
    return evacuate, verify


def _config(
    tmp_path: Path,
    *,
    trainer: tuple[str, ...] = ("/usr/bin/true",),
    trainer_cancel: tuple[str, ...] = ("/usr/bin/true",),
    transport_cleanup: tuple[str, ...] = ("/usr/bin/true",),
    evacuate: tuple[str, ...] | None = None,
    verify: tuple[str, ...] | None = None,
    emergency: tuple[str, ...] = ("/usr/bin/true",),
    post_delete_probe: tuple[str, ...] = ("/usr/bin/false",),
    attempts: int = 2,
) -> supervisor.SupervisorConfig:
    default_evac, default_verify = _artifact_commands(tmp_path)
    return supervisor.SupervisorConfig(
        pod_id=POD_ID,
        trainer=trainer,
        trainer_cancel=trainer_cancel,
        transport_cleanup=transport_cleanup,
        evacuate=evacuate or default_evac,
        verify=verify or default_verify,
        emergency_evacuate=emergency,
        post_delete_reachability_probe=post_delete_probe,
        event_log=tmp_path / "lifecycle.jsonl",
        trainer_timeout_seconds=2,
        trainer_cancel_timeout_seconds=2,
        transport_cleanup_timeout_seconds=2,
        evacuation_timeout_seconds=2,
        verification_timeout_seconds=2,
        emergency_timeout_seconds=2,
        delete_timeout_seconds=2,
        post_delete_probe_timeout_seconds=2,
        delete_attempts=attempts,
        delete_retry_delay_seconds=0,
        cleanup_deadline_seconds=100,
    )


def _events(tmp_path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (tmp_path / "lifecycle.jsonl").read_text().splitlines()
    ]


def test_success_evacuates_verifies_then_deletes_exact_owned_id(tmp_path: Path) -> None:
    fake_safe = _fake_safe(tmp_path)

    result = supervisor.supervise(_config(tmp_path), safe_wrapper=fake_safe)

    assert result == 0
    calls = [
        json.loads(line)
        for line in (tmp_path / "safe-calls.jsonl").read_text().splitlines()
    ]
    assert calls == [["audit"], ["delete", POD_ID], ["audit"]]
    events = _events(tmp_path)
    names = [event["event"] for event in events]
    assert names.index("trainer_finished") < names.index("trainer_cancel_started")
    assert names.index("trainer_cancel_finished") < names.index("transport_cleanup_started")
    assert names.index("transport_cleanup_finished") < names.index("evacuation_started")
    assert names.index("evacuation_finished") < names.index(
        "artifact_verification_started"
    )
    assert names.index("artifact_verification_finished") < names.index(
        "safe_delete_attempt"
    )
    assert events[-1]["event"] == "final_audit"
    assert events[-1]["pod_absent"] is True
    assert events[-1]["contradictory_reachability"] is False
    assert events[-1]["transport_cleaned"] is True
    assert events[-1]["remote_cancelled"] is True
    assert "emergency_evacuation_started" not in names


def test_failed_primary_evacuation_gets_bounded_emergency_attempt_then_delete(
    tmp_path: Path,
) -> None:
    fake_safe = _fake_safe(tmp_path)
    emergency_marker = tmp_path / "emergency-ran"
    verify_marker = tmp_path / "verify-ran"
    config = _config(
        tmp_path,
        evacuate=("/usr/bin/false",),
        verify=("/usr/bin/touch", str(verify_marker)),
        emergency=("/usr/bin/touch", str(emergency_marker)),
    )

    result = supervisor.supervise(config, safe_wrapper=fake_safe)

    assert result == supervisor.EXIT_ARTIFACT_UNCERTAIN
    assert emergency_marker.exists()
    assert not verify_marker.exists()
    assert (tmp_path / "deleted").exists()
    assert _events(tmp_path)[-1]["pod_absent"] is True


def test_transport_cleanup_failure_is_recorded_but_never_pins_paid_compute(
    tmp_path: Path,
) -> None:
    fake_safe = _fake_safe(tmp_path)
    config = _config(tmp_path, transport_cleanup=("/usr/bin/false",))

    result = supervisor.supervise(config, safe_wrapper=fake_safe)

    assert result == supervisor.EXIT_INTERNAL
    assert (tmp_path / "deleted").exists()
    assert _events(tmp_path)[-1]["transport_cleaned"] is False


def test_cancel_failure_makes_artifacts_uncertain_but_still_deletes(
    tmp_path: Path,
) -> None:
    fake_safe = _fake_safe(tmp_path)
    config = _config(tmp_path, trainer_cancel=("/usr/bin/false",))

    result = supervisor.supervise(config, safe_wrapper=fake_safe)

    assert result == supervisor.EXIT_ARTIFACT_UNCERTAIN
    assert (tmp_path / "deleted").exists()
    events = _events(tmp_path)
    names = [event["event"] for event in events]
    assert names.index("trainer_cancel_finished") < names.index("evacuation_started")
    assert names.index("evacuation_finished") < names.index("safe_delete_attempt")
    assert events[-1]["remote_cancelled"] is False
    assert events[-1]["pod_absent"] is True


def test_cancel_timeout_still_evacuates_and_deletes(tmp_path: Path) -> None:
    fake_safe = _fake_safe(tmp_path)
    config = _config(tmp_path, trainer_cancel=("/bin/sleep", "10"))
    config = supervisor.SupervisorConfig(
        **{**config.__dict__, "trainer_cancel_timeout_seconds": 0.05}
    )

    result = supervisor.supervise(config, safe_wrapper=fake_safe)

    assert result == supervisor.EXIT_ARTIFACT_UNCERTAIN
    assert (tmp_path / "deleted").exists()
    cancel = next(
        event for event in _events(tmp_path) if event["event"] == "trainer_cancel_finished"
    )
    assert cancel["timed_out"] is True
    assert _events(tmp_path)[-1]["remote_cancelled"] is False


def test_trainer_timeout_runs_cancel_before_evacuation_and_returns_timeout(
    tmp_path: Path,
) -> None:
    fake_safe = _fake_safe(tmp_path)
    cancel_marker = tmp_path / "cancel-ran"
    config = _config(
        tmp_path,
        trainer=("/bin/sleep", "10"),
        trainer_cancel=("/usr/bin/touch", str(cancel_marker)),
    )
    config = supervisor.SupervisorConfig(
        **{**config.__dict__, "trainer_timeout_seconds": 0.05}
    )

    result = supervisor.supervise(config, safe_wrapper=fake_safe)

    assert result == 124
    assert cancel_marker.exists()
    assert (tmp_path / "deleted").exists()
    names = [event["event"] for event in _events(tmp_path)]
    assert names.index("trainer_finished") < names.index("trainer_cancel_started")
    assert names.index("trainer_cancel_finished") < names.index("evacuation_started")


def test_delete_unproven_takes_precedence_over_cancel_failure(tmp_path: Path) -> None:
    fake_safe = _fake_safe(tmp_path, delete_fails=True)
    config = _config(tmp_path, trainer_cancel=("/usr/bin/false",))

    result = supervisor.supervise(config, safe_wrapper=fake_safe)

    assert result == supervisor.EXIT_DELETE_UNPROVEN
    final = _events(tmp_path)[-1]
    assert final["remote_cancelled"] is False
    assert final["pod_absent"] is False


def test_timed_out_evacuation_is_killed_and_cannot_prevent_delete(
    tmp_path: Path,
) -> None:
    fake_safe = _fake_safe(tmp_path)
    emergency_marker = tmp_path / "emergency-ran"
    config = _config(
        tmp_path,
        evacuate=("/bin/sleep", "10"),
        emergency=("/usr/bin/touch", str(emergency_marker)),
    )
    config = supervisor.SupervisorConfig(
        **{**config.__dict__, "evacuation_timeout_seconds": 0.05}
    )

    result = supervisor.supervise(config, safe_wrapper=fake_safe)

    assert result == supervisor.EXIT_ARTIFACT_UNCERTAIN
    assert emergency_marker.exists()
    assert (tmp_path / "deleted").exists()
    timeout_event = next(
        event for event in _events(tmp_path) if event["event"] == "evacuation_finished"
    )
    assert timeout_event["timed_out"] is True


def test_signal_interrupts_trainer_but_cleanup_and_delete_still_run(
    tmp_path: Path,
) -> None:
    fake_safe = _fake_safe(tmp_path)
    config = _config(
        tmp_path,
        trainer=("/bin/sh", "-c", 'kill -TERM "$PPID"; sleep 10'),
    )

    result = supervisor.supervise(config, safe_wrapper=fake_safe)

    assert result == 128 + signal.SIGTERM
    assert (tmp_path / "deleted").exists()
    final = _events(tmp_path)[-1]
    assert final["deferred_signals"] == [signal.SIGTERM]
    assert final["pod_absent"] is True
    names = [event["event"] for event in _events(tmp_path)]
    assert names.index("trainer_interrupted") < names.index("trainer_cancel_started")
    assert names.index("trainer_cancel_finished") < names.index("evacuation_started")


def test_failed_delete_is_retried_and_never_claimed_absent(tmp_path: Path) -> None:
    fake_safe = _fake_safe(tmp_path, delete_fails=True)

    result = supervisor.supervise(_config(tmp_path), safe_wrapper=fake_safe)

    assert result == supervisor.EXIT_DELETE_UNPROVEN
    calls = [
        json.loads(line)
        for line in (tmp_path / "safe-calls.jsonl").read_text().splitlines()
    ]
    assert calls.count(["delete", POD_ID]) == 2
    assert _events(tmp_path)[-1]["pod_absent"] is False


def test_audit_absent_but_target_still_reachable_is_delete_unproven(
    tmp_path: Path,
) -> None:
    fake_safe = _fake_safe(tmp_path)
    config = _config(tmp_path, post_delete_probe=("/usr/bin/true",))

    result = supervisor.supervise(config, safe_wrapper=fake_safe)

    assert result == supervisor.EXIT_DELETE_UNPROVEN
    final = _events(tmp_path)[-1]
    assert final["pod_absent"] is True
    assert final["contradictory_reachability"] is True


def test_timed_out_post_delete_probe_is_delete_unproven(tmp_path: Path) -> None:
    fake_safe = _fake_safe(tmp_path)
    config = _config(tmp_path, post_delete_probe=("/bin/sleep", "10"))
    config = supervisor.SupervisorConfig(
        **{**config.__dict__, "post_delete_probe_timeout_seconds": 0.05}
    )

    result = supervisor.supervise(config, safe_wrapper=fake_safe)

    assert result == supervisor.EXIT_DELETE_UNPROVEN
    final = _events(tmp_path)[-1]
    assert final["pod_absent"] is True
    assert final["contradictory_reachability"] is False
    assert final["reachability_probe_conclusive"] is False


def test_final_audit_launch_failure_is_delete_unproven(tmp_path: Path) -> None:
    fake_safe = _fake_safe(tmp_path, vanish_after_delete=True)

    result = supervisor.supervise(_config(tmp_path), safe_wrapper=fake_safe)

    assert result == supervisor.EXIT_DELETE_UNPROVEN
    final = _events(tmp_path)[-1]
    assert final["returncode"] == 127
    assert final["pod_absent"] is False


def test_cleanup_budget_must_reserve_bounded_delete_and_probe_time(
    tmp_path: Path,
) -> None:
    fake_safe = _fake_safe(tmp_path)
    config = _config(tmp_path)
    config = supervisor.SupervisorConfig(
        **{**config.__dict__, "cleanup_deadline_seconds": 60}
    )

    with pytest.raises(supervisor.SupervisorError, match="positive evacuation time"):
        supervisor.supervise(config, safe_wrapper=fake_safe)

    assert not (tmp_path / "safe-calls.jsonl").exists()


def test_untracked_id_is_refused_before_trainer_or_delete(tmp_path: Path) -> None:
    fake_safe = _fake_safe(tmp_path, owned=False)
    trainer_marker = tmp_path / "trainer-ran"
    config = _config(
        tmp_path,
        trainer=("/usr/bin/touch", str(trainer_marker)),
    )

    with pytest.raises(supervisor.SupervisorError, match="not one exact"):
        supervisor.supervise(config, safe_wrapper=fake_safe)

    assert not trainer_marker.exists()
    calls = [
        json.loads(line)
        for line in (tmp_path / "safe-calls.jsonl").read_text().splitlines()
    ]
    assert calls == [["audit"]]


def test_expired_server_deadline_is_refused_before_trainer(tmp_path: Path) -> None:
    fake_safe = _fake_safe(tmp_path, expired=True)
    trainer_marker = tmp_path / "trainer-ran"
    config = _config(
        tmp_path,
        trainer=("/usr/bin/touch", str(trainer_marker)),
    )

    with pytest.raises(supervisor.SupervisorError, match="deadline is expired"):
        supervisor.supervise(config, safe_wrapper=fake_safe)

    assert not trainer_marker.exists()
    assert not (tmp_path / "deleted").exists()


@pytest.mark.parametrize(
    ("safe_kwargs", "message"),
    [
        ({"expires_at": "not-a-timestamp"}, "malformed server expiry"),
        ({"include_expiry": False}, "missing or invalid server expiry"),
        ({"include_expired": False}, "missing or invalid expired state"),
    ],
)
def test_malformed_or_missing_server_deadline_is_refused_before_trainer(
    tmp_path: Path,
    safe_kwargs: dict[str, object],
    message: str,
) -> None:
    fake_safe = _fake_safe(tmp_path, **safe_kwargs)
    trainer_marker = tmp_path / "trainer-ran"
    config = _config(
        tmp_path,
        trainer=("/usr/bin/touch", str(trainer_marker)),
    )

    with pytest.raises(supervisor.SupervisorError, match=message):
        supervisor.supervise(config, safe_wrapper=fake_safe)

    assert not trainer_marker.exists()
    assert not (tmp_path / "deleted").exists()


def test_insufficient_server_ttl_is_refused_before_trainer(tmp_path: Path) -> None:
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=60)
    ).isoformat().replace("+00:00", "Z")
    fake_safe = _fake_safe(tmp_path, expires_at=expires_at)
    trainer_marker = tmp_path / "trainer-ran"
    config = _config(
        tmp_path,
        trainer=("/usr/bin/touch", str(trainer_marker)),
    )

    with pytest.raises(supervisor.SupervisorError, match="does not cover"):
        supervisor.supervise(config, safe_wrapper=fake_safe)

    assert not trainer_marker.exists()
    assert not (tmp_path / "deleted").exists()


def test_command_parser_rejects_shell_resolution_and_nul() -> None:
    with pytest.raises(supervisor.SupervisorError, match="absolute"):
        supervisor._parse_command('["ssh", "host"]', label="trainer")
    with pytest.raises(supervisor.SupervisorError, match="array of strings"):
        supervisor._parse_command('["/usr/bin/ssh", "bad\\u0000arg"]', label="trainer")
