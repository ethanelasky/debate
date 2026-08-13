from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

import pytest

from scripts import runpod_remote_job as remote_job


ROOT_LINUX_REMOTE = (
    Path("/proc/self/stat").is_file()
    and Path("/usr/bin/python3").is_file()
    and os.geteuid() == 0
)
ROOT_LINUX_PIDFD_REMOTE = ROOT_LINUX_REMOTE and hasattr(os, "pidfd_open")


def _config(tmp_path: Path) -> remote_job.JobConfig:
    executable = tmp_path / "ssh"
    executable.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    executable.chmod(0o700)
    identity = tmp_path / "id"
    known_hosts = tmp_path / "known_hosts"
    identity.write_text("test", encoding="utf-8")
    known_hosts.write_text("test", encoding="utf-8")
    identity.chmod(0o600)
    return remote_job.JobConfig(
        job_id="test-job",
        ssh_argv=(
            str(executable),
            "-T",
            "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={known_hosts}",
            "-o", "ConnectTimeout=5",
            "-o", "ConnectionAttempts=1",
            "-o", "ControlMaster=no",
            "-o", "ControlPath=none",
            "-o", "RequestTTY=no",
            "-i", str(identity),
            "root@example.test",
        ),
        remote_state_dir="/workspace/run-control/test-job",
        remote_launch_script="/root/run-control/test-job/launch.sh",
        launch_script_sha256="0" * 64,
        poll_interval_seconds=1,
        outage_timeout_seconds=10,
        ssh_command_timeout_seconds=10,
        initialization_timeout_seconds=5,
        cancel_timeout_seconds=2,
    )


def test_load_config_requires_strict_ssh_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    body = {
        "format": remote_job.CONFIG_FORMAT,
        "job_id": config.job_id,
        "ssh_argv": list(config.ssh_argv),
        "remote_state_dir": config.remote_state_dir,
        "remote_launch_script": config.remote_launch_script,
        "launch_script_sha256": config.launch_script_sha256,
        "poll_interval_seconds": config.poll_interval_seconds,
        "outage_timeout_seconds": config.outage_timeout_seconds,
        "ssh_command_timeout_seconds": config.ssh_command_timeout_seconds,
        "initialization_timeout_seconds": config.initialization_timeout_seconds,
        "cancel_timeout_seconds": config.cancel_timeout_seconds,
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    path.chmod(0o600)
    assert remote_job.load_config(path) == config

    body["ssh_argv"][body["ssh_argv"].index("StrictHostKeyChecking=yes")] = (
        "StrictHostKeyChecking=no"
    )
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(remote_job.ConfigError, match="StrictHostKeyChecking=yes"):
        remote_job.load_config(path)


@pytest.mark.parametrize("unsafe", ["relative", "/", "//root/job", "/root/../job", "/root//job"])
def test_remote_paths_must_be_normalized_absolute(unsafe: str) -> None:
    with pytest.raises(remote_job.ConfigError, match="normalized absolute"):
        remote_job._remote_path(unsafe, "remote path")


def test_retries_uncertain_start_and_poll_without_local_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []
    answers: list[object] = [
        remote_job.TransportError("lost start response"),
        {"ok": True, "state": "running", "start_disposition": "existing"},
        remote_job.TransportError("wifi down"),
        {"ok": True, "state": "completed", "returncode": 7},
    ]

    def request(_config: remote_job.JobConfig, operation: str) -> dict:
        calls.append(operation)
        answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        assert isinstance(answer, dict)
        return answer

    monkeypatch.setattr(remote_job, "_remote_request", request)
    monkeypatch.setattr(remote_job.time, "sleep", lambda _seconds: None)
    assert remote_job.run(config, mode="start-and-monitor") == 7
    # A lost response retries the idempotent remote start operation.  Once the
    # remote says the atomic state exists, all subsequent requests are status.
    assert calls == ["start", "start", "status", "status"]


def test_continuous_transport_outage_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config = remote_job.JobConfig(**{**config.__dict__, "outage_timeout_seconds": 2})
    clock = [0.0]

    def fail(_config: remote_job.JobConfig, _operation: str) -> dict:
        raise remote_job.TransportError("offline")

    monkeypatch.setattr(remote_job, "_remote_request", fail)
    monkeypatch.setattr(remote_job.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(remote_job.time, "monotonic_ns", lambda: int(clock[0] * 1e9))
    monkeypatch.setattr(remote_job.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    assert remote_job.run(config, mode="monitor") == remote_job.EXIT_TRANSPORT_OUTAGE


def test_monitor_only_never_starts_missing_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def request(_config: remote_job.JobConfig, operation: str) -> dict:
        calls.append(operation)
        return {"ok": True, "state": "missing"}

    monkeypatch.setattr(remote_job, "_remote_request", request)
    assert remote_job.run(config, mode="monitor") == remote_job.EXIT_REMOTE_PROTOCOL
    assert calls == ["status"]


def test_dead_pid_without_rc_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        remote_job,
        "_remote_request",
        lambda _config, _operation: {
            "ok": True,
            "state": "dead_no_rc",
            "orphan_group_terminated": True,
        },
    )
    assert remote_job.run(config, mode="monitor") == remote_job.EXIT_DEAD_WITHOUT_RC


def _direct_remote(payload: dict[str, str]) -> dict:
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).decode("ascii")
    result = subprocess.run(
        ["/usr/bin/python3", "-B", "-", encoded],
        input=remote_job.REMOTE_CONTROLLER_SOURCE.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=10,
    )
    return remote_job._parse_response(result.stdout)


def _start_direct_remote(payload: dict[str, object]) -> subprocess.Popen[bytes]:
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).decode("ascii")
    return subprocess.Popen(
        ["/usr/bin/python3", "-B", "-", encoded],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _live_group_members(pgid: int, session: int) -> list[int]:
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            body = (entry / "stat").read_text(encoding="utf-8")
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
        close = body.rfind(")")
        fields = body[close + 2 :].split()
        if close >= 0 and len(fields) >= 20 and fields[0] != "Z":
            if int(fields[2]) == pgid and int(fields[3]) == session:
                members.append(int(entry.name))
    return members


@pytest.mark.skipif(
    not ROOT_LINUX_REMOTE,
    reason="remote controller integration requires root Linux /proc and /usr/bin/python3",
)
def test_remote_atomic_single_start_rc_and_digest_freeze(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    counter = tmp_path / "counter"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"printf x >> {shlex.quote(str(counter))}\n"
        "sleep 0.2\n"
        "exit 7\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(launch.read_bytes()).hexdigest()
    payload = {
        "operation": "start",
        "job_id": "atomic-test",
        "state_dir": str(tmp_path / "state"),
        "launch_script": str(launch),
        "launch_sha256": digest,
        "cancel_timeout_seconds": 2,
    }
    first = _direct_remote(payload)
    second = _direct_remote(payload)
    assert first["ok"] and first["start_disposition"] == "started"
    assert second["ok"] and second["start_disposition"] == "existing"

    payload["operation"] = "status"
    deadline = time.monotonic() + 5
    while True:
        status = _direct_remote(payload)
        if status.get("state") == "completed":
            break
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert status["returncode"] == 7
    assert counter.read_text(encoding="utf-8") == "x"
    assert not Path("/root/palaestra_remote_job_busy.json").exists()

    launch.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    changed = _direct_remote(payload)
    assert changed["ok"] is False
    assert "SHA-256 mismatch" in changed["message"]


@pytest.mark.skipif(
    not ROOT_LINUX_REMOTE,
    reason="remote controller integration requires root Linux /proc and /usr/bin/python3",
)
def test_remote_killed_wrapper_reports_dead_without_rc(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    launch.write_text("#!/usr/bin/env bash\nsleep 60\n", encoding="utf-8")
    payload = {
        "operation": "start",
        "job_id": "dead-test",
        "state_dir": str(tmp_path / "state"),
        "launch_script": str(launch),
        "launch_sha256": hashlib.sha256(launch.read_bytes()).hexdigest(),
        "cancel_timeout_seconds": 2,
    }
    started = _direct_remote(payload)
    pid = int(started["pid"])
    os.kill(pid, signal.SIGKILL)
    payload["operation"] = "status"
    deadline = time.monotonic() + 5
    while True:
        status = _direct_remote(payload)
        if status.get("state") == "dead_no_rc":
            break
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert not (tmp_path / "state" / "rc").exists()
    assert status["identity_ambiguous"] is False


@pytest.mark.skipif(
    not ROOT_LINUX_PIDFD_REMOTE,
    reason="exact cancellation requires root Linux /proc and pidfds",
)
@pytest.mark.parametrize("mutate", ["unchanged", "replace", "delete"])
def test_remote_exact_cancel_survives_mutable_launch_path(
    tmp_path: Path, mutate: str
) -> None:
    launch = tmp_path / "launch.sh"
    child = tmp_path / "child-pid"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$$\" > {shlex.quote(str(child))}\n"
        "while :; do sleep 1; done\n",
        encoding="utf-8",
    )
    payload = {
        "operation": "start",
        "job_id": "cancel-path-test",
        "state_dir": str(tmp_path / "state"),
        "launch_script": str(launch),
        "launch_sha256": hashlib.sha256(launch.read_bytes()).hexdigest(),
        "cancel_timeout_seconds": 2,
    }
    started = _direct_remote(payload)
    assert started["ok"] and started["state"] == "running"
    wrapper = int(started["pid"])
    deadline = time.monotonic() + 5
    while not child.exists():
        assert time.monotonic() < deadline
        time.sleep(0.02)
    if mutate == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        replacement.replace(launch)
    elif mutate == "delete":
        launch.unlink()

    payload["operation"] = "cancel"
    cancelled = _direct_remote(payload)
    assert cancelled["ok"] and cancelled["cancelled"] is True
    deadline = time.monotonic() + 5
    while _live_group_members(wrapper, wrapper):
        assert time.monotonic() < deadline
        time.sleep(0.02)


@pytest.mark.skipif(
    not ROOT_LINUX_PIDFD_REMOTE,
    reason="exact cancellation requires root Linux /proc and pidfds",
)
def test_cancel_kills_surviving_child_after_wrapper_leader_dies(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    child_path = tmp_path / "child-pid"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$$\" > {shlex.quote(str(child_path))}\n"
        "trap '' TERM\n"
        "while :; do sleep 1; done\n",
        encoding="utf-8",
    )
    payload = {
        "operation": "start",
        "job_id": "orphan-cancel-test",
        "state_dir": str(tmp_path / "state"),
        "launch_script": str(launch),
        "launch_sha256": hashlib.sha256(launch.read_bytes()).hexdigest(),
        "cancel_timeout_seconds": 1,
    }
    started = _direct_remote(payload)
    wrapper = int(started["pid"])
    deadline = time.monotonic() + 5
    while not child_path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.02)
    child = int(child_path.read_text(encoding="utf-8"))
    os.kill(wrapper, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while wrapper in _live_group_members(wrapper, wrapper):
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert child in _live_group_members(wrapper, wrapper)

    payload["operation"] = "cancel"
    cancelled = _direct_remote(payload)
    assert cancelled["ok"] and cancelled["cancelled"] is True
    assert not _live_group_members(wrapper, wrapper)


@pytest.mark.skipif(
    not ROOT_LINUX_PIDFD_REMOTE,
    reason="descriptor race probe requires root Linux /proc and pidfds",
)
def test_digest_verified_descriptor_executes_after_path_replacement(
    tmp_path: Path,
) -> None:
    launch = tmp_path / "launch.sh"
    result_path = tmp_path / "result"
    # Stay below the controller's 16 MiB cap but keep hashing long enough to
    # observe the verified descriptor in /proc before replacing its pathname.
    launch.write_bytes(
        b"#!/usr/bin/env bash\n#"
        + b"x" * (15 * 1024 * 1024)
        + b"\n"
        + f"printf ORIGINAL > {shlex.quote(str(result_path))}\n".encode()
    )
    original_identity = (launch.stat().st_dev, launch.stat().st_ino)
    payload: dict[str, object] = {
        "operation": "start",
        "job_id": "fd-freeze-test",
        "state_dir": str(tmp_path / "state"),
        "launch_script": str(launch),
        "launch_sha256": hashlib.sha256(launch.read_bytes()).hexdigest(),
        "cancel_timeout_seconds": 2,
    }
    controller = _start_direct_remote(payload)
    assert controller.stdin is not None
    controller.stdin.write(remote_job.REMOTE_CONTROLLER_SOURCE.encode())
    controller.stdin.close()

    pid_path = tmp_path / "state" / "pid.json"
    deadline = time.monotonic() + 10
    while not pid_path.exists():
        assert controller.poll() is None
        assert time.monotonic() < deadline
        time.sleep(0.001)
    wrapper = json.loads(pid_path.read_text(encoding="utf-8"))["pid"]
    observed_verified_fd = False
    while time.monotonic() < deadline and not observed_verified_fd:
        try:
            descriptors = list(Path(f"/proc/{wrapper}/fd").iterdir())
        except (FileNotFoundError, ProcessLookupError):
            break
        for descriptor in descriptors:
            try:
                details = descriptor.stat()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if (details.st_dev, details.st_ino) == original_identity:
                observed_verified_fd = True
                break
    assert observed_verified_fd, "never observed wrapper's verified launch fd"
    replacement = tmp_path / "replacement"
    replacement.write_text(
        f"#!/usr/bin/env bash\nprintf MUTATED > {shlex.quote(str(result_path))}\n",
        encoding="utf-8",
    )
    replacement.replace(launch)

    while not result_path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert result_path.read_text(encoding="utf-8") == "ORIGINAL"
    controller.wait(timeout=10)
