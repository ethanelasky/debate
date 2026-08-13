from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import plistlib
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import install_codecontests_tunnel_supervisor as installer
from scripts import supervise_codecontests_tunnel as tunnel


def _write(path: Path, data: bytes, mode: int = 0o600) -> None:
    path.write_bytes(data)
    path.chmod(mode)


def _config(tmp_path: Path, *, ssh_config: bool = False) -> Path:
    identity = {
        "kind": "identity",
        "protocol_version": "palaestra.codecontests.executor.v2",
        "service_id": "executor-test",
    }
    _write(tmp_path / "identity.json", json.dumps(identity).encode())
    _write(tmp_path / "bearer", b"b" * 32 + b"\n")
    _write(tmp_path / "hmac-key", b"h" * 32 + b"\n")
    _write(tmp_path / "id_ed25519", b"private-key-placeholder")
    _write(tmp_path / "known_hosts", b"host key-placeholder\n", 0o644)
    _write(tmp_path / "ssh.config", b"Host executor\n", 0o600)
    ssh: dict[str, Any]
    if ssh_config:
        ssh = {
            "target": "executor",
            "port": None,
            "identity_file": None,
            "known_hosts_file": None,
            "config_file": str(tmp_path / "ssh.config"),
        }
    else:
        ssh = {
            "target": "root@203.0.113.2",
            "port": 2222,
            "identity_file": str(tmp_path / "id_ed25519"),
            "known_hosts_file": str(tmp_path / "known_hosts"),
            "config_file": None,
        }
    value = {
        "format": tunnel.CONFIG_FORMAT,
        "name": "executor-test-tunnel",
        "ssh": ssh,
        "reverse": {
            "remote_bind_host": "127.0.0.1",
            "remote_port": 18081,
            "local_host": "127.0.0.1",
            "local_port": 18081,
        },
        "attestation": {
            "identity_file": str(tmp_path / "identity.json"),
            "bearer_file": str(tmp_path / "bearer"),
            "hmac_key_file": str(tmp_path / "hmac-key"),
        },
        "state_file": str(tmp_path / "state.json"),
        "connect_timeout_seconds": 2,
        "startup_timeout_seconds": 5,
        "health_interval_seconds": 0.2,
        "health_failure_limit": 2,
        "backoff_initial_seconds": 0.05,
        "backoff_max_seconds": 1,
    }
    path = tmp_path / "config.json"
    _write(path, json.dumps(value).encode())
    return path


def test_config_builds_loopback_reverse_tunnel_with_strict_ssh(tmp_path: Path) -> None:
    supervisor = tunnel.TunnelSupervisor(tunnel._load_config(_config(tmp_path)))
    command = supervisor._tunnel_command()
    assert command[0] == "/usr/bin/ssh"
    assert "BatchMode=yes" in command
    assert "ExitOnForwardFailure=yes" in command
    assert "StrictHostKeyChecking=yes" in command
    assert command[command.index("-R") + 1] == (
        "127.0.0.1:18081:127.0.0.1:18081"
    )
    serialized = " ".join(command)
    assert (b"b" * 32).decode() not in serialized
    assert (b"h" * 32).decode() not in serialized


def test_config_file_mode_is_supported_without_insecure_hostkey_override(
    tmp_path: Path,
) -> None:
    supervisor = tunnel.TunnelSupervisor(
        tunnel._load_config(_config(tmp_path, ssh_config=True))
    )
    command = supervisor._tunnel_command()
    assert command[command.index("-F") + 1] == str(tmp_path / "ssh.config")
    assert not any(value.startswith("UserKnownHostsFile=") for value in command)


def test_non_loopback_forward_and_loose_config_permissions_are_rejected(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text())
    config["reverse"]["remote_bind_host"] = "0.0.0.0"
    _write(config_path, json.dumps(config).encode())
    with pytest.raises(tunnel.ConfigError, match="loopback"):
        tunnel._load_config(config_path)

    config["reverse"]["remote_bind_host"] = "127.0.0.1"
    _write(config_path, json.dumps(config).encode(), 0o644)
    with pytest.raises(tunnel.ConfigError, match="permissions"):
        tunnel._load_config(config_path)


def test_probe_attestation_requires_exact_hmac_signed_frozen_identity(
    tmp_path: Path,
) -> None:
    supervisor = tunnel.TunnelSupervisor(tunnel._load_config(_config(tmp_path)))
    signature = hmac.new(
        b"h" * 32,
        tunnel._canonical_json(supervisor.identity),
        hashlib.sha256,
    ).hexdigest()
    envelope = {
        "payload": supervisor.identity,
        "signature": f"hmac-sha256:{signature}",
    }
    supervisor._verify_envelope(tunnel._canonical_json(envelope))

    tampered = json.loads(json.dumps(envelope))
    tampered["payload"]["service_id"] = "wrong-service"
    with pytest.raises(RuntimeError, match="signature mismatch"):
        supervisor._verify_envelope(tunnel._canonical_json(tampered))


def test_state_is_atomic_private_and_contains_no_credentials(tmp_path: Path) -> None:
    supervisor = tunnel.TunnelSupervisor(tunnel._load_config(_config(tmp_path)))
    supervisor.generation = 7
    supervisor._write_state("connecting")
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text())
    assert state["status"] == "connecting"
    assert state["generation"] == 7
    assert state["identity_service_id"] == "executor-test"
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    serialized = state_path.read_text()
    assert (b"b" * 32).decode() not in serialized
    assert (b"h" * 32).decode() not in serialized
    assert not list(tmp_path.glob(".state.json.*"))


class _DeadAfterReadyChild:
    pid = 4242

    def __init__(self) -> None:
        self.stderr = io.BytesIO()
        self.polls = 0

    def poll(self) -> int | None:
        self.polls += 1
        return None if self.polls <= 3 else -9

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return -9


def test_ready_child_death_returns_to_reconnect_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = tunnel.TunnelSupervisor(tunnel._load_config(_config(tmp_path)))
    child = _DeadAfterReadyChild()
    monkeypatch.setattr(tunnel.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(supervisor, "_probe", lambda: (True, None))
    monkeypatch.setattr(supervisor, "_wait_or_stop", lambda _seconds: False)
    was_ready, reason = supervisor._run_generation()
    assert was_ready is True
    assert reason == "ssh_exit"
    assert supervisor.generation == 1
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "disconnected"
    assert state["reason"] == "ssh_exit"


def test_launchd_plist_keeps_foreground_supervisor_alive(tmp_path: Path) -> None:
    python = tmp_path / "python3"
    supervisor = tmp_path / "supervisor.py"
    config = tmp_path / "config.json"
    log_file = tmp_path / "supervisor.log"
    data = installer.build_launchd_plist(
        label="com.palaestra.codecontests-tunnel.test",
        python=python,
        supervisor=supervisor,
        config=config,
        log_file=log_file,
    )
    value = plistlib.loads(data)
    assert value["KeepAlive"] is True
    assert value["RunAtLoad"] is True
    assert "--daemonize" not in value["ProgramArguments"]
    assert value["StandardOutPath"] == str(log_file)
    assert value["StandardErrorPath"] == str(log_file)


def test_exact_pid_validation_rejects_pid_reuse_or_wrong_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    supervisor = REPOSITORY_ROOT / "scripts" / "supervise_codecontests_tunnel.py"
    state = {"supervisor_pid": 4242}
    monkeypatch.setattr(installer, "_pid_alive", lambda _pid: True)

    class Result:
        returncode = 0
        stderr = ""
        stdout = (
            f"/usr/bin/python3 {supervisor} --config "
            f"{tmp_path / 'different.json'}\n"
        )

    monkeypatch.setattr(installer.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(ValueError, match="exact supervisor/config"):
        installer._validated_supervisor_pid(
            state, supervisor=supervisor, config=config
        )


def test_nohup_remove_signals_only_validated_exact_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    supervisor = REPOSITORY_ROOT / "scripts" / "supervise_codecontests_tunnel.py"
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "name": "executor-test-tunnel",
                "status": "ready",
                "supervisor_pid": 4242,
                "tunnel_pid": 4243,
            }
        )
    )
    monkeypatch.setattr(
        installer,
        "_pid_alive",
        lambda pid: pid in {4242, 4243},
    )
    monkeypatch.setattr(
        installer,
        "_validated_supervisor_pid",
        lambda state, **kwargs: state["supervisor_pid"],
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(installer.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(
        installer,
        "_wait_stopped",
        lambda **kwargs: {"status": "stopped"},
    )
    result = installer._remove_supervisor(
        mode="nohup",
        label="unused",
        supervisor=supervisor,
        config=config,
        state_path=state_path,
        name="executor-test-tunnel",
        timeout_seconds=1,
    )
    assert signals == [(4242, installer.signal.SIGTERM)]
    assert result["former_supervisor_pid"] == 4242
    assert result["former_tunnel_pid"] == 4243
    assert result["state_preserved"] is True
