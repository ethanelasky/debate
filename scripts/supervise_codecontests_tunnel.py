#!/usr/bin/env python3
"""Keep an authenticated CodeContests reverse SSH tunnel alive.

The supervisor owns one foreground ``ssh -N`` child at a time.  It does not
mark that child ready until a second SSH connection reaches the reverse
listener and returns the exact frozen, HMAC-signed executor identity.  A dead
child or repeatedly failing health probe is replaced forever, with bounded
backoff.  Run this process under launchd (preferred on macOS) or nohup so its
lifetime is independent of the process that launched an experiment.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_FORMAT = "palaestra.codecontests.tunnel-supervisor.v1"
STATE_FORMAT = "palaestra.codecontests.tunnel-state.v1"
MAX_IDENTITY_BYTES = 256 * 1024
REMOTE_IDENTITY_PROBE = r"""
import http.client
import sys

port = int(sys.argv[1])
token = sys.stdin.buffer.readline(65537).rstrip(b"\r\n")
if not token or len(token) > 65536:
    raise SystemExit(20)
try:
    auth = "Bearer " + token.decode("ascii", errors="strict")
except UnicodeDecodeError:
    raise SystemExit(21)
connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3.0)
connection.request(
    "GET",
    "/v1/identity",
    headers={"Authorization": auth, "Accept": "application/json", "Connection": "close"},
)
response = connection.getresponse()
body = response.read(262145)
if response.status != 200 or len(body) > 262144:
    raise SystemExit(22)
sys.stdout.buffer.write(body)
""".strip()


class ConfigError(ValueError):
    """The local supervisor configuration is unsafe or malformed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ConfigError(
            f"{label} keys mismatch "
            f"(missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)})"
        )


def _number(
    value: Any, label: str, *, minimum: float, maximum: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ConfigError(f"{label} must be in [{minimum}, {maximum}]")
    return result


def _port(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ConfigError(f"{label} must be a TCP port")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ConfigError(f"{label} must be an absolute path")
    return Path(value)


def _read_file(path: Path, *, label: str, max_bytes: int, secret: bool) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ConfigError(f"cannot stat {label}") from exc
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ConfigError(f"{label} must be a non-symlink regular file")
    forbidden = stat.S_IRWXG | stat.S_IRWXO if secret else stat.S_IWGRP | stat.S_IWOTH
    if before.st_mode & forbidden:
        raise ConfigError(f"{label} permissions are unsafe")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConfigError(f"cannot open {label}") from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ConfigError(f"{label} changed while opening")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ConfigError(f"{label} exceeds byte limit")
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        stable = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(opened, field) != getattr(after, field) for field in stable):
            raise ConfigError(f"{label} changed while reading")
        if len(data) != opened.st_size:
            raise ConfigError(f"{label} was not read completely")
        return data
    finally:
        os.close(descriptor)


def _strict_json(data: bytes, label: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ConfigError(f"duplicate key in {label}: {key}")
            result[key] = value
        return result

    try:
        return json.loads(data.decode("utf-8", errors="strict"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid JSON in {label}") from exc


def _load_config(path: Path) -> dict[str, Any]:
    value = _strict_json(
        _read_file(path, label="supervisor config", max_bytes=256 * 1024, secret=True),
        "supervisor config",
    )
    if not isinstance(value, dict):
        raise ConfigError("supervisor config must be an object")
    _exact_keys(
        value,
        {
            "format",
            "name",
            "ssh",
            "reverse",
            "attestation",
            "state_file",
            "connect_timeout_seconds",
            "startup_timeout_seconds",
            "health_interval_seconds",
            "health_failure_limit",
            "backoff_initial_seconds",
            "backoff_max_seconds",
        },
        "config",
    )
    if value["format"] != CONFIG_FORMAT:
        raise ConfigError("unsupported supervisor config format")
    if not isinstance(value["name"], str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value["name"]) is None:
        raise ConfigError("name is invalid")

    ssh = value["ssh"]
    if not isinstance(ssh, dict):
        raise ConfigError("ssh must be an object")
    _exact_keys(
        ssh,
        {"target", "port", "identity_file", "known_hosts_file", "config_file"},
        "ssh",
    )
    if not isinstance(ssh["target"], str) or re.fullmatch(r"[A-Za-z0-9_.@:-]{1,255}", ssh["target"]) is None:
        raise ConfigError("ssh.target is invalid")
    for optional_path in ("identity_file", "known_hosts_file", "config_file"):
        if ssh[optional_path] is not None:
            _absolute_path(ssh[optional_path], f"ssh.{optional_path}")
    if ssh["config_file"] is None:
        if ssh["port"] is None or ssh["identity_file"] is None or ssh["known_hosts_file"] is None:
            raise ConfigError("direct SSH requires port, identity_file, and known_hosts_file")
        _port(ssh["port"], "ssh.port")
    elif any(ssh[key] is not None for key in ("port", "identity_file", "known_hosts_file")):
        raise ConfigError("ssh.config_file cannot be combined with direct SSH fields")

    reverse = value["reverse"]
    if not isinstance(reverse, dict):
        raise ConfigError("reverse must be an object")
    _exact_keys(reverse, {"remote_bind_host", "remote_port", "local_host", "local_port"}, "reverse")
    if reverse["remote_bind_host"] != "127.0.0.1" or reverse["local_host"] != "127.0.0.1":
        raise ConfigError("both tunnel endpoints must be IPv4 loopback")
    _port(reverse["remote_port"], "reverse.remote_port")
    _port(reverse["local_port"], "reverse.local_port")

    attestation = value["attestation"]
    if not isinstance(attestation, dict):
        raise ConfigError("attestation must be an object")
    _exact_keys(attestation, {"identity_file", "bearer_file", "hmac_key_file"}, "attestation")
    for field in attestation:
        _absolute_path(attestation[field], f"attestation.{field}")
    _absolute_path(value["state_file"], "state_file")
    _number(value["connect_timeout_seconds"], "connect_timeout_seconds", minimum=1, maximum=30)
    _number(value["startup_timeout_seconds"], "startup_timeout_seconds", minimum=1, maximum=120)
    _number(value["health_interval_seconds"], "health_interval_seconds", minimum=0.2, maximum=300)
    if isinstance(value["health_failure_limit"], bool) or not isinstance(value["health_failure_limit"], int) or not 1 <= value["health_failure_limit"] <= 20:
        raise ConfigError("health_failure_limit must be in [1, 20]")
    initial = _number(value["backoff_initial_seconds"], "backoff_initial_seconds", minimum=0.05, maximum=60)
    maximum = _number(value["backoff_max_seconds"], "backoff_max_seconds", minimum=initial, maximum=300)
    del maximum
    return value


def _open_daemon_log(path: Path) -> int:
    if not path.is_absolute():
        raise ConfigError("daemon log path must be absolute")
    if not path.parent.is_dir():
        raise ConfigError("daemon log parent directory does not exist")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ConfigError("cannot open daemon log safely") from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode):
        os.close(descriptor)
        raise ConfigError("daemon log must be a regular file")
    os.fchmod(descriptor, 0o600)
    return descriptor


def _daemonize(log_path: Path) -> bool:
    """Double-fork and detach; return True only in the original process."""

    log_descriptor = _open_daemon_log(log_path)
    first_pid = os.fork()
    if first_pid > 0:
        os.close(log_descriptor)
        _pid, wait_status = os.waitpid(first_pid, 0)
        if not os.WIFEXITED(wait_status) or os.WEXITSTATUS(wait_status) != 0:
            raise ConfigError("daemon bootstrap failed")
        return True

    try:
        os.setsid()
        second_pid = os.fork()
        if second_pid > 0:
            os._exit(0)
        os.chdir("/")
        os.umask(0o077)
        null_descriptor = os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC)
        os.dup2(null_descriptor, 0)
        os.dup2(log_descriptor, 1)
        os.dup2(log_descriptor, 2)
        if null_descriptor > 2:
            os.close(null_descriptor)
        if log_descriptor > 2:
            os.close(log_descriptor)
        return False
    except BaseException:
        os._exit(1)


class TunnelSupervisor:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stop = threading.Event()
        self.child: subprocess.Popen[bytes] | None = None
        self.generation = 0
        self.state_path = Path(config["state_file"])
        ssh = config["ssh"]
        if ssh["config_file"] is not None:
            _read_file(
                Path(ssh["config_file"]),
                label="SSH config",
                max_bytes=1024 * 1024,
                secret=False,
            )
        else:
            _read_file(
                Path(ssh["identity_file"]),
                label="SSH identity",
                max_bytes=1024 * 1024,
                secret=True,
            )
            _read_file(
                Path(ssh["known_hosts_file"]),
                label="known-hosts file",
                max_bytes=4 * 1024 * 1024,
                secret=False,
            )
        self.identity = self._load_identity()
        self.identity_digest = hashlib.sha256(_canonical_json(self.identity)).hexdigest()
        self.bearer = _read_file(
            Path(config["attestation"]["bearer_file"]),
            label="bearer token",
            max_bytes=64 * 1024,
            secret=True,
        ).rstrip(b"\r\n")
        self.hmac_key = _read_file(
            Path(config["attestation"]["hmac_key_file"]),
            label="HMAC key",
            max_bytes=64 * 1024,
            secret=True,
        ).rstrip(b"\r\n")
        if len(self.bearer) < 32 or len(self.hmac_key) < 32:
            raise ConfigError("attestation secrets must contain at least 32 bytes")
        try:
            self.bearer.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise ConfigError("bearer token must be ASCII") from exc

    def _load_identity(self) -> dict[str, Any]:
        path = Path(self.config["attestation"]["identity_file"])
        value = _strict_json(
            _read_file(path, label="frozen identity", max_bytes=MAX_IDENTITY_BYTES, secret=False),
            "frozen identity",
        )
        if not isinstance(value, dict) or value.get("kind") != "identity":
            raise ConfigError("frozen identity payload is absent")
        return value

    def _ssh_prefix(self) -> list[str]:
        ssh = self.config["ssh"]
        connect_timeout = int(self.config["connect_timeout_seconds"])
        command = [
            "/usr/bin/ssh",
            "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes",
            "-o", "ControlMaster=no",
            "-o", "ControlPath=none",
            "-o", "ControlPersist=no",
            "-o", f"ConnectTimeout={connect_timeout}",
            "-o", "ConnectionAttempts=1",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=3",
            "-o", "TCPKeepAlive=yes",
            "-o", "RequestTTY=no",
        ]
        if ssh["config_file"] is not None:
            command.extend(["-F", ssh["config_file"]])
        else:
            command.extend(
                [
                    "-o", "StrictHostKeyChecking=yes",
                    "-o", f"UserKnownHostsFile={ssh['known_hosts_file']}",
                    "-i", ssh["identity_file"],
                    "-p", str(ssh["port"]),
                ]
            )
        return command

    def _tunnel_command(self) -> list[str]:
        reverse = self.config["reverse"]
        specification = (
            f"{reverse['remote_bind_host']}:{reverse['remote_port']}:"
            f"{reverse['local_host']}:{reverse['local_port']}"
        )
        return self._ssh_prefix() + [
            "-o", "ExitOnForwardFailure=yes",
            "-N",
            "-T",
            "-R", specification,
            self.config["ssh"]["target"],
        ]

    def _probe_command(self) -> list[str]:
        remote = "python3 -c " + shlex.quote(REMOTE_IDENTITY_PROBE) + " " + str(self.config["reverse"]["remote_port"])
        return self._ssh_prefix() + ["-T", self.config["ssh"]["target"], remote]

    def _log(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": _utc_now(),
            "name": self.config["name"],
            "event": event,
            "generation": self.generation,
            **fields,
        }
        print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)

    def _write_state(self, status: str, **fields: Any) -> None:
        state = {
            "format": STATE_FORMAT,
            "timestamp": _utc_now(),
            "name": self.config["name"],
            "status": status,
            "supervisor_pid": os.getpid(),
            "generation": self.generation,
            "tunnel_pid": self.child.pid if self.child is not None and self.child.poll() is None else None,
            "identity_service_id": self.identity.get("service_id"),
            "identity_sha256": self.identity_digest,
            "remote_listener": f"127.0.0.1:{self.config['reverse']['remote_port']}",
            **fields,
        }
        parent = self.state_path.parent
        if not parent.is_dir():
            raise ConfigError("state_file parent directory does not exist")
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.state_path.name}.", dir=parent)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(_canonical_json(state) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def _verify_envelope(self, body: bytes) -> None:
        envelope = _strict_json(body, "remote identity response")
        if not isinstance(envelope, dict) or set(envelope) != {"payload", "signature"}:
            raise RuntimeError("remote identity envelope is malformed")
        payload = envelope["payload"]
        signature = envelope["signature"]
        if not isinstance(payload, dict) or not isinstance(signature, str):
            raise RuntimeError("remote identity envelope is malformed")
        expected = "hmac-sha256:" + hmac.new(
            self.hmac_key, _canonical_json(payload), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise RuntimeError("remote identity signature mismatch")
        if payload != self.identity:
            raise RuntimeError("remote identity payload mismatch")

    def _probe(self) -> tuple[bool, str | None]:
        try:
            result = subprocess.run(
                self._probe_command(),
                input=self.bearer + b"\n",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=float(self.config["connect_timeout_seconds"]) + 5.0,
                check=False,
            )
            if result.returncode != 0:
                return False, f"ssh_probe_rc_{result.returncode}"
            if len(result.stdout) > MAX_IDENTITY_BYTES:
                return False, "identity_response_too_large"
            self._verify_envelope(result.stdout)
            return True, None
        except subprocess.TimeoutExpired:
            return False, "ssh_probe_timeout"
        except (ConfigError, RuntimeError, OSError, ValueError) as exc:
            return False, type(exc).__name__

    def _stderr_reader(self, child: subprocess.Popen[bytes]) -> None:
        assert child.stderr is not None
        for raw_line in iter(child.stderr.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").rstrip()[:1000]
            if line:
                self._log("ssh_stderr", tunnel_pid=child.pid, message=line)

    def _terminate_child(self) -> None:
        child = self.child
        if child is None or child.poll() is not None:
            return
        child.terminate()
        try:
            child.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=3.0)

    def _wait_or_stop(self, seconds: float) -> bool:
        return self.stop.wait(seconds)

    def _run_generation(self) -> tuple[bool, str]:
        self.generation += 1
        self._write_state("connecting")
        self._log("tunnel_starting")
        try:
            self.child = subprocess.Popen(
                self._tunnel_command(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            self._write_state("disconnected", reason=type(exc).__name__)
            self._log("tunnel_spawn_failed", error=type(exc).__name__)
            return False, "spawn_failed"
        threading.Thread(target=self._stderr_reader, args=(self.child,), daemon=True).start()
        self._write_state("attesting")
        deadline = time.monotonic() + float(self.config["startup_timeout_seconds"])
        last_probe_error: str | None = None
        while not self.stop.is_set() and time.monotonic() < deadline:
            returncode = self.child.poll()
            if returncode is not None:
                self._write_state("disconnected", reason="ssh_exit", ssh_returncode=returncode)
                self._log("tunnel_exited_before_ready", tunnel_pid=self.child.pid, ssh_returncode=returncode)
                return False, "ssh_exit"
            healthy, last_probe_error = self._probe()
            if healthy:
                break
            self._wait_or_stop(0.2)
        else:
            self._terminate_child()
            self._write_state("disconnected", reason="startup_attestation_failed", probe_error=last_probe_error)
            self._log("startup_attestation_failed", probe_error=last_probe_error)
            return False, "startup_attestation_failed"

        ready_since = time.monotonic()
        self._write_state("ready", ready_since_monotonic_ns=time.monotonic_ns())
        self._log("tunnel_ready", tunnel_pid=self.child.pid, identity_sha256=self.identity_digest)
        failures = 0
        while not self.stop.is_set():
            if self._wait_or_stop(float(self.config["health_interval_seconds"])):
                break
            returncode = self.child.poll()
            if returncode is not None:
                self._write_state("disconnected", reason="ssh_exit", ssh_returncode=returncode)
                self._log("tunnel_exited", tunnel_pid=self.child.pid, ssh_returncode=returncode)
                return True, "ssh_exit"
            healthy, probe_error = self._probe()
            if healthy:
                failures = 0
                self._write_state("ready", ready_since_monotonic_ns=int(ready_since * 1_000_000_000))
                continue
            failures += 1
            self._write_state("degraded", probe_failures=failures, probe_error=probe_error)
            self._log("health_probe_failed", probe_failures=failures, probe_error=probe_error)
            if failures >= self.config["health_failure_limit"]:
                self._terminate_child()
                self._write_state("disconnected", reason="health_probe_failed", probe_error=probe_error)
                return True, "health_probe_failed"
        self._terminate_child()
        return True, "stopped"

    def run_forever(self) -> int:
        lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")
        lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ConfigError("another supervisor owns this state file") from exc
            backoff = float(self.config["backoff_initial_seconds"])
            maximum = float(self.config["backoff_max_seconds"])
            while not self.stop.is_set():
                was_ready, reason = self._run_generation()
                if self.stop.is_set():
                    break
                delay = float(self.config["backoff_initial_seconds"]) if was_ready else backoff
                self._write_state("backoff", reason=reason, retry_in_seconds=delay)
                self._log("reconnect_backoff", reason=reason, retry_in_seconds=delay)
                if self._wait_or_stop(delay):
                    break
                backoff = float(self.config["backoff_initial_seconds"]) if was_ready else min(maximum, backoff * 2.0)
            self._terminate_child()
            self._write_state("stopped")
            self._log("supervisor_stopped")
            return 0
        finally:
            os.close(lock_descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--daemonize", action="store_true")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    if bool(args.daemonize) != bool(args.log_file):
        parser.error("--daemonize and --log-file must be used together")
    try:
        supervisor = TunnelSupervisor(_load_config(args.config))
        if args.daemonize and _daemonize(args.log_file):
            return 0
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    def stop(_signum: int, _frame: Any) -> None:
        supervisor.stop.set()
        supervisor._terminate_child()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        return supervisor.run_forever()
    except ConfigError as exc:
        print(f"supervisor error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
