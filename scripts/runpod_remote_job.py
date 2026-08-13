#!/usr/bin/env python3
"""Start one remote trainer exactly once, then monitor it across SSH outages.

The remote launch script is staged separately.  This command verifies its
SHA-256, creates one atomic run directory, detaches a small return-code wrapper,
and then polls durable state.  Retrying the start request is safe: an existing
run directory is never relaunched.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


CONFIG_FORMAT = "palaestra.runpod-remote-job.v1"
RESPONSE_PREFIX = "PALAESTRA_REMOTE_JOB_V1:"
EXIT_TRANSPORT_OUTAGE = 75
EXIT_REMOTE_PROTOCOL = 76
EXIT_INITIALIZATION_STUCK = 77
EXIT_DEAD_WITHOUT_RC = 78
EXIT_CANCEL_UNPROVEN = 79
MAX_RESPONSE_BYTES = 64 * 1024


class ConfigError(ValueError):
    """The local job ownership configuration is unsafe or malformed."""


class TransportError(RuntimeError):
    """The bounded SSH request did not produce an authenticated response."""


@dataclass(frozen=True)
class JobConfig:
    job_id: str
    ssh_argv: tuple[str, ...]
    remote_state_dir: str
    remote_launch_script: str
    launch_script_sha256: str
    poll_interval_seconds: float
    outage_timeout_seconds: float
    ssh_command_timeout_seconds: float
    initialization_timeout_seconds: float
    cancel_timeout_seconds: float


# This source is sent on stdin to ``python3 -B -`` over strict SSH.  It uses
# only the remote Python standard library and emits one prefixed JSON record.
REMOTE_CONTROLLER_SOURCE = r"""
import base64
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time

PREFIX = "PALAESTRA_REMOTE_JOB_V1:"
MANIFEST_FORMAT = "palaestra.runpod-remote-job-state.v1"
PID_FORMAT = "palaestra.runpod-remote-job-pid.v1"


def emit(value):
    print(PREFIX + json.dumps(value, sort_keys=True, separators=(",", ":")), flush=True)


def fail(code, message):
    emit({"ok": False, "error": code, "message": message})
    raise SystemExit(0)


def atomic_write(path, data, mode=0o600):
    parent = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", dir=parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_regular(path, maximum):
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise RuntimeError("not a regular non-symlink file: " + path)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError("file changed while opening: " + path)
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise RuntimeError("file exceeds byte limit: " + path)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(opened, field) != getattr(after, field) for field in stable):
            raise RuntimeError("file changed while reading: " + path)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def read_json(path):
    value = json.loads(read_regular(path, 1024 * 1024).decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("JSON state is not an object: " + path)
    return value


def verify_script(path, expected):
    body = read_regular(path, 16 * 1024 * 1024)
    actual = hashlib.sha256(body).hexdigest()
    if actual != expected:
        raise RuntimeError("launch script SHA-256 mismatch")


def proc_identity(pid):
    try:
        body = read_regular("/proc/%d/stat" % pid, 64 * 1024).decode("utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    close = body.rfind(")")
    if close < 0:
        raise RuntimeError("malformed /proc stat")
    fields = body[close + 2:].split()
    if len(fields) < 20:
        raise RuntimeError("short /proc stat")
    return {"pid": pid, "state": fields[0], "pgrp": int(fields[2]), "session": int(fields[3]), "start_ticks": int(fields[19])}


def exact_live_identity(recorded):
    live = proc_identity(recorded["pid"])
    if live is None or live["state"] == "Z":
        return None
    if (live["start_ticks"], live["pgrp"], live["session"]) != (
        recorded["start_ticks"], recorded["pgid"], recorded["pid"]
    ):
        return False
    return live


def exact_group_members(recorded):
    # Return live members of the private session, or False after PID reuse.
    leader = proc_identity(recorded["pid"])
    if leader is not None and (
        leader["start_ticks"] != recorded["start_ticks"]
        or leader["pgrp"] != recorded["pgid"]
        or leader["session"] != recorded["pid"]
    ):
        return False
    members = []
    try:
        names = os.listdir("/proc")
    except OSError as exc:
        raise RuntimeError("cannot census /proc") from exc
    for name in names:
        if not name.isdigit():
            continue
        member = proc_identity(int(name))
        if member is None or member["state"] == "Z":
            continue
        if member["pgrp"] == recorded["pgid"] and member["session"] == recorded["pid"]:
            members.append(member)
    return members


def signal_exact_member(member, signum):
    # Signal one unchanged process through a pidfd, never a reused PID.
    try:
        descriptor = os.pidfd_open(member["pid"], 0)
    except ProcessLookupError:
        return
    try:
        current = proc_identity(member["pid"])
        if current is None or current["state"] == "Z":
            return
        fields = ("pid", "pgrp", "session", "start_ticks")
        if any(current[field] != member[field] for field in fields):
            return
        signal.pidfd_send_signal(descriptor, signum, None, 0)
    except ProcessLookupError:
        return
    finally:
        os.close(descriptor)


def stop_exact_group(recorded, timeout):
    # Signal each member through a pidfd rather than killpg().  This catches
    # Bash/trainer descendants when the Python group leader died first, without
    # ever signalling a process that reused a numeric PID.
    for signum, duration, label in (
        (signal.SIGTERM, timeout, "TERM"),
        (signal.SIGKILL, min(5.0, timeout), "KILL"),
    ):
        deadline = time.monotonic() + duration
        while True:
            members = exact_group_members(recorded)
            if members is False:
                return {"cancelled": False, "ambiguous": True}
            if not members:
                return {"cancelled": True, "signal": label}
            for member in members:
                signal_exact_member(member, signum)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    members = exact_group_members(recorded)
    if members is False:
        return {"cancelled": False, "ambiguous": True}
    return {"cancelled": not members, "ambiguous": bool(members), "signal": "KILL"}


WRAPPER_SOURCE = r'''
import base64
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile

payload = json.loads(base64.urlsafe_b64decode(sys.argv[1].encode("ascii")))
script = payload["launch_script"]
expected = payload["launch_sha256"]
state = payload["state_dir"]


def atomic_marker(path, data):
    parent = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", dir=parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def remove_exact_marker(path, expected):
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            return
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                return
            body = os.read(descriptor, len(expected) + 1)
        finally:
            os.close(descriptor)
        current = os.lstat(path)
        if body == expected and (current.st_dev, current.st_ino) == (before.st_dev, before.st_ino):
            os.unlink(path)
    except OSError:
        return


# The detached process records itself.  If the SSH-side controller is killed
# immediately after Popen, this closes the otherwise-dangerous window where a
# live trainer has an atomic start directory but no monitorable PID marker.
pid = os.getpid()
with open("/proc/%d/stat" % pid, "rb") as process_stat:
    body = process_stat.read(65537).decode("utf-8")
close = body.rfind(")")
fields = body[close + 2:].split()
if close < 0 or len(fields) < 20:
    raise RuntimeError("cannot read wrapper process identity")
marker = {
    "format": "palaestra.runpod-remote-job-pid.v1",
    "pid": pid,
    "pgid": int(fields[2]),
    "start_ticks": int(fields[19]),
}
if marker["pgid"] != pid or int(fields[3]) != pid:
    raise RuntimeError("wrapper does not own a private session")
marker_body = (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
atomic_marker(os.path.join(state, "pid.json"), marker_body)

# The paid-compute watchdog treats this as busy only while this exact PID,
# start-time, session, and process group still identify the detached wrapper.
atomic_marker(
    "/root/palaestra_remote_job_busy.json",
    marker_body,
)

flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(script, flags)
try:
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode):
        raise RuntimeError("launch script is not regular")
    digest = hashlib.sha256()
    for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
        digest.update(chunk)
    if digest.hexdigest() != expected:
        rc = 126
    else:
        os.lseek(descriptor, 0, os.SEEK_SET)
        log_path = os.path.join(state, "launch.log")
        with open(os.devnull, "rb") as source, open(log_path, "ab", buffering=0) as log:
            try:
                result = subprocess.run(
                    ["/usr/bin/bash", "/proc/self/fd/%d" % descriptor],
                    stdin=source, stdout=log, stderr=subprocess.STDOUT,
                    check=False, pass_fds=(descriptor,),
                )
                rc = result.returncode if result.returncode >= 0 else 128 + (-result.returncode)
            except OSError:
                rc = 126
finally:
    os.close(descriptor)
rc = max(0, min(255, int(rc)))
target = os.path.join(state, "rc")
atomic_marker(target, (str(rc) + "\n").encode("ascii"))
remove_exact_marker("/root/palaestra_remote_job_busy.json", marker_body)
raise SystemExit(rc)
'''


def state_status(payload, verify_launch=True):
    state = payload["state_dir"]
    manifest_path = os.path.join(state, "manifest.json")
    try:
        details = os.lstat(state)
    except FileNotFoundError:
        return {"ok": True, "state": "missing"}
    if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
        raise RuntimeError("state path is not a non-symlink directory")
    deadline = time.monotonic() + 2.0
    while not os.path.exists(manifest_path) and time.monotonic() < deadline:
        time.sleep(0.02)
    if not os.path.exists(manifest_path):
        raise RuntimeError("atomic start marker exists without manifest")
    manifest = read_json(manifest_path)
    expected_manifest = {
        "format": MANIFEST_FORMAT,
        "job_id": payload["job_id"],
        "launch_script": payload["launch_script"],
        "launch_sha256": payload["launch_sha256"],
    }
    if manifest != expected_manifest:
        raise RuntimeError("existing state belongs to a different launch")
    if verify_launch:
        verify_script(payload["launch_script"], payload["launch_sha256"])
    rc_path = os.path.join(state, "rc")
    if os.path.exists(rc_path):
        raw = read_regular(rc_path, 32).decode("ascii").strip()
        if not raw.isdigit() or not 0 <= int(raw) <= 255:
            raise RuntimeError("invalid return-code marker")
        return {"ok": True, "state": "completed", "returncode": int(raw)}
    pid_path = os.path.join(state, "pid.json")
    if not os.path.exists(pid_path):
        return {"ok": True, "state": "initializing"}
    recorded = read_json(pid_path)
    if recorded.get("format") != PID_FORMAT:
        raise RuntimeError("invalid PID marker")
    try:
        pid = int(recorded["pid"])
        pgid = int(recorded["pgid"])
        start_ticks = int(recorded["start_ticks"])
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("invalid PID marker fields")
    if pid <= 1 or pgid != pid:
        raise RuntimeError("unsafe PID marker")
    recorded_identity = {"pid": pid, "pgid": pgid, "start_ticks": start_ticks}
    live = exact_live_identity(recorded_identity)
    if live not in (None, False):
        return {"ok": True, "state": "running", "pid": pid}
    return {"ok": True, "state": "dead_no_rc", "pid": pid, "identity_ambiguous": live is False}


def cancel(payload):
    # Cancellation must remain possible when the mutable staged script path was
    # replaced or removed after the exact verified file descriptor executed.
    status = state_status(payload, verify_launch=False)
    if status["state"] in ("missing", "completed"):
        return {"ok": True, "state": status["state"], "cancelled": True}
    if status["state"] == "initializing":
        return {"ok": True, "state": "initializing", "cancelled": False, "ambiguous": True}
    # A dead wrapper does not prove its Bash/trainer descendants stopped.  The
    # private process group can outlive its leader, so every state with a PID
    # marker is censused and cancelled below.
    recorded = read_json(os.path.join(payload["state_dir"], "pid.json"))
    result = stop_exact_group(recorded, float(payload["cancel_timeout_seconds"]))
    return {"ok": True, "state": "cancelled" if result["cancelled"] else "ambiguous", **result}


def start(payload):
    status = state_status(payload)
    if status["state"] != "missing":
        status["start_disposition"] = "existing"
        return status
    verify_script(payload["launch_script"], payload["launch_sha256"])
    parent = os.path.dirname(payload["state_dir"])
    if not os.path.isdir(parent):
        raise RuntimeError("remote state parent does not exist")
    try:
        os.mkdir(payload["state_dir"], 0o700)
    except FileExistsError:
        status = state_status(payload)
        status["start_disposition"] = "existing"
        return status
    manifest = {
        "format": MANIFEST_FORMAT,
        "job_id": payload["job_id"],
        "launch_script": payload["launch_script"],
        "launch_sha256": payload["launch_sha256"],
    }
    atomic_write(os.path.join(payload["state_dir"], "manifest.json"), (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    wrapper_payload = base64.urlsafe_b64encode(json.dumps({
        "launch_script": payload["launch_script"],
        "launch_sha256": payload["launch_sha256"],
        "state_dir": payload["state_dir"],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).decode("ascii")
    process = subprocess.Popen(
        ["/usr/bin/python3", "-B", "-c", WRAPPER_SOURCE, wrapper_payload],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    try:
        process_pidfd = os.pidfd_open(process.pid, 0)
    except (AttributeError, OSError) as exc:
        # pidfds are part of the Linux safety contract: without one, an
        # initialization failure cannot terminate the wrapper without a PID
        # reuse race.
        raise RuntimeError("cannot open detached-wrapper pidfd") from exc
    # The wrapper, not this short-lived SSH controller, owns the PID marker.
    # It therefore remains observable even if this controller loses its SSH
    # session in the milliseconds immediately following Popen.
    pid_path = os.path.join(payload["state_dir"], "pid.json")
    for _ in range(100):
        if os.path.exists(pid_path) or os.path.exists(os.path.join(payload["state_dir"], "rc")):
            break
        if process.poll() is not None:
            break
        time.sleep(0.01)
    status = state_status(payload)
    if status["state"] == "initializing":
        live = proc_identity(process.pid)
        if live is not None and live["state"] != "Z" and live["pgrp"] == process.pid and live["session"] == process.pid:
            try:
                signal.pidfd_send_signal(process_pidfd, signal.SIGKILL, None, 0)
            except ProcessLookupError:
                pass
        os.close(process_pidfd)
        raise RuntimeError("detached wrapper did not publish its PID marker")
    if status.get("pid", process.pid) != process.pid:
        os.close(process_pidfd)
        raise RuntimeError("detached wrapper PID marker mismatch")
    os.close(process_pidfd)
    status["start_disposition"] = "started"
    return status


try:
    if len(sys.argv) != 2:
        fail("arguments", "expected one encoded payload")
    payload = json.loads(base64.urlsafe_b64decode(sys.argv[1].encode("ascii")))
    required = {"operation", "job_id", "state_dir", "launch_script", "launch_sha256", "cancel_timeout_seconds"}
    if not isinstance(payload, dict) or set(payload) != required:
        fail("payload", "payload keys mismatch")
    result = start(payload) if payload["operation"] == "start" else state_status(payload) if payload["operation"] == "status" else cancel(payload) if payload["operation"] == "cancel" else None
    if result is None:
        fail("operation", "unsupported operation")
    emit(result)
except SystemExit:
    raise
except BaseException as exc:
    fail(type(exc).__name__, str(exc)[:500])
"""


def _strict_json(data: str, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ConfigError(f"duplicate key in {label}: {key}")
            result[key] = value
        return result

    try:
        return json.loads(data, object_pairs_hook=pairs)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {label}") from exc


def _positive_number(value: Any, label: str, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    result = float(value)
    if result <= 0 or result > maximum:
        raise ConfigError(f"{label} must be in (0, {maximum}]")
    return result


def _remote_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value or "\n" in value:
        raise ConfigError(f"{label} must be a non-empty path")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or value == "/"
        or ".." in path.parts
        or value.startswith("//")
    ):
        raise ConfigError(f"{label} must be a normalized absolute POSIX path")
    return value


def _validate_ssh_argv(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < 2 or any(
        not isinstance(item, str) or not item or "\0" in item or "\n" in item
        for item in value
    ):
        raise ConfigError("ssh_argv must be a non-empty array of strings")
    executable = Path(value[0])
    if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
        raise ConfigError("ssh_argv executable must be an existing absolute executable")
    target = value[-1]
    if re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+", target) is None:
        raise ConfigError("ssh_argv must end with one explicit user@host target")

    options: dict[str, str] = {}
    identity: str | None = None
    port_seen = False
    index = 1
    while index < len(value) - 1:
        token = value[index]
        if token == "-o":
            if index + 1 >= len(value) - 1 or "=" not in value[index + 1]:
                raise ConfigError("each ssh -o must be followed by Key=Value")
            option = value[index + 1]
            index += 2
            key, option_value = option.split("=", 1)
            if key in options:
                raise ConfigError(f"duplicate SSH option: {key}")
            options[key] = option_value
            continue
        if token.startswith("-o") and "=" in token[2:]:
            key, option_value = token[2:].split("=", 1)
            if key in options:
                raise ConfigError(f"duplicate SSH option: {key}")
            options[key] = option_value
            index += 1
            continue
        if token in ("-i", "-p"):
            if index + 1 >= len(value) - 1:
                raise ConfigError(f"{token} requires a value")
            argument = value[index + 1]
            if token == "-i":
                if identity is not None:
                    raise ConfigError("SSH requires exactly one -i identity path")
                identity = argument
            else:
                if port_seen:
                    raise ConfigError("SSH port may be supplied only once")
                port_seen = True
                if not argument.isdigit() or not 1 <= int(argument) <= 65535:
                    raise ConfigError("SSH port is invalid")
            index += 2
            continue
        if token in ("-T", "-4", "-6"):
            index += 1
            continue
        raise ConfigError(f"unsupported SSH argv token: {token}")

    allowed_options = {
        "BatchMode", "IdentitiesOnly", "StrictHostKeyChecking",
        "UserKnownHostsFile", "ConnectTimeout", "ConnectionAttempts",
        "ControlMaster", "ControlPath", "ControlPersist", "RequestTTY",
        "ServerAliveInterval", "ServerAliveCountMax", "TCPKeepAlive", "LogLevel",
    }
    unknown = set(options) - allowed_options
    if unknown:
        raise ConfigError(f"unsupported SSH options: {sorted(unknown)}")
    required = {
        "BatchMode": "yes",
        "IdentitiesOnly": "yes",
        "StrictHostKeyChecking": "yes",
        "ConnectionAttempts": "1",
        "ControlMaster": "no",
        "ControlPath": "none",
        "RequestTTY": "no",
    }
    for key, expected in required.items():
        if options.get(key, "").lower() != expected:
            raise ConfigError(f"SSH option {key}={expected} is required")
    known_hosts = options.get("UserKnownHostsFile")
    if not known_hosts or not Path(known_hosts).is_absolute() or known_hosts == "/dev/null":
        raise ConfigError("SSH requires an absolute, non-/dev/null UserKnownHostsFile")
    if identity is None or not Path(identity).is_absolute():
        raise ConfigError("SSH requires an absolute -i identity path")
    for raw_path, label, secret in (
        (known_hosts, "UserKnownHostsFile", False),
        (identity, "SSH identity", True),
    ):
        path = Path(raw_path)
        try:
            details = path.lstat()
        except OSError as exc:
            raise ConfigError(f"{label} is not readable") from exc
        if not stat.S_ISREG(details.st_mode) or path.is_symlink():
            raise ConfigError(f"{label} must be a non-symlink regular file")
        forbidden = (stat.S_IRWXG | stat.S_IRWXO) if secret else (stat.S_IWGRP | stat.S_IWOTH)
        if details.st_mode & forbidden:
            raise ConfigError(f"{label} permissions are unsafe")
    try:
        connect_timeout = int(options.get("ConnectTimeout", ""))
    except ValueError as exc:
        raise ConfigError("SSH ConnectTimeout must be an integer") from exc
    if not 1 <= connect_timeout <= 30:
        raise ConfigError("SSH ConnectTimeout must be in [1, 30]")
    return tuple(value)


def load_config(path: Path) -> JobConfig:
    try:
        details = path.lstat()
    except OSError as exc:
        raise ConfigError("cannot stat config") from exc
    if not stat.S_ISREG(details.st_mode) or path.is_symlink() or details.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ConfigError("config must be a non-symlink file not writable by group/other")
    try:
        value = _strict_json(path.read_text(encoding="utf-8"), "config")
    except (OSError, UnicodeError) as exc:
        raise ConfigError("cannot read config") from exc
    expected = {
        "format", "job_id", "ssh_argv", "remote_state_dir",
        "remote_launch_script", "launch_script_sha256",
        "poll_interval_seconds", "outage_timeout_seconds",
        "ssh_command_timeout_seconds", "initialization_timeout_seconds",
        "cancel_timeout_seconds",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ConfigError("config keys mismatch")
    if value["format"] != CONFIG_FORMAT:
        raise ConfigError("unsupported config format")
    job_id = value["job_id"]
    if not isinstance(job_id, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", job_id) is None:
        raise ConfigError("job_id is invalid")
    digest = value["launch_script_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ConfigError("launch_script_sha256 must be lowercase SHA-256")
    return JobConfig(
        job_id=job_id,
        ssh_argv=_validate_ssh_argv(value["ssh_argv"]),
        remote_state_dir=_remote_path(value["remote_state_dir"], "remote_state_dir"),
        remote_launch_script=_remote_path(value["remote_launch_script"], "remote_launch_script"),
        launch_script_sha256=digest,
        poll_interval_seconds=_positive_number(value["poll_interval_seconds"], "poll_interval_seconds", maximum=300),
        outage_timeout_seconds=_positive_number(value["outage_timeout_seconds"], "outage_timeout_seconds", maximum=7200),
        ssh_command_timeout_seconds=_positive_number(value["ssh_command_timeout_seconds"], "ssh_command_timeout_seconds", maximum=300),
        initialization_timeout_seconds=_positive_number(value["initialization_timeout_seconds"], "initialization_timeout_seconds", maximum=300),
        cancel_timeout_seconds=_positive_number(value["cancel_timeout_seconds"], "cancel_timeout_seconds", maximum=300),
    )


def _payload(config: JobConfig, operation: str) -> str:
    body = json.dumps(
        {
            "operation": operation,
            "job_id": config.job_id,
            "state_dir": config.remote_state_dir,
            "launch_script": config.remote_launch_script,
            "launch_sha256": config.launch_script_sha256,
            "cancel_timeout_seconds": config.cancel_timeout_seconds,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(body).decode("ascii")


def _parse_response(stdout: bytes) -> dict[str, Any]:
    if len(stdout) > MAX_RESPONSE_BYTES:
        raise TransportError("remote response exceeded byte limit")
    try:
        lines = stdout.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise TransportError("remote response was not UTF-8") from exc
    records = [line[len(RESPONSE_PREFIX):] for line in lines if line.startswith(RESPONSE_PREFIX)]
    if len(records) != 1:
        raise TransportError("remote response did not contain exactly one protocol record")
    try:
        value = json.loads(records[0])
    except json.JSONDecodeError as exc:
        raise TransportError("remote protocol record was invalid JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("ok"), bool):
        raise TransportError("remote protocol record had invalid shape")
    return value


def _remote_request(config: JobConfig, operation: str) -> dict[str, Any]:
    encoded = _payload(config, operation)
    remote_command = "/usr/bin/python3 -B - " + shlex.quote(encoded)
    try:
        result = subprocess.run(
            [*config.ssh_argv, remote_command],
            input=REMOTE_CONTROLLER_SOURCE.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=None,
            check=False,
            timeout=config.ssh_command_timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TransportError(type(exc).__name__) from exc
    if result.returncode != 0:
        raise TransportError(f"SSH exited {result.returncode}")
    return _parse_response(result.stdout)


def _event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "monotonic_ns": time.monotonic_ns(), **fields}, sort_keys=True, separators=(",", ":")), flush=True)


def run(config: JobConfig, *, mode: str) -> int:
    last_contact = time.monotonic()
    initialization_started: float | None = None
    operation = "status" if mode == "monitor" else "cancel" if mode == "cancel" else "start"

    while True:
        try:
            response = _remote_request(config, operation)
        except TransportError as exc:
            outage = time.monotonic() - last_contact
            _event("transport_retry", operation=operation, outage_seconds=outage, reason=str(exc))
            if outage >= config.outage_timeout_seconds:
                _event("transport_outage_exceeded", outage_seconds=outage)
                return EXIT_TRANSPORT_OUTAGE
            time.sleep(config.poll_interval_seconds)
            continue

        last_contact = time.monotonic()
        if response.get("ok") is not True:
            _event("remote_protocol_failure", error=response.get("error"), message=response.get("message"))
            return EXIT_REMOTE_PROTOCOL
        if mode == "cancel":
            cancelled = response.get("cancelled")
            _event("remote_cancel", cancelled=cancelled, ambiguous=response.get("ambiguous"))
            return 0 if cancelled is True else EXIT_CANCEL_UNPROVEN
        state = response.get("state")
        _event("remote_state", state=state, start_disposition=response.get("start_disposition"))
        if mode == "start-only":
            return 0
        operation = "status"
        if state == "completed":
            returncode = response.get("returncode")
            if isinstance(returncode, bool) or not isinstance(returncode, int) or not 0 <= returncode <= 255:
                return EXIT_REMOTE_PROTOCOL
            return returncode
        if state == "running":
            initialization_started = None
        elif state == "initializing":
            if initialization_started is None:
                initialization_started = time.monotonic()
            elif time.monotonic() - initialization_started >= config.initialization_timeout_seconds:
                _event("initialization_timeout")
                return EXIT_INITIALIZATION_STUCK
        elif state == "dead_no_rc":
            _event("trainer_dead_without_returncode", orphan_group_terminated=response.get("orphan_group_terminated"))
            return EXIT_DEAD_WITHOUT_RC
        elif state == "missing":
            # Monitor-only is deliberately non-creating.  Missing state means
            # the separately staged start never committed.
            return EXIT_REMOTE_PROTOCOL
        else:
            return EXIT_REMOTE_PROTOCOL
        time.sleep(config.poll_interval_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("start-and-monitor", "start-only", "monitor", "cancel"), default="start-and-monitor")
    args = parser.parse_args(argv)
    try:
        return run(load_config(args.config), mode=args.mode)
    except ConfigError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
