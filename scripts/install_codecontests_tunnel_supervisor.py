#!/usr/bin/env python3
"""Install a persistent host lifecycle for a CodeContests tunnel supervisor."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _absolute_existing_file(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label} must be an existing absolute file")
    return path


def _absolute_output(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError(f"{label} must have an existing absolute parent")
    return path


def _atomic_write(path: Path, data: bytes, mode: int, *, replace: bool) -> None:
    if path.exists() and not replace:
        raise ValueError(f"refusing to replace existing file: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_launchd_plist(
    *,
    label: str,
    python: Path,
    supervisor: Path,
    config: Path,
    log_file: Path,
) -> bytes:
    if re.fullmatch(r"[A-Za-z0-9.-]{1,128}", label) is None:
        raise ValueError("launchd label is invalid")
    payload: dict[str, Any] = {
        "Label": label,
        "ProgramArguments": [
            str(python),
            "-u",
            str(supervisor),
            "--config",
            str(config),
        ],
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "ThrottleInterval": 1,
        "StandardOutPath": str(log_file),
        "StandardErrorPath": str(log_file),
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def _read_state(config: Path) -> tuple[Path, str]:
    try:
        value = json.loads(config.read_text(encoding="utf-8"))
        state_path = Path(value["state_file"])
        name = value["name"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("cannot read state_file/name from supervisor config") from exc
    if not state_path.is_absolute() or not state_path.parent.is_dir():
        raise ValueError("configured state_file must have an existing absolute parent")
    if not isinstance(name, str):
        raise ValueError("configured supervisor name is invalid")
    return state_path, name


def _pid_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _live_state(state_path: Path, expected_name: str) -> dict[str, Any] | None:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(state, dict)
        or state.get("name") != expected_name
        or not _pid_alive(state.get("supervisor_pid"))
    ):
        return None
    return state


def _load_state(state_path: Path, expected_name: str) -> dict[str, Any] | None:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read supervisor state") from exc
    if not isinstance(state, dict) or state.get("name") != expected_name:
        raise ValueError("supervisor state does not match configured name")
    return state


def _validated_supervisor_pid(
    state: dict[str, Any], *, supervisor: Path, config: Path
) -> int:
    pid = state.get("supervisor_pid")
    if not _pid_alive(pid):
        raise ValueError("configured supervisor is not live")
    assert isinstance(pid, int)
    result = subprocess.run(
        ["/bin/ps", "-ww", "-p", str(pid), "-o", "command="],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("cannot validate configured supervisor process")
    try:
        arguments = shlex.split(result.stdout.strip())
    except ValueError as exc:
        raise ValueError("configured supervisor command line is malformed") from exc
    try:
        config_index = arguments.index("--config")
    except ValueError as exc:
        raise ValueError("live PID is not the configured tunnel supervisor") from exc
    if (
        str(supervisor) not in arguments
        or config_index + 1 >= len(arguments)
        or arguments[config_index + 1] != str(config)
    ):
        raise ValueError("live PID does not match exact supervisor/config")
    return pid


def _wait_stopped(
    *,
    state_path: Path,
    expected_name: str,
    supervisor_pid: int | None,
    tunnel_pid: int | None,
    timeout_seconds: float,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = _load_state(state_path, expected_name)
        supervisor_dead = supervisor_pid is None or not _pid_alive(supervisor_pid)
        tunnel_dead = tunnel_pid is None or not _pid_alive(tunnel_pid)
        if supervisor_dead and tunnel_dead and (
            state is None or state.get("status") == "stopped"
        ):
            return state
        time.sleep(0.1)
    raise RuntimeError("tunnel supervisor did not stop cleanly")


def _wait_ready(
    state_path: Path,
    expected_name: str,
    *,
    previous_pid: int | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = _live_state(state_path, expected_name)
        if (
            state is not None
            and state.get("status") == "ready"
            and state.get("supervisor_pid") != previous_pid
            and _pid_alive(state.get("tunnel_pid"))
        ):
            return state
        time.sleep(0.1)
    raise RuntimeError(f"supervisor did not become ready; inspect {state_path}")


def _install_launchd(
    *,
    label: str,
    python: Path,
    supervisor: Path,
    config: Path,
    log_file: Path,
    replace: bool,
) -> None:
    if sys.platform != "darwin":
        raise ValueError("launchd mode is only available on macOS")
    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(mode=0o700, parents=True, exist_ok=True)
    plist_path = launch_agents / f"{label}.plist"
    _atomic_write(
        plist_path,
        build_launchd_plist(
            label=label,
            python=python,
            supervisor=supervisor,
            config=config,
            log_file=log_file,
        ),
        0o644,
        replace=replace,
    )
    domain = f"gui/{os.getuid()}"
    service = f"{domain}/{label}"
    if replace:
        subprocess.run(
            ["/bin/launchctl", "bootout", service],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    result = subprocess.run(
        ["/bin/launchctl", "bootstrap", domain, str(plist_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"launchd bootstrap failed ({message}); use --mode nohup when no GUI domain is active"
        )
    subprocess.run(["/bin/launchctl", "enable", service], check=True)
    subprocess.run(["/bin/launchctl", "kickstart", "-k", service], check=True)


def _install_nohup(
    *, python: Path, supervisor: Path, config: Path, log_file: Path
) -> None:
    result = subprocess.run(
        [
            "/usr/bin/nohup",
            str(python),
            "-u",
            str(supervisor),
            "--config",
            str(config),
            "--daemonize",
            "--log-file",
            str(log_file),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nohup daemon bootstrap failed: {result.stderr.strip()}")


def _validate_launchd_plist(
    *, label: str, supervisor: Path, config: Path
) -> Path:
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    try:
        value = plistlib.loads(plist_path.read_bytes())
    except (OSError, plistlib.InvalidFileException) as exc:
        raise ValueError("cannot read the exact launchd plist") from exc
    if not isinstance(value, dict):
        raise ValueError("launchd plist is not a dictionary")
    arguments = value.get("ProgramArguments")
    if (
        value.get("Label") != label
        or not isinstance(arguments, list)
        or str(supervisor) not in arguments
    ):
        raise ValueError("launchd plist does not match exact supervisor")
    try:
        config_index = arguments.index("--config")
    except ValueError as exc:
        raise ValueError("launchd plist has no supervisor config") from exc
    if config_index + 1 >= len(arguments) or arguments[config_index + 1] != str(config):
        raise ValueError("launchd plist does not match exact config")
    return plist_path


def _remove_supervisor(
    *,
    mode: str,
    label: str,
    supervisor: Path,
    config: Path,
    state_path: Path,
    name: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    state = _load_state(state_path, name)
    supervisor_pid: int | None = None
    tunnel_pid: int | None = None
    if state is not None:
        raw_tunnel_pid = state.get("tunnel_pid")
        if _pid_alive(raw_tunnel_pid):
            assert isinstance(raw_tunnel_pid, int)
            tunnel_pid = raw_tunnel_pid

    if mode == "launchd":
        if sys.platform != "darwin":
            raise ValueError("launchd mode is only available on macOS")
        _validate_launchd_plist(label=label, supervisor=supervisor, config=config)
        if state is not None and _pid_alive(state.get("supervisor_pid")):
            supervisor_pid = _validated_supervisor_pid(
                state, supervisor=supervisor, config=config
            )
        service = f"gui/{os.getuid()}/{label}"
        disable = subprocess.run(
            ["/bin/launchctl", "disable", service],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if disable.returncode != 0:
            raise RuntimeError(f"launchd disable failed: {disable.stderr.strip()}")
        bootout = subprocess.run(
            ["/bin/launchctl", "bootout", service],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if bootout.returncode != 0 and supervisor_pid is not None:
            raise RuntimeError(f"launchd bootout failed: {bootout.stderr.strip()}")
    else:
        if state is not None and _pid_alive(state.get("supervisor_pid")):
            supervisor_pid = _validated_supervisor_pid(
                state, supervisor=supervisor, config=config
            )
            os.kill(supervisor_pid, signal.SIGTERM)
        elif tunnel_pid is not None:
            raise ValueError("orphaned tunnel is live but its supervisor is absent")

    final_state = _wait_stopped(
        state_path=state_path,
        expected_name=name,
        supervisor_pid=supervisor_pid,
        tunnel_pid=tunnel_pid,
        timeout_seconds=timeout_seconds,
    )
    return {
        "mode": mode,
        "name": name,
        "status": "stopped",
        "former_supervisor_pid": supervisor_pid,
        "former_tunnel_pid": tunnel_pid,
        "state_file": str(state_path),
        "state_preserved": final_state is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-file")
    parser.add_argument("--mode", choices=("launchd", "nohup"), default="launchd")
    parser.add_argument(
        "--label", default="com.palaestra.codecontests-tunnel-supervisor"
    )
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--ready-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    try:
        config = _absolute_existing_file(args.config, "config")
        python = _absolute_existing_file(args.python, "python")
        supervisor = Path(__file__).resolve().with_name(
            "supervise_codecontests_tunnel.py"
        )
        supervisor = _absolute_existing_file(str(supervisor), "supervisor")
        state_path, name = _read_state(config)
        if args.remove:
            if args.replace:
                raise ValueError("--replace cannot be combined with --remove")
            state = _remove_supervisor(
                mode=args.mode,
                label=args.label,
                supervisor=supervisor,
                config=config,
                state_path=state_path,
                name=name,
                timeout_seconds=args.ready_timeout_seconds,
            )
            print(json.dumps(state, indent=2, sort_keys=True))
            return 0
        if not args.log_file:
            raise ValueError("--log-file is required when installing")
        log_file = _absolute_output(args.log_file, "log file")
        existing = _live_state(state_path, name)
        if existing is not None:
            raise ValueError(
                f"supervisor is already live with PID {existing['supervisor_pid']}"
            )
        previous_pid: int | None = None
        try:
            stale = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(stale, dict) and isinstance(stale.get("supervisor_pid"), int):
                previous_pid = stale["supervisor_pid"]
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass

        if args.mode == "launchd":
            _install_launchd(
                label=args.label,
                python=python,
                supervisor=supervisor,
                config=config,
                log_file=log_file,
                replace=args.replace,
            )
        else:
            _install_nohup(
                python=python,
                supervisor=supervisor,
                config=config,
                log_file=log_file,
            )
        state = _wait_ready(
            state_path,
            name,
            previous_pid=previous_pid,
            timeout_seconds=args.ready_timeout_seconds,
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"install failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "mode": args.mode,
                "name": state["name"],
                "status": state["status"],
                "supervisor_pid": state["supervisor_pid"],
                "tunnel_pid": state["tunnel_pid"],
                "generation": state["generation"],
                "identity_service_id": state["identity_service_id"],
                "state_file": str(state_path),
                "log_file": str(log_file),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
