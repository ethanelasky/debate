#!/usr/bin/env python3
"""Bound one owned RunPod job to evacuation, verification, and safe deletion.

The command is intentionally local: it uses the ownership-aware
``/opt/homebrew/bin/runpod-safe`` wrapper and never calls ``runpodctl``.  Each
work command is a JSON argv array and is executed directly without a shell.
``RUNPOD_POD_ID`` is added to its environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SAFE_WRAPPER = Path("/opt/homebrew/bin/runpod-safe")
POD_ID_RE = re.compile(r"^[A-Za-z0-9]+$")
EXIT_PREFLIGHT = 70
EXIT_ARTIFACT_UNCERTAIN = 71
EXIT_DELETE_UNPROVEN = 72
EXIT_INTERNAL = 73
PROCESS_TERMINATION_GRACE_SECONDS = 10.0
MAX_CLEANUP_PROCESS_TERMINATIONS = 5
CLOCK_SKEW_MARGIN_SECONDS = 30.0


class SupervisorError(RuntimeError):
    """The lifecycle contract could not be established or completed."""


class _Interrupted(RuntimeError):
    def __init__(self, signum: int):
        super().__init__(f"received signal {signum}")
        self.signum = signum


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    timed_out: bool
    duration_seconds: float
    launch_failed: bool = False


@dataclass(frozen=True)
class SupervisorConfig:
    pod_id: str
    trainer: tuple[str, ...]
    trainer_cancel: tuple[str, ...]
    transport_cleanup: tuple[str, ...]
    evacuate: tuple[str, ...]
    verify: tuple[str, ...]
    emergency_evacuate: tuple[str, ...]
    post_delete_reachability_probe: tuple[str, ...]
    event_log: Path
    trainer_timeout_seconds: float
    trainer_cancel_timeout_seconds: float
    transport_cleanup_timeout_seconds: float
    evacuation_timeout_seconds: float
    verification_timeout_seconds: float
    emergency_timeout_seconds: float
    delete_timeout_seconds: float
    post_delete_probe_timeout_seconds: float
    delete_attempts: int
    delete_retry_delay_seconds: float
    cleanup_deadline_seconds: float


class EventLog:
    def __init__(self, path: Path):
        if not path.is_absolute():
            raise SupervisorError("event log path must be absolute")
        if path.exists() or path.is_symlink():
            raise SupervisorError("event log already exists")
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except OSError as exc:
            raise SupervisorError(f"could not create event log: {path}") from exc
        self._handle = os.fdopen(descriptor, "w", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        payload = {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        self._handle.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


def _command_metadata(argv: Sequence[str]) -> dict[str, Any]:
    encoded = json.dumps(list(argv), separators=(",", ":")).encode()
    return {
        "executable": argv[0],
        "argc": len(argv),
        "argv_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _parse_command(value: str, *, label: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SupervisorError(f"{label} must be a JSON argv array") from exc
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(not isinstance(item, str) or not item or "\0" in item for item in parsed)
    ):
        raise SupervisorError(f"{label} must be a non-empty JSON array of strings")
    executable = Path(parsed[0])
    if not executable.is_absolute():
        raise SupervisorError(f"{label} executable must be absolute")
    return tuple(parsed)


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # The remote Pod deletion path must still run even if a local process is
        # stuck in an uninterruptible state after SIGKILL.
        return


def _run_work_command(
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    pod_id: str,
    log: EventLog,
    phase: str,
) -> CommandResult:
    metadata = _command_metadata(argv)
    log.emit(f"{phase}_started", timeout_seconds=timeout_seconds, **metadata)
    started = time.monotonic()
    environment = dict(os.environ)
    environment["RUNPOD_POD_ID"] = pod_id
    try:
        process = subprocess.Popen(
            list(argv),
            env=environment,
            start_new_session=True,
        )
    except OSError as exc:
        duration = time.monotonic() - started
        log.emit(
            f"{phase}_finished",
            returncode=127,
            timed_out=False,
            duration_seconds=duration,
            launch_error=type(exc).__name__,
            **metadata,
        )
        return CommandResult(127, False, duration, launch_failed=True)
    timed_out = False
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_group(process)
        returncode = 124
    except BaseException:
        _terminate_process_group(process)
        raise
    duration = time.monotonic() - started
    log.emit(
        f"{phase}_finished",
        returncode=returncode,
        timed_out=timed_out,
        duration_seconds=duration,
        **metadata,
    )
    return CommandResult(returncode, timed_out, duration)


def _run_safe(
    safe_wrapper: Path,
    args: Sequence[str],
    *,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [os.fspath(safe_wrapper), *args],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
    )


def _remaining_timeout(deadline: float, requested: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(cmd="lifecycle cleanup", timeout=0)
    return min(requested, remaining)


def _audit(
    safe_wrapper: Path,
    *,
    timeout_seconds: float,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
    result = _run_safe(safe_wrapper, ["audit"], timeout_seconds=timeout_seconds)
    try:
        payload = json.loads(result.stdout) if result.returncode == 0 else None
    except json.JSONDecodeError:
        payload = None
    return result, payload if isinstance(payload, dict) else None


def _preflight_owned(
    payload: dict[str, Any] | None,
    *,
    pod_id: str,
) -> dict[str, Any]:
    if not _audit_payload_has_expected_lists(payload):
        raise SupervisorError(
            "runpod-safe audit did not return its expected JSON shape"
        )
    assert payload is not None
    tracked = [
        item
        for item in payload.get("tracked_live", [])
        if isinstance(item, dict) and item.get("id") == pod_id
    ]
    if len(tracked) != 1 or tracked[0].get("allocation_exact") is not True:
        raise SupervisorError(
            f"pod {pod_id} is not one exact, live, safe-wrapper-owned allocation"
        )
    conflicting_sections = (
        "untracked_live_untouched",
        "pending_live",
        "deletion_pending",
        "archived_owned_live",
        "ownership_mismatches",
    )
    if any(
        isinstance(item, dict) and item.get("id") == pod_id
        for section in conflicting_sections
        for item in payload.get(section, [])
    ):
        raise SupervisorError(f"pod {pod_id} has conflicting ownership state")
    return tracked[0]


def _audit_proves_absent(payload: dict[str, Any] | None, *, pod_id: str) -> bool:
    if not _audit_payload_has_expected_lists(payload):
        return False
    assert payload is not None
    live_or_ambiguous_sections = (
        "tracked_live",
        "untracked_live_untouched",
        "pending_live",
        "deletion_pending",
        "archived_owned_live",
        "ownership_mismatches",
    )
    return not any(
        isinstance(item, dict) and item.get("id") == pod_id
        for section in live_or_ambiguous_sections
        for item in payload.get(section, [])
    )


def _audit_payload_has_expected_lists(payload: dict[str, Any] | None) -> bool:
    expected = (
        "tracked_live",
        "untracked_live_untouched",
        "stale_receipts",
        "pending_transactions",
        "pending_live",
        "deletion_pending",
        "archived_owned_live",
        "ownership_mismatches",
        "malformed_state",
    )
    return payload is not None and all(
        isinstance(payload.get(section), list) for section in expected
    )


def _require_server_ttl_budget(
    owned: dict[str, Any],
    *,
    trainer_timeout_seconds: float,
    cleanup_deadline_seconds: float,
    now: datetime | None = None,
) -> tuple[datetime, float]:
    expired = owned.get("expired")
    if not isinstance(expired, bool):
        raise SupervisorError("owned Pod audit has missing or invalid expired state")
    raw_expiry = owned.get("expires_at")
    if not isinstance(raw_expiry, str) or not raw_expiry:
        raise SupervisorError("owned Pod audit has missing or invalid server expiry")
    try:
        expiry = datetime.fromisoformat(raw_expiry.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SupervisorError("owned Pod audit has malformed server expiry") from exc
    if expiry.tzinfo is None or expiry.utcoffset() is None:
        raise SupervisorError("owned Pod audit server expiry must include a timezone")
    expiry = expiry.astimezone(timezone.utc)
    checked_at = now or datetime.now(timezone.utc)
    remaining_seconds = (expiry - checked_at).total_seconds()
    required_seconds = (
        trainer_timeout_seconds
        + cleanup_deadline_seconds
        + CLOCK_SKEW_MARGIN_SECONDS
    )
    if expired or remaining_seconds <= 0:
        raise SupervisorError("owned Pod server termination deadline is expired")
    if remaining_seconds < required_seconds:
        raise SupervisorError(
            "owned Pod server termination deadline does not cover trainer timeout, "
            "cleanup deadline, and clock-skew margin"
        )
    return expiry, remaining_seconds


def _safe_wrapper_is_usable(path: Path) -> bool:
    try:
        details = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return path.is_absolute() and path.is_file() and not path.is_symlink() and bool(
        details.st_mode & 0o111
    )


def supervise(config: SupervisorConfig, *, safe_wrapper: Path = SAFE_WRAPPER) -> int:
    if not POD_ID_RE.fullmatch(config.pod_id):
        raise SupervisorError("pod ID must contain ASCII letters and digits only")
    if not _safe_wrapper_is_usable(safe_wrapper):
        raise SupervisorError("runpod-safe wrapper is missing or unsafe")
    if config.delete_attempts < 1:
        raise SupervisorError("delete attempts must be positive")
    delete_reserve_seconds = (
        config.delete_attempts * config.delete_timeout_seconds
        + (config.delete_attempts - 1) * config.delete_retry_delay_seconds
        + config.delete_timeout_seconds
        + config.post_delete_probe_timeout_seconds
        + (
            MAX_CLEANUP_PROCESS_TERMINATIONS
            * PROCESS_TERMINATION_GRACE_SECONDS
        )
    )
    if config.cleanup_deadline_seconds <= (
        delete_reserve_seconds + config.trainer_cancel_timeout_seconds
    ):
        raise SupervisorError(
            "cleanup deadline must leave positive evacuation time after reserving "
            "trainer cancellation, all delete attempts, final audit, reachability "
            "probe, process shutdown, and retry delays"
        )

    log = EventLog(config.event_log)
    deferred_signals: list[int] = []
    trainer_active = False

    def handle_signal(signum: int, _frame: Any) -> None:
        deferred_signals.append(signum)
        if trainer_active:
            raise _Interrupted(signum)

    old_handlers = {
        signum: signal.signal(signum, handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        try:
            audit_result, audit_payload = _audit(
                safe_wrapper,
                timeout_seconds=config.delete_timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SupervisorError("initial runpod-safe audit failed") from exc
        log.emit("ownership_audit", returncode=audit_result.returncode)
        owned = _preflight_owned(audit_payload, pod_id=config.pod_id)
        server_expiry, ttl_remaining_seconds = _require_server_ttl_budget(
            owned,
            trainer_timeout_seconds=config.trainer_timeout_seconds,
            cleanup_deadline_seconds=config.cleanup_deadline_seconds,
        )
        log.emit(
            "ownership_confirmed",
            pod_id=config.pod_id,
            server_expiry=server_expiry.isoformat().replace("+00:00", "Z"),
            ttl_remaining_seconds=ttl_remaining_seconds,
            ttl_required_seconds=(
                config.trainer_timeout_seconds
                + config.cleanup_deadline_seconds
                + CLOCK_SKEW_MARGIN_SECONDS
            ),
        )

        trainer_returncode = EXIT_INTERNAL
        trainer_active = True
        try:
            if deferred_signals:
                trainer_returncode = 128 + deferred_signals[0]
                log.emit("trainer_skipped", reason="signal_before_trainer")
            else:
                trainer_result = _run_work_command(
                    config.trainer,
                    timeout_seconds=config.trainer_timeout_seconds,
                    pod_id=config.pod_id,
                    log=log,
                    phase="trainer",
                )
                trainer_returncode = trainer_result.returncode
        except _Interrupted as exc:
            trainer_returncode = 128 + exc.signum
            log.emit("trainer_interrupted", signal=exc.signum)
        except KeyboardInterrupt:
            deferred_signals.append(signal.SIGINT)
            trainer_returncode = 130
            log.emit("trainer_interrupted", signal=signal.SIGINT)
        finally:
            trainer_active = False

        cleanup_started_at = time.monotonic()
        cleanup_deadline = cleanup_started_at + config.cleanup_deadline_seconds
        predelete_deadline = cleanup_deadline - delete_reserve_seconds
        artifact_certain = False
        remote_cancelled = False
        transport_cleaned = False
        try:
            trainer_cancel_result = _run_work_command(
                config.trainer_cancel,
                timeout_seconds=_remaining_timeout(
                    predelete_deadline,
                    config.trainer_cancel_timeout_seconds,
                ),
                pod_id=config.pod_id,
                log=log,
                phase="trainer_cancel",
            )
            remote_cancelled = trainer_cancel_result.returncode == 0
        except subprocess.TimeoutExpired:
            log.emit("trainer_cancel_skipped", reason="cleanup_deadline")
        try:
            transport_cleanup_result = _run_work_command(
                config.transport_cleanup,
                timeout_seconds=_remaining_timeout(
                    predelete_deadline,
                    config.transport_cleanup_timeout_seconds,
                ),
                pod_id=config.pod_id,
                log=log,
                phase="transport_cleanup",
            )
            transport_cleaned = transport_cleanup_result.returncode == 0
        except subprocess.TimeoutExpired:
            log.emit("transport_cleanup_skipped", reason="cleanup_deadline")

        try:
            evacuation_result = _run_work_command(
                config.evacuate,
                timeout_seconds=_remaining_timeout(
                    predelete_deadline,
                    config.evacuation_timeout_seconds,
                ),
                pod_id=config.pod_id,
                log=log,
                phase="evacuation",
            )
            if evacuation_result.returncode == 0:
                verification_result = _run_work_command(
                    config.verify,
                    timeout_seconds=_remaining_timeout(
                        predelete_deadline,
                        config.verification_timeout_seconds,
                    ),
                    pod_id=config.pod_id,
                    log=log,
                    phase="artifact_verification",
                )
                artifact_certain = (
                    remote_cancelled and verification_result.returncode == 0
                )
        except subprocess.TimeoutExpired:
            log.emit("primary_artifact_path_skipped", reason="cleanup_deadline")

        if not artifact_certain:
            log.emit("primary_artifact_path_failed")
            try:
                emergency_result = _run_work_command(
                    config.emergency_evacuate,
                    timeout_seconds=_remaining_timeout(
                        predelete_deadline,
                        config.emergency_timeout_seconds,
                    ),
                    pod_id=config.pod_id,
                    log=log,
                    phase="emergency_evacuation",
                )
                emergency_success = emergency_result.returncode == 0
            except subprocess.TimeoutExpired:
                emergency_success = False
                log.emit("emergency_evacuation_skipped", reason="cleanup_deadline")
            log.emit(
                "emergency_artifact_attempt_complete",
                successful=emergency_success,
            )

        delete_command_succeeded = False
        for attempt in range(1, config.delete_attempts + 1):
            try:
                delete_result = _run_safe(
                    safe_wrapper,
                    ["delete", config.pod_id],
                    timeout_seconds=config.delete_timeout_seconds,
                )
                delete_returncode = delete_result.returncode
            except subprocess.TimeoutExpired:
                delete_returncode = 124
            except OSError:
                delete_returncode = 127
            log.emit(
                "safe_delete_attempt",
                pod_id=config.pod_id,
                attempt=attempt,
                returncode=delete_returncode,
            )
            if delete_returncode == 0:
                delete_command_succeeded = True
                break
            if attempt < config.delete_attempts:
                time.sleep(config.delete_retry_delay_seconds)

        try:
            final_audit_result, final_audit_payload = _audit(
                safe_wrapper,
                timeout_seconds=config.delete_timeout_seconds,
            )
            final_audit_returncode = final_audit_result.returncode
        except subprocess.TimeoutExpired:
            final_audit_payload = None
            final_audit_returncode = 124
        except (OSError, subprocess.SubprocessError):
            final_audit_payload = None
            final_audit_returncode = 127
        absent = _audit_proves_absent(final_audit_payload, pod_id=config.pod_id)
        reachability_result = _run_work_command(
            config.post_delete_reachability_probe,
            timeout_seconds=config.post_delete_probe_timeout_seconds,
            pod_id=config.pod_id,
            log=log,
            phase="post_delete_reachability_probe",
        )
        contradictory_reachability = reachability_result.returncode == 0
        reachability_probe_conclusive = (
            not reachability_result.timed_out
            and not reachability_result.launch_failed
        )
        log.emit(
            "final_audit",
            pod_id=config.pod_id,
            returncode=final_audit_returncode,
            pod_absent=absent,
            contradictory_reachability=contradictory_reachability,
            reachability_probe_conclusive=reachability_probe_conclusive,
            remote_cancelled=remote_cancelled,
            transport_cleaned=transport_cleaned,
            deferred_signals=deferred_signals,
        )
        if (
            not delete_command_succeeded
            or not absent
            or contradictory_reachability
            or not reachability_probe_conclusive
        ):
            return EXIT_DELETE_UNPROVEN
        if not artifact_certain:
            return EXIT_ARTIFACT_UNCERTAIN
        if not transport_cleaned:
            return EXIT_INTERNAL
        if deferred_signals:
            return 128 + deferred_signals[0]
        return trainer_returncode
    finally:
        for signum, old_handler in old_handlers.items():
            signal.signal(signum, old_handler)
        log.close()


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--trainer-command-json", required=True)
    parser.add_argument("--trainer-cancel-command-json", required=True)
    parser.add_argument("--transport-cleanup-command-json", required=True)
    parser.add_argument("--evacuate-command-json", required=True)
    parser.add_argument("--verify-command-json", required=True)
    parser.add_argument("--emergency-evacuate-command-json", required=True)
    parser.add_argument("--post-delete-reachability-probe-command-json", required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument(
        "--trainer-timeout-seconds", type=_positive_float, required=True
    )
    parser.add_argument(
        "--trainer-cancel-timeout-seconds", type=_positive_float, required=True
    )
    parser.add_argument(
        "--transport-cleanup-timeout-seconds", type=_positive_float, default=60
    )
    parser.add_argument(
        "--evacuation-timeout-seconds", type=_positive_float, default=1800
    )
    parser.add_argument(
        "--verification-timeout-seconds", type=_positive_float, default=300
    )
    parser.add_argument(
        "--emergency-timeout-seconds", type=_positive_float, default=300
    )
    parser.add_argument("--delete-timeout-seconds", type=_positive_float, default=180)
    parser.add_argument(
        "--post-delete-probe-timeout-seconds", type=_positive_float, default=30
    )
    parser.add_argument("--delete-attempts", type=int, default=3)
    parser.add_argument(
        "--delete-retry-delay-seconds", type=_nonnegative_float, default=5
    )
    parser.add_argument(
        "--cleanup-deadline-seconds", type=_positive_float, default=2700
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = SupervisorConfig(
            pod_id=args.pod_id,
            trainer=_parse_command(args.trainer_command_json, label="trainer command"),
            trainer_cancel=_parse_command(
                args.trainer_cancel_command_json,
                label="trainer cancel command",
            ),
            transport_cleanup=_parse_command(
                args.transport_cleanup_command_json,
                label="transport cleanup command",
            ),
            evacuate=_parse_command(
                args.evacuate_command_json, label="evacuation command"
            ),
            verify=_parse_command(args.verify_command_json, label="verify command"),
            emergency_evacuate=_parse_command(
                args.emergency_evacuate_command_json,
                label="emergency evacuation command",
            ),
            post_delete_reachability_probe=_parse_command(
                args.post_delete_reachability_probe_command_json,
                label="post-delete reachability probe command",
            ),
            event_log=args.event_log,
            trainer_timeout_seconds=args.trainer_timeout_seconds,
            trainer_cancel_timeout_seconds=args.trainer_cancel_timeout_seconds,
            transport_cleanup_timeout_seconds=args.transport_cleanup_timeout_seconds,
            evacuation_timeout_seconds=args.evacuation_timeout_seconds,
            verification_timeout_seconds=args.verification_timeout_seconds,
            emergency_timeout_seconds=args.emergency_timeout_seconds,
            delete_timeout_seconds=args.delete_timeout_seconds,
            post_delete_probe_timeout_seconds=args.post_delete_probe_timeout_seconds,
            delete_attempts=args.delete_attempts,
            delete_retry_delay_seconds=args.delete_retry_delay_seconds,
            cleanup_deadline_seconds=args.cleanup_deadline_seconds,
        )
        return supervise(config)
    except SupervisorError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return EXIT_PREFLIGHT


if __name__ == "__main__":
    raise SystemExit(main())
