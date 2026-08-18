"""Small, strict client for executing one Python case with Piston.

Piston owns only untrusted process execution.  The caller retains the total
solution deadline, expected output, normalization, and verdict aggregation.

The source is intentionally sent directly for the first production probe.
Consequently Piston cannot distinguish a normal top-level return from
``os._exit(0)`` the way CodeContests' local bootstrap can.  Both are reported
as a clean exit; live parity testing must account for that known difference.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from infra.envs.tasks.base import GraderInfrastructureError

# Candidate source and stdin must go only to the explicitly configured judge.
# urllib's process-wide proxy environment is inappropriate for this trusted
# execution channel, including a loopback SSH forward.
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# Bump whenever the reward-affecting deployment contract changes. Version 1
# is defined in deploy/piston/README.md (pinned image/runtime, exact stdin,
# resource ceilings, and network policy).
PROTOCOL_ID = "codecontests-piston-v1"
# Fixed by the v1 deployment contract in deploy/piston/docker-compose.yml.
MAX_CONCURRENT_JOBS = 4
MAX_RUN_TIMEOUT_MILLISECONDS = 90_000

_EXECUTE_PATH = "/api/v2/execute"
_MAX_ATTEMPTS = 2
# Piston enforces the candidate deadline.  This additional trusted-side window
# only lets isolate cleanup, JSON serialization, and an SSH-forwarded response
# reach the trainer.  A two-second window proved too tight under sustained
# four-way production traffic even though Piston completed every sandbox job.
_RESPONSE_GRACE_SECONDS = 10.0
_OUTPUT_LIMIT_BYTES = 1024 * 1024
# Piston returns stdout, stderr, and an interleaved ``output`` copy.  Permit
# the worst-case six-byte JSON escaping of all four output-limit-sized regions,
# plus modest metadata overhead, while still bounding a buggy server response.
_MAX_RESPONSE_BYTES = 24 * _OUTPUT_LIMIT_BYTES + 1024 * 1024
_NORMAL_RETURN_CODE = 120

_EXACT_VERSION_RE = re.compile(
    r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?"
    r"(?:\+[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?$"
)

_TOP_LEVEL_FIELDS = {"language", "version", "run"}
_RUN_FIELDS = {
    "stdout",
    "stderr",
    "output",
    "code",
    "signal",
    "message",
    "status",
    "cpu_time",
    "wall_time",
    "memory",
}
_CANDIDATE_STATUSES = {None, "RE", "SG", "TO", "OL", "EL"}


def run_python_case(
    *,
    base_url: str,
    runtime_version: str,
    solution_code: str,
    test_input: str,
    remaining_seconds: float,
) -> dict[str, Any]:
    """Execute one Python source/input pair and return the case-result schema.

    A transport error or HTTP 5xx is retried once while the supplied deadline
    still has budget.  HTTP 4xx and protocol errors are deterministic and are
    never retried.  All trusted-side failures raise
    :class:`GraderInfrastructureError`; candidate failures are returned as a
    non-successful case result.
    """
    execute_url = _validate_inputs(
        base_url=base_url,
        runtime_version=runtime_version,
        solution_code=solution_code,
        test_input=test_input,
        remaining_seconds=remaining_seconds,
    )
    deadline = time.perf_counter() + float(remaining_seconds)
    last_retryable_error: BaseException | None = None
    last_attempt_timed_out = False

    for attempt in range(_MAX_ATTEMPTS):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            if last_attempt_timed_out:
                raise GraderInfrastructureError(
                    "Piston did not return a response within the "
                    "execution-plus-transport deadline"
                ) from last_retryable_error
            raise GraderInfrastructureError(
                "Piston transport exhausted the remaining solution deadline"
            ) from last_retryable_error

        # A positive sub-millisecond budget still gets one millisecond, the
        # finest unit accepted by Piston.  It can overshoot by less than 1 ms.
        run_timeout_ms = max(
            1,
            min(MAX_RUN_TIMEOUT_MILLISECONDS, math.floor(remaining * 1000)),
        )
        body = json.dumps(
            {
                "language": "python",
                "version": runtime_version,
                "files": [
                    {
                        "name": "solution.py",
                        "content": solution_code,
                        "encoding": "utf8",
                    }
                ],
                "stdin": test_input,
                "run_timeout": run_timeout_ms,
                "run_cpu_time": run_timeout_ms,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            execute_url,
            data=body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )

        try:
            # Piston must first stop/reap isolate and serialize its TO result.
            # This trusted cleanup allowance mirrors the local verifier's
            # post-timeout reap grace; it never enters ``run_timeout`` and is
            # never refunded when a retry recomputes candidate time above.
            # The deployment can run a single case for at most 90 seconds.
            # Keep the trusted HTTP wait tied to that clamped request rather
            # than a potentially much larger solution-level budget.
            transport_timeout = (
                run_timeout_ms / 1000 + _RESPONSE_GRACE_SECONDS
            )
            with _DIRECT_OPENER.open(
                request, timeout=transport_timeout
            ) as response:
                status_code = response.getcode()
                if isinstance(status_code, bool) or not isinstance(status_code, int):
                    raise GraderInfrastructureError(
                        "Piston returned an invalid HTTP status"
                    )
                if status_code != 200:
                    if 500 <= status_code <= 599:
                        raise _RetryablePistonError(
                            f"Piston returned HTTP {status_code}"
                        )
                    raise GraderInfrastructureError(
                        f"Piston returned non-success HTTP {status_code}"
                    )
                content_type = response.headers.get("Content-Type", "")
                if not content_type.lower().startswith("application/json"):
                    raise GraderInfrastructureError(
                        "Piston returned a non-JSON content type"
                    )
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    raise GraderInfrastructureError(
                        "Piston response exceeded the verifier response limit"
                    )
        except urllib.error.HTTPError as exc:
            last_attempt_timed_out = False
            if 500 <= exc.code <= 599:
                last_retryable_error = exc
            else:
                raise GraderInfrastructureError(
                    f"Piston rejected the execution request with HTTP {exc.code}"
                ) from exc
        except _RetryablePistonError as exc:
            last_retryable_error = exc
            last_attempt_timed_out = False
        except TimeoutError as exc:
            last_retryable_error = exc
            last_attempt_timed_out = True
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            ConnectionError,
            OSError,
        ) as exc:
            last_retryable_error = exc
            last_attempt_timed_out = False
        else:
            return _decode_case_result(raw, runtime_version=runtime_version)

        if attempt + 1 == _MAX_ATTEMPTS:
            raise GraderInfrastructureError(
                f"Piston transport failed after {_MAX_ATTEMPTS} attempts"
            ) from last_retryable_error

    raise AssertionError("unreachable")


class _RetryablePistonError(Exception):
    """Internal marker for a 5xx returned as a normal response object."""


def validate_settings(*, base_url: str, runtime_version: str) -> str:
    """Validate trusted endpoint settings and return the execute URL."""
    if not isinstance(base_url, str) or not base_url:
        raise GraderInfrastructureError("Piston base URL must be a non-empty string")
    try:
        parsed = urllib.parse.urlsplit(base_url)
        hostname = parsed.hostname
        # Accessing port also validates a supplied value/range.
        parsed.port
    except ValueError as exc:
        raise GraderInfrastructureError("Piston base URL is invalid") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise GraderInfrastructureError("Piston base URL is invalid")
    if parsed.scheme == "http" and not _is_loopback_host(hostname):
        raise GraderInfrastructureError(
            "plaintext Piston URLs are allowed only on loopback (use an SSH "
            "forward or HTTPS)"
        )
    if not isinstance(runtime_version, str) or not _EXACT_VERSION_RE.fullmatch(
        runtime_version
    ):
        raise GraderInfrastructureError(
            "Piston runtime version must be an exact semantic version"
        )
    return base_url.rstrip("/") + _EXECUTE_PATH


def _validate_inputs(
    *,
    base_url: str,
    runtime_version: str,
    solution_code: str,
    test_input: str,
    remaining_seconds: float,
) -> str:
    execute_url = validate_settings(
        base_url=base_url, runtime_version=runtime_version
    )
    if not isinstance(solution_code, str) or not isinstance(test_input, str):
        raise GraderInfrastructureError("Piston source and stdin must be strings")
    if (
        isinstance(remaining_seconds, bool)
        or not isinstance(remaining_seconds, (int, float))
        or not math.isfinite(remaining_seconds)
        or remaining_seconds <= 0
    ):
        raise GraderInfrastructureError(
            "Piston execution requires a positive finite remaining deadline"
        )
    return execute_url


def _decode_case_result(raw: bytes, *, runtime_version: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraderInfrastructureError("Piston returned malformed JSON") from exc

    if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL_FIELDS:
        raise GraderInfrastructureError("Piston returned an invalid response schema")
    if payload["language"] != "python" or payload["version"] != runtime_version:
        raise GraderInfrastructureError("Piston returned the wrong runtime identity")

    run = payload["run"]
    if not isinstance(run, dict) or set(run) != _RUN_FIELDS:
        raise GraderInfrastructureError("Piston returned an invalid run schema")
    if not all(isinstance(run[name], str) for name in ("stdout", "stderr", "output")):
        raise GraderInfrastructureError("Piston returned non-string process output")

    code = run["code"]
    piston_signal = run["signal"]
    status = run["status"]
    if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
        raise GraderInfrastructureError("Piston returned an invalid process exit code")
    if piston_signal is not None and (
        not isinstance(piston_signal, str) or not piston_signal
    ):
        raise GraderInfrastructureError("Piston returned an invalid process signal")
    if status is not None and not isinstance(status, str):
        raise GraderInfrastructureError("Piston returned an invalid process status")
    if run["message"] is not None and not isinstance(run["message"], str):
        raise GraderInfrastructureError("Piston returned an invalid process message")
    if run["memory"] is not None and (
        isinstance(run["memory"], bool) or not isinstance(run["memory"], int)
    ):
        raise GraderInfrastructureError("Piston returned invalid memory telemetry")
    for name in ("cpu_time", "wall_time"):
        value = run[name]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise GraderInfrastructureError(f"Piston returned invalid {name} telemetry")

    if status == "XX":
        raise GraderInfrastructureError("Piston reported an internal execution failure")
    if status not in _CANDIDATE_STATUSES:
        raise GraderInfrastructureError(
            f"Piston returned unsupported process status {status!r}"
        )
    if (code is None) == (piston_signal is None):
        raise GraderInfrastructureError(
            "Piston returned an inconsistent exit code/signal pair"
        )

    # The caller only distinguishes trusted clean completion from every other
    # candidate outcome; it does not expose raw exit/signal values. Collapsing
    # failures to 1 is both sufficient and prevents candidate exit 120 from
    # colliding with the local runner's normal-completion sentinel.
    clean_exit = status is None and code == 0 and piston_signal is None
    returncode = _NORMAL_RETURN_CODE if clean_exit else 1

    return {
        "returncode": returncode,
        "timed_out": status == "TO",
        "output_limited": status in {"OL", "EL"},
        "stdout": run["stdout"],
        "stderr": run["stderr"],
    }
def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False
