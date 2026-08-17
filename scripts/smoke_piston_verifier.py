#!/usr/bin/env python3
"""Fail-fast production-path smoke test for the CodeContests Piston verifier."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import infra.envs.tasks.codecontests as codecontests  # noqa: E402
from infra.envs.base import _fail_fast_thread_map  # noqa: E402

_DEFAULT_RUNTIME = "3.12.0"
_PROTOCOL_MAX_CONCURRENCY = codecontests.piston.MAX_CONCURRENT_JOBS

_BYTE_ECHO_SOURCE = (
    "import sys\n"
    "sys.stdout.write(sys.stdin.buffer.read().hex())\n"
)
_PASS_SOURCE = "print('piston-preflight-ok')\n"
_EXIT_1_SOURCE = "import sys\nprint('piston-preflight-ok')\nsys.exit(1)\n"
_EXIT_120_SOURCE = "import sys\nprint('piston-preflight-ok')\nsys.exit(120)\n"
_TIMEOUT_SOURCE = "while True:\n    pass\n"
_LARGE_REQUEST_SOURCE = (
    "import sys\n"
    "print(len(sys.stdin.buffer.read()))\n"
)
_EARLY_EXIT_SOURCE = "pass\n"
_OUTPUT_FLOOD_SOURCE = "import sys\nsys.stdout.write('x' * (2 * 1024 * 1024))\n"
_FOLLOWER_SOURCE = "print('piston-preflight-follower')\n"

Verifier = Callable[..., dict[str, Any]]


class PreflightFailure(RuntimeError):
    """A production verifier behavior did not match the deployment contract."""


def _settings(url: str, runtime: str) -> dict[str, str]:
    return {
        "verifier": "piston",
        "piston_url": url,
        "piston_python_version": runtime,
    }


def _summary(result: dict[str, Any]) -> str:
    return (
        f"status={result.get('status')!r}, passed={result.get('passed')!r}, "
        f"timeout={result.get('timeout')!r}, "
        f"tests={result.get('tests_passed')!r}/{result.get('tests_total')!r}"
    )


def _require(
    phase: str,
    result: dict[str, Any],
    *,
    status: str,
    passed: bool,
    timed_out: bool,
    tests_passed: int | None = None,
    tests_total: int | None = None,
) -> None:
    matches = (
        result.get("status") == status
        and result.get("passed") is passed
        and result.get("timeout") is timed_out
        and (tests_passed is None or result.get("tests_passed") == tests_passed)
        and (tests_total is None or result.get("tests_total") == tests_total)
    )
    if not matches:
        raise PreflightFailure(f"{phase} mismatch: {_summary(result)}")


def _check_saturation(
    *,
    verifier: Verifier,
    settings: dict[str, str],
    concurrency: int,
) -> None:
    leaders_ready = threading.Barrier(concurrency)
    jobs = [("leader", index) for index in range(1, concurrency + 1)]
    jobs.extend(("follower", index) for index in range(1, 29))

    def run_job(job: tuple[str, int]) -> None:
        role, index = job
        if role == "leader":
            try:
                leaders_ready.wait(timeout=5)
            except threading.BrokenBarrierError as exc:
                raise PreflightFailure(
                    "saturation leaders did not start together"
                ) from exc
            result = verifier(
                _TIMEOUT_SOURCE,
                [""],
                [""],
                # With four real slots all leaders finish in three seconds.
                # A hidden two-slot queue needs two waves (six seconds), past
                # the client's five-second execution-plus-cleanup window.
                timeout=3,
                **settings,
            )
            _require(
                f"saturation leader {index}",
                result,
                status="timeout",
                passed=False,
                timed_out=True,
                tests_passed=0,
                tests_total=1,
            )
            return

        result = verifier(
            _FOLLOWER_SOURCE,
            [""],
            ["piston-preflight-follower"],
            timeout=10,
            **settings,
        )
        _require(
            f"saturation follower {index}",
            result,
            status="passed",
            passed=True,
            timed_out=False,
            tests_passed=1,
            tests_total=1,
        )

    # At most ``concurrency`` calls enter the supervisor. A failure sets the
    # shared abort flag before any worker can pull another queued job, so a
    # partial judge outage cannot drain all 28 followers in waves.
    _fail_fast_thread_map(run_job, jobs, max_workers=concurrency)


def run_preflight(
    *,
    url: str,
    runtime: str = _DEFAULT_RUNTIME,
    verifier: Verifier = codecontests.run_stdin_tests,
) -> None:
    """Exercise the real supervisor contract using only synthetic cases."""
    effective = codecontests._MAX_CONCURRENT_PISTON_VERIFIERS
    if (
        isinstance(effective, bool)
        or not isinstance(effective, int)
        or not 1 <= effective <= _PROTOCOL_MAX_CONCURRENCY
    ):
        raise PreflightFailure(
            "effective MAX_CONCURRENT_PISTON_VERIFIERS must be an integer "
            f"from 1 through {_PROTOCOL_MAX_CONCURRENCY}; "
            f"the imported verifier is using {effective!r}"
        )

    settings = _settings(url, runtime)

    print("[1/7] byte-exact stdin", flush=True)
    stdin_cases = ["", "unterminated", "line\n", "line\r\n"]
    result = verifier(
        _BYTE_ECHO_SOURCE,
        stdin_cases,
        [case.encode("utf-8").hex() for case in stdin_cases],
        timeout=20,
        **settings,
    )
    _require(
        "byte-exact stdin",
        result,
        status="passed",
        passed=True,
        timed_out=False,
        tests_passed=4,
        tests_total=4,
    )

    print("[2/7] normal exit", flush=True)
    result = verifier(
        _PASS_SOURCE,
        [""],
        ["piston-preflight-ok"],
        timeout=10,
        **settings,
    )
    _require(
        "normal exit",
        result,
        status="passed",
        passed=True,
        timed_out=False,
        tests_passed=1,
        tests_total=1,
    )

    print("[3/7] nonzero exits", flush=True)
    for exit_code, source in ((1, _EXIT_1_SOURCE), (120, _EXIT_120_SOURCE)):
        result = verifier(
            source,
            [""],
            ["piston-preflight-ok"],
            timeout=10,
            **settings,
        )
        _require(
            f"exit {exit_code}",
            result,
            status="failed",
            passed=False,
            timed_out=False,
            tests_passed=0,
            tests_total=1,
        )

    print("[4/7] timeout mapping", flush=True)
    result = verifier(
        _TIMEOUT_SOURCE,
        [""],
        [""],
        timeout=1,
        **settings,
    )
    _require(
        "timeout mapping",
        result,
        status="timeout",
        passed=False,
        timed_out=True,
        tests_passed=0,
        tests_total=1,
    )

    print("[5/7] 600 KiB request", flush=True)
    large_input = "a" * (600 * 1024)
    result = verifier(
        _LARGE_REQUEST_SOURCE,
        [large_input],
        [str(len(large_input))],
        timeout=10,
        **settings,
    )
    _require(
        "600 KiB request",
        result,
        status="passed",
        passed=True,
        timed_out=False,
        tests_passed=1,
        tests_total=1,
    )

    # The child may legally finish without consuming stdin. Stock Piston's
    # pipe handling used to turn this combination into an API-wide EPIPE
    # crash; keep it adjacent to the full-read probe so both directions are
    # required by every production preflight.
    result = verifier(
        _EARLY_EXIT_SOURCE,
        [large_input],
        [""],
        timeout=10,
        **settings,
    )
    _require(
        "600 KiB early exit",
        result,
        status="passed",
        passed=True,
        timed_out=False,
        tests_passed=1,
        tests_total=1,
    )

    print("[6/7] 2 MiB output cap", flush=True)
    result = verifier(
        _OUTPUT_FLOOD_SOURCE,
        [""],
        [""],
        timeout=10,
        **settings,
    )
    _require(
        "2 MiB output cap",
        result,
        status="candidate_error",
        passed=False,
        timed_out=False,
        tests_passed=0,
        tests_total=1,
    )

    print(f"[7/7] {effective}-slot saturation", flush=True)
    _check_saturation(
        verifier=verifier,
        settings=settings,
        concurrency=effective,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise the production CodeContests supervisor against a Piston "
            "service and fail on any contract mismatch."
        )
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Piston base URL (use an SSH-forwarded loopback URL for HTTP)",
    )
    parser.add_argument(
        "--runtime",
        default=_DEFAULT_RUNTIME,
        help=f"exact Piston Python runtime version (default: {_DEFAULT_RUNTIME})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_preflight(url=args.url, runtime=args.runtime)
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed
        print(f"PISTON PREFLIGHT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("PISTON PREFLIGHT PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
