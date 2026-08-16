"""Black-box contract for one durable, idempotent remote job attempt.

The behavior under test is the claim, the detach, the stream capture, and the
durable terminal result -- exercised with real subprocesses and real files
rather than mocks.

Everything about *how* the wrapper is addressed here is a PRIVATE test adapter
convention, not approved public contract: the ``python -m`` entry point and its
flags in ``_launcher_argv``, and the per-attempt record names in
``_attempt_dir``/``RECORDS``.  An implementation may choose other flags or
filenames; only those two private helpers should then need editing.  What *is*
contract: an attempt identity is claimed exactly once, its command runs as
exact argv, and its terminal result is absent until that command has actually
terminated.

TODO: how a caller other than these tests discovers the streams and the
terminal result is an open design decision, deliberately not asserted here.
"""

from __future__ import annotations

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest


MODULE = "tools.scheduler.attempt"
POLL_INTERVAL = 0.01
POLL_TIMEOUT = 5.0

STDOUT, STDERR, TERMINAL = "stdout.log", "stderr.log", "exit_code"
RECORDS = (STDOUT, STDERR, TERMINAL)

# Private adapter payload: append one execution marker, emit both streams, exit.
_RECORD_SCRIPT = """\
from pathlib import Path
import sys

with Path(sys.argv[1]).open("ab", buffering=0) as handle:
    handle.write((sys.argv[2] + "\\n").encode())
sys.stdout.write(sys.argv[3])
sys.stderr.write(sys.argv[4])
raise SystemExit(int(sys.argv[5]))
"""

# Private adapter payload: announce, block until released, then record one run.
_BARRIER_SCRIPT = """\
from pathlib import Path
import os
import sys
import time

ready, release, executions = map(Path, sys.argv[1:])
with ready.open("ab", buffering=0) as handle:
    handle.write(b"ready\\n")
while not release.exists():
    time.sleep(0.01)
fd = os.open(executions, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
try:
    os.write(fd, b"executed\\n")
    os.fsync(fd)
finally:
    os.close(fd)
"""


@pytest.fixture(autouse=True)
def require_attempt_wrapper() -> None:
    """Probe lazily so a missing future module does not abort collection."""
    try:
        spec = importlib.util.find_spec(MODULE)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        pytest.fail(
            f"{MODULE} is not implemented; these are the red contract tests "
            "for the remote attempt wrapper",
            pytrace=False,
        )


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    """An attempt root and a command working directory, both with spaces."""
    cwd = tmp_path / "working directory"
    cwd.mkdir()
    return tmp_path / "attempt root", cwd


def _attempt_dir(root: Path, job_id: str, attempt_id: str) -> Path:
    return root / job_id / attempt_id


def _records(attempt_dir: Path) -> dict[str, bytes]:
    return {name: (attempt_dir / name).read_bytes() for name in RECORDS}


def _launcher_argv(
    root: Path, job_id: str, attempt_id: str, cwd: Path, command: list[str]
) -> list[str]:
    return [
        sys.executable,
        "-m",
        MODULE,
        "--root",
        str(root),
        "--job-id",
        job_id,
        "--attempt-id",
        attempt_id,
        "--cwd",
        str(cwd),
        "--",
        *command,
    ]


def _dispatch(
    root: Path,
    job_id: str,
    attempt_id: str,
    cwd: Path,
    command: list[str],
    *,
    require_success: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        _launcher_argv(root, job_id, attempt_id, cwd, command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3.0,
        check=False,
    )
    if require_success:
        assert completed.returncode == 0, (
            f"attempt launcher exited {completed.returncode}: "
            f"{completed.stderr.decode(errors='replace')}"
        )
    return completed


def _script(cwd: Path, name: str, source: str) -> str:
    path = cwd / name
    path.write_text(source, encoding="utf-8")
    return str(path)


def _wait_until(predicate, description: str, timeout: float = POLL_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"timed out waiting for {description}")


def _await_terminal(attempt_dir: Path, expected: int) -> None:
    terminal = attempt_dir / TERMINAL
    expected_bytes = str(expected).encode("ascii")

    _wait_until(
        lambda: terminal.exists() and terminal.read_bytes().strip() == expected_bytes,
        f"terminal exit code {expected}",
    )


def test_command_runs_as_exact_argv_and_cwd_without_shell_interpolation(
    workspace, tmp_path: Path
):
    root, cwd = workspace
    observed = cwd / "observed argv.json"
    semicolon_sentinel = tmp_path / "semicolon-injection"
    expansion_sentinel = tmp_path / "expansion-injection"
    script = _script(
        cwd,
        "record argv.py",
        """\
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(
    json.dumps({"argv": sys.argv[2:], "cwd": str(Path.cwd())}, ensure_ascii=False),
    encoding="utf-8",
)
""",
    )
    arguments = [
        "has spaces",
        'both "double" and \'single\' quotes',
        f"; touch {semicolon_sentinel}",
        f"$(touch {expansion_sentinel})",
        "$HOME and ${PATH} stay literal",
        "first line\nsecond line",
        "trailing-backslash\\",
    ]

    _dispatch(
        root,
        "job-exact-argv",
        "attempt-1",
        cwd,
        [sys.executable, script, str(observed), *arguments],
    )
    _await_terminal(_attempt_dir(root, "job-exact-argv", "attempt-1"), 0)

    assert json.loads(observed.read_text(encoding="utf-8")) == {
        "argv": arguments,
        "cwd": str(cwd),
    }
    assert not semicolon_sentinel.exists()
    assert not expansion_sentinel.exists()


def test_records_exact_separate_streams_and_terminal_exit_code(workspace):
    root, cwd = workspace
    for exit_code in (0, 23):
        expected_stdout = b"stdout: first line\nstdout without trailing newline"
        expected_stderr = b"stderr: first line\nstderr without trailing newline"
        script = _script(
            cwd,
            "emit.py",
            """\
import sys

sys.stdout.buffer.write(bytes.fromhex(sys.argv[1]))
sys.stdout.buffer.flush()
sys.stderr.buffer.write(bytes.fromhex(sys.argv[2]))
sys.stderr.buffer.flush()
raise SystemExit(int(sys.argv[3]))
""",
        )
        attempt_id = f"attempt-exit-{exit_code}"

        _dispatch(
            root,
            "job-streams",
            attempt_id,
            cwd,
            [
                sys.executable,
                script,
                expected_stdout.hex(),
                expected_stderr.hex(),
                str(exit_code),
            ],
        )
        attempt_dir = _attempt_dir(root, "job-streams", attempt_id)
        _await_terminal(attempt_dir, exit_code)

        assert (attempt_dir / STDOUT).read_bytes() == expected_stdout
        assert (attempt_dir / STDERR).read_bytes() == expected_stderr


def test_concurrent_dispatch_executes_once_and_redispatch_is_a_noop(
    workspace, tmp_path: Path
):
    root, cwd = workspace
    ready = tmp_path / "payload-ready"
    release = tmp_path / "release-payload"
    executions = tmp_path / "executions"
    command = [
        sys.executable,
        _script(cwd, "barrier.py", _BARRIER_SCRIPT),
        str(ready),
        str(release),
        str(executions),
    ]
    attempt_dir = _attempt_dir(root, "same-job", "same-attempt")
    caller_barrier = threading.Barrier(2)

    def dispatch():
        caller_barrier.wait(timeout=POLL_TIMEOUT)
        return _dispatch(root, "same-job", "same-attempt", cwd, command)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(dispatch) for _ in range(2)]
        try:
            _wait_until(ready.exists, "the claimed payload to reach its barrier")
            done, pending = wait(futures, timeout=2.0)
            assert not pending, "dispatch must detach while the payload still runs"
            assert all(future.result().returncode == 0 for future in done)

            assert attempt_dir.is_dir(), "the durable attempt claim must be observable"
            assert not (attempt_dir / TERMINAL).exists(), (
                "a non-terminal attempt has an unknown outcome; do not publish a "
                "guessed result"
            )
        finally:
            release.touch(exist_ok=True)

    _await_terminal(attempt_dir, 0)
    terminal_records = _records(attempt_dir)

    # Same identity, same command, already terminal: accepted and does nothing.
    _dispatch(root, "same-job", "same-attempt", cwd, command)

    assert ready.read_bytes().splitlines() == [b"ready"]
    assert executions.read_bytes().splitlines() == [b"executed"]
    assert _records(attempt_dir) == terminal_records


def test_same_identity_with_a_different_command_fails_closed(
    workspace, tmp_path: Path
):
    root, cwd = workspace
    executions = tmp_path / "executions"
    script = _script(cwd, "record.py", _RECORD_SCRIPT)
    attempt_dir = _attempt_dir(root, "immutable-job", "immutable-attempt")

    def dispatch(marker: str, stdout: str, stderr: str, exit_code: str, **kwargs):
        return _dispatch(
            root,
            "immutable-job",
            "immutable-attempt",
            cwd,
            [sys.executable, script, str(executions), marker, stdout, stderr, exit_code],
            **kwargs,
        )

    dispatch("first", "original stdout", "original stderr", "0")
    _await_terminal(attempt_dir, 0)
    original_records = _records(attempt_dir)

    rejected = dispatch(
        "second-command-must-not-run",
        "replacement stdout",
        "replacement stderr",
        "23",
        require_success=False,
    )

    assert rejected.returncode != 0
    assert executions.read_bytes().splitlines() == [b"first"]
    assert _records(attempt_dir) == original_records


def test_attempt_identity_cannot_escape_the_attempt_root(
    workspace, tmp_path: Path
):
    root, cwd = workspace
    root.mkdir()
    cases = (
        ("../escaped-by-job", "attempt", "escaped-by-job"),
        ("safe-job", "../../escaped-by-attempt", "escaped-by-attempt"),
        (None, "attempt", "escaped-absolute-job"),
    )
    for job_id, attempt_id, escaped_name in cases:
        escaped = tmp_path / escaped_name
        sentinel = tmp_path / f"{escaped_name}-command-ran"
        if job_id is None:
            job_id = str(escaped)

        rejected = _dispatch(
            root,
            job_id,
            attempt_id,
            cwd,
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).touch()",
            ],
            require_success=False,
        )

        assert rejected.returncode != 0
        assert not sentinel.exists()
        assert not escaped.exists()
