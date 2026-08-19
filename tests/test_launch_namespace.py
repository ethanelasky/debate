"""The launch namespace is one exact, immutable path component."""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import os
import socket
import subprocess
import sys
import textwrap
import time
import uuid
import warnings
from pathlib import Path

import pytest

from infra.launch_namespace import (
    ENV_VAR,
    claim_directory,
    open_claimed_read_fd,
    open_claimed_text_file,
    require_claimed_directory,
    resolve_launch_namespace,
    safe_path_component,
    validate_launch_namespace,
)
from infra.run_common import run_identity_suffix


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "value",
    ["a", "Attempt_12-3.4", "9" + "x" * 127],
)
def test_namespace_validation_preserves_approved_bytes(value):
    assert validate_launch_namespace(value) == value


@pytest.mark.parametrize(
    "value",
    ["", "_leading", "-leading", ".", "../escape", "has space", "é", "a" * 129],
)
def test_namespace_validation_rejects_non_components(value):
    with pytest.raises(ValueError, match="DEBATE_LAUNCH_NAMESPACE"):
        validate_launch_namespace(value)


def test_scheduler_environment_namespace_wins_unchanged():
    assert resolve_launch_namespace(environ={ENV_VAR: "scheduler-attempt.7"}) == (
        "scheduler-attempt.7"
    )


def test_explicit_namespace_wins_over_environment():
    assert resolve_launch_namespace(
        "explicit-attempt", environ={ENV_VAR: "environment-attempt"}
    ) == "explicit-attempt"


def test_present_but_empty_environment_namespace_refuses():
    with pytest.raises(ValueError, match="DEBATE_LAUNCH_NAMESPACE"):
        resolve_launch_namespace(environ={ENV_VAR: ""})


def test_manual_fallback_is_canonical_uuid4():
    value = resolve_launch_namespace(environ={})
    parsed = uuid.UUID(value)

    assert parsed.version == 4
    assert str(parsed) == value


def test_claim_directory_is_atomic_and_never_adopts(tmp_path):
    target = tmp_path / "run" / "attempt"
    assert claim_directory(target) == target
    assert target.is_dir()

    with pytest.raises(FileExistsError, match="refusing existing"):
        claim_directory(target)


def test_claim_directory_refuses_existing_file(tmp_path):
    target = tmp_path / "attempt"
    target.write_text("evidence")

    with pytest.raises(FileExistsError, match="refusing existing"):
        claim_directory(target)
    assert target.read_text() == "evidence"


def test_claim_directory_never_follows_symlinked_parent(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError, match="refusing unsafe directory component"):
        claim_directory(linked_parent / "attempt")

    assert not (outside / "attempt").exists()


def test_claim_directory_rejects_parent_components_without_normalizing(tmp_path):
    target = tmp_path / "inside" / ".." / "escaped"

    with pytest.raises(ValueError, match="parent components"):
        claim_directory(target)

    assert not (tmp_path / "escaped").exists()


def test_claim_directory_never_follows_symlinked_leaf(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    leaf = tmp_path / "attempt"
    leaf.symlink_to(outside, target_is_directory=True)

    with pytest.raises(FileExistsError, match="refusing existing"):
        claim_directory(leaf)

    assert list(outside.iterdir()) == []


def test_replaced_ancestor_is_refused_without_writing_outside(tmp_path):
    anchor = tmp_path / "anchor"
    parent = anchor / "parent"
    parent.mkdir(parents=True)
    retained = anchor / "parent-retained"
    parent.rename(retained)
    outside = tmp_path / "outside"
    outside.mkdir()
    parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError, match="refusing unsafe directory component"):
        claim_directory(parent / "nested" / "attempt")

    assert not (outside / "nested").exists()
    assert retained.is_dir()


def test_preclaimed_directory_refuses_ancestor_replacement(tmp_path):
    anchor = tmp_path / "anchor"
    parent = anchor / "parent"
    target = parent / "attempt"
    claim_directory(target)
    retained = anchor / "parent-retained"
    parent.rename(retained)
    outside = tmp_path / "outside"
    outside.mkdir()
    parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="no longer safely reachable"):
        require_claimed_directory(target)

    assert (retained / "attempt").is_dir()
    assert not (outside / "attempt").exists()


def test_existing_directory_cannot_be_adopted_as_a_process_claim(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()

    with pytest.raises(ValueError, match="not claimed by this process"):
        require_claimed_directory(target)


def test_retained_writer_refuses_real_directory_replacement(tmp_path):
    parent = tmp_path / "artifacts"
    target = parent / "run" / "attempt"
    claim_directory(target)
    retained = tmp_path / "artifacts-retained"
    parent.rename(retained)
    replacement = tmp_path / "artifacts" / "run" / "attempt"
    replacement.mkdir(parents=True)

    with pytest.raises(ValueError, match="identity changed"):
        open_claimed_text_file(target, "step-00001.jsonl")

    assert list(replacement.iterdir()) == []
    assert list((retained / "run" / "attempt").iterdir()) == []


def test_retained_writer_exclusively_creates_regular_output(tmp_path):
    target = tmp_path / "artifacts" / "run" / "attempt"
    claim_directory(target)

    with open_claimed_text_file(target, "step-00001.jsonl") as handle:
        handle.write("evidence\n")
    original = (target / "step-00001.jsonl").read_bytes()

    with pytest.raises(FileExistsError, match="refusing existing launch output"):
        open_claimed_text_file(target, "step-00001.jsonl")

    assert (target / "step-00001.jsonl").read_bytes() == original


def test_retained_reader_returns_blocking_read_only_fd_at_byte_zero(tmp_path):
    target = tmp_path / "artifacts" / "run" / "attempt"
    claim_directory(target)
    payload = b"complete evidence\n"
    source = target / "step-00001.jsonl"
    source.write_bytes(payload)
    source.chmod(0o640)

    file_fd = open_claimed_read_fd(target, source.name)
    try:
        status_flags = fcntl.fcntl(file_fd, fcntl.F_GETFL)
        assert status_flags & os.O_ACCMODE == os.O_RDONLY
        assert not status_flags & os.O_NONBLOCK
        assert not os.get_inheritable(file_fd)
        assert os.lseek(file_fd, 0, os.SEEK_CUR) == 0
        assert os.read(file_fd, len(payload) + 1) == payload
    finally:
        os.close(file_fd)


@pytest.mark.parametrize(
    "filename",
    ["", ".", "..", "../escape", "nested/file", "has space", "é", "a" * 256],
)
def test_retained_reader_refuses_unsafe_leaf_names(tmp_path, filename):
    target = tmp_path / "attempt"
    claim_directory(target)

    with pytest.raises(ValueError, match="one safe component"):
        open_claimed_read_fd(target, filename)


def test_retained_reader_requires_this_process_claim(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    (target / "result.json").write_text("evidence")

    with pytest.raises(ValueError, match="not claimed by this process"):
        open_claimed_read_fd(target, "result.json")


def test_retained_reader_refuses_replaced_claimed_directory(tmp_path):
    parent = tmp_path / "artifacts"
    target = parent / "run" / "attempt"
    claim_directory(target)
    (target / "result.json").write_text("retained")
    retained = tmp_path / "artifacts-retained"
    parent.rename(retained)
    replacement = tmp_path / "artifacts" / "run" / "attempt"
    replacement.mkdir(parents=True)
    (replacement / "result.json").write_text("replacement")

    with pytest.raises(ValueError, match="identity changed"):
        open_claimed_read_fd(target, "result.json")


def test_retained_reader_refuses_symlink_and_hardlink(tmp_path):
    target = tmp_path / "attempt"
    claim_directory(target)
    source = target / "source.json"
    source.write_text("evidence")
    (target / "symlink.json").symlink_to(source.name)
    os.link(source, target / "hardlink.json")

    with pytest.raises(RuntimeError, match="unsafe claimed input"):
        open_claimed_read_fd(target, "symlink.json")
    with pytest.raises(RuntimeError, match="exactly one hard link"):
        open_claimed_read_fd(target, "hardlink.json")


def test_retained_reader_refuses_fifo_without_blocking(tmp_path):
    target = tmp_path / "attempt"
    claim_directory(target)
    os.mkfifo(target / "stream")

    with pytest.raises(RuntimeError, match="not a regular file"):
        open_claimed_read_fd(target, "stream")


def test_retained_reader_refuses_unix_socket(tmp_path):
    target = tmp_path / "attempt"
    claim_directory(target)
    server = socket.socket(socket.AF_UNIX)
    try:
        server.bind(str(target / "service.sock"))
        with pytest.raises(RuntimeError, match="unsafe claimed input"):
            open_claimed_read_fd(target, "service.sock")
    finally:
        server.close()


def test_retained_reader_refuses_group_or_other_writable_file(tmp_path):
    target = tmp_path / "attempt"
    claim_directory(target)
    source = target / "result.json"
    source.write_text("evidence")
    source.chmod(0o660)

    with pytest.raises(RuntimeError, match="group- or other-writable"):
        open_claimed_read_fd(target, source.name)


def test_retained_reader_fd_pins_inode_across_named_path_replacement(tmp_path):
    target = tmp_path / "attempt"
    claim_directory(target)
    source = target / "result.json"
    source.write_bytes(b"original")
    original_inode = source.stat().st_ino

    file_fd = open_claimed_read_fd(target, source.name)
    try:
        source.rename(target / "original-retained.json")
        source.write_bytes(b"replacement")

        assert os.fstat(file_fd).st_ino == original_inode
        assert os.read(file_fd, 32) == b"original"
        assert source.read_bytes() == b"replacement"
    finally:
        os.close(file_fd)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_forked_child_does_not_inherit_parent_claim_authority(tmp_path):
    target = tmp_path / "attempt"
    claim_directory(target)
    read_fd, write_fd = os.pipe()
    with warnings.catch_warnings():
        # Python 3.12 warns about fork in a multithreaded parent. The module's
        # at-fork hook replaces its lock; the child exercises only that narrow
        # contract and exits immediately without touching pytest state.
        warnings.simplefilter("ignore", DeprecationWarning)
        child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        try:
            require_claimed_directory(target)
        except ValueError:
            os.write(write_fd, b"refused")
            os._exit(0)
        os.write(write_fd, b"adopted")
        os._exit(1)

    os.close(write_fd)
    message = os.read(read_fd, 64)
    os.close(read_fd)
    waited_pid, status = os.waitpid(child_pid, 0)

    assert waited_pid == child_pid
    assert os.waitstatus_to_exitcode(status) == 0
    assert message == b"refused"
    assert require_claimed_directory(target) == target


def test_safe_path_component_does_not_collapse_unsafe_run_names():
    slash = safe_path_component("../../arm/a", fallback="run")
    backslash = safe_path_component(r"..\..\arm\a", fallback="run")

    assert slash != backslash
    for value in (slash, backslash):
        assert "/" not in value and "\\" not in value
        assert value not in {"", ".", ".."}


def test_manual_fallback_is_unique_across_real_processes():
    env = os.environ.copy()
    env.pop(ENV_VAR, None)
    code = (
        "from infra.launch_namespace import resolve_launch_namespace; "
        "print(resolve_launch_namespace())"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(32)
    ]
    results = [process.communicate(timeout=20) for process in processes]

    assert all(process.returncode == 0 for process in processes), results
    values = [stdout.strip() for stdout, _ in results]
    assert len(set(values)) == 32
    for value in values:
        parsed = uuid.UUID(value)
        assert parsed.version == 4
        assert str(parsed) == value


def test_real_processes_racing_for_one_directory_get_exactly_one_winner(tmp_path):
    target = tmp_path / "checkpoints" / "run" / "scheduler-attempt"
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    gate = tmp_path / "go"
    code = textwrap.dedent(
        """
        import pathlib
        import sys
        import time

        from infra.launch_namespace import claim_directory

        target = pathlib.Path(sys.argv[1])
        ready = pathlib.Path(sys.argv[2])
        gate = pathlib.Path(sys.argv[3])
        token = sys.argv[4]
        ready.write_text(token)
        deadline = time.monotonic() + 20
        while not gate.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("barrier was never released")
            time.sleep(0.005)
        try:
            claimed = claim_directory(target)
        except FileExistsError:
            raise SystemExit(17)
        with open(claimed / "winner", "x", encoding="utf-8") as handle:
            handle.write(token)
        print(token)
        """
    )
    env = os.environ.copy()
    env.pop(ENV_VAR, None)
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                code,
                str(target),
                str(ready_dir / f"ready-{index}"),
                str(gate),
                f"process-{index}",
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(16)
    ]
    deadline = time.monotonic() + 20
    while len(list(ready_dir.iterdir())) != len(processes):
        if time.monotonic() >= deadline:
            gate.touch()
            outputs = [process.communicate(timeout=5) for process in processes]
            pytest.fail(f"children did not reach barrier: {outputs!r}")
        time.sleep(0.01)
    gate.touch()
    results = [process.communicate(timeout=20) for process in processes]

    winners = [
        stdout.strip()
        for process, (stdout, _) in zip(processes, results)
        if process.returncode == 0
    ]
    losers = [process for process in processes if process.returncode != 0]
    assert len(winners) == 1
    assert len(losers) == 15
    assert all(process.returncode == 17 for process in losers), results
    sentinel = target / "winner"
    assert sentinel.read_text() == winners[0]

    before = (
        target.stat().st_ino,
        target.stat().st_mode,
        target.stat().st_mtime_ns,
        sentinel.stat().st_ino,
        sentinel.stat().st_mode,
        sentinel.stat().st_mtime_ns,
        sentinel.read_bytes(),
    )
    reuse = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from infra.launch_namespace import claim_directory; "
                "claim_directory(sys.argv[1])"
            ),
            str(target),
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
        check=False,
    )
    after = (
        target.stat().st_ino,
        target.stat().st_mode,
        target.stat().st_mtime_ns,
        sentinel.stat().st_ino,
        sentinel.stat().st_mode,
        sentinel.stat().st_mtime_ns,
        sentinel.read_bytes(),
    )
    assert reuse.returncode != 0
    assert before == after


def test_run_identity_suffix_source_is_byte_for_byte_historical_definition():
    source = inspect.getsource(run_identity_suffix).encode("utf-8")
    assert hashlib.sha256(source).hexdigest() == (
        "3dfdbd08d614c7182755ed21b3ef1779387998d9f778440350399649e853a176"
    )
