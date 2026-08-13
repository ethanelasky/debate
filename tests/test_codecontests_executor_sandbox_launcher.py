"""Pure regressions for the trusted guest monitor's syscall policy."""

from __future__ import annotations

import errno
import io
import signal
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any, cast

import pytest

from codecontests_executor import sandbox_launcher as launcher

_V2_SECCOMP_CREATION_MODE = launcher._seccomp_creation_mode


def _legacy_process_creation_decoder(
    event_data: int,
) -> tuple[int, tuple[str, int] | None]:
    """Decode now-unreachable process tags for generic teardown regressions.

    The v2 candidate BPF policy never emits these tags: process creation is
    rejected directly with EPERM.  A few lower-level tracer tests intentionally
    inject historical in-flight fork/vfork transitions to prove teardown stays
    fail-closed if policy drift ever reintroduces one.
    """

    if event_data == launcher._SECCOMP_AARCH64_FORK:
        return 220, ("cpython_fork_clone", launcher._PTRACE_EVENT_FORK)
    if event_data == launcher._SECCOMP_AARCH64_VFORK:
        return 220, ("glibc_vfork_clone", launcher._PTRACE_EVENT_VFORK)
    return _V2_SECCOMP_CREATION_MODE(event_data)


def _stopped(signo: int, event: int = 0) -> int:
    return (event << 16) | (signo << 8) | 0x7F


def _signaled(signo: int) -> int:
    return signo


def _exited(code: int) -> int:
    return code << 8


def _exercise_trace(
    monkeypatch: pytest.MonkeyPatch,
    waits: Iterable[tuple[int, int]],
    *,
    event_messages: Iterable[int],
    syscall_results: Iterable[int] = (),
    siginfos: Iterable[tuple[int, int]] = (),
    candidate_tasks: Callable[[], set[int]] = set,
    teardown_control: launcher._TeardownControl | None = None,
    ptrace_esrch_calls: frozenset[int] = frozenset(),
    ptrace_hook: Callable[[int, int], None] | None = None,
    group_kills: list[int] | None = None,
    group_stops: list[int] | None = None,
    group_kill_hook: Callable[[], None] | None = None,
    repeat_no_wait: bool = False,
    monotonic_values: Iterable[float] | None = None,
    seccomp_creation_mode: Callable[[int], tuple[int, Any]] | None = None,
) -> tuple[tuple[Any, ...], list[tuple[int, int, Any]]]:
    usage = SimpleNamespace(ru_utime=0.001, ru_stime=0.002)
    wait_results = iter((pid, status, usage) for pid, status in waits)
    event_results = iter(event_messages)
    syscall_result_values = iter(syscall_results)
    siginfo_values = iter(siginfos)
    ptrace_calls: list[tuple[int, int, Any]] = []
    proc = cast(Any, SimpleNamespace(pid=101, returncode=None))
    if teardown_control is None:
        teardown_control = launcher._TeardownControl()
    if group_kills is None:
        group_kills = []
    if group_stops is None:
        group_stops = []

    def next_wait(*_args: Any) -> tuple[int, int, Any]:
        try:
            return next(wait_results)
        except StopIteration:
            if repeat_no_wait:
                return 0, 0, usage
            raise

    monkeypatch.setattr(launcher.os, "wait4", next_wait)
    monkeypatch.setattr(
        launcher,
        "_ptrace_event_message",
        lambda _pid: next(event_results),
    )
    monkeypatch.setattr(
        launcher,
        "_ptrace_syscall_result",
        lambda _pid: next(syscall_result_values),
    )
    monkeypatch.setattr(
        launcher,
        "_ptrace_siginfo",
        lambda _pid: next(siginfo_values),
    )
    monkeypatch.setattr(launcher, "_candidate_task_ids", candidate_tasks)
    monkeypatch.setattr(launcher, "_candidate_proc_attested", lambda *_args: True)
    monkeypatch.setattr(launcher, "_candidate_open_file_size", lambda _pid: 4096)
    monkeypatch.setattr(
        launcher,
        "_seccomp_creation_mode",
        seccomp_creation_mode or (lambda event_data: (event_data, None)),
    )
    monkeypatch.setattr(launcher, "_ptrace_skip_syscall", lambda _pid: None)
    monkeypatch.setattr(launcher, "_ptrace_set_syscall_result", lambda *_args: None)
    monkeypatch.setattr(launcher, "_kill_candidate_tree", lambda _proc: None)

    def record_group_kill(observed_proc: Any) -> None:
        group_kills.append(observed_proc.pid)
        if group_kill_hook is not None:
            group_kill_hook()

    monkeypatch.setattr(launcher, "_kill_candidate_group_once", record_group_kill)
    monkeypatch.setattr(
        launcher,
        "_stop_candidate_group_once",
        lambda observed_proc: group_stops.append(observed_proc.pid),
    )
    if monotonic_values is not None:
        monotonic_results = iter(monotonic_values)
        monkeypatch.setattr(launcher.time, "monotonic", lambda: next(monotonic_results))

    def record_ptrace(
        request: int,
        pid: int,
        address: int = 0,
        data: Any = None,
    ) -> int:
        del address
        ptrace_calls.append((request, pid, data))
        if ptrace_hook is not None:
            ptrace_hook(request, pid)
        if len(ptrace_calls) in ptrace_esrch_calls:
            raise launcher._PtraceError(request, pid, errno.ESRCH)
        return 0

    monkeypatch.setattr(launcher, "_ptrace", record_ptrace)
    result = launcher._trace_and_measure(
        proc,
        limits=(1024, 1, 4096, 16, 32),
        process_limit=16,
        file_size_limit=4096,
        teardown_control=teardown_control,
    )
    return result, ptrace_calls


@pytest.mark.parametrize(
    ("machine", "syscall_number", "arguments"),
    [
        (
            "x86_64",
            56,
            (
                launcher._CPYTHON_THREAD_CLONE_FLAGS,
                0x1000,
                0x2000,
                0x3000,
                0x4000,
                0,
            ),
        ),
        (
            "aarch64",
            220,
            (
                launcher._CPYTHON_THREAD_CLONE_FLAGS,
                0x1000,
                0x2000,
                0x4000,
                0x3000,
                0,
            ),
        ),
    ],
)
def test_only_pinned_raw_cpython_pthread_clone_layout_is_accepted(
    machine: str,
    syscall_number: int,
    arguments: tuple[int, ...],
) -> None:
    assert launcher._clone_mode(machine, syscall_number, arguments) == (
        "cpython_pthread_clone",
        launcher._PTRACE_EVENT_CLONE,
    )


@pytest.mark.parametrize(
    ("machine", "syscall_number", "arguments"),
    [
        ("x86_64", 57, (0, 0, 0, 0, 0, 0)),
        ("x86_64", 58, (0, 0, 0, 0, 0, 0)),
        (
            "x86_64",
            56,
            (launcher._CPYTHON_FORK_CLONE_FLAGS, 0, 0, 0x1000, 0, 0),
        ),
        (
            "aarch64",
            220,
            (launcher._CPYTHON_FORK_CLONE_FLAGS, 0, 0, 0, 0x1000, 0),
        ),
    ],
)
def test_every_legacy_process_creation_layout_is_denied(
    machine: str,
    syscall_number: int,
    arguments: tuple[int, ...],
) -> None:
    assert launcher._clone_mode(machine, syscall_number, arguments) is None


@pytest.mark.parametrize(
    ("machine", "syscall_number"),
    [("x86_64", 56), ("aarch64", 220)],
)
@pytest.mark.parametrize(
    "escape_flag",
    [
        0x00002000,
        0x00008000,
        0x00020000,
        0x00800000,
        0x02000000,
        0x10000000,
        0x20000000,
        0x40000000,
    ],
)
def test_raw_legacy_clone_escape_flags_are_denied(
    machine: str,
    syscall_number: int,
    escape_flag: int,
) -> None:
    assert (
        launcher._clone_mode(
            machine,
            syscall_number,
            (
                launcher._CPYTHON_THREAD_CLONE_FLAGS | escape_flag,
                1,
                1,
                1,
                1,
                0,
            ),
        )
        is None
    )


def test_candidate_seccomp_denies_namespace_mount_and_process_group_changes() -> None:
    namespace: dict[str, object] = {}
    source = launcher._CANDIDATE_RUNNER.rsplit("\nmain()", 1)[0]
    exec(  # noqa: S102 - execute the trusted checked-in runner constant only
        compile(source, "<candidate-policy>", "exec"), namespace
    )
    policy = cast(
        Callable[
            [str],
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        ],
        namespace["syscall_policy"],
    )
    assert namespace["AUDIT_ARCH_AARCH64"] == 0xC00000B7
    assert namespace["AUDIT_ARCH_X86_64"] == 0xC000003E
    assert namespace["SECCOMP_RET_KILL_PROCESS"] == 0x80000000
    arm_deny, arm_enosys, arm_trace = policy("aarch64")
    x86_deny, x86_enosys, x86_trace = policy("x86_64")
    assert arm_enosys == x86_enosys == (435,)
    assert 435 not in arm_trace
    assert 435 not in x86_trace
    assert 220 in arm_trace
    assert 56 in x86_trace
    assert {57, 58} <= set(x86_deny)
    assert {57, 58}.isdisjoint(x86_trace)
    assert {
        39,
        40,
        41,
        51,
        97,
        154,
        157,
        268,
        428,
        429,
        430,
        431,
        432,
        433,
        442,
    } <= set(arm_deny)
    assert {
        109,
        112,
        155,
        161,
        165,
        166,
        272,
        308,
        428,
        429,
        430,
        431,
        432,
        433,
        442,
    } <= set(x86_deny)


def test_pthread_clone_layout_is_validated_in_bpf_before_ptrace_register_loss() -> None:
    namespace: dict[str, object] = {}
    source = launcher._CANDIDATE_RUNNER.rsplit("\nmain()", 1)[0]
    exec(  # noqa: S102 - execute the trusted checked-in runner constant only
        compile(source, "<candidate-policy>", "exec"), namespace
    )
    clone_filter = cast(Callable[[int], list[Any]], namespace["pthread_clone_filter"])
    load = cast(int, namespace["BPF_LD_W_ABS"])
    equal = cast(int, namespace["BPF_JMP_JEQ_K"])
    ret = cast(int, namespace["BPF_RET_K"])
    trace = cast(int, namespace["SECCOMP_RET_TRACE"])
    errno_result = cast(int, namespace["SECCOMP_RET_ERRNO"]) | cast(
        int, namespace["EPERM"]
    )
    thread_flags = cast(int, namespace["CPYTHON_THREAD_CLONE_FLAGS"])
    thread_tag = cast(int, namespace["AARCH64_THREAD_TAG"])
    trace_value = trace | thread_tag
    instructions = clone_filter(trace_value)

    def evaluate(arguments: tuple[int, ...]) -> int:
        seccomp_data = bytearray(64)
        for index, argument in enumerate(arguments):
            struct.pack_into("=Q", seccomp_data, 16 + 8 * index, argument)
        accumulator = 0
        instruction_pointer = 0
        while True:
            instruction = instructions[instruction_pointer]
            if instruction.code == load:
                accumulator = struct.unpack_from("=I", seccomp_data, instruction.k)[0]
                instruction_pointer += 1
                continue
            if instruction.code == equal:
                instruction_pointer += (
                    instruction.jt if accumulator == instruction.k else instruction.jf
                ) + 1
                continue
            if instruction.code == ret:
                return int(instruction.k)
            raise AssertionError(f"unexpected BPF opcode: {instruction.code}")

    assert evaluate((thread_flags, 1, 1, 1, 1, 0)) == trace_value
    assert evaluate((thread_flags, 1 << 32, 1, 1, 1, 0)) == trace_value
    assert evaluate((cast(int, namespace["CPYTHON_FORK_CLONE_FLAGS"]), 1, 1, 1, 1, 0)) == errno_result
    assert evaluate((cast(int, namespace["GLIBC_VFORK_CLONE_FLAGS"]), 1, 1, 1, 1, 0)) == errno_result
    assert evaluate((0x10000000 | 17, 0, 0, 0, 1, 0)) == errno_result
    assert evaluate((thread_flags | (1 << 32), 1, 1, 1, 1, 0)) == errno_result
    for missing_pointer in range(1, 5):
        arguments = [thread_flags, 1, 1, 1, 1, 0]
        arguments[missing_pointer] = 0
        assert evaluate(tuple(arguments)) == errno_result

    assert thread_tag == launcher._SECCOMP_AARCH64_THREAD


def test_aarch64_seccomp_clone_tags_bind_exact_ptrace_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher.platform, "machine", lambda: "aarch64")

    assert launcher._seccomp_creation_mode(launcher._SECCOMP_AARCH64_THREAD) == (
        220,
        ("cpython_pthread_clone", launcher._PTRACE_EVENT_CLONE),
    )
    for forbidden_tag in (
        launcher._SECCOMP_AARCH64_FORK,
        launcher._SECCOMP_AARCH64_VFORK,
    ):
        with pytest.raises(RuntimeError, match="process-clone.*forbidden"):
            launcher._seccomp_creation_mode(forbidden_tag)
    with pytest.raises(RuntimeError, match="lacks its seccomp layout tag"):
        launcher._seccomp_creation_mode(220)

    monkeypatch.setattr(launcher.platform, "machine", lambda: "x86_64")
    assert launcher._seccomp_creation_mode(56) == (56, None)


def test_candidate_drops_and_attests_all_capabilities_before_source_bytes() -> None:
    source = launcher._CANDIDATE_RUNNER
    drop_bounding = source.index("for capability in range(CAP_LAST_CAP + 1):")
    drop_uid = source.index("os.setresuid(UID, UID, UID)")
    attest_caps = source.index(
        'for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):'
    )
    read_source = source.index("source = read_exact(code_fd, source_size)")
    compile_source = source.index(
        'compiled = compile(source, "/tmp/solution.py", "exec")'
    )

    assert drop_bounding < drop_uid < attest_caps < read_source < compile_source


def test_candidate_proc_attestation_requires_zero_status_capability_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits = (4_294_967_296, 2, 2_097_152, 2, 64)
    status_template = """\
Name:\tpython3
Uid:\t65534\t65534\t65534\t65534
Gid:\t65534\t65534\t65534\t65534
Groups:\t
CapInh:\t{cap_inh}
CapPrm:\t{cap_prm}
CapEff:\t{cap_eff}
CapBnd:\t{cap_bnd}
CapAmb:\t{cap_amb}
"""
    limits_text = """\
Limit                     Soft Limit           Hard Limit           Units
Max cpu time              2                    3                    seconds
Max file size             2097152              2097152              bytes
Max processes             1                    1                    processes
Max open files            64                   64                   files
Max address space         4294967296           4294967296           bytes
"""
    capability_values = {
        "cap_inh": "0000000000000000",
        "cap_prm": "0000000000000000",
        "cap_eff": "0000000000000000",
        "cap_bnd": "0000000000000000",
        "cap_amb": "0000000000000000",
    }

    def attest(values: dict[str, str]) -> bool:
        def fake_open(path: str, **_kwargs: Any) -> io.StringIO:
            if path.endswith("/status"):
                return io.StringIO(status_template.format(**values))
            if path.endswith("/limits"):
                return io.StringIO(limits_text)
            raise AssertionError(f"unexpected proc path: {path}")

        monkeypatch.setattr("builtins.open", fake_open)
        return launcher._candidate_proc_attested(123, limits)

    assert attest(capability_values) is True
    for name in capability_values:
        inherited_kill = {**capability_values, name: "0000000000000020"}
        assert attest(inherited_kill) is False


def test_every_successful_creation_requires_exact_event_and_task_equality() -> None:
    pending = launcher._PendingSyscall(
        number=220,
        saw_entry=True,
        creation_mode="cpython_pthread_clone",
        expected_event=launcher._PTRACE_EVENT_CLONE,
        event_child=22,
    )
    launcher._validate_creation_result(pending, 22, {11, 22}, {11, 22})
    with pytest.raises(RuntimeError, match="exact ptrace child"):
        launcher._validate_creation_result(pending, 23, {11, 22}, {11, 22})
    with pytest.raises(RuntimeError, match="inventory"):
        launcher._validate_creation_result(pending, 22, {11, 22}, {11})

    pending.event_child = None
    launcher._validate_creation_result(pending, -11, {11}, {11})
    pending.event_child = 22
    with pytest.raises(RuntimeError, match="failed creation"):
        launcher._validate_creation_result(pending, -11, {11, 22}, {11, 22})


def test_ptrace_failure_exposes_structured_request_pid_and_errno(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingPtrace:
        restype: Any = None

        def __call__(self, *_args: Any) -> int:
            return -1

    monkeypatch.setattr(
        launcher.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(ptrace=FailingPtrace()),
    )
    monkeypatch.setattr(launcher.ctypes, "get_errno", lambda: errno.ESRCH)

    with pytest.raises(launcher._PtraceError) as caught:
        launcher._ptrace(launcher._PTRACE_SYSCALL, 101)

    assert caught.value.request == launcher._PTRACE_SYSCALL
    assert caught.value.pid == 101
    assert caught.value.error_number == errno.ESRCH


def test_general_ptrace_esrch_remains_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_syscall = min(launcher._file_write_syscalls())

    with pytest.raises(launcher._PtraceError) as caught:
        _exercise_trace(
            monkeypatch,
            [
                (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)),
                (101, _stopped(signal.SIGTRAP | 0x80)),
                (101, _stopped(signal.SIGSTOP)),
            ],
            event_messages=[write_syscall],
            syscall_results=[-launcher._ENOSYS],
            ptrace_esrch_calls=frozenset({3}),
        )

    assert caught.value.request == launcher._PTRACE_SYSCALL
    assert caught.value.pid == 101
    assert caught.value.error_number == errno.ESRCH


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux ptrace")
def test_monitor_failure_hard_exits_with_ptraced_relay_pipe_open() -> None:
    program = r"""
import os
import signal
import threading

from codecontests_executor import sandbox_launcher as launcher


def fail_with_ptraced_relay(*_args, **_kwargs):
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        launcher._ptrace(launcher._PTRACE_TRACEME, 0)
        os.kill(os.getpid(), signal.SIGSTOP)
        signal.pause()
        os._exit(0)

    os.close(write_fd)
    waited, status, _usage = os.wait4(
        child,
        os.WUNTRACED | launcher._WAIT_ALL,
    )
    if (
        waited != child
        or not os.WIFSTOPPED(status)
        or os.WSTOPSIG(status) != signal.SIGSTOP
    ):
        os._exit(97)
    launcher._ptrace(
        launcher._PTRACE_SETOPTIONS,
        child,
        data=launcher._PTRACE_O_TRACEEXIT | launcher._PTRACE_O_EXITKILL,
    )
    launcher._ptrace(launcher._PTRACE_CONT, child)
    threading.Thread(target=lambda: os.read(read_fd, 1)).start()
    os.kill(child, signal.SIGKILL)
    waited, status, _usage = os.wait4(
        child,
        os.WUNTRACED | launcher._WAIT_ALL,
    )
    if (
        waited != child
        or not os.WIFSTOPPED(status)
        or status >> 16 != launcher._PTRACE_EVENT_EXIT
    ):
        os._exit(98)
    os.write(2, b"PTRACED_RELAY_ARMED\n")
    raise RuntimeError("forced trusted-monitor failure")


def fail_status_emission(*_args, **_kwargs):
    raise BrokenPipeError("forced status failure")


launcher._prepare_monitor = lambda: (
    b"",
    "a" * 64,
    (1024, 1, 4096, 16, 32),
)
launcher._read_exact = lambda *_args: b"G"
launcher._run_candidate = fail_with_ptraced_relay
launcher._emit_status = fail_status_emission
launcher.main()
"""
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        check=False,
        timeout=2,
    )
    elapsed = time.monotonic() - started

    assert completed.returncode == launcher._INFRA_EXIT
    assert b"PTRACED_RELAY_ARMED\n" in completed.stderr
    assert elapsed < 1.5


def test_output_relay_records_evidence_and_never_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {"bytes": 0, "exceeded": False}
    teardown_control = launcher._TeardownControl()
    emitted: list[bytes] = []
    readable = cast(Any, SimpleNamespace(fileno=lambda: 99))

    monkeypatch.setattr(launcher.os, "read", lambda *_args: b"abc")
    monkeypatch.setattr(
        launcher,
        "_write_all",
        lambda _fd, value: emitted.append(value),
    )
    monkeypatch.setattr(
        launcher,
        "_kill_candidate_tree",
        lambda _proc: pytest.fail("relay must not signal the candidate"),
    )
    monkeypatch.setattr(
        launcher,
        "_kill_candidate_group_once",
        lambda _proc: pytest.fail("relay must not claim the tracer kill"),
    )

    launcher._relay(readable, 1, 2, state, teardown_control)

    assert emitted == [b"ab"]
    assert state == {"bytes": 2, "exceeded": True}
    assert teardown_control.output_limit_requested() is True
    assert teardown_control.kill_record() is None


def test_concurrent_output_relays_only_wake_one_evidence_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teardown_control = launcher._TeardownControl()
    states: list[dict[str, Any]] = [
        {"bytes": 0, "exceeded": False},
        {"bytes": 0, "exceeded": False},
    ]
    barrier = threading.Barrier(3)

    def read_after_both_relays_arrive(_fd: int, _size: int) -> bytes:
        barrier.wait()
        return b"x"

    monkeypatch.setattr(launcher.os, "read", read_after_both_relays_arrive)
    monkeypatch.setattr(
        launcher,
        "_kill_candidate_group_once",
        lambda _proc: pytest.fail("relay must never kill"),
    )
    threads = []
    for index, fd in enumerate((10, 11)):
        threads.append(
            threading.Thread(
                target=launcher._relay,
                args=(
                    cast(Any, SimpleNamespace(fileno=lambda fd=fd: fd)),
                    fd,
                    0,
                    states[index],
                    teardown_control,
                ),
            )
        )
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert not any(thread.is_alive() for thread in threads)
    assert states == [
        {"bytes": 0, "exceeded": True},
        {"bytes": 0, "exceeded": True},
    ]
    assert teardown_control.output_limit_requested() is True
    assert teardown_control.kill_record() is None


def test_tracer_owns_one_output_kill_after_exact_inventory_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teardown_control = launcher._TeardownControl()
    teardown_control.record_output_limit()
    group_kills: list[int] = []
    group_stops: list[int] = []
    live_tasks = {101}

    def waits() -> Iterable[tuple[int, int]]:
        yield 0, 0
        yield 101, _stopped(signal.SIGSTOP)
        live_tasks.clear()
        yield 101, _signaled(signal.SIGKILL)

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=[],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
        group_stops=group_stops,
    )

    assert result[0] == -signal.SIGKILL
    assert result[12] is True
    assert ptrace_calls == []
    assert group_kills == [101]
    assert group_stops == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        targets=frozenset({101}),
    )


def test_relay_evidence_during_ptrace_only_causes_a_later_tracer_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_syscall = min(launcher._file_write_syscalls())
    teardown_control = launcher._TeardownControl()
    relay_state: dict[str, Any] = {"bytes": 0, "exceeded": False}
    readable = cast(Any, SimpleNamespace(fileno=lambda: 99))
    group_kills: list[int] = []
    relay_ran = False

    monkeypatch.setattr(launcher.os, "read", lambda *_args: b"x")

    def run_relay_during_first_ptrace(_request: int, _pid: int) -> None:
        nonlocal relay_ran
        if relay_ran:
            return
        relay_ran = True
        launcher._relay(readable, 1, 0, relay_state, teardown_control)
        assert group_kills == []

    live_tasks = {101}

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 0, 0
        yield 101, _stopped(signal.SIGSTOP)
        live_tasks.clear()
        yield 101, _signaled(signal.SIGKILL)

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=[write_syscall],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        ptrace_hook=run_relay_during_first_ptrace,
        group_kills=group_kills,
    )

    assert result[0] == -signal.SIGKILL
    assert relay_state == {"bytes": 0, "exceeded": True}
    assert ptrace_calls == [(launcher._PTRACE_SYSCALL, 101, None)]
    assert group_kills == [101]


def test_multitask_teardown_accepts_reversed_waits_after_proc_disappearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_FORK)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        teardown_control.record_output_limit()
        yield 0, 0
        yield 202, _stopped(signal.SIGSTOP)
        yield 101, _stopped(signal.SIGSTOP)
        live_tasks.remove(202)
        yield 202, _signaled(signal.SIGKILL)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield creation_syscall
        live_tasks.add(202)
        yield 202

    result, _ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
    )

    assert result[0] == -signal.SIGKILL
    assert result[2] == 2
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        targets=frozenset({101, 202}),
    )


def test_teardown_ledger_rejects_task_addition_after_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teardown_control = launcher._TeardownControl()
    teardown_control.record_output_limit()
    inventories = iter(({101}, {101, 404}))
    with pytest.raises(RuntimeError, match="without an exact ptrace event"):
        _exercise_trace(
            monkeypatch,
            [(0, 0), (0, 0)],
            event_messages=[],
            candidate_tasks=lambda: next(inventories),
            teardown_control=teardown_control,
        )


def test_teardown_rejects_task_addition_before_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teardown_control = launcher._TeardownControl()
    teardown_control.record_output_limit()
    with pytest.raises(RuntimeError, match="without an exact ptrace event"):
        _exercise_trace(
            monkeypatch,
            [(0, 0), (101, _stopped(signal.SIGSTOP))],
            event_messages=[],
            candidate_tasks=lambda: {101, 404},
            teardown_control=teardown_control,
        )


def test_teardown_continues_one_exit_stop_then_requires_sigkill_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teardown_control = launcher._TeardownControl()
    teardown_control.record_output_limit()
    live_tasks = {101}
    group_kills: list[int] = []

    def waits() -> Iterable[tuple[int, int]]:
        yield 0, 0
        yield 101, _stopped(signal.SIGSTOP)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXIT)
        live_tasks.clear()
        yield 101, _signaled(signal.SIGKILL)

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=[],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
    )

    assert result[0] == -signal.SIGKILL
    assert group_kills == [101]
    assert ptrace_calls == [(launcher._PTRACE_CONT, 101, None)]


def test_teardown_exit_stop_esrch_still_requires_exact_terminal_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teardown_control = launcher._TeardownControl()
    teardown_control.record_output_limit()
    live_tasks = {101}
    group_kills: list[int] = []

    def waits() -> Iterable[tuple[int, int]]:
        yield 0, 0
        yield 101, _stopped(signal.SIGSTOP)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXIT)
        live_tasks.clear()
        yield 101, _signaled(signal.SIGKILL)

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=[],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        ptrace_esrch_calls=frozenset({1}),
        group_kills=group_kills,
    )

    assert result[0] == -signal.SIGKILL
    assert ptrace_calls == [(launcher._PTRACE_CONT, 101, None)]
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        targets=frozenset({101}),
    )


def test_teardown_exit_stop_esrch_without_terminal_wait_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teardown_control = launcher._TeardownControl()
    teardown_control.record_output_limit()
    with pytest.raises(RuntimeError, match="terminal drain timed out"):
        _exercise_trace(
            monkeypatch,
            [
                (0, 0),
                (101, _stopped(signal.SIGSTOP)),
                (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXIT)),
            ],
            event_messages=[],
            candidate_tasks=lambda: {101},
            teardown_control=teardown_control,
            ptrace_esrch_calls=frozenset({1}),
            repeat_no_wait=True,
            monotonic_values=(10.0, 10.0, 10.0, 11.0),
        )


def test_main_exit_cleanup_keeps_vanished_known_child_in_immutable_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )
    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield creation_syscall
        live_tasks.add(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_FORK)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        live_tasks.remove(101)
        yield 101, _exited(0)
        yield 202, _stopped(signal.SIGSTOP)
        live_tasks.remove(202)
        yield 202, _exited(23)

    result, _ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
    )

    assert result[0] == 0
    assert result[2] == 2
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_MAIN_EXIT_CLEANUP,
        targets=frozenset({202}),
    )


def test_output_teardown_accepts_known_child_natural_exit_after_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )
    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield creation_syscall
        live_tasks.add(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_FORK)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        teardown_control.record_output_limit()
        yield 0, 0
        yield 101, _stopped(signal.SIGSTOP)
        yield 202, _stopped(signal.SIGSTOP)
        live_tasks.remove(202)
        yield 202, _exited(17)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    result, _ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
    )

    assert result[0] == -signal.SIGKILL
    assert result[2] == 2
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        targets=frozenset({101, 202}),
    )


def test_aarch64_attributed_clone_reconciles_output_teardown_before_zero_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher.platform, "machine", lambda: "aarch64")
    decode_seccomp = _legacy_process_creation_decoder
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield launcher._SECCOMP_AARCH64_FORK
        live_tasks.add(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 0, 0
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_FORK)
        yield 202, _stopped(signal.SIGSTOP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        live_tasks.remove(202)
        yield 202, _signaled(signal.SIGKILL)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    output_recorded = False

    def record_output_after_clone_resume(request: int, pid: int) -> None:
        nonlocal output_recorded
        if not output_recorded and request == launcher._PTRACE_SYSCALL and pid == 101:
            output_recorded = True
            teardown_control.record_output_limit()

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        ptrace_hook=record_output_after_clone_resume,
        group_kills=group_kills,
        seccomp_creation_mode=decode_seccomp,
    )

    assert result[0] == -signal.SIGKILL
    assert result[2] == 2
    assert ptrace_calls == [
        (launcher._PTRACE_SYSCALL, 101, None),
        (launcher._PTRACE_SYSCALL, 101, None),
        (launcher._PTRACE_SYSCALL, 101, None),
    ]
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        targets=frozenset({101, 202}),
    )


def test_reconciled_clone_initial_stop_is_held_before_exact_kill_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    live_tasks = {101}
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )

    def event_messages() -> Iterable[int]:
        yield creation_syscall
        live_tasks.add(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 0, 0
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_FORK)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 202, _stopped(signal.SIGSTOP)
        live_tasks.remove(202)
        yield 202, _signaled(signal.SIGKILL)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    output_recorded = False

    def record_output_after_clone_resume(request: int, pid: int) -> None:
        nonlocal output_recorded
        if not output_recorded and request == launcher._PTRACE_SYSCALL and pid == 101:
            output_recorded = True
            teardown_control.record_output_limit()

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        ptrace_hook=record_output_after_clone_resume,
        group_kills=group_kills,
    )

    assert result[0] == -signal.SIGKILL
    assert group_kills == [101]
    assert ptrace_calls[-1] == (launcher._PTRACE_SYSCALL, 101, None)
    assert all(call[:2] != (launcher._PTRACE_CONT, 202) for call in ptrace_calls)
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        targets=frozenset({101, 202}),
    )


def test_reconciled_clone_initial_stop_needs_no_esrch_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    live_tasks = {101}
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )

    def event_messages() -> Iterable[int]:
        yield creation_syscall
        live_tasks.add(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 0, 0
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_FORK)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 202, _stopped(signal.SIGSTOP)
        live_tasks.remove(202)
        yield 202, _signaled(signal.SIGKILL)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    output_recorded = False

    def record_output_after_clone_resume(request: int, pid: int) -> None:
        nonlocal output_recorded
        if not output_recorded and request == launcher._PTRACE_SYSCALL and pid == 101:
            output_recorded = True
            teardown_control.record_output_limit()

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        ptrace_hook=record_output_after_clone_resume,
    )

    assert result[0] == -signal.SIGKILL
    assert ptrace_calls[-1] == (launcher._PTRACE_SYSCALL, 101, None)
    assert all(call[:2] != (launcher._PTRACE_CONT, 202) for call in ptrace_calls)


def test_reconciled_initial_stop_esrch_without_terminal_wait_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    live_tasks = {101}
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )

    def event_messages() -> Iterable[int]:
        yield creation_syscall
        live_tasks.add(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 0, 0
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_FORK)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 202, _stopped(signal.SIGSTOP)

    output_recorded = False

    def record_output_after_clone_resume(request: int, pid: int) -> None:
        nonlocal output_recorded
        if not output_recorded and request == launcher._PTRACE_SYSCALL and pid == 101:
            output_recorded = True
            teardown_control.record_output_limit()

    with pytest.raises(RuntimeError, match="terminal drain timed out"):
        _exercise_trace(
            monkeypatch,
            waits(),
            event_messages=event_messages(),
            syscall_results=[-launcher._ENOSYS, 202],
            candidate_tasks=lambda: set(live_tasks),
            teardown_control=teardown_control,
            ptrace_hook=record_output_after_clone_resume,
            repeat_no_wait=True,
            monotonic_values=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 11.0),
        )


def test_preledger_creation_initial_stop_esrch_remains_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    live_tasks = {101}
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )

    def event_messages() -> Iterable[int]:
        yield creation_syscall
        live_tasks.add(202)
        yield 202

    with pytest.raises(launcher._PtraceError) as caught:
        _exercise_trace(
            monkeypatch,
            [
                (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)),
                (101, _stopped(signal.SIGTRAP | 0x80)),
                (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_FORK)),
                (202, _stopped(signal.SIGSTOP)),
            ],
            event_messages=event_messages(),
            syscall_results=[-launcher._ENOSYS],
            candidate_tasks=lambda: set(live_tasks),
            ptrace_esrch_calls=frozenset({4}),
        )

    assert caught.value.request == launcher._PTRACE_CONT
    assert caught.value.pid == 202
    assert caught.value.error_number == errno.ESRCH


@pytest.mark.parametrize(
    ("resource_signal", "expected_reason"),
    [
        (signal.SIGXCPU, launcher._TEARDOWN_CPU_LIMIT),
        (signal.SIGXFSZ, launcher._TEARDOWN_FILE_SPACE_LIMIT),
    ],
)
def test_aarch64_attributed_clone_reconciles_kernel_resource_teardown(
    monkeypatch: pytest.MonkeyPatch,
    resource_signal: int,
    expected_reason: str,
) -> None:
    monkeypatch.setattr(launcher.platform, "machine", lambda: "aarch64")
    decode_seccomp = _legacy_process_creation_decoder
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield launcher._SECCOMP_AARCH64_FORK
        live_tasks.add(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(resource_signal)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_FORK)
        yield 202, _stopped(signal.SIGSTOP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        live_tasks.remove(202)
        yield 202, _signaled(signal.SIGKILL)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        siginfos=[(resource_signal, launcher._SI_KERNEL)],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
        seccomp_creation_mode=decode_seccomp,
    )

    assert result[0] == -signal.SIGKILL
    assert result[3] is (resource_signal == signal.SIGXFSZ)
    assert result[9] is (resource_signal == signal.SIGXCPU)
    assert ptrace_calls == [
        (launcher._PTRACE_SYSCALL, 101, None),
        (launcher._PTRACE_SYSCALL, 101, None),
        (launcher._PTRACE_SYSCALL, 101, None),
        (launcher._PTRACE_SYSCALL, 101, None),
    ]
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=expected_reason,
        targets=frozenset({101, 202}),
    )


def test_teardown_reconciliation_rejects_second_unattributed_child_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    live_tasks = {101}
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        teardown_control.record_output_limit()
        yield 0, 0
        live_tasks.add(202)
        yield 202, _stopped(signal.SIGSTOP)
        live_tasks.add(404)
        yield 404, _stopped(signal.SIGSTOP)

    with pytest.raises(RuntimeError, match="ptrace stop is unattributed"):
        _exercise_trace(
            monkeypatch,
            waits(),
            event_messages=[creation_syscall],
            candidate_tasks=lambda: set(live_tasks),
            teardown_control=teardown_control,
        )


def test_teardown_drain_deadline_fails_before_outer_wall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teardown_control = launcher._TeardownControl()
    teardown_control.record_output_limit()
    with pytest.raises(RuntimeError, match="terminal drain timed out"):
        _exercise_trace(
            monkeypatch,
            [(0, 0), (101, _stopped(signal.SIGSTOP))],
            event_messages=[],
            candidate_tasks=lambda: {101},
            teardown_control=teardown_control,
            repeat_no_wait=True,
            monotonic_values=(10.0, 10.0, 11.0),
        )


def test_main_exit_cleanup_preserves_main_result_and_kills_exact_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )
    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield creation_syscall
        live_tasks.add(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_FORK)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        live_tasks.remove(101)
        yield 101, _exited(0)
        yield 202, _stopped(signal.SIGSTOP)
        live_tasks.remove(202)
        yield 202, _signaled(signal.SIGKILL)

    result, _ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
    )

    assert result[0] == 0
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_MAIN_EXIT_CLEANUP,
        targets=frozenset({202}),
    )


@pytest.mark.parametrize(
    ("evidence_timing", "expected_reason"),
    [
        ("before_claim", launcher._TEARDOWN_OUTPUT_LIMIT),
        ("after_claim", launcher._TEARDOWN_MAIN_EXIT_CLEANUP),
    ],
)
def test_output_evidence_racing_natural_main_exit_preserves_exact_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    evidence_timing: str,
    expected_reason: str,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )
    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield creation_syscall
        live_tasks.add(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_FORK)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        live_tasks.remove(101)
        yield 101, _exited(0)
        if evidence_timing == "before_claim":
            teardown_control.record_output_limit()
        yield 202, _stopped(signal.SIGSTOP)
        live_tasks.remove(202)
        yield 202, _signaled(signal.SIGKILL)

    result, _ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
        group_kill_hook=(
            teardown_control.record_output_limit
            if evidence_timing == "after_claim"
            else None
        ),
    )

    assert result[0] == 0
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=expected_reason,
        targets=frozenset({202}),
    )
    launcher._validate_teardown_evidence(
        teardown_control,
        stdout_truncated=True,
        stderr_truncated=False,
        cpu_limit_hit=False,
        file_space_limit_source=None,
        file_limit_signal=None,
    )


def test_teardown_evidence_is_reconciled_after_relays_drain() -> None:
    output_control = launcher._TeardownControl()
    output_control.record_output_limit()
    launcher._validate_teardown_evidence(
        output_control,
        stdout_truncated=True,
        stderr_truncated=False,
        cpu_limit_hit=False,
        file_space_limit_source=None,
        file_limit_signal=None,
    )

    cpu_control = launcher._TeardownControl()
    cpu_control.claim_kill(launcher._TEARDOWN_CPU_LIMIT, frozenset({101}))
    launcher._validate_teardown_evidence(
        cpu_control,
        stdout_truncated=False,
        stderr_truncated=False,
        cpu_limit_hit=True,
        file_space_limit_source=None,
        file_limit_signal=None,
    )

    with pytest.raises(RuntimeError, match="evidence is inconsistent"):
        launcher._validate_teardown_evidence(
            launcher._TeardownControl(),
            stdout_truncated=True,
            stderr_truncated=False,
            cpu_limit_hit=False,
            file_space_limit_source=None,
            file_limit_signal=None,
        )


def test_normal_exit_with_pending_write_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_syscall = min(launcher._file_write_syscalls())
    with pytest.raises(RuntimeError, match="candidate exited during a traced syscall"):
        _exercise_trace(
            monkeypatch,
            [
                (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)),
                (101, _stopped(signal.SIGTRAP | 0x80)),
                (101, _exited(0)),
            ],
            event_messages=[write_syscall],
            syscall_results=[-launcher._ENOSYS],
        )


def test_unowned_self_sigkill_has_no_resource_teardown_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teardown_control = launcher._TeardownControl()
    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        [(101, _signaled(signal.SIGKILL))],
        event_messages=[],
        teardown_control=teardown_control,
    )

    assert result[0] == -signal.SIGKILL
    assert result[3] is False
    assert result[9] is False
    assert result[12] is False
    assert ptrace_calls == []
    assert teardown_control.kill_record() is None
    assert teardown_control.output_limit_requested() is False


def test_nonmain_writer_sigkill_terminal_first_remains_unowned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    write_syscall = min(launcher._file_write_syscalls())
    teardown_control = launcher._TeardownControl()
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )
    inventories = iter(
        (
            {101, 202},
            {101, 202},
            {101},
            set(),
            set(),
        )
    )

    result, _ptrace_calls = _exercise_trace(
        monkeypatch,
        [
            (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)),
            (101, _stopped(signal.SIGTRAP | 0x80)),
            (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_FORK)),
            (101, _stopped(signal.SIGTRAP | 0x80)),
            (202, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)),
            (202, _stopped(signal.SIGTRAP | 0x80)),
            (202, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXIT)),
            (202, _signaled(signal.SIGKILL)),
            (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXIT)),
            (101, _signaled(signal.SIGKILL)),
        ],
        event_messages=[creation_syscall, 202, write_syscall],
        syscall_results=[-launcher._ENOSYS, 202, -launcher._ENOSYS],
        candidate_tasks=lambda: next(inventories),
        teardown_control=teardown_control,
    )

    assert result[0] == -signal.SIGKILL
    assert result[3] is False
    assert result[9] is False
    assert teardown_control.kill_record() is None
    assert teardown_control.output_limit_requested() is False


@pytest.mark.parametrize("signal_phase", ["before_entry", "before_exit"])
def test_generic_signal_delivery_preserves_pending_syscall_exit_stop(
    monkeypatch: pytest.MonkeyPatch,
    signal_phase: str,
) -> None:
    write_syscall = min(launcher._file_write_syscalls())
    signal_stop = (101, _stopped(signal.SIGUSR1))
    waits = [
        (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)),
    ]
    if signal_phase == "before_entry":
        waits.append(signal_stop)
    waits.append((101, _stopped(signal.SIGTRAP | 0x80)))
    if signal_phase == "before_exit":
        waits.append(signal_stop)
    waits.extend(
        [
            (101, _stopped(signal.SIGTRAP | 0x80)),
            (101, _exited(0)),
        ]
    )

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits,
        event_messages=[write_syscall],
        syscall_results=[-launcher._ENOSYS, 7],
    )

    assert result[0] == 0
    assert (launcher._PTRACE_SYSCALL, 101, signal.SIGUSR1) in ptrace_calls
    assert (launcher._PTRACE_CONT, 101, signal.SIGUSR1) not in ptrace_calls


@pytest.mark.parametrize("internal_signal", [signal.SIGSTOP, signal.SIGTRAP])
def test_internal_signal_stop_preserves_pending_syscall_exit_stop(
    monkeypatch: pytest.MonkeyPatch,
    internal_signal: int,
) -> None:
    write_syscall = min(launcher._file_write_syscalls())
    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        [
            (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)),
            (101, _stopped(signal.SIGTRAP | 0x80)),
            (101, _stopped(internal_signal)),
            (101, _stopped(signal.SIGTRAP | 0x80)),
            (101, _exited(0)),
        ],
        event_messages=[write_syscall],
        syscall_results=[-launcher._ENOSYS, 7],
    )

    assert result[0] == 0
    assert ptrace_calls == [
        (launcher._PTRACE_SYSCALL, 101, None),
        (launcher._PTRACE_SYSCALL, 101, None),
        (launcher._PTRACE_SYSCALL, 101, None),
        (launcher._PTRACE_CONT, 101, None),
    ]


@pytest.mark.parametrize("resource_signal", [signal.SIGXCPU, signal.SIGXFSZ])
def test_user_resource_signal_delivery_preserves_pending_syscall_exit_stop(
    monkeypatch: pytest.MonkeyPatch,
    resource_signal: int,
) -> None:
    write_syscall = min(launcher._file_write_syscalls())
    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        [
            (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)),
            (101, _stopped(signal.SIGTRAP | 0x80)),
            (101, _stopped(resource_signal)),
            (101, _stopped(signal.SIGTRAP | 0x80)),
            (101, _exited(0)),
        ],
        event_messages=[write_syscall],
        syscall_results=[-launcher._ENOSYS, 7],
        siginfos=[(resource_signal, 0)],
    )

    assert result[0] == 0
    assert (launcher._PTRACE_SYSCALL, 101, resource_signal) in ptrace_calls
    assert (launcher._PTRACE_CONT, 101, resource_signal) not in ptrace_calls


def test_kernel_cpu_limit_during_pending_write_keeps_signed_limit_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_syscall = min(launcher._file_write_syscalls())
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    inventories = iter(({101}, set(), set(), set()))
    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        [
            (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)),
            (101, _stopped(signal.SIGTRAP | 0x80)),
            (101, _stopped(signal.SIGXCPU)),
            (101, _signaled(signal.SIGKILL)),
        ],
        event_messages=[write_syscall],
        syscall_results=[-launcher._ENOSYS],
        siginfos=[(signal.SIGXCPU, launcher._SI_KERNEL)],
        candidate_tasks=lambda: next(inventories),
        teardown_control=teardown_control,
        group_kills=group_kills,
    )

    assert result[0] == -signal.SIGKILL
    assert result[9] is True
    assert result[12] is True
    assert (launcher._PTRACE_KILL, 101, None) not in ptrace_calls
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_CPU_LIMIT,
        targets=frozenset({101}),
    )


def test_kernel_file_limit_during_pending_write_keeps_signed_limit_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_syscall = min(launcher._file_write_syscalls())
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    inventories = iter(({101}, set(), set(), set()))
    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        [
            (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)),
            (101, _stopped(signal.SIGTRAP | 0x80)),
            (101, _stopped(signal.SIGXFSZ)),
            (101, _signaled(signal.SIGKILL)),
        ],
        event_messages=[write_syscall],
        syscall_results=[-launcher._ENOSYS],
        siginfos=[(signal.SIGXFSZ, launcher._SI_KERNEL)],
        candidate_tasks=lambda: next(inventories),
        teardown_control=teardown_control,
        group_kills=group_kills,
    )

    assert result[0] == -signal.SIGKILL
    assert result[3] is True
    assert result[4] == "guest_monitor_ptrace_siginfo_fsize"
    assert result[5] == signal.SIGXFSZ
    assert result[12] is True
    assert (launcher._PTRACE_KILL, 101, None) not in ptrace_calls
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_FILE_SPACE_LIMIT,
        targets=frozenset({101}),
    )


def test_runtime_signal_during_pending_write_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_syscall = min(launcher._file_write_syscalls())
    with pytest.raises(RuntimeError, match="candidate exited during a traced syscall"):
        _exercise_trace(
            monkeypatch,
            [
                (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)),
                (101, _stopped(signal.SIGTRAP | 0x80)),
                (101, _stopped(signal.SIGSEGV)),
                (101, _signaled(signal.SIGSEGV)),
            ],
            event_messages=[write_syscall],
            syscall_results=[-launcher._ENOSYS],
        )


@pytest.mark.parametrize("denied", [False, True])
def test_teardown_reconciliation_requires_exact_pending_creation_result(
    monkeypatch: pytest.MonkeyPatch,
    denied: bool,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        (
            (lambda *_args: None)
            if denied
            else (lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK))
        ),
    )

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        teardown_control.record_output_limit()
        yield 0, 0

    with pytest.raises(RuntimeError, match="reconciliation timed out"):
        _exercise_trace(
            monkeypatch,
            waits(),
            event_messages=[creation_syscall],
            teardown_control=teardown_control,
            repeat_no_wait=True,
            monotonic_values=(10.0, 10.0, 11.0),
        )


@pytest.mark.parametrize("denied", [False, True])
def test_teardown_reconciliation_rejects_terminal_without_creation_result(
    monkeypatch: pytest.MonkeyPatch,
    denied: bool,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    live_tasks = {101}
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        (
            (lambda *_args: None)
            if denied
            else (lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK))
        ),
    )

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        teardown_control.record_output_limit()
        yield 0, 0
        live_tasks.clear()
        yield 101, _exited(0)

    with pytest.raises(RuntimeError, match="creation result is absent"):
        _exercise_trace(
            monkeypatch,
            waits(),
            event_messages=[creation_syscall],
            candidate_tasks=lambda: set(live_tasks),
            teardown_control=teardown_control,
        )


def test_teardown_reconciliation_requires_child_initial_and_syscall_exit_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_FORK)
        teardown_control.record_output_limit()
        yield 0, 0

    with pytest.raises(RuntimeError, match="reconciliation timed out"):
        _exercise_trace(
            monkeypatch,
            waits(),
            event_messages=[creation_syscall, 202],
            syscall_results=[-launcher._ENOSYS],
            candidate_tasks=lambda: {101, 202},
            teardown_control=teardown_control,
            repeat_no_wait=True,
            monotonic_values=(10.0, 10.0, 10.0, 10.0, 11.0),
        )


def test_teardown_reconciliation_requires_provisional_child_parent_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 202, _stopped(signal.SIGSTOP)
        teardown_control.record_output_limit()
        yield 0, 0

    with pytest.raises(RuntimeError, match="reconciliation timed out"):
        _exercise_trace(
            monkeypatch,
            waits(),
            event_messages=[creation_syscall],
            syscall_results=[-launcher._ENOSYS],
            candidate_tasks=lambda: {101, 202},
            teardown_control=teardown_control,
            repeat_no_wait=True,
            monotonic_values=(10.0, 10.0, 10.0, 10.0, 11.0),
        )


@pytest.mark.parametrize(
    ("trigger", "expected_reason"),
    [
        ("output", launcher._TEARDOWN_OUTPUT_LIMIT),
        ("cpu", launcher._TEARDOWN_CPU_LIMIT),
        ("file", launcher._TEARDOWN_FILE_SPACE_LIMIT),
    ],
)
def test_successful_vfork_during_resource_reconciliation_freezes_exact_ledger(
    monkeypatch: pytest.MonkeyPatch,
    trigger: str,
    expected_reason: str,
) -> None:
    monkeypatch.setattr(launcher.platform, "machine", lambda: "aarch64")
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield launcher._SECCOMP_AARCH64_VFORK
        live_tasks.add(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        if trigger == "output":
            teardown_control.record_output_limit()
        else:
            yield 101, _stopped(signal.SIGXCPU if trigger == "cpu" else signal.SIGXFSZ)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_VFORK)
        yield 202, _stopped(signal.SIGSTOP)
        live_tasks.remove(202)
        yield 202, _signaled(signal.SIGKILL)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    resource_signal = signal.SIGXCPU if trigger == "cpu" else signal.SIGXFSZ
    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS],
        siginfos=(
            [(resource_signal, launcher._SI_KERNEL)] if trigger != "output" else []
        ),
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
        seccomp_creation_mode=_legacy_process_creation_decoder,
    )

    assert result[0] == -signal.SIGKILL
    assert result[2] == 2
    assert result[3] is (trigger == "file")
    assert result[9] is (trigger == "cpu")
    assert (launcher._PTRACE_SYSCALL, 101, None) in ptrace_calls
    assert ptrace_calls[-1] == (launcher._PTRACE_SYSCALL, 101, None)
    assert all(call[:2] != (launcher._PTRACE_CONT, 202) for call in ptrace_calls)
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=expected_reason,
        targets=frozenset({101, 202}),
    )


@pytest.mark.parametrize(
    "captured_child_boundary",
    ["zero", "exec", "exit", "terminal"],
)
def test_preobserved_vfork_reconciles_without_releasing_its_blocked_parent(
    monkeypatch: pytest.MonkeyPatch,
    captured_child_boundary: str,
) -> None:
    monkeypatch.setattr(launcher.platform, "machine", lambda: "aarch64")
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield launcher._SECCOMP_AARCH64_VFORK
        live_tasks.add(202)
        yield 202
        if captured_child_boundary == "exec":
            yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_VFORK)
        yield 202, _stopped(signal.SIGSTOP)
        teardown_control.record_output_limit()
        if captured_child_boundary == "zero":
            yield 0, 0
            yield 202, _stopped(signal.SIGSTOP)
            yield 101, _stopped(signal.SIGSTOP)
        elif captured_child_boundary == "terminal":
            live_tasks.remove(202)
            yield 202, _exited(17)
            yield 101, _stopped(signal.SIGSTOP)
        else:
            yield (
                202,
                _stopped(
                    signal.SIGTRAP,
                    (
                        launcher._PTRACE_EVENT_EXEC
                        if captured_child_boundary == "exec"
                        else launcher._PTRACE_EVENT_EXIT
                    ),
                ),
            )
            yield 101, _stopped(signal.SIGTRAP | 0x80)
        if captured_child_boundary == "terminal":
            yield 101, _stopped(signal.SIGTRAP | 0x80)
        else:
            live_tasks.remove(202)
            yield (
                202,
                (
                    _exited(17)
                    if captured_child_boundary == "exit"
                    else _signaled(signal.SIGKILL)
                ),
            )
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[
            -launcher._ENOSYS,
            *([202] if captured_child_boundary != "zero" else []),
        ],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
        seccomp_creation_mode=_legacy_process_creation_decoder,
    )

    assert result[0] == -signal.SIGKILL
    assert result[2] == 2
    assert ptrace_calls.count((launcher._PTRACE_SYSCALL, 101, None)) == (
        4 if captured_child_boundary == "terminal" else 3
    )
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        targets=(
            frozenset({101, 202})
            if captured_child_boundary in {"zero", "exec"}
            else frozenset({101})
        ),
    )


@pytest.mark.parametrize(
    "wake_order",
    [
        "zero_before_parent",
        "on_parent",
        "child_before_parent",
        "on_child_initial",
        "zero_after_initial",
        "child_before_parent_then_zero",
    ],
)
def test_vfork_output_teardown_folds_pre_release_wait_orders_before_capture(
    monkeypatch: pytest.MonkeyPatch,
    wake_order: str,
) -> None:
    monkeypatch.setattr(launcher.platform, "machine", lambda: "aarch64")
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield launcher._SECCOMP_AARCH64_VFORK
        if 202 not in live_tasks:
            live_tasks.add(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        child_first = wake_order in {
            "child_before_parent",
            "child_before_parent_then_zero",
        }
        if child_first:
            live_tasks.add(202)
            yield 202, _stopped(signal.SIGSTOP)
        if wake_order == "zero_before_parent":
            teardown_control.record_output_limit()
            yield 0, 0
        if wake_order in {"on_parent", "child_before_parent"}:
            teardown_control.record_output_limit()
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_VFORK)
        if wake_order == "on_child_initial":
            teardown_control.record_output_limit()
            yield 202, _stopped(signal.SIGSTOP)
            yield 101, _stopped(signal.SIGSTOP)
        elif wake_order in {
            "zero_after_initial",
            "child_before_parent_then_zero",
        }:
            if not child_first:
                yield 202, _stopped(signal.SIGSTOP)
            teardown_control.record_output_limit()
            yield 0, 0
            yield 202, _stopped(signal.SIGSTOP)
            yield 101, _stopped(signal.SIGSTOP)
        elif not child_first:
            # The immutable ledger owns this exact post-kill initial stop.
            yield 202, _stopped(signal.SIGSTOP)
        live_tasks.remove(202)
        yield 202, _signaled(signal.SIGKILL)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    result, _ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
        seccomp_creation_mode=_legacy_process_creation_decoder,
    )

    assert result[0] == -signal.SIGKILL
    assert result[2] == 2
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        targets=frozenset({101, 202}),
    )


@pytest.mark.parametrize(
    ("trigger", "expected_reason"),
    [
        ("output", launcher._TEARDOWN_OUTPUT_LIMIT),
        ("cpu", launcher._TEARDOWN_CPU_LIMIT),
        ("file", launcher._TEARDOWN_FILE_SPACE_LIMIT),
    ],
)
@pytest.mark.parametrize("child_release", ["exec", "exit", "terminal"])
def test_vfork_release_requires_parent_result_across_every_teardown_trigger(
    monkeypatch: pytest.MonkeyPatch,
    trigger: str,
    expected_reason: str,
    child_release: str,
) -> None:
    monkeypatch.setattr(launcher.platform, "machine", lambda: "aarch64")
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield launcher._SECCOMP_AARCH64_VFORK
        live_tasks.add(202)
        yield 202
        if child_release == "exec":
            yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_VFORK)
        yield 202, _stopped(signal.SIGSTOP)
        if child_release == "exec":
            yield 202, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXEC)
        elif child_release == "exit":
            yield 202, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXIT)
        else:
            live_tasks.remove(202)
            yield 202, _exited(17)
        if trigger == "output":
            teardown_control.record_output_limit()
        else:
            yield 101, _stopped(signal.SIGXCPU if trigger == "cpu" else signal.SIGXFSZ)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        if child_release == "exec":
            yield 202, _stopped(signal.SIGSTOP)
        if child_release != "terminal":
            live_tasks.remove(202)
            yield (
                202,
                (_exited(17) if child_release == "exit" else _signaled(signal.SIGKILL)),
            )
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    resource_signal = signal.SIGXCPU if trigger == "cpu" else signal.SIGXFSZ
    result, _ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        siginfos=(
            [(resource_signal, launcher._SI_KERNEL)] if trigger != "output" else []
        ),
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
        seccomp_creation_mode=_legacy_process_creation_decoder,
    )

    assert result[0] == -signal.SIGKILL
    assert result[3] is (trigger == "file")
    assert result[9] is (trigger == "cpu")
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=expected_reason,
        targets=(
            frozenset({101, 202}) if child_release == "exec" else frozenset({101})
        ),
    )


@pytest.mark.parametrize(
    ("creation_mode", "creation_event"),
    [
        ("cpython_fork", launcher._PTRACE_EVENT_FORK),
        ("cpython_vfork", launcher._PTRACE_EVENT_VFORK),
        ("cpython_pthread_clone", launcher._PTRACE_EVENT_CLONE),
    ],
)
def test_candidate_sigkill_during_creation_preserves_child_and_main_provenance(
    monkeypatch: pytest.MonkeyPatch,
    creation_mode: str,
    creation_event: int,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    group_stops: list[int] = []
    live_tasks = {101}
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: (creation_mode, creation_event),
    )

    def event_messages() -> Iterable[int]:
        yield creation_syscall
        live_tasks.add(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, creation_event)
        yield 202, _stopped(signal.SIGSTOP)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)
        yield 202, _stopped(signal.SIGSTOP)
        live_tasks.remove(202)
        yield 202, _signaled(signal.SIGKILL)

    result, _ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
        group_stops=group_stops,
    )

    assert result[0] == -signal.SIGKILL
    assert result[12] is False
    assert group_stops == [101]
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_MAIN_EXIT_CLEANUP,
        targets=frozenset({202}),
    )


def test_candidate_sigkill_before_creation_event_remains_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    group_stops: list[int] = []
    live_tasks = {101}
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        live_tasks.clear()
        yield 101, _signaled(signal.SIGKILL)
        yield 0, 0

    with pytest.raises(
        RuntimeError,
        match="candidate teardown reconciliation timed out",
    ):
        _exercise_trace(
            monkeypatch,
            waits(),
            event_messages=[creation_syscall],
            syscall_results=[-launcher._ENOSYS],
            candidate_tasks=lambda: set(live_tasks),
            teardown_control=teardown_control,
            group_kills=group_kills,
            group_stops=group_stops,
            repeat_no_wait=True,
            monotonic_values=[0.0, 0.1, 0.2, 0.3, 1.3],
        )

    assert group_stops == [101]
    assert group_kills == []
    assert teardown_control.kill_record() is None


def test_non_sigkill_creation_parent_terminal_remains_unattributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    group_stops: list[int] = []
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_fork", launcher._PTRACE_EVENT_FORK),
    )

    with pytest.raises(RuntimeError, match="exited during a traced syscall"):
        _exercise_trace(
            monkeypatch,
            [
                (101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)),
                (101, _stopped(signal.SIGTRAP | 0x80)),
                (101, _signaled(signal.SIGTERM)),
            ],
            event_messages=[creation_syscall],
            syscall_results=[-launcher._ENOSYS],
            candidate_tasks=set,
            teardown_control=teardown_control,
            group_stops=group_stops,
        )

    assert group_stops == []
    assert teardown_control.kill_record() is None


@pytest.mark.parametrize(
    ("creation_mode", "creation_event"),
    [
        ("cpython_fork", launcher._PTRACE_EVENT_FORK),
        ("cpython_vfork", launcher._PTRACE_EVENT_VFORK),
        ("cpython_pthread_clone", launcher._PTRACE_EVENT_CLONE),
    ],
)
def test_zero_wait_procfs_appearance_needs_exact_creation_correlation(
    monkeypatch: pytest.MonkeyPatch,
    creation_mode: str,
    creation_event: int,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    group_stops: list[int] = []
    live_tasks = {101}
    output_recorded = False
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: (creation_mode, creation_event),
    )

    def event_messages() -> Iterable[int]:
        yield creation_syscall
        yield 202

    def record_output_after_seccomp_resume(request: int, pid: int) -> None:
        nonlocal output_recorded
        if not output_recorded and request == launcher._PTRACE_SYSCALL and pid == 101:
            output_recorded = True
            teardown_control.record_output_limit()

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 0, 0
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        live_tasks.add(202)
        yield 0, 0
        yield 202, _stopped(signal.SIGSTOP)
        yield 101, _stopped(signal.SIGTRAP, creation_event)
        if creation_event != launcher._PTRACE_EVENT_VFORK:
            yield 101, _stopped(signal.SIGTRAP | 0x80)
        live_tasks.remove(202)
        yield 202, _signaled(signal.SIGKILL)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    result, _ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[
            -launcher._ENOSYS,
            *([202] if creation_event != launcher._PTRACE_EVENT_VFORK else []),
        ],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        ptrace_hook=record_output_after_seccomp_resume,
        group_kills=group_kills,
        group_stops=group_stops,
    )

    assert result[0] == -signal.SIGKILL
    assert result[2] == 2
    assert group_stops == [101]
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        targets=frozenset({101, 202}),
    )


def test_normal_nonleader_exec_folds_former_tid_before_terminal_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher.platform, "machine", lambda: "aarch64")
    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield launcher._SECCOMP_AARCH64_THREAD
        live_tasks.add(202)
        yield 202
        live_tasks.remove(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_CLONE)
        yield 202, _stopped(signal.SIGSTOP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXEC)
        live_tasks.remove(101)
        yield 101, _exited(0)

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        candidate_tasks=lambda: set(live_tasks),
        seccomp_creation_mode=launcher._seccomp_creation_mode,
    )

    assert result[0] == 0
    assert result[2] == 2
    assert ptrace_calls[-1] == (launcher._PTRACE_CONT, 101, None)


def test_normal_nonleader_exec_discards_destroyed_leader_pending_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher.platform, "machine", lambda: "aarch64")
    write_syscall = min(launcher._file_write_syscalls())
    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield launcher._SECCOMP_AARCH64_THREAD
        live_tasks.add(202)
        yield 202
        yield write_syscall
        live_tasks.remove(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_CLONE)
        yield 202, _stopped(signal.SIGSTOP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXEC)
        live_tasks.remove(101)
        yield 101, _exited(0)

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202, -launcher._ENOSYS],
        candidate_tasks=lambda: set(live_tasks),
        seccomp_creation_mode=launcher._seccomp_creation_mode,
    )

    assert result[0] == 0
    assert result[2] == 2
    assert ptrace_calls[-1] == (launcher._PTRACE_CONT, 101, None)


def test_nonleader_exec_remaps_only_the_former_tasks_pending_state() -> None:
    file_write_syscalls = launcher._file_write_syscalls()
    fork_syscalls = launcher._fork_syscalls()
    write_syscall = min(file_write_syscalls)
    former_pending = launcher._PendingSyscall(number=write_syscall)
    traced = {101, 202}
    awaiting = {202: former_pending}

    launcher._fold_nonleader_exec_state(
        current_tid=101,
        former_tid=202,
        traced=traced,
        awaiting_syscall_exit=awaiting,
        provisional_children=set(),
        file_write_syscalls=file_write_syscalls,
        fork_syscalls=fork_syscalls,
    )

    assert traced == {101}
    assert awaiting == {101: former_pending}

    old_leader_pending = launcher._PendingSyscall(number=write_syscall)
    awaiting = {101: old_leader_pending}
    launcher._fold_nonleader_exec_state(
        current_tid=101,
        former_tid=202,
        traced={101, 202},
        awaiting_syscall_exit=awaiting,
        provisional_children=set(),
        file_write_syscalls=file_write_syscalls,
        fork_syscalls=fork_syscalls,
    )
    assert awaiting == {}

    creation_pending = launcher._PendingSyscall(
        number=min(fork_syscalls),
        creation_mode="cpython_fork",
        expected_event=launcher._PTRACE_EVENT_FORK,
    )
    with pytest.raises(
        RuntimeError,
        match="candidate exec replaced a pending syscall task",
    ):
        launcher._fold_nonleader_exec_state(
            current_tid=101,
            former_tid=202,
            traced={101, 202},
            awaiting_syscall_exit={101: creation_pending},
            provisional_children=set(),
            file_write_syscalls=file_write_syscalls,
            fork_syscalls=fork_syscalls,
        )


@pytest.mark.parametrize("terminal_tid", [101, 202])
def test_nonleader_exec_rejects_terminal_evidence_for_either_live_identity(
    terminal_tid: int,
) -> None:
    reconciliation = launcher._TeardownReconciliation(
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        transitions={},
        snapshotted_pending=set(),
        resolved_transitions=set(),
        attributed_targets={101, 202},
        terminal_statuses={terminal_tid: 0},
        children_needing_initial_stop=set(),
        held_tasks=set(),
        continued_exit_stops=set(),
        deadline=10.0,
    )

    with pytest.raises(
        RuntimeError,
        match="candidate exec has terminal evidence for a live task",
    ):
        launcher._fold_nonleader_exec_state(
            current_tid=101,
            former_tid=202,
            traced={101, 202},
            awaiting_syscall_exit={},
            provisional_children=set(),
            file_write_syscalls=launcher._file_write_syscalls(),
            fork_syscalls=launcher._fork_syscalls(),
            reconciliation=reconciliation,
        )


def test_nonleader_exec_discards_old_leader_held_and_all_old_exit_markers() -> None:
    reconciliation = launcher._TeardownReconciliation(
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        transitions={},
        snapshotted_pending=set(),
        resolved_transitions=set(),
        attributed_targets={101, 202, 303},
        terminal_statuses={},
        children_needing_initial_stop=set(),
        held_tasks={101, 303},
        continued_exit_stops={101, 202, 303},
        deadline=10.0,
    )

    launcher._fold_nonleader_exec_state(
        current_tid=101,
        former_tid=202,
        traced={101, 202, 303},
        awaiting_syscall_exit={},
        provisional_children=set(),
        file_write_syscalls=launcher._file_write_syscalls(),
        fork_syscalls=launcher._fork_syscalls(),
        reconciliation=reconciliation,
    )

    assert reconciliation.held_tasks == {303}
    assert reconciliation.continued_exit_stops == {303}
    assert reconciliation.attributed_targets == {101, 303}


def test_reconciliation_nonleader_exec_folds_held_and_attribution_ledgers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher.platform, "machine", lambda: "aarch64")
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    group_stops: list[int] = []
    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield launcher._SECCOMP_AARCH64_THREAD
        live_tasks.add(202)
        yield 202
        live_tasks.remove(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_CLONE)
        yield 202, _stopped(signal.SIGSTOP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        teardown_control.record_output_limit()
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXEC)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    result, _ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
        group_stops=group_stops,
        seccomp_creation_mode=launcher._seccomp_creation_mode,
    )

    assert result[0] == -signal.SIGKILL
    assert result[2] == 2
    assert result[12] is True
    assert group_stops == [101]
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        targets=frozenset({101}),
    )


def test_nonleader_exec_discards_old_exit_continuation_before_new_leader_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher.platform, "machine", lambda: "aarch64")
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    group_stops: list[int] = []
    live_tasks = {101}

    def event_messages() -> Iterable[int]:
        yield launcher._SECCOMP_AARCH64_THREAD
        live_tasks.add(202)
        yield 202
        live_tasks.remove(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_CLONE)
        yield 202, _stopped(signal.SIGSTOP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        teardown_control.record_output_limit()
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXIT)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXEC)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXIT)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
        group_stops=group_stops,
        seccomp_creation_mode=launcher._seccomp_creation_mode,
    )

    assert result[0] == -signal.SIGKILL
    assert result[2] == 2
    assert result[12] is True
    assert ptrace_calls[-2:] == [
        (launcher._PTRACE_CONT, 101, None),
        (launcher._PTRACE_CONT, 101, None),
    ]
    assert group_stops == [101]
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        targets=frozenset({101}),
    )


def test_vfork_parent_result_before_child_exec_is_order_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    live_tasks = {101}
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_vfork", launcher._PTRACE_EVENT_VFORK),
    )

    def event_messages() -> Iterable[int]:
        yield creation_syscall
        live_tasks.add(202)
        yield 202
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_VFORK)
        yield 202, _stopped(signal.SIGSTOP)
        teardown_control.record_output_limit()
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 202, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXEC)
        live_tasks.remove(202)
        yield 202, _signaled(signal.SIGKILL)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        group_kills=group_kills,
    )

    assert result[0] == -signal.SIGKILL
    assert ptrace_calls.count((launcher._PTRACE_CONT, 202, None)) == 1
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        targets=frozenset({101, 202}),
    )


@pytest.mark.parametrize(
    "wait_order",
    ["parent_before_exit", "parent_before_terminal", "parent_after_terminal"],
)
def test_vfork_exit_and_parent_result_are_order_free_with_exact_esrch(
    monkeypatch: pytest.MonkeyPatch,
    wait_order: str,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    live_tasks = {101}
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_vfork", launcher._PTRACE_EVENT_VFORK),
    )

    def event_messages() -> Iterable[int]:
        yield creation_syscall
        live_tasks.add(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_VFORK)
        yield 202, _stopped(signal.SIGSTOP)
        teardown_control.record_output_limit()
        if wait_order == "parent_before_exit":
            yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 202, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXIT)
        if wait_order == "parent_before_terminal":
            yield 101, _stopped(signal.SIGTRAP | 0x80)
        live_tasks.remove(202)
        yield 202, _exited(17)
        if wait_order == "parent_after_terminal":
            yield 101, _stopped(signal.SIGTRAP | 0x80)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        ptrace_esrch_calls=frozenset({5}),
        group_kills=group_kills,
    )

    assert result[0] == -signal.SIGKILL
    assert ptrace_calls[4] == (launcher._PTRACE_CONT, 202, None)
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        targets=frozenset({101}),
    )


@pytest.mark.parametrize(
    ("trigger", "expected_reason"),
    [
        ("output", launcher._TEARDOWN_OUTPUT_LIMIT),
        ("cpu", launcher._TEARDOWN_CPU_LIMIT),
        ("file", launcher._TEARDOWN_FILE_SPACE_LIMIT),
    ],
)
def test_preobserved_vfork_exit_esrch_reconciles_every_trigger(
    monkeypatch: pytest.MonkeyPatch,
    trigger: str,
    expected_reason: str,
) -> None:
    creation_syscall = min(launcher._fork_syscalls())
    teardown_control = launcher._TeardownControl()
    group_kills: list[int] = []
    live_tasks = {101}
    monkeypatch.setattr(
        launcher,
        "_creation_mode",
        lambda *_args: ("cpython_vfork", launcher._PTRACE_EVENT_VFORK),
    )

    def event_messages() -> Iterable[int]:
        yield creation_syscall
        live_tasks.add(202)
        yield 202

    def waits() -> Iterable[tuple[int, int]]:
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_SECCOMP)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        yield 101, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_VFORK)
        yield 202, _stopped(signal.SIGSTOP)
        yield 202, _stopped(signal.SIGTRAP, launcher._PTRACE_EVENT_EXIT)
        if trigger == "output":
            teardown_control.record_output_limit()
        else:
            yield 101, _stopped(signal.SIGXCPU if trigger == "cpu" else signal.SIGXFSZ)
        yield 101, _stopped(signal.SIGTRAP | 0x80)
        live_tasks.remove(202)
        yield 202, _exited(17)
        live_tasks.remove(101)
        yield 101, _signaled(signal.SIGKILL)

    resource_signal = signal.SIGXCPU if trigger == "cpu" else signal.SIGXFSZ
    result, ptrace_calls = _exercise_trace(
        monkeypatch,
        waits(),
        event_messages=event_messages(),
        syscall_results=[-launcher._ENOSYS, 202],
        siginfos=(
            [(resource_signal, launcher._SI_KERNEL)] if trigger != "output" else []
        ),
        candidate_tasks=lambda: set(live_tasks),
        teardown_control=teardown_control,
        ptrace_esrch_calls=frozenset({5}),
        group_kills=group_kills,
    )

    assert result[0] == -signal.SIGKILL
    assert result[3] is (trigger == "file")
    assert result[9] is (trigger == "cpu")
    assert ptrace_calls[4] == (launcher._PTRACE_CONT, 202, None)
    assert group_kills == [101]
    assert teardown_control.kill_record() == launcher._TeardownKillRecord(
        epoch=1,
        reason=expected_reason,
        targets=frozenset({101}),
    )


@pytest.mark.parametrize(
    "child_phase",
    [
        launcher._VFORK_CHILD_INITIAL_STOP,
        launcher._VFORK_CHILD_RUNNING,
        launcher._VFORK_CHILD_RELEASED,
        launcher._VFORK_CHILD_TERMINAL,
    ],
)
def test_vfork_lifecycle_phase_alone_never_proves_physical_quiescence(
    child_phase: str,
) -> None:
    pending = launcher._PendingSyscall(
        number=220,
        saw_entry=True,
        creation_mode="glibc_vfork_clone",
        expected_event=launcher._PTRACE_EVENT_VFORK,
        event_child=202,
        child_initial_stop_pending=(child_phase == launcher._VFORK_CHILD_INITIAL_STOP),
        vfork_child_phase=child_phase,
        vfork_child_release=(
            launcher._VFORK_RELEASE_EXEC
            if child_phase == launcher._VFORK_CHILD_RELEASED
            else (
                launcher._VFORK_RELEASE_TERMINAL
                if child_phase == launcher._VFORK_CHILD_TERMINAL
                else None
            )
        ),
        vfork_child_terminal_seen=(child_phase == launcher._VFORK_CHILD_TERMINAL),
    )
    awaiting = {101: pending}
    reconciliation = launcher._snapshot_teardown_reconciliation(
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        traced=({101} if child_phase == launcher._VFORK_CHILD_TERMINAL else {101, 202}),
        awaiting_syscall_exit=awaiting,
        provisional_children=set(),
        file_write_syscalls=launcher._file_write_syscalls(),
        fork_syscalls=launcher._fork_syscalls(),
        current_time=10.0,
    )
    launcher._refresh_creation_transition_resolution(
        reconciliation,
        awaiting,
    )

    assert reconciliation.resolved_transitions == set()
    assert awaiting == {101: pending}


def test_later_vfork_exit_cannot_replace_earlier_exec_release_proof() -> None:
    pending = launcher._PendingSyscall(
        number=220,
        saw_entry=True,
        creation_mode="glibc_vfork_clone",
        expected_event=launcher._PTRACE_EVENT_VFORK,
        event_child=202,
        vfork_child_phase=launcher._VFORK_CHILD_RUNNING,
    )

    launcher._record_vfork_child_release(pending, launcher._VFORK_RELEASE_EXEC)
    launcher._record_vfork_child_release(pending, launcher._VFORK_RELEASE_EXIT)

    assert pending.vfork_child_phase == launcher._VFORK_CHILD_RELEASED
    assert pending.vfork_child_release == launcher._VFORK_RELEASE_EXEC


@pytest.mark.parametrize("denied", [False, True])
def test_eventless_vfork_and_denied_creation_remain_syscall_exit_gated(
    monkeypatch: pytest.MonkeyPatch,
    denied: bool,
) -> None:
    monkeypatch.setattr(launcher.platform, "machine", lambda: "aarch64")
    pending = launcher._PendingSyscall(
        number=220,
        saw_entry=True,
        creation_mode=None if denied else "glibc_vfork_clone",
        expected_event=None if denied else launcher._PTRACE_EVENT_VFORK,
        denied=denied,
    )
    awaiting = {101: pending}
    monkeypatch.setattr(launcher, "_candidate_task_ids", lambda: {101})

    reconciliation = launcher._snapshot_teardown_reconciliation(
        reason=launcher._TEARDOWN_OUTPUT_LIMIT,
        traced={101},
        awaiting_syscall_exit=awaiting,
        provisional_children=set(),
        file_write_syscalls=launcher._file_write_syscalls(),
        fork_syscalls=launcher._fork_syscalls(),
        current_time=10.0,
    )

    assert reconciliation.resolved_transitions == set()
    assert awaiting == {101: pending}
    if not denied:
        launcher._validate_creation_result(pending, -launcher._EAGAIN, {101}, {101})
