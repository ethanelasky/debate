"""Trusted in-sandbox monitor for one Python candidate.

The OCI initial process is trusted setup/monitor code.  It never executes
candidate source in its interpreter.  After the host releases the outer
cgroup gate, it starts a fresh interpreter, proves that interpreter has the
candidate limits, UID/GID, empty capability sets, and ``no_new_privs``, and
only then releases a second gate into candidate compilation/execution.

The monitor retains the launch nonce in a non-dumpable process.  The fresh
candidate interpreter receives neither that nonce nor a writable trusted
evidence descriptor.  Candidate stdout/stderr are relayed through the monitor;
the nonce-bound terminal record is appended only after the candidate and its
descendants are dead.
"""

from __future__ import annotations

import base64
import ctypes
import errno
import json
import os
import platform
import selectors
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

_NONCE_ENV = "PALAESTRA_EXECUTOR_LAUNCH_NONCE"
_READY_PREFIX = "PALAESTRA_EXECUTOR_READY:"
_STATUS_PREFIX = "PALAESTRA_EXECUTOR_STATUS:"
_INFRA_EXIT = 125
_MAX_CODE_BYTES = 1024 * 1024
_STDOUT_CAP_BYTES = 2 * 1024 * 1024
_STDERR_CAP_BYTES = 2 * 1024 * 1024
_CANDIDATE_UID = 65534
_CANDIDATE_GID = 65534
_PR_SET_DUMPABLE = 4
_PR_GET_DUMPABLE = 3
_PR_SET_NO_NEW_PRIVS = 38
_PR_GET_NO_NEW_PRIVS = 39
_PR_CAPBSET_DROP = 24
_CAP_LAST_CAP = 40
_CANDIDATE_READY_TIMEOUT_SECONDS = 5.0
_MONITOR_FAILURE_SITES = frozenset(
    {
        "pipe_setup",
        "candidate_spawn",
        "source_delivery",
        "setup_readiness",
        "initial_attestation",
        "gate_release",
        "ptrace_continue",
        "relay_start",
        "trace_measure",
        "relay_drain",
        "teardown_evidence",
        "candidate_boundary",
    }
)
_PTRACE_TRACEME = 0
_PTRACE_CONT = 7
_PTRACE_KILL = 8
_PTRACE_SETOPTIONS = 0x4200
_PTRACE_GETSIGINFO = 0x4202
_PTRACE_GETEVENTMSG = 0x4201
_PTRACE_GETREGSET = 0x4204
_PTRACE_SETREGSET = 0x4205
_PTRACE_SYSCALL = 24
_PTRACE_O_TRACESYSGOOD = 0x00000001
_PTRACE_O_TRACEFORK = 0x00000002
_PTRACE_O_TRACEVFORK = 0x00000004
_PTRACE_O_TRACECLONE = 0x00000008
_PTRACE_O_TRACEEXEC = 0x00000010
_PTRACE_O_TRACEEXIT = 0x00000040
_PTRACE_O_TRACESECCOMP = 0x00000080
_PTRACE_O_EXITKILL = 0x00100000
_PTRACE_EVENT_FORK = 1
_PTRACE_EVENT_VFORK = 2
_PTRACE_EVENT_CLONE = 3
_PTRACE_EVENT_EXEC = 4
_PTRACE_EVENT_EXIT = 6
_PTRACE_EVENT_SECCOMP = 7
_NT_PRSTATUS = 1
_SI_KERNEL = 0x80
_EAGAIN = 11
_EPERM = 1
_EFBIG = 27
_ENOSYS = 38
_WAIT_ALL = 0x40000000
_SECCOMP_AARCH64_FORK = 0xFF01
_SECCOMP_AARCH64_THREAD = 0xFF02
_SECCOMP_AARCH64_VFORK = 0xFF03
_TEARDOWN_OUTPUT_LIMIT = "output_limit"
_TEARDOWN_CPU_LIMIT = "cpu_limit"
_TEARDOWN_FILE_SPACE_LIMIT = "file_space_limit"
_TEARDOWN_MAIN_EXIT_CLEANUP = "main_exit_cleanup"
_TRACER_TEARDOWN_REASONS = frozenset(
    {
        _TEARDOWN_OUTPUT_LIMIT,
        _TEARDOWN_CPU_LIMIT,
        _TEARDOWN_FILE_SPACE_LIMIT,
        _TEARDOWN_MAIN_EXIT_CLEANUP,
    }
)
_TEARDOWN_DRAIN_TIMEOUT_SECONDS = 1.0
_VFORK_CHILD_INITIAL_STOP = "initial_stop"
_VFORK_CHILD_RUNNING = "running"
_VFORK_CHILD_RELEASED = "released"
_VFORK_CHILD_TERMINAL = "terminal"
_VFORK_CHILD_PHASES = frozenset(
    {
        _VFORK_CHILD_INITIAL_STOP,
        _VFORK_CHILD_RUNNING,
        _VFORK_CHILD_RELEASED,
        _VFORK_CHILD_TERMINAL,
    }
)
_VFORK_RELEASE_EXEC = "exec"
_VFORK_RELEASE_EXIT = "exit"
_VFORK_RELEASE_TERMINAL = "terminal"


class _MonitorStageError(RuntimeError):
    """Attach one bounded, stable operator site to a monitor failure."""

    def __init__(self, site: str, cause: BaseException):
        if site not in _MONITOR_FAILURE_SITES:
            site = "candidate_boundary"
        error_type = type(cause).__name__
        if (
            not error_type
            or len(error_type) > 128
            or not error_type.isascii()
            or not error_type.replace("_", "a").isalnum()
        ):
            error_type = "Exception"
        self.site = site
        self.error_type = error_type
        self.source_line = _trusted_monitor_source_line(cause)
        super().__init__(f"{site}:{error_type}")


def _trusted_monitor_source_line(exc: BaseException) -> int | None:
    """Return the terminal traceback line only when it belongs to this module."""
    source_line: int | None = None
    traceback = exc.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_globals is globals():
            candidate_line = traceback.tb_lineno
            if 1 <= candidate_line <= 999_999:
                source_line = candidate_line
        traceback = traceback.tb_next
    return source_line


def _monitor_failure_status(exc: BaseException) -> dict[str, Any]:
    """Build diagnostics without including exception text or candidate data."""
    if isinstance(exc, _MonitorStageError):
        site = exc.site
        error_type = exc.error_type
        source_line = exc.source_line
    else:
        site = "candidate_boundary"
        error_type = type(exc).__name__
        source_line = _trusted_monitor_source_line(exc)
        if (
            not error_type
            or len(error_type) > 128
            or not error_type.isascii()
            or not error_type.replace("_", "a").isalnum()
        ):
            error_type = "Exception"
    status: dict[str, Any] = {
        "version": 1,
        "candidate_ready_attested": False,
        "error": error_type,
        "site": site,
    }
    if source_line is not None:
        status["source_line"] = source_line
    return status

_SIGCHLD = 17
_CLONE_VM = 0x00000100
_CLONE_FS = 0x00000200
_CLONE_FILES = 0x00000400
_CLONE_SIGHAND = 0x00000800
_CLONE_THREAD = 0x00010000
_CLONE_SYSVSEM = 0x00040000
_CLONE_SETTLS = 0x00080000
_CLONE_PARENT_SETTID = 0x00100000
_CLONE_CHILD_CLEARTID = 0x00200000
_CLONE_CHILD_SETTID = 0x01000000
_CPYTHON_FORK_CLONE_FLAGS = _CLONE_CHILD_SETTID | _CLONE_CHILD_CLEARTID | _SIGCHLD
_CPYTHON_THREAD_CLONE_FLAGS = (
    _CLONE_VM
    | _CLONE_FS
    | _CLONE_FILES
    | _CLONE_SIGHAND
    | _CLONE_THREAD
    | _CLONE_SYSVSEM
    | _CLONE_SETTLS
    | _CLONE_PARENT_SETTID
    | _CLONE_CHILD_CLEARTID
)


@dataclass(frozen=True)
class _TeardownKillRecord:
    epoch: int
    reason: str
    targets: frozenset[int]


class _TeardownControl:
    """Cross-thread output evidence and tracer-owned kill authority."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._output_limit_requested = False
        self._kill_record: _TeardownKillRecord | None = None

    def record_output_limit(self) -> None:
        with self._lock:
            self._output_limit_requested = True
        self._wake.set()

    def output_limit_requested(self) -> bool:
        with self._lock:
            return self._output_limit_requested

    def claim_kill(
        self,
        reason: str,
        targets: frozenset[int],
    ) -> _TeardownKillRecord:
        if reason not in _TRACER_TEARDOWN_REASONS or not targets:
            raise RuntimeError("invalid tracer teardown claim")
        with self._lock:
            if self._kill_record is not None:
                raise RuntimeError("candidate teardown kill was already claimed")
            if reason == _TEARDOWN_OUTPUT_LIMIT and not self._output_limit_requested:
                raise RuntimeError("output teardown lacks relay evidence")
            if reason == _TEARDOWN_MAIN_EXIT_CLEANUP and self._output_limit_requested:
                reason = _TEARDOWN_OUTPUT_LIMIT
            self._kill_record = _TeardownKillRecord(
                epoch=1,
                reason=reason,
                targets=targets,
            )
            return self._kill_record

    def kill_record(self) -> _TeardownKillRecord | None:
        with self._lock:
            return self._kill_record

    def wait_for_wake(self, timeout: float) -> None:
        self._wake.wait(timeout)
        self._wake.clear()


def _validate_teardown_evidence(
    teardown_control: _TeardownControl,
    *,
    stdout_truncated: bool,
    stderr_truncated: bool,
    cpu_limit_hit: bool,
    file_space_limit_source: str | None,
    file_limit_signal: int | None,
) -> None:
    output_limit_hit = stdout_truncated or stderr_truncated
    file_signal_hit = (
        file_space_limit_source == "guest_monitor_ptrace_siginfo_fsize"
        and file_limit_signal == signal.SIGXFSZ
    )
    record = teardown_control.kill_record()
    if output_limit_hit != teardown_control.output_limit_requested():
        raise RuntimeError("candidate forced-teardown evidence is inconsistent")
    if record is None:
        if cpu_limit_hit or file_signal_hit:
            raise RuntimeError("candidate forced-teardown evidence is inconsistent")
        return
    if record.epoch != 1 or not record.targets:
        raise RuntimeError("candidate forced-teardown evidence is inconsistent")
    if record.reason == _TEARDOWN_OUTPUT_LIMIT and not output_limit_hit:
        raise RuntimeError("candidate forced-teardown evidence is inconsistent")
    if record.reason == _TEARDOWN_CPU_LIMIT and not cpu_limit_hit:
        raise RuntimeError("candidate forced-teardown evidence is inconsistent")
    if record.reason == _TEARDOWN_FILE_SPACE_LIMIT and not file_signal_hit:
        raise RuntimeError("candidate forced-teardown evidence is inconsistent")
    if cpu_limit_hit and record.reason != _TEARDOWN_CPU_LIMIT:
        raise RuntimeError("candidate forced-teardown evidence is inconsistent")
    if file_signal_hit and record.reason != _TEARDOWN_FILE_SPACE_LIMIT:
        raise RuntimeError("candidate forced-teardown evidence is inconsistent")


def _read_exact(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(fd, min(remaining, 64 * 1024))
        if not chunk:
            raise RuntimeError("truncated candidate frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_all(fd: int, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = os.write(fd, view[: 64 * 1024])
        if written <= 0:
            raise RuntimeError("short trusted-pipe write")
        view = view[written:]


def _integer_arg(index: int, label: str) -> int:
    try:
        value = int(sys.argv[index], 10)
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"invalid {label}") from exc
    if value <= 0:
        raise RuntimeError(f"invalid {label}")
    return value


def _prctl(operation: int, argument: int = 0) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(operation, argument, 0, 0, 0)
    if result < 0:
        raise RuntimeError(f"prctl_{operation}_failed")
    return int(result)


def _set_no_new_privs() -> None:
    if _prctl(_PR_SET_NO_NEW_PRIVS, 1) != 0:
        raise RuntimeError("PR_SET_NO_NEW_PRIVS failed")
    if _prctl(_PR_GET_NO_NEW_PRIVS) != 1:
        raise RuntimeError("PR_GET_NO_NEW_PRIVS verification failed")


def _set_nondumpable() -> None:
    if _prctl(_PR_SET_DUMPABLE, 0) != 0:
        raise RuntimeError("PR_SET_DUMPABLE failed")
    if _prctl(_PR_GET_DUMPABLE) != 0:
        raise RuntimeError("PR_GET_DUMPABLE verification failed")


def _prepare_monitor() -> tuple[bytes, str, tuple[int, int, int, int, int]]:
    address_space = _integer_arg(1, "address-space limit")
    cpu_seconds = _integer_arg(2, "CPU limit")
    file_size = _integer_arg(3, "file-size limit")
    process_count = _integer_arg(4, "process limit")
    open_files = _integer_arg(5, "open-file limit")
    nonce = os.environ.get(_NONCE_ENV, "")
    if not nonce or len(nonce) != 64 or not nonce.isascii():
        raise RuntimeError("missing launch nonce")
    _set_nondumpable()
    header = _read_exact(0, 8)
    code_size = struct.unpack("!Q", header)[0]
    if code_size > _MAX_CODE_BYTES:
        raise RuntimeError("candidate frame exceeds code limit")
    source_bytes = _read_exact(0, code_size)
    source_bytes.decode("utf-8", errors="strict")
    os.environ.pop(_NONCE_ENV, None)
    os.umask(0o077)
    os.chdir("/tmp")
    return (
        source_bytes,
        nonce,
        (address_space, cpu_seconds, file_size, process_count, open_files),
    )


# This source runs in a fresh interpreter.  It deliberately contains no launch
# nonce and closes its one-way setup evidence descriptor before candidate bytes
# can compile or execute.
_CANDIDATE_RUNNER = r"""
import builtins
import ctypes
import os
import platform
import resource
import site
import struct
import sys

PR_SET_NO_NEW_PRIVS = 38
PR_GET_NO_NEW_PRIVS = 39
PR_CAPBSET_DROP = 24
CAP_LAST_CAP = 40
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
PTRACE_TRACEME = 0
BPF_LD_W_ABS = 0x20
BPF_JMP_JEQ_K = 0x15
BPF_RET_K = 0x06
SECCOMP_RET_ALLOW = 0x7fff0000
SECCOMP_RET_TRACE = 0x7ff00000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_KILL_PROCESS = 0x80000000
AUDIT_ARCH_AARCH64 = 0xc00000b7
AUDIT_ARCH_X86_64 = 0xc000003e
EPERM = 1
ENOSYS = 38
UID = 65534
GID = 65534
SIGCHLD = 17
CLONE_VM = 0x00000100
CLONE_FS = 0x00000200
CLONE_FILES = 0x00000400
CLONE_SIGHAND = 0x00000800
CLONE_VFORK = 0x00004000
CLONE_THREAD = 0x00010000
CLONE_SYSVSEM = 0x00040000
CLONE_SETTLS = 0x00080000
CLONE_PARENT_SETTID = 0x00100000
CLONE_CHILD_CLEARTID = 0x00200000
CLONE_CHILD_SETTID = 0x01000000
CPYTHON_FORK_CLONE_FLAGS = CLONE_CHILD_SETTID | CLONE_CHILD_CLEARTID | SIGCHLD
GLIBC_VFORK_CLONE_FLAGS = CLONE_VM | CLONE_VFORK | SIGCHLD
CPYTHON_THREAD_CLONE_FLAGS = (
    CLONE_VM
    | CLONE_FS
    | CLONE_FILES
    | CLONE_SIGHAND
    | CLONE_THREAD
    | CLONE_SYSVSEM
    | CLONE_SETTLS
    | CLONE_PARENT_SETTID
    | CLONE_CHILD_CLEARTID
)
AARCH64_THREAD_TAG = 0xff02

def read_exact(fd, size):
    chunks = []
    while size:
        chunk = os.read(fd, min(size, 65536))
        if not chunk:
            raise RuntimeError("truncated private candidate frame")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)

def write_all(fd, value):
    view = memoryview(value)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise RuntimeError("short private setup write")
        view = view[count:]

def prctl(operation, argument=0):
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(operation, argument, 0, 0, 0)
    if result < 0:
        raise RuntimeError("candidate prctl failed")
    return int(result)

class SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint),
    ]

class SockFprog(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]

def syscall_policy(machine):
    if machine in {"aarch64", "arm64"}:
        # setrlimit/prlimit64; umount2/mount/pivot_root/chroot; unshare;
        # setpgid/setsid; setns; open_tree/move_mount/fsopen/fsconfig/
        # fsmount/fspick/mount_setattr.  These remain denied after exec.
        deny = (
            39, 40, 41, 51, 97, 154, 157, 164, 261, 268,
            428, 429, 430, 431, 432, 433, 442,
        )
        enosys = (435,)
        trace = (45, 46, 64, 66, 68, 70, 71, 76, 220, 285, 287)
    elif machine in {"x86_64", "amd64"}:
        # setrlimit/prlimit64; pivot_root/mount/umount2/chroot; setpgid/
        # setsid; unshare/setns; open_tree/move_mount/fsopen/fsconfig/
        # fsmount/fspick/mount_setattr.  These remain denied after exec.
        deny = (
            57, 58, 109, 112, 155, 160, 161, 165, 166, 272, 302, 308,
            428, 429, 430, 431, 432, 433, 442,
        )
        enosys = (435,)
        trace = (1, 18, 20, 40, 56, 76, 77, 296, 326, 328)
    else:
        raise RuntimeError("unsupported seccomp architecture")
    return deny, enosys, trace

def pthread_clone_filter(trace_value):
    # Admit only the exact pinned glibc/CPython pthread clone layout.  Fork,
    # vfork, process-clone layouts, missing task pointers, and namespace flags
    # receive EPERM directly from seccomp.  All offsets are into little-endian
    # struct seccomp_data u64 arguments.  x86_64 and aarch64 order the final
    # child-tid/TLS fields differently, but both must merely be nonzero, so the
    # same complete layout checks apply to both architectures.
    deny = SECCOMP_RET_ERRNO | EPERM
    instructions = []
    reject_if_not_equal = []
    reject_if_zero = []

    def require_equal(offset, value):
        instructions.append(SockFilter(BPF_LD_W_ABS, 0, 0, offset))
        reject_if_not_equal.append(len(instructions))
        instructions.append(SockFilter(BPF_JMP_JEQ_K, 0, 0, value))

    def require_nonzero_u64(low_offset):
        instructions.append(SockFilter(BPF_LD_W_ABS, 0, 0, low_offset + 4))
        # A nonzero high word skips the low-word zero check.
        instructions.append(SockFilter(BPF_JMP_JEQ_K, 0, 2, 0))
        instructions.append(SockFilter(BPF_LD_W_ABS, 0, 0, low_offset))
        reject_if_zero.append(len(instructions))
        instructions.append(SockFilter(BPF_JMP_JEQ_K, 0, 0, 0))

    require_equal(20, 0)
    require_equal(16, CPYTHON_THREAD_CLONE_FLAGS)
    for offset in (24, 32, 40, 48):
        require_nonzero_u64(offset)
    instructions.append(SockFilter(BPF_RET_K, 0, 0, trace_value))
    deny_index = len(instructions)
    instructions.append(SockFilter(BPF_RET_K, 0, 0, deny))
    for index in reject_if_not_equal:
        instructions[index].jf = deny_index - index - 1
    for index in reject_if_zero:
        instructions[index].jt = deny_index - index - 1
    return instructions

def install_filter():
    machine = platform.machine()
    deny, enosys, trace = syscall_policy(machine)
    expected_arch = (
        AUDIT_ARCH_AARCH64
        if machine in {"aarch64", "arm64"}
        else AUDIT_ARCH_X86_64
    )
    instructions = [
        SockFilter(BPF_LD_W_ABS, 0, 0, 4),
        SockFilter(BPF_JMP_JEQ_K, 1, 0, expected_arch),
        SockFilter(BPF_RET_K, 0, 0, SECCOMP_RET_KILL_PROCESS),
        SockFilter(BPF_LD_W_ABS, 0, 0, 0),
    ]
    for number in deny:
        instructions.extend([
            SockFilter(BPF_JMP_JEQ_K, 0, 1, number),
            SockFilter(BPF_RET_K, 0, 0, SECCOMP_RET_ERRNO | EPERM),
        ])
    # clone3 takes a pointer to candidate-owned memory.  Returning ENOSYS in
    # seccomp, before ptrace or dereference, both removes that TOCTOU surface
    # and selects the pinned glibc fallback to inspectable legacy clone.
    for number in enosys:
        instructions.extend([
            SockFilter(BPF_JMP_JEQ_K, 0, 1, number),
            SockFilter(BPF_RET_K, 0, 0, SECCOMP_RET_ERRNO | ENOSYS),
        ])
    for number in trace:
        clone_number = 220 if machine in {"aarch64", "arm64"} else 56
        if number == clone_number:
            trace_value = SECCOMP_RET_TRACE | (
                AARCH64_THREAD_TAG
                if machine in {"aarch64", "arm64"}
                else number
            )
            clone_filter = pthread_clone_filter(trace_value)
            instructions.append(
                SockFilter(BPF_JMP_JEQ_K, 0, len(clone_filter), number)
            )
            instructions.extend(clone_filter)
        else:
            instructions.extend([
                SockFilter(BPF_JMP_JEQ_K, 0, 1, number),
                SockFilter(BPF_RET_K, 0, 0, SECCOMP_RET_TRACE | number),
            ])
    instructions.append(SockFilter(BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW))
    array = (SockFilter * len(instructions))(*instructions)
    program = SockFprog(len(instructions), array)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(program), 0, 0) != 0:
        raise RuntimeError("candidate seccomp installation failed")

def ptrace_traceme():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.ptrace(PTRACE_TRACEME, 0, None, None) != 0:
        raise RuntimeError("candidate ptrace setup failed")

def main():
    code_fd, ready_fd, gate_fd = (int(sys.argv[index], 10) for index in (1, 2, 3))
    address_space, cpu_seconds, file_size, process_count, open_files = (
        int(sys.argv[index], 10) for index in (4, 5, 6, 7, 8)
    )
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_files, open_files))
    # Pinned gVisor accounts descendants created after this process changes
    # real UID, but not this already-existing process itself.  Reserve that
    # initial task so the semantic process tree can contain exactly
    # ``process_count`` tasks, never ``process_count + 1``.
    resource.setrlimit(
        resource.RLIMIT_NPROC, (process_count - 1, process_count - 1)
    )
    resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    if prctl(PR_SET_NO_NEW_PRIVS, 1) != 0 or prctl(PR_GET_NO_NEW_PRIVS) != 1:
        raise RuntimeError("candidate no_new_privs failed")
    for capability in range(CAP_LAST_CAP + 1):
        prctl(PR_CAPBSET_DROP, capability)
    os.setgroups([])
    os.setresgid(GID, GID, GID)
    os.setresuid(UID, UID, UID)
    if os.getresuid() != (UID, UID, UID) or os.getresgid() != (GID, GID, GID):
        raise RuntimeError("candidate identity drop failed")
    status = {}
    with open("/proc/self/status", encoding="ascii") as handle:
        for line in handle:
            if ":" in line:
                key, value = line.split(":", 1)
                status[key] = value.strip()
    if status.get("Groups", "") != "":
        raise RuntimeError("candidate supplementary groups remain")
    for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        if int(status.get(key, "-1"), 16) != 0:
            raise RuntimeError("candidate capability set is not empty")
    # Pinned gVisor does not expose Linux's ``NoNewPrivs`` status field.
    # The immediately preceding PR_GET_NO_NEW_PRIVS check is the authoritative
    # guest-kernel interface; this runner cannot reach the private ready write
    # unless that check returned exactly one.
    ptrace_traceme()
    header = read_exact(code_fd, 8)
    source_size = struct.unpack("!Q", header)[0]
    source = read_exact(code_fd, source_size).decode("utf-8", errors="strict")
    os.close(code_fd)
    # ``-I`` enables safe-path and ignores candidate-controlled environment,
    # but (unlike ``-S``) still imports the standard ``site`` module.  Assert
    # both parts of the contest-Python contract before emitting readiness:
    # generated solutions may use site-provided exit()/quit(), while neither
    # the read-only /tmp working directory nor an empty cwd entry may become
    # an import root.  The explicit site import above makes this dependency
    # visible to static measurement as well as to this runtime attestation.
    if not callable(getattr(builtins, "exit", None)) or not callable(
        getattr(builtins, "quit", None)
    ):
        raise RuntimeError("candidate site conveniences are unavailable")
    if "" in sys.path or "/tmp" in sys.path:
        raise RuntimeError("candidate working directory entered sys.path")
    write_all(ready_fd, b"R")
    os.close(ready_fd)
    os.kill(os.getpid(), 19)
    if read_exact(gate_fd, 1) != b"G":
        raise RuntimeError("candidate gate protocol failed")
    os.close(gate_fd)
    install_filter()
    sys.argv = ["/tmp/solution.py"]
    globals_for_candidate = {
        "__builtins__": __builtins__,
        "__cached__": None,
        "__doc__": None,
        "__loader__": None,
        "__name__": "__main__",
        "__file__": "/tmp/solution.py",
        "__package__": None,
        "__spec__": None,
    }
    compiled = compile(source, "/tmp/solution.py", "exec")
    exec(compiled, globals_for_candidate, globals_for_candidate)

main()
"""


def _relay(
    readable: Any,
    target_fd: int,
    cap: int,
    state: dict[str, Any],
    teardown_control: _TeardownControl,
) -> None:
    try:
        while True:
            chunk = os.read(readable.fileno(), 64 * 1024)
            if not chunk:
                return
            remaining = max(0, cap - state["bytes"])
            if remaining:
                emitted = chunk[:remaining]
                _write_all(target_fd, emitted)
                state["bytes"] += len(emitted)
            # The local verifier treats a capture that reaches exactly the
            # per-stream cap as saturated.  Match that boundary rather than
            # requiring a cap+1 byte read from the pipe.
            if state["bytes"] >= cap:
                state["exceeded"] = True
                # This flag is the pre-kill output-limit evidence consumed by
                # the signed terminal record.  Relays never signal or ptrace:
                # they wake the tracer, which snapshots the task inventory and
                # owns the one classified kill between ptrace operations.
                teardown_control.record_output_limit()
                return
    except (BrokenPipeError, OSError):
        return


def _candidate_task_ids() -> set[int]:
    tasks: set[int] = set()
    for entry in os.scandir("/proc"):
        if not entry.name.isdecimal():
            continue
        try:
            with open(f"/proc/{entry.name}/status", encoding="ascii") as handle:
                status = {
                    line.split(":", 1)[0]: line.split(":", 1)[1].split()
                    for line in handle
                    if ":" in line
                }
            if int(status["Uid"][0], 10) != _CANDIDATE_UID:
                continue
            tasks.update(
                int(task, 10)
                for task in os.listdir(f"/proc/{entry.name}/task")
                if task.isdecimal()
            )
        except (KeyError, OSError, ValueError):
            continue
    return tasks


def _candidate_open_file_size(pid: int) -> int:
    largest = 0
    try:
        entries = tuple(os.scandir(f"/proc/{pid}/fd"))
    except OSError:
        return 0
    for entry in entries:
        try:
            metadata = os.stat(entry.path)
        except OSError:
            continue
        if stat.S_ISREG(metadata.st_mode):
            largest = max(largest, metadata.st_size)
    return largest


def _candidate_proc_attested(pid: int, limits: tuple[int, int, int, int, int]) -> bool:
    address_space, cpu_seconds, file_size, process_count, open_files = limits
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as handle:
            status = {
                line.split(":", 1)[0]: line.split(":", 1)[1].split()
                for line in handle
                if ":" in line
            }
        if [int(value, 10) for value in status["Uid"][:3]] != [_CANDIDATE_UID] * 3:
            return False
        if [int(value, 10) for value in status["Gid"][:3]] != [_CANDIDATE_GID] * 3:
            return False
        if status.get("Groups", []) != []:
            return False
        if any(
            int(status.get(key, ["-1"])[0], 16) != 0
            for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
        ):
            return False
        expected_limits = {
            "Max cpu time": (str(cpu_seconds), str(cpu_seconds + 1), "seconds"),
            "Max file size": (str(file_size), str(file_size), "bytes"),
            "Max processes": (
                str(process_count - 1),
                str(process_count - 1),
                "processes",
            ),
            "Max open files": (str(open_files), str(open_files), "files"),
            "Max address space": (
                str(address_space),
                str(address_space),
                "bytes",
            ),
        }
        measured_limits: dict[str, tuple[str, str, str]] = {}
        with open(f"/proc/{pid}/limits", encoding="ascii") as handle:
            header = next(handle).split()
            if header != ["Limit", "Soft", "Limit", "Hard", "Limit", "Units"]:
                return False
            for line in handle:
                fields = line.split()
                if len(fields) < 4:
                    return False
                label = " ".join(fields[:-3])
                if label in expected_limits:
                    measured_limits[label] = (
                        fields[-3],
                        fields[-2],
                        fields[-1],
                    )
        return measured_limits == expected_limits
    except (KeyError, OSError, ValueError):
        return False


class _PtraceError(RuntimeError):
    def __init__(self, request: int, pid: int, error_number: int) -> None:
        self.request = request
        self.pid = pid
        self.error_number = error_number
        super().__init__(f"ptrace_{request}_failed_{error_number}")


def _ptrace(request: int, pid: int, address: int = 0, data: Any = None) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.ptrace.restype = ctypes.c_long
    data_value = ctypes.c_void_p(data) if isinstance(data, int) else data
    result = libc.ptrace(
        request,
        pid,
        ctypes.c_void_p(address),
        data_value,
    )
    if result == -1:
        raise _PtraceError(request, pid, ctypes.get_errno())
    return int(result)


def _ptrace_siginfo(pid: int) -> tuple[int, int]:
    raw = ctypes.create_string_buffer(128)
    _ptrace(_PTRACE_GETSIGINFO, pid, data=ctypes.byref(raw))
    signo, _error, code = struct.unpack_from("=iii", raw.raw, 0)
    return signo, code


def _ptrace_event_message(pid: int) -> int:
    value = ctypes.c_ulonglong()
    _ptrace(_PTRACE_GETEVENTMSG, pid, data=ctypes.byref(value))
    return int(value.value)


class _IOVec(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
    ]


def _register_layout() -> tuple[int, int, int, tuple[int, ...]]:
    machine = platform.machine()
    if machine in {"aarch64", "arm64"}:
        # x0 is both syscall argument zero and the result; x8 is the number.
        return 34 * 8, 0, 8 * 8, tuple(index * 8 for index in range(6))
    if machine in {"x86_64", "amd64"}:
        # Linux x86_64 user_regs_struct: rax/orig_rax are words 10/15.
        return 27 * 8, 10 * 8, 15 * 8, (14 * 8, 13 * 8, 12 * 8, 7 * 8, 9 * 8, 8 * 8)
    raise RuntimeError("unsupported ptrace architecture")


def _ptrace_registers(pid: int) -> bytearray:
    register_bytes, _result_offset, _syscall_offset, _argument_offsets = (
        _register_layout()
    )
    raw = ctypes.create_string_buffer(register_bytes)
    vector = _IOVec(ctypes.cast(raw, ctypes.c_void_p), len(raw))
    _ptrace(
        _PTRACE_GETREGSET,
        pid,
        address=_NT_PRSTATUS,
        data=ctypes.byref(vector),
    )
    if vector.length != register_bytes:
        raise RuntimeError("ptrace register set is truncated")
    return bytearray(raw.raw[:register_bytes])


def _ptrace_set_registers(pid: int, registers: bytearray) -> None:
    register_bytes, _result_offset, _syscall_offset, _argument_offsets = (
        _register_layout()
    )
    if len(registers) != register_bytes:
        raise RuntimeError("invalid ptrace register set")
    raw = ctypes.create_string_buffer(bytes(registers), register_bytes)
    vector = _IOVec(ctypes.cast(raw, ctypes.c_void_p), register_bytes)
    _ptrace(
        _PTRACE_SETREGSET,
        pid,
        address=_NT_PRSTATUS,
        data=ctypes.byref(vector),
    )
    if vector.length != register_bytes:
        raise RuntimeError("ptrace register update was truncated")


def _ptrace_syscall_result(pid: int) -> int:
    _register_bytes, result_offset, _syscall_offset, _argument_offsets = (
        _register_layout()
    )
    return struct.unpack_from("=q", _ptrace_registers(pid), result_offset)[0]


def _ptrace_syscall_arguments(pid: int) -> tuple[int, ...]:
    _register_bytes, _result_offset, _syscall_offset, argument_offsets = (
        _register_layout()
    )
    registers = _ptrace_registers(pid)
    return tuple(
        struct.unpack_from("=Q", registers, offset)[0] for offset in argument_offsets
    )


def _ptrace_skip_syscall(pid: int) -> None:
    _register_bytes, _result_offset, syscall_offset, _argument_offsets = (
        _register_layout()
    )
    registers = _ptrace_registers(pid)
    struct.pack_into("=q", registers, syscall_offset, -1)
    _ptrace_set_registers(pid, registers)


def _ptrace_set_syscall_result(pid: int, result: int) -> None:
    _register_bytes, result_offset, _syscall_offset, _argument_offsets = (
        _register_layout()
    )
    registers = _ptrace_registers(pid)
    struct.pack_into("=q", registers, result_offset, result)
    _ptrace_set_registers(pid, registers)


def _clone_mode(
    machine: str,
    syscall_number: int,
    arguments: tuple[int, ...],
) -> tuple[str, int] | None:
    """Return only the pinned pthread mode; every process clone is denied."""

    if machine in {"x86_64", "amd64"}:
        clone_number = 56
    elif machine in {"aarch64", "arm64"}:
        clone_number = 220
    else:
        raise RuntimeError("unsupported ptrace architecture")
    if syscall_number == clone_number:
        flags, child_stack = arguments[:2]
        if flags == _CPYTHON_THREAD_CLONE_FLAGS and child_stack != 0:
            if machine in {"aarch64", "arm64"}:
                parent_tid, tls, child_tid = arguments[2:5]
            else:
                parent_tid, child_tid, tls = arguments[2:5]
            if parent_tid and child_tid and tls:
                return "cpython_pthread_clone", _PTRACE_EVENT_CLONE
        return None
    return None


def _creation_mode(pid: int, syscall_number: int) -> tuple[str, int] | None:
    arguments = _ptrace_syscall_arguments(pid)
    return _clone_mode(
        platform.machine(),
        syscall_number,
        arguments,
    )


def _seccomp_creation_mode(
    event_data: int,
) -> tuple[int, tuple[str, int] | None]:
    if platform.machine() in {"aarch64", "arm64"}:
        if event_data == _SECCOMP_AARCH64_THREAD:
            return 220, ("cpython_pthread_clone", _PTRACE_EVENT_CLONE)
        if event_data in {_SECCOMP_AARCH64_FORK, _SECCOMP_AARCH64_VFORK}:
            raise RuntimeError("aarch64 process-clone seccomp tag is forbidden")
        if event_data == 220:
            raise RuntimeError("aarch64 clone lacks its seccomp layout tag")
    return event_data, None


@dataclass
class _PendingSyscall:
    number: int
    saw_entry: bool = False
    creation_mode: str | None = None
    expected_event: int | None = None
    event_child: int | None = None
    child_initial_stop_pending: bool = False
    vfork_child_phase: str | None = None
    vfork_child_release: str | None = None
    vfork_child_terminal_seen: bool = False
    denied: bool = False


def _resume_after_ptrace_stop(
    pid: int,
    pending: _PendingSyscall | None,
    *,
    deliver_signal: int | None = None,
) -> None:
    request = _PTRACE_SYSCALL if pending is not None else _PTRACE_CONT
    if deliver_signal is None:
        _ptrace(request, pid)
    else:
        _ptrace(request, pid, data=deliver_signal)


def _pending_write_is_unambiguous(
    pending: _PendingSyscall,
    file_write_syscalls: set[int],
    fork_syscalls: set[int],
) -> bool:
    return (
        pending.number in file_write_syscalls
        and pending.number not in fork_syscalls
        and pending.creation_mode is None
        and pending.expected_event is None
        and pending.event_child is None
        and not pending.denied
    )


@dataclass
class _TeardownLedger:
    kill: _TeardownKillRecord
    attributed_targets: frozenset[int]
    remaining: set[int]
    terminal_statuses: dict[int, int]
    continued_exit_stops: set[int]
    deadline: float


@dataclass(frozen=True)
class _CreationTransitionSnapshot:
    number: int
    creation_mode: str | None
    expected_event: int | None
    denied: bool
    initial_event_child: int | None


@dataclass
class _CreationTransitionState:
    snapshot: _CreationTransitionSnapshot
    pending: _PendingSyscall
    parent_result_seen: bool = False
    parent_result: int | None = None
    parent_terminal: int | None = None
    child_release: str | None = None
    child_terminal_seen: bool = False


@dataclass
class _TeardownReconciliation:
    reason: str
    transitions: dict[int, _CreationTransitionState]
    snapshotted_pending: set[int]
    resolved_transitions: set[int]
    attributed_targets: set[int]
    terminal_statuses: dict[int, int]
    children_needing_initial_stop: set[int]
    held_tasks: set[int]
    continued_exit_stops: set[int]
    deadline: float
    quiescence_sent: bool = False
    candidate_main_sigkill: bool = False


def _remap_pid_set(values: set[int], former_tid: int, current_tid: int) -> None:
    if former_tid in values:
        values.remove(former_tid)
        values.add(current_tid)


def _remap_pid_mapping(
    values: dict[int, Any],
    former_tid: int,
    current_tid: int,
    *,
    label: str,
) -> None:
    if former_tid not in values:
        return
    former_value = values[former_tid]
    if current_tid in values:
        current_value = values[current_tid]
        same_value = current_value is former_value or (
            isinstance(current_value, int)
            and isinstance(former_value, int)
            and current_value == former_value
        )
        if not same_value:
            raise RuntimeError(f"candidate exec has conflicting {label} state")
    del values[former_tid]
    values[current_tid] = former_value


def _remap_pending_child(
    pending: _PendingSyscall,
    former_tid: int,
    current_tid: int,
) -> None:
    if pending.event_child == former_tid:
        pending.event_child = current_tid


def _remap_transition_snapshot_child(
    transition: _CreationTransitionState,
    former_tid: int,
    current_tid: int,
) -> None:
    snapshot = transition.snapshot
    if snapshot.initial_event_child != former_tid:
        return
    transition.snapshot = _CreationTransitionSnapshot(
        number=snapshot.number,
        creation_mode=snapshot.creation_mode,
        expected_event=snapshot.expected_event,
        denied=snapshot.denied,
        initial_event_child=current_tid,
    )


def _fold_nonleader_exec_state(
    *,
    current_tid: int,
    former_tid: int,
    traced: set[int],
    awaiting_syscall_exit: dict[int, _PendingSyscall],
    provisional_children: set[int],
    file_write_syscalls: set[int],
    fork_syscalls: set[int],
    reconciliation: _TeardownReconciliation | None = None,
    ledger: _TeardownLedger | None = None,
) -> None:
    """Fold Linux's former nonleader TID into the post-exec leader TID.

    Linux reports a non-thread-group-leader exec stop under the thread-group
    leader's TID and exposes the execing thread's former TID through
    ``PTRACE_GETEVENTMSG``.  No later wait will arrive under that former TID,
    so every live ownership/teardown ledger must be folded at this exact
    kernel-authenticated boundary.
    """

    if former_tid in {0, current_tid}:
        return
    exec_tids = {former_tid, current_tid}
    if (
        former_tid <= 1
        or current_tid <= 1
        or current_tid not in traced
        or former_tid not in traced
    ):
        raise RuntimeError("candidate exec former TID is unattributed")
    if exec_tids & provisional_children:
        raise RuntimeError("candidate exec has ambiguous provisional state")
    if reconciliation is not None and exec_tids & set(reconciliation.terminal_statuses):
        raise RuntimeError("candidate exec has terminal evidence for a live task")
    if ledger is not None and exec_tids & set(ledger.terminal_statuses):
        raise RuntimeError("candidate exec has terminal evidence for a live task")
    current_transition = (
        reconciliation.transitions.get(current_tid)
        if reconciliation is not None
        else None
    )
    if current_transition is not None:
        assert reconciliation is not None
        if current_tid not in reconciliation.resolved_transitions:
            raise RuntimeError("candidate exec replaced a creation transition task")
    current_pending = awaiting_syscall_exit.get(current_tid)
    if current_pending is not None:
        if not _pending_write_is_unambiguous(
            current_pending,
            file_write_syscalls,
            fork_syscalls,
        ):
            raise RuntimeError("candidate exec replaced a pending syscall task")
        # ``current_tid`` named the pre-exec group leader, which Linux has
        # destroyed.  Its unambiguous write cannot complete, and must not be
        # attributed to the former-TID task that now owns this numeric PID.
        awaiting_syscall_exit.pop(current_tid)

    traced.remove(former_tid)
    _remap_pid_set(provisional_children, former_tid, current_tid)
    _remap_pid_mapping(
        awaiting_syscall_exit,
        former_tid,
        current_tid,
        label="pending syscall",
    )

    pending_values: list[_PendingSyscall] = list(awaiting_syscall_exit.values())
    if reconciliation is not None:
        reconciliation.transitions.pop(current_tid, None)
        reconciliation.snapshotted_pending.discard(current_tid)
        reconciliation.resolved_transitions.discard(current_tid)
        reconciliation.held_tasks.discard(current_tid)
        _remap_pid_mapping(
            reconciliation.transitions,
            former_tid,
            current_tid,
            label="creation transition",
        )
        _remap_pid_set(
            reconciliation.snapshotted_pending,
            former_tid,
            current_tid,
        )
        _remap_pid_set(
            reconciliation.resolved_transitions,
            former_tid,
            current_tid,
        )
        _remap_pid_set(
            reconciliation.attributed_targets,
            former_tid,
            current_tid,
        )
        _remap_pid_set(
            reconciliation.children_needing_initial_stop,
            former_tid,
            current_tid,
        )
        _remap_pid_set(reconciliation.held_tasks, former_tid, current_tid)
        # An EXIT continuation belongs to one exact task lifetime.  Neither
        # pre-exec identity can prove that the replacement leader's later EXIT
        # has already been continued.
        reconciliation.continued_exit_stops.difference_update(exec_tids)
        pending_values.extend(
            transition.pending for transition in reconciliation.transitions.values()
        )

    if ledger is not None:
        attributed_targets = set(ledger.attributed_targets)
        _remap_pid_set(attributed_targets, former_tid, current_tid)
        ledger.attributed_targets = frozenset(attributed_targets)
        _remap_pid_set(ledger.remaining, former_tid, current_tid)
        ledger.continued_exit_stops.difference_update(exec_tids)

    seen_pending: set[int] = set()
    for pending in pending_values:
        identity = id(pending)
        if identity in seen_pending:
            continue
        seen_pending.add(identity)
        _remap_pending_child(pending, former_tid, current_tid)
    if reconciliation is not None:
        for transition in reconciliation.transitions.values():
            _remap_transition_snapshot_child(
                transition,
                former_tid,
                current_tid,
            )


def _terminal_returncode(status: int) -> int:
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    raise RuntimeError("candidate teardown wait is not terminal")


def _continue_captured_exit_stop(pid: int) -> None:
    try:
        _ptrace(_PTRACE_CONT, pid)
    except _PtraceError as exc:
        # At an exact PTRACE_EVENT_EXIT stop the tracee can become unrestartable
        # before the tracer's CONT reaches the kernel.  ESRCH is safe only here:
        # the immutable ledger still requires the exact terminal wait for ``pid``.
        if (
            exc.request != _PTRACE_CONT
            or exc.pid != pid
            or exc.error_number != errno.ESRCH
        ):
            raise


def _validate_vfork_child_state(pending: _PendingSyscall) -> None:
    if pending.expected_event != _PTRACE_EVENT_VFORK:
        if (
            pending.vfork_child_phase is not None
            or pending.vfork_child_release is not None
            or pending.vfork_child_terminal_seen
        ):
            raise RuntimeError("non-vfork transition has vfork child state")
        return
    if pending.event_child is None:
        if (
            pending.child_initial_stop_pending
            or pending.vfork_child_phase is not None
            or pending.vfork_child_release is not None
            or pending.vfork_child_terminal_seen
        ):
            raise RuntimeError("eventless vfork transition has child state")
        return
    if pending.vfork_child_phase not in _VFORK_CHILD_PHASES:
        raise RuntimeError("vfork transition lacks exact child phase")
    if pending.child_initial_stop_pending != (
        pending.vfork_child_phase == _VFORK_CHILD_INITIAL_STOP
    ):
        raise RuntimeError("vfork child phase and initial stop disagree")
    if pending.vfork_child_phase in {
        _VFORK_CHILD_INITIAL_STOP,
        _VFORK_CHILD_RUNNING,
    } and (
        pending.vfork_child_release is not None or pending.vfork_child_terminal_seen
    ):
        raise RuntimeError("unreleased vfork child has release evidence")
    if pending.vfork_child_phase == _VFORK_CHILD_RELEASED and (
        pending.vfork_child_release not in {_VFORK_RELEASE_EXEC, _VFORK_RELEASE_EXIT}
        or pending.vfork_child_terminal_seen
    ):
        raise RuntimeError("released vfork child evidence is inconsistent")
    if pending.vfork_child_phase == _VFORK_CHILD_TERMINAL and (
        pending.vfork_child_release
        not in {
            _VFORK_RELEASE_EXEC,
            _VFORK_RELEASE_EXIT,
            _VFORK_RELEASE_TERMINAL,
        }
        or not pending.vfork_child_terminal_seen
    ):
        raise RuntimeError("terminal vfork child evidence is inconsistent")


def _vfork_child_owner(
    awaiting_syscall_exit: dict[int, _PendingSyscall],
    child: int,
) -> _PendingSyscall | None:
    owners = [
        pending
        for pending in awaiting_syscall_exit.values()
        if pending.expected_event == _PTRACE_EVENT_VFORK
        and pending.event_child == child
    ]
    if len(owners) > 1:
        raise RuntimeError("vfork child has multiple parent transitions")
    if not owners:
        return None
    pending = owners[0]
    _validate_vfork_child_state(pending)
    return pending


def _reconciliation_vfork_owner(
    reconciliation: _TeardownReconciliation,
    child: int,
) -> tuple[int, _CreationTransitionState] | None:
    owners = [
        (parent, transition)
        for parent, transition in reconciliation.transitions.items()
        if transition.pending.expected_event == _PTRACE_EVENT_VFORK
        and transition.pending.event_child == child
    ]
    if len(owners) > 1:
        raise RuntimeError("vfork child has multiple teardown transitions")
    if not owners:
        return None
    _validate_vfork_child_state(owners[0][1].pending)
    return owners[0]


def _resume_released_vfork_parent(
    reconciliation: _TeardownReconciliation,
    *,
    parent: int,
    transition: _CreationTransitionState,
    traced: set[int],
) -> None:
    if (
        transition.child_release is None
        or transition.parent_result_seen
        or transition.parent_terminal is not None
        or parent not in traced
        or parent not in reconciliation.held_tasks
    ):
        return
    reconciliation.held_tasks.remove(parent)
    _resume_after_ptrace_stop(parent, transition.pending)


def _set_vfork_child_initial_phase(
    pending: _PendingSyscall,
    *,
    held: bool,
) -> None:
    if pending.expected_event != _PTRACE_EVENT_VFORK or pending.event_child is None:
        raise RuntimeError("vfork child phase lacks its exact event")
    pending.child_initial_stop_pending = not held
    pending.vfork_child_phase = (
        _VFORK_CHILD_RUNNING if held else _VFORK_CHILD_INITIAL_STOP
    )
    pending.vfork_child_release = None
    pending.vfork_child_terminal_seen = False
    _validate_vfork_child_state(pending)


def _mark_vfork_child_running(pending: _PendingSyscall) -> None:
    _validate_vfork_child_state(pending)
    if pending.vfork_child_phase != _VFORK_CHILD_INITIAL_STOP:
        raise RuntimeError("vfork child running phase is inconsistent")
    pending.child_initial_stop_pending = False
    pending.vfork_child_phase = _VFORK_CHILD_RUNNING


def _record_vfork_child_release(
    pending: _PendingSyscall,
    release: str,
) -> None:
    _validate_vfork_child_state(pending)
    if release not in {_VFORK_RELEASE_EXEC, _VFORK_RELEASE_EXIT}:
        raise RuntimeError("vfork child release kind is invalid")
    if pending.vfork_child_phase == _VFORK_CHILD_RUNNING:
        pending.vfork_child_phase = _VFORK_CHILD_RELEASED
        pending.vfork_child_release = release
    elif (
        pending.vfork_child_phase == _VFORK_CHILD_RELEASED
        and pending.vfork_child_release == _VFORK_RELEASE_EXEC
        and release == _VFORK_RELEASE_EXIT
    ):
        # EXEC already released the parent. A later queued EXIT stop must be
        # continued, but it cannot replace the stronger earlier proof.
        pass
    elif (
        pending.vfork_child_phase != _VFORK_CHILD_RELEASED
        or pending.vfork_child_release != release
    ):
        raise RuntimeError("vfork child release preceded its runnable phase")
    _validate_vfork_child_state(pending)


def _mark_vfork_child_released(
    awaiting_syscall_exit: dict[int, _PendingSyscall],
    child: int,
    release: str,
) -> bool:
    pending = _vfork_child_owner(awaiting_syscall_exit, child)
    if pending is None:
        return False
    _record_vfork_child_release(pending, release)
    return True


def _mark_vfork_child_terminal(
    awaiting_syscall_exit: dict[int, _PendingSyscall],
    child: int,
) -> bool:
    pending = _vfork_child_owner(awaiting_syscall_exit, child)
    if pending is None:
        return False
    if pending.vfork_child_phase not in {
        _VFORK_CHILD_RUNNING,
        _VFORK_CHILD_RELEASED,
    }:
        raise RuntimeError("vfork child terminal preceded its runnable phase")
    pending.vfork_child_phase = _VFORK_CHILD_TERMINAL
    if pending.vfork_child_release is None:
        pending.vfork_child_release = _VFORK_RELEASE_TERMINAL
    pending.vfork_child_terminal_seen = True
    _validate_vfork_child_state(pending)
    return True


def _creation_transition_snapshot(
    pending: _PendingSyscall,
    fork_syscalls: set[int],
) -> _CreationTransitionSnapshot | None:
    if pending.number not in fork_syscalls:
        return None
    if pending.denied:
        if (
            pending.creation_mode is not None
            or pending.expected_event is not None
            or pending.event_child is not None
            or pending.child_initial_stop_pending
        ):
            raise RuntimeError("denied creation transition is inconsistent")
    elif pending.creation_mode is None or pending.expected_event not in {
        _PTRACE_EVENT_FORK,
        _PTRACE_EVENT_VFORK,
        _PTRACE_EVENT_CLONE,
    }:
        raise RuntimeError("candidate teardown has ambiguous creation transition")
    elif pending.child_initial_stop_pending and pending.event_child is None:
        raise RuntimeError("candidate teardown child stop evidence is inconsistent")
    _validate_vfork_child_state(pending)
    return _CreationTransitionSnapshot(
        number=pending.number,
        creation_mode=pending.creation_mode,
        expected_event=pending.expected_event,
        denied=pending.denied,
        initial_event_child=pending.event_child,
    )


def _transition_matches_snapshot(
    pending: _PendingSyscall,
    snapshot: _CreationTransitionSnapshot,
) -> bool:
    _validate_vfork_child_state(pending)
    return (
        pending.number == snapshot.number
        and pending.creation_mode == snapshot.creation_mode
        and pending.expected_event == snapshot.expected_event
        and pending.denied == snapshot.denied
        and (
            pending.event_child == snapshot.initial_event_child
            or (
                snapshot.initial_event_child is None
                and not snapshot.denied
                and pending.event_child is not None
            )
        )
    )


def _snapshot_teardown_reconciliation(
    *,
    reason: str,
    traced: set[int],
    awaiting_syscall_exit: dict[int, _PendingSyscall],
    provisional_children: set[int],
    file_write_syscalls: set[int],
    fork_syscalls: set[int],
    current_time: float,
) -> _TeardownReconciliation:
    if not traced or not set(awaiting_syscall_exit).issubset(traced):
        raise RuntimeError("candidate teardown has creation or task ambiguity")
    transitions: dict[int, _CreationTransitionState] = {}
    for pid, pending in awaiting_syscall_exit.items():
        snapshot = _creation_transition_snapshot(pending, fork_syscalls)
        if snapshot is not None:
            transitions[pid] = _CreationTransitionState(
                snapshot=snapshot,
                pending=pending,
                child_release=pending.vfork_child_release,
                child_terminal_seen=pending.vfork_child_terminal_seen,
            )
            continue
        if not _pending_write_is_unambiguous(
            pending,
            file_write_syscalls,
            fork_syscalls,
        ):
            raise RuntimeError("candidate teardown has ambiguous pending syscall")
    available_child_slots = sum(
        not transition.snapshot.denied
        and transition.snapshot.initial_event_child is None
        for transition in transitions.values()
    )
    if len(provisional_children) > available_child_slots:
        raise RuntimeError("candidate teardown has creation or task ambiguity")
    return _TeardownReconciliation(
        reason=reason,
        transitions=transitions,
        snapshotted_pending=set(awaiting_syscall_exit),
        resolved_transitions=set(),
        # A child-first initial SIGSTOP is held but remains unattributed until
        # its exact parent ptrace event names it.
        attributed_targets=set(traced),
        terminal_statuses={},
        children_needing_initial_stop={
            pending.event_child
            for pid in transitions
            if (pending := awaiting_syscall_exit[pid]).event_child is not None
            and pending.child_initial_stop_pending
        },
        # Provisional children are exact consumed initial SIGSTOP waits.  They
        # are physically held, but remain unattributed until a parent event
        # names the same pid.
        held_tasks=set(provisional_children),
        continued_exit_stops=set(),
        deadline=current_time + _TEARDOWN_DRAIN_TIMEOUT_SECONDS,
    )


def _register_quiescence_transition(
    reconciliation: _TeardownReconciliation,
    *,
    pid: int,
    pending: _PendingSyscall,
    fork_syscalls: set[int],
) -> None:
    if pid in reconciliation.snapshotted_pending:
        raise RuntimeError("candidate teardown syscall transition was duplicated")
    snapshot = _creation_transition_snapshot(pending, fork_syscalls)
    reconciliation.snapshotted_pending.add(pid)
    if snapshot is not None:
        reconciliation.transitions[pid] = _CreationTransitionState(
            snapshot=snapshot,
            pending=pending,
            child_release=pending.vfork_child_release,
            child_terminal_seen=pending.vfork_child_terminal_seen,
        )


def _refresh_creation_transition_resolution(
    reconciliation: _TeardownReconciliation,
    awaiting_syscall_exit: dict[int, _PendingSyscall],
) -> None:
    for pid, transition in reconciliation.transitions.items():
        if pid in reconciliation.resolved_transitions:
            continue
        pending = transition.pending
        if not _transition_matches_snapshot(pending, transition.snapshot):
            raise RuntimeError("candidate teardown creation transition changed")
        if transition.parent_terminal is not None:
            if transition.parent_terminal != -signal.SIGKILL:
                raise RuntimeError(
                    "creation parent terminal signal is not attributable"
                )
            if pending.event_child is None:
                continue
            if pending.child_initial_stop_pending:
                continue
            awaiting_syscall_exit.pop(pid, None)
            reconciliation.resolved_transitions.add(pid)
            continue

        if pending.event_child is None:
            if transition.parent_result_seen:
                reconciliation.resolved_transitions.add(pid)
            continue
        if pending.child_initial_stop_pending:
            continue
        if pending.expected_event != _PTRACE_EVENT_VFORK:
            if transition.parent_result_seen:
                reconciliation.resolved_transitions.add(pid)
            continue
        if not pending.saw_entry:
            raise RuntimeError("captured vfork transition lacks exact entry evidence")
        if transition.parent_result_seen:
            if transition.child_release == _VFORK_RELEASE_EXEC or (
                transition.child_release
                in {_VFORK_RELEASE_EXIT, _VFORK_RELEASE_TERMINAL}
                and transition.child_terminal_seen
            ):
                reconciliation.resolved_transitions.add(pid)
            continue
        if (
            transition.child_release is None
            and pending.event_child in reconciliation.held_tasks
            and pid in reconciliation.held_tasks
        ):
            # Physical quiescence, rather than a procfs snapshot, proves that
            # the pre-release vfork parent cannot produce a syscall result.
            awaiting_syscall_exit.pop(pid, None)
            reconciliation.resolved_transitions.add(pid)


def _reconciliation_ready(
    reconciliation: _TeardownReconciliation,
    traced: set[int],
    awaiting_syscall_exit: dict[int, _PendingSyscall],
    provisional_children: set[int],
    file_write_syscalls: set[int],
    fork_syscalls: set[int],
) -> bool:
    if not reconciliation.quiescence_sent:
        raise RuntimeError("candidate teardown quiescence pulse is absent")
    observed_tasks = _candidate_task_ids()
    unknown_tasks = observed_tasks - traced
    if unknown_tasks:
        available_child_slots = sum(
            parent not in reconciliation.resolved_transitions
            and not transition.pending.denied
            and transition.pending.event_child is None
            for parent, transition in reconciliation.transitions.items()
        )
        if len(unknown_tasks) > available_child_slots:
            raise RuntimeError("candidate task appeared without an exact ptrace event")
        return False
    if not traced.issubset(reconciliation.attributed_targets):
        raise RuntimeError("candidate teardown attribution ledger changed")
    if provisional_children or not traced.issubset(reconciliation.held_tasks):
        return False
    _refresh_creation_transition_resolution(
        reconciliation,
        awaiting_syscall_exit,
    )
    if (
        reconciliation.resolved_transitions != set(reconciliation.transitions)
        or reconciliation.children_needing_initial_stop
    ):
        return False
    if not set(awaiting_syscall_exit).issubset(reconciliation.snapshotted_pending):
        raise RuntimeError("candidate created a syscall transition during teardown")
    for pending in awaiting_syscall_exit.values():
        if not _pending_write_is_unambiguous(
            pending,
            file_write_syscalls,
            fork_syscalls,
        ):
            raise RuntimeError("candidate teardown transition did not reconcile")
    return True


def _finalize_teardown_reconciliation(
    proc: subprocess.Popen[bytes],
    *,
    reconciliation: _TeardownReconciliation,
    traced: set[int],
    awaiting_syscall_exit: dict[int, _PendingSyscall],
    provisional_children: set[int],
    file_write_syscalls: set[int],
    fork_syscalls: set[int],
    teardown_control: _TeardownControl,
) -> tuple[bool, _TeardownLedger | None]:
    if not _reconciliation_ready(
        reconciliation,
        traced,
        awaiting_syscall_exit,
        provisional_children,
        file_write_syscalls,
        fork_syscalls,
    ):
        return False, None
    if reconciliation.attributed_targets != (
        traced | set(reconciliation.terminal_statuses)
    ):
        raise RuntimeError("candidate teardown reconciliation ledger is incomplete")
    targets = frozenset(traced)
    if not targets:
        if (
            reconciliation.reason == _TEARDOWN_MAIN_EXIT_CLEANUP
            and reconciliation.transitions
            and not reconciliation.candidate_main_sigkill
        ):
            raise RuntimeError("main-exit teardown lacks its exact terminal evidence")
        return True, None
    kill = teardown_control.claim_kill(reconciliation.reason, targets)
    _kill_candidate_group_once(proc)
    return (
        True,
        _TeardownLedger(
            kill=kill,
            attributed_targets=frozenset(reconciliation.attributed_targets),
            remaining=set(traced),
            terminal_statuses=dict(reconciliation.terminal_statuses),
            continued_exit_stops=set(reconciliation.continued_exit_stops),
            deadline=reconciliation.deadline,
        ),
    )


def _begin_teardown_quiescence(
    proc: subprocess.Popen[bytes],
    *,
    reason: str,
    traced: set[int],
    awaiting_syscall_exit: dict[int, _PendingSyscall],
    provisional_children: set[int],
    file_write_syscalls: set[int],
    fork_syscalls: set[int],
    current_time: float,
) -> _TeardownReconciliation:
    reconciliation = _snapshot_teardown_reconciliation(
        reason=reason,
        traced=traced,
        awaiting_syscall_exit=awaiting_syscall_exit,
        provisional_children=provisional_children,
        file_write_syscalls=file_write_syscalls,
        fork_syscalls=fork_syscalls,
        current_time=current_time,
    )
    _stop_candidate_group_once(proc)
    reconciliation.quiescence_sent = True
    return reconciliation


def _validate_teardown_inventory(ledger: _TeardownLedger) -> set[int]:
    observed_tasks = _candidate_task_ids()
    if not observed_tasks.issubset(ledger.remaining):
        raise RuntimeError("candidate task appeared after teardown capture")
    return observed_tasks


def _teardown_ledger_is_complete(
    ledger: _TeardownLedger,
    *,
    observed_tasks: set[int],
    traced: set[int],
    awaiting_syscall_exit: dict[int, _PendingSyscall],
    provisional_children: set[int],
) -> bool:
    if ledger.remaining:
        return False
    if (
        observed_tasks
        or traced
        or awaiting_syscall_exit
        or provisional_children
        or set(ledger.terminal_statuses) != set(ledger.attributed_targets)
    ):
        raise RuntimeError("candidate teardown ledger is incomplete")
    return True


def _validate_creation_result(
    pending: _PendingSyscall,
    result: int,
    traced: set[int],
    observed_tasks: set[int],
) -> None:
    _validate_vfork_child_state(pending)
    if result >= 0:
        if pending.event_child is None or result != pending.event_child:
            raise RuntimeError("successful creation lacks its exact ptrace child")
        if observed_tasks != traced:
            raise RuntimeError("successful creation task inventory is incomplete")
    elif pending.event_child is not None:
        raise RuntimeError("failed creation emitted a ptrace child")


def _fork_syscalls() -> set[int]:
    machine = platform.machine()
    if machine in {"aarch64", "arm64"}:
        return {220}
    if machine in {"x86_64", "amd64"}:
        return {56}
    raise RuntimeError("unsupported ptrace architecture")


def _file_write_syscalls() -> set[int]:
    machine = platform.machine()
    if machine in {"aarch64", "arm64"}:
        return {45, 46, 64, 66, 68, 70, 71, 76, 285, 287}
    if machine in {"x86_64", "amd64"}:
        return {1, 18, 20, 40, 76, 77, 296, 326, 328}
    raise RuntimeError("unsupported ptrace architecture")


def _attest_initial_trace_stop(
    proc: subprocess.Popen[bytes],
    limits: tuple[int, int, int, int, int],
) -> None:
    waited_pid, status, _usage = os.wait4(proc.pid, os.WUNTRACED | _WAIT_ALL)
    if (
        waited_pid != proc.pid
        or not os.WIFSTOPPED(status)
        or os.WSTOPSIG(status) != signal.SIGSTOP
    ):
        raise RuntimeError("candidate initial trace stop is invalid")
    if not _candidate_proc_attested(proc.pid, limits):
        raise RuntimeError("candidate process attestation failed")
    if _candidate_task_ids() != {proc.pid}:
        raise RuntimeError("candidate initial task inventory is ambiguous")
    options = (
        _PTRACE_O_TRACESYSGOOD
        | _PTRACE_O_TRACEFORK
        | _PTRACE_O_TRACEVFORK
        | _PTRACE_O_TRACECLONE
        | _PTRACE_O_TRACEEXEC
        | _PTRACE_O_TRACEEXIT
        | _PTRACE_O_TRACESECCOMP
        | _PTRACE_O_EXITKILL
    )
    _ptrace(_PTRACE_SETOPTIONS, proc.pid, data=options)


class _CandidateReadyEOF(RuntimeError):
    """The trusted candidate runner exited before its readiness marker."""


class _CandidateReadyTimeout(RuntimeError):
    """The trusted candidate runner did not become ready within its deadline."""


class _CandidateReadyProtocolError(RuntimeError):
    """The trusted candidate runner emitted an invalid readiness marker."""


def _wait_ready(fd: int) -> None:
    selector = selectors.DefaultSelector()
    try:
        selector.register(fd, selectors.EVENT_READ)
        deadline = time.monotonic() + _CANDIDATE_READY_TIMEOUT_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _CandidateReadyTimeout
            if selector.select(min(0.05, remaining)):
                marker = os.read(fd, 2)
                if marker == b"R":
                    return
                if not marker:
                    raise _CandidateReadyEOF
                raise _CandidateReadyProtocolError
    finally:
        selector.close()


def _trace_and_measure(
    proc: subprocess.Popen[bytes],
    *,
    limits: tuple[int, int, int, int, int],
    process_limit: int,
    file_size_limit: int,
    teardown_control: _TeardownControl,
) -> tuple[
    int,
    int,
    int,
    bool,
    str | None,
    int | None,
    int | None,
    int,
    int,
    bool,
    int | None,
    bool,
    bool,
]:
    traced = {proc.pid}
    provisional_children: set[int] = set()
    # Pinned gVisor emits a syscall-entry stop after its seccomp event and
    # before the eventual syscall-exit stop.
    awaiting_syscall_exit: dict[int, _PendingSyscall] = {}
    main_returncode: int | None = None
    total_cpu_usage_us = 0
    process_peak = 1
    file_space_limit_source: str | None = None
    file_limit_signal: int | None = None
    file_limit_errno: int | None = None
    file_size_observed_bytes = 0
    writable_available_bytes = 0
    cpu_limit_hit = False
    process_limit_hit = False
    process_limit_syscall: int | None = None
    fork_syscalls = _fork_syscalls()
    file_write_syscalls = _file_write_syscalls()
    teardown_ledger: _TeardownLedger | None = None
    teardown_reconciliation: _TeardownReconciliation | None = None

    while (
        traced
        or provisional_children
        or teardown_ledger is not None
        or teardown_reconciliation is not None
    ):
        waited_pid, status, usage = os.wait4(-1, os.WNOHANG | os.WUNTRACED | _WAIT_ALL)
        current_time = time.monotonic()
        if teardown_ledger is not None:
            observed_tasks = _validate_teardown_inventory(teardown_ledger)
            if waited_pid == 0:
                if current_time >= teardown_ledger.deadline:
                    raise RuntimeError("candidate teardown terminal drain timed out")
                teardown_control.wait_for_wake(
                    min(
                        0.001,
                        max(0.0, teardown_ledger.deadline - current_time),
                    )
                )
                continue
            if waited_pid not in teardown_ledger.remaining:
                raise RuntimeError("candidate teardown wait is outside target ledger")
            if os.WIFSTOPPED(status):
                # Darwin's wait macros retain ptrace event bits here while
                # Linux masks them.  The low byte is the portable stop signal.
                stopped_signal = os.WSTOPSIG(status) & 0xFF
                event = status >> 16
                if event == _PTRACE_EVENT_EXEC and stopped_signal == signal.SIGTRAP:
                    _fold_nonleader_exec_state(
                        current_tid=waited_pid,
                        former_tid=_ptrace_event_message(waited_pid),
                        traced=traced,
                        awaiting_syscall_exit=awaiting_syscall_exit,
                        provisional_children=provisional_children,
                        file_write_syscalls=file_write_syscalls,
                        fork_syscalls=fork_syscalls,
                        ledger=teardown_ledger,
                    )
                    _mark_vfork_child_released(
                        awaiting_syscall_exit,
                        waited_pid,
                        _VFORK_RELEASE_EXEC,
                    )
                    _resume_after_ptrace_stop(
                        waited_pid,
                        awaiting_syscall_exit.get(waited_pid),
                    )
                    continue
                if (
                    event == _PTRACE_EVENT_EXIT
                    and stopped_signal == signal.SIGTRAP
                    and waited_pid not in teardown_ledger.continued_exit_stops
                ):
                    teardown_ledger.continued_exit_stops.add(waited_pid)
                    _continue_captured_exit_stop(waited_pid)
                    continue
                raise RuntimeError("candidate teardown emitted a nonterminal wait")
            returncode = _terminal_returncode(status)
            total_cpu_usage_us += round((usage.ru_utime + usage.ru_stime) * 1_000_000)
            pending = awaiting_syscall_exit.get(waited_pid)
            if pending is not None:
                if not _pending_write_is_unambiguous(
                    pending,
                    file_write_syscalls,
                    fork_syscalls,
                ):
                    raise RuntimeError(
                        "candidate teardown terminal has ambiguous pending syscall"
                    )
                awaiting_syscall_exit.pop(waited_pid)
            teardown_ledger.terminal_statuses[waited_pid] = returncode
            teardown_ledger.remaining.remove(waited_pid)
            traced.remove(waited_pid)
            if waited_pid == proc.pid:
                if main_returncode is not None:
                    raise RuntimeError("candidate main exit status was duplicated")
                main_returncode = returncode
                proc.returncode = main_returncode
            observed_tasks = _validate_teardown_inventory(teardown_ledger)
            if _teardown_ledger_is_complete(
                teardown_ledger,
                observed_tasks=observed_tasks,
                traced=traced,
                awaiting_syscall_exit=awaiting_syscall_exit,
                provisional_children=provisional_children,
            ):
                teardown_ledger = None
            continue
        if (
            teardown_reconciliation is None
            and teardown_control.output_limit_requested()
        ):
            teardown_reconciliation = _begin_teardown_quiescence(
                proc,
                reason=_TEARDOWN_OUTPUT_LIMIT,
                traced=traced,
                awaiting_syscall_exit=awaiting_syscall_exit,
                provisional_children=provisional_children,
                file_write_syscalls=file_write_syscalls,
                fork_syscalls=fork_syscalls,
                current_time=current_time,
            )
        pending_main = awaiting_syscall_exit.get(proc.pid)
        if (
            teardown_reconciliation is None
            and waited_pid == proc.pid
            and os.WIFSIGNALED(status)
            and os.WTERMSIG(status) == signal.SIGKILL
            and pending_main is not None
            and _creation_transition_snapshot(pending_main, fork_syscalls) is not None
        ):
            teardown_reconciliation = _begin_teardown_quiescence(
                proc,
                reason=_TEARDOWN_MAIN_EXIT_CLEANUP,
                traced=traced,
                awaiting_syscall_exit=awaiting_syscall_exit,
                provisional_children=provisional_children,
                file_write_syscalls=file_write_syscalls,
                fork_syscalls=fork_syscalls,
                current_time=current_time,
            )
        if teardown_reconciliation is not None:
            reconciliation = teardown_reconciliation
            if current_time >= reconciliation.deadline:
                raise RuntimeError("candidate teardown reconciliation timed out")
            if waited_pid == 0:
                completed, finalized_ledger = _finalize_teardown_reconciliation(
                    proc,
                    reconciliation=reconciliation,
                    traced=traced,
                    awaiting_syscall_exit=awaiting_syscall_exit,
                    provisional_children=provisional_children,
                    file_write_syscalls=file_write_syscalls,
                    fork_syscalls=fork_syscalls,
                    teardown_control=teardown_control,
                )
                if completed:
                    teardown_reconciliation = None
                    teardown_ledger = finalized_ledger
                    if finalized_ledger is not None:
                        observed_tasks = _validate_teardown_inventory(finalized_ledger)
                        if _teardown_ledger_is_complete(
                            finalized_ledger,
                            observed_tasks=observed_tasks,
                            traced=traced,
                            awaiting_syscall_exit=awaiting_syscall_exit,
                            provisional_children=provisional_children,
                        ):
                            teardown_ledger = None
                    continue
                teardown_control.wait_for_wake(
                    min(
                        0.001,
                        max(0.0, reconciliation.deadline - current_time),
                    )
                )
                continue
            if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                if waited_pid not in traced:
                    raise RuntimeError(
                        "candidate teardown reconciliation wait is unattributed"
                    )
                returncode = _terminal_returncode(status)
                total_cpu_usage_us += round(
                    (usage.ru_utime + usage.ru_stime) * 1_000_000
                )
                transition = reconciliation.transitions.get(waited_pid)
                if transition is not None:
                    pending = transition.pending
                    if not _transition_matches_snapshot(pending, transition.snapshot):
                        raise RuntimeError(
                            "candidate teardown creation transition changed"
                        )
                    if (
                        waited_pid != proc.pid
                        or returncode != -signal.SIGKILL
                        or transition.parent_result_seen
                    ):
                        raise RuntimeError(
                            "candidate teardown creation result is absent"
                        )
                    transition.parent_terminal = returncode
                    reconciliation.candidate_main_sigkill = True
                    awaiting_syscall_exit.pop(waited_pid, None)
                vfork_owner = _reconciliation_vfork_owner(
                    reconciliation,
                    waited_pid,
                )
                if vfork_owner is not None:
                    owner_parent, owner_transition = vfork_owner
                    owner_pending = owner_transition.pending
                    if owner_pending.vfork_child_phase not in {
                        _VFORK_CHILD_RUNNING,
                        _VFORK_CHILD_RELEASED,
                    }:
                        raise RuntimeError(
                            "vfork child terminal preceded its runnable phase"
                        )
                    owner_pending.vfork_child_phase = _VFORK_CHILD_TERMINAL
                    if owner_pending.vfork_child_release is None:
                        owner_pending.vfork_child_release = _VFORK_RELEASE_TERMINAL
                    owner_pending.vfork_child_terminal_seen = True
                    owner_transition.child_terminal_seen = True
                    if owner_transition.child_release is None:
                        owner_transition.child_release = _VFORK_RELEASE_TERMINAL
                    _resume_released_vfork_parent(
                        reconciliation,
                        parent=owner_parent,
                        transition=owner_transition,
                        traced=traced,
                    )
                pending = awaiting_syscall_exit.get(waited_pid)
                if (
                    transition is None
                    and pending is not None
                    and not _pending_write_is_unambiguous(
                        pending,
                        file_write_syscalls,
                        fork_syscalls,
                    )
                ):
                    raise RuntimeError(
                        "candidate teardown terminal has ambiguous pending syscall"
                    )
                if transition is None:
                    awaiting_syscall_exit.pop(waited_pid, None)
                traced.remove(waited_pid)
                reconciliation.held_tasks.discard(waited_pid)
                reconciliation.children_needing_initial_stop.discard(waited_pid)
                if waited_pid in reconciliation.terminal_statuses:
                    raise RuntimeError(
                        "candidate teardown terminal status was duplicated"
                    )
                reconciliation.terminal_statuses[waited_pid] = returncode
                if waited_pid == proc.pid:
                    if main_returncode is not None:
                        raise RuntimeError("candidate main exit status was duplicated")
                    main_returncode = returncode
                    proc.returncode = returncode
            elif os.WIFSTOPPED(status):
                stopped_signal = os.WSTOPSIG(status) & 0xFF
                event = status >> 16
                if event == _PTRACE_EVENT_EXEC and stopped_signal == signal.SIGTRAP:
                    _fold_nonleader_exec_state(
                        current_tid=waited_pid,
                        former_tid=_ptrace_event_message(waited_pid),
                        traced=traced,
                        awaiting_syscall_exit=awaiting_syscall_exit,
                        provisional_children=provisional_children,
                        file_write_syscalls=file_write_syscalls,
                        fork_syscalls=fork_syscalls,
                        reconciliation=reconciliation,
                    )
                if waited_pid not in traced:
                    available_child_slots = sum(
                        pid not in reconciliation.resolved_transitions
                        and not transition.pending.denied
                        and transition.pending.event_child is None
                        for pid, transition in reconciliation.transitions.items()
                    )
                    if (
                        waited_pid in provisional_children
                        or stopped_signal != signal.SIGSTOP
                        or event != 0
                        or len(provisional_children) >= available_child_slots
                    ):
                        raise RuntimeError(
                            "candidate teardown ptrace stop is unattributed"
                        )
                    provisional_children.add(waited_pid)
                    reconciliation.held_tasks.add(waited_pid)
                    if not _candidate_proc_attested(waited_pid, limits):
                        raise RuntimeError(
                            "provisional teardown task attestation failed"
                        )
                    process_peak = max(
                        process_peak,
                        len(traced | provisional_children),
                    )
                    if process_peak > process_limit:
                        raise RuntimeError("candidate task count exceeded semantic cap")
                elif (
                    event == _PTRACE_EVENT_EXIT
                    and stopped_signal == signal.SIGTRAP
                    and waited_pid not in reconciliation.continued_exit_stops
                ):
                    vfork_owner = _reconciliation_vfork_owner(
                        reconciliation,
                        waited_pid,
                    )
                    released_owner: tuple[int, _CreationTransitionState] | None = None
                    if vfork_owner is not None:
                        owner_parent, owner_transition = vfork_owner
                        _record_vfork_child_release(
                            owner_transition.pending,
                            _VFORK_RELEASE_EXIT,
                        )
                        owner_transition.child_release = (
                            owner_transition.pending.vfork_child_release
                        )
                        released_owner = (owner_parent, owner_transition)
                    reconciliation.held_tasks.discard(waited_pid)
                    reconciliation.continued_exit_stops.add(waited_pid)
                    _continue_captured_exit_stop(waited_pid)
                    if released_owner is not None:
                        owner_parent, owner_transition = released_owner
                        _resume_released_vfork_parent(
                            reconciliation,
                            parent=owner_parent,
                            transition=owner_transition,
                            traced=traced,
                        )
                elif event == _PTRACE_EVENT_EXEC and stopped_signal == signal.SIGTRAP:
                    vfork_owner = _reconciliation_vfork_owner(
                        reconciliation,
                        waited_pid,
                    )
                    if vfork_owner is not None:
                        owner_parent, owner_transition = vfork_owner
                        _record_vfork_child_release(
                            owner_transition.pending,
                            _VFORK_RELEASE_EXEC,
                        )
                        owner_transition.child_release = (
                            owner_transition.pending.vfork_child_release
                        )
                        _resume_released_vfork_parent(
                            reconciliation,
                            parent=owner_parent,
                            transition=owner_transition,
                            traced=traced,
                        )
                    reconciliation.held_tasks.add(waited_pid)
                elif waited_pid in reconciliation.children_needing_initial_stop:
                    if stopped_signal != signal.SIGSTOP or event != 0:
                        raise RuntimeError(
                            "created teardown task initial stop is invalid"
                        )
                    reconciliation.children_needing_initial_stop.remove(waited_pid)
                    initial_stop_owners = [
                        transition.pending
                        for transition in reconciliation.transitions.values()
                        if transition.pending.event_child == waited_pid
                        and transition.pending.child_initial_stop_pending
                    ]
                    if len(initial_stop_owners) != 1:
                        raise RuntimeError(
                            "created teardown task initial stop is unattributed"
                        )
                    initial_owner = initial_stop_owners[0]
                    if initial_owner.expected_event == _PTRACE_EVENT_VFORK:
                        _mark_vfork_child_running(initial_owner)
                    else:
                        initial_owner.child_initial_stop_pending = False
                    reconciliation.held_tasks.add(waited_pid)
                elif event in {
                    _PTRACE_EVENT_FORK,
                    _PTRACE_EVENT_VFORK,
                    _PTRACE_EVENT_CLONE,
                }:
                    transition = reconciliation.transitions.get(waited_pid)
                    pending = transition.pending if transition is not None else None
                    if (
                        transition is None
                        or waited_pid in reconciliation.resolved_transitions
                        or pending is None
                        or not _transition_matches_snapshot(
                            pending, transition.snapshot
                        )
                        or pending.creation_mode is None
                        or pending.denied
                        or pending.expected_event != event
                        or pending.event_child is not None
                    ):
                        raise RuntimeError(
                            "candidate teardown creation event is unattributed"
                        )
                    child = _ptrace_event_message(waited_pid)
                    if child <= 1 or child in traced:
                        raise RuntimeError(
                            "candidate teardown creation child is invalid"
                        )
                    unassigned_slots = sum(
                        pid not in reconciliation.resolved_transitions
                        and not candidate_transition.pending.denied
                        and candidate_transition.pending.event_child is None
                        for pid, candidate_transition in reconciliation.transitions.items()
                    )
                    if (
                        provisional_children
                        and child not in provisional_children
                        and len(provisional_children) >= unassigned_slots
                    ):
                        raise RuntimeError(
                            "candidate teardown provisional child did not reconcile"
                        )
                    was_provisional = child in provisional_children
                    provisional_children.discard(child)
                    pending.event_child = child
                    if event == _PTRACE_EVENT_VFORK:
                        _set_vfork_child_initial_phase(
                            pending,
                            held=was_provisional,
                        )
                    else:
                        pending.child_initial_stop_pending = not was_provisional
                    traced.add(child)
                    reconciliation.attributed_targets.add(child)
                    if not _candidate_proc_attested(child, limits):
                        raise RuntimeError("created teardown task attestation failed")
                    process_peak = max(
                        process_peak,
                        len(traced | provisional_children),
                    )
                    if process_peak > process_limit:
                        raise RuntimeError("candidate task count exceeded semantic cap")
                    if not was_provisional:
                        reconciliation.children_needing_initial_stop.add(child)
                    if event == _PTRACE_EVENT_VFORK:
                        if not pending.saw_entry:
                            raise RuntimeError(
                                "captured vfork transition lacks exact entry evidence"
                            )
                        reconciliation.held_tasks.add(waited_pid)
                    else:
                        reconciliation.held_tasks.discard(waited_pid)
                        _resume_after_ptrace_stop(waited_pid, pending)
                elif stopped_signal == (signal.SIGTRAP | 0x80):
                    pending = awaiting_syscall_exit.get(waited_pid)
                    if (
                        pending is None
                        or waited_pid not in reconciliation.snapshotted_pending
                    ):
                        raise RuntimeError(
                            "candidate teardown syscall stop is unattributed"
                        )
                    transition = reconciliation.transitions.get(waited_pid)
                    if transition is not None and not _transition_matches_snapshot(
                        pending, transition.snapshot
                    ):
                        raise RuntimeError(
                            "candidate teardown creation transition changed"
                        )
                    result = _ptrace_syscall_result(waited_pid)
                    if not pending.saw_entry:
                        if result != -_ENOSYS:
                            raise RuntimeError("ptrace syscall-entry marker is invalid")
                        pending.saw_entry = True
                        if transition is not None:
                            reconciliation.held_tasks.discard(waited_pid)
                            _resume_after_ptrace_stop(waited_pid, pending)
                        else:
                            reconciliation.held_tasks.add(waited_pid)
                    else:
                        awaiting_syscall_exit.pop(waited_pid)
                        if pending.denied:
                            if pending.event_child is not None or result != -_ENOSYS:
                                raise RuntimeError(
                                    "denied clone unexpectedly created a task"
                                )
                            _ptrace_set_syscall_result(waited_pid, -_EPERM)
                            assert transition is not None
                            transition.parent_result_seen = True
                            transition.parent_result = -_EPERM
                        elif pending.number in fork_syscalls:
                            _validate_creation_result(
                                pending,
                                result,
                                traced | provisional_children,
                                traced | provisional_children,
                            )
                            assert transition is not None
                            transition.parent_result_seen = True
                            transition.parent_result = result
                        if pending.number in fork_syscalls and result == -_EAGAIN:
                            candidate_tasks = _candidate_task_ids()
                            tasks_at_denial = len(traced | provisional_children)
                            if candidate_tasks and not candidate_tasks.issubset(
                                traced | provisional_children
                            ):
                                raise RuntimeError(
                                    "process-limit task inventory is ambiguous"
                                )
                            if tasks_at_denial != process_limit:
                                raise RuntimeError(
                                    "thread clone EAGAIN did not occur at the exact process cap"
                                )
                            process_limit_hit = True
                            process_limit_syscall = pending.number
                            process_peak = max(process_peak, tasks_at_denial)
                        if pending.number in file_write_syscalls and result == -_EFBIG:
                            observed_size = _candidate_open_file_size(waited_pid)
                            if observed_size != file_size_limit:
                                raise RuntimeError(
                                    "write EFBIG did not occur at the exact file-size cap"
                                )
                            file_space_limit_source = "guest_monitor_ptrace_write_efbig"
                            file_limit_errno = _EFBIG
                            file_size_observed_bytes = max(
                                file_size_observed_bytes,
                                observed_size,
                            )
                        reconciliation.held_tasks.add(waited_pid)
                elif event == _PTRACE_EVENT_SECCOMP:
                    syscall_number, validated_creation = _seccomp_creation_mode(
                        _ptrace_event_message(waited_pid)
                    )
                    if (
                        syscall_number not in fork_syscalls
                        and syscall_number not in file_write_syscalls
                    ):
                        raise RuntimeError("unexpected traced seccomp syscall")
                    if waited_pid in awaiting_syscall_exit:
                        raise RuntimeError("candidate seccomp stop is unattributed")
                    pending = _PendingSyscall(number=syscall_number)
                    if syscall_number in fork_syscalls:
                        creation = validated_creation
                        if creation is None:
                            creation = _creation_mode(waited_pid, syscall_number)
                        if creation is None:
                            pending.denied = True
                            _ptrace_skip_syscall(waited_pid)
                        else:
                            pending.creation_mode, pending.expected_event = creation
                    awaiting_syscall_exit[waited_pid] = pending
                    _register_quiescence_transition(
                        reconciliation,
                        pid=waited_pid,
                        pending=pending,
                        fork_syscalls=fork_syscalls,
                    )
                    if syscall_number in fork_syscalls:
                        reconciliation.held_tasks.discard(waited_pid)
                        _resume_after_ptrace_stop(waited_pid, pending)
                    else:
                        reconciliation.held_tasks.add(waited_pid)
                else:
                    pending = awaiting_syscall_exit.get(waited_pid)
                    transition = reconciliation.transitions.get(waited_pid)
                    if (
                        transition is not None
                        and waited_pid not in reconciliation.resolved_transitions
                        and not transition.parent_result_seen
                        and transition.parent_terminal is None
                    ):
                        pending = transition.pending
                        if not _transition_matches_snapshot(
                            pending, transition.snapshot
                        ):
                            raise RuntimeError(
                                "candidate teardown creation transition changed"
                            )
                        if (
                            pending.expected_event == _PTRACE_EVENT_VFORK
                            and pending.event_child is not None
                            and transition.child_release is None
                        ):
                            reconciliation.held_tasks.add(waited_pid)
                        else:
                            reconciliation.held_tasks.discard(waited_pid)
                            _resume_after_ptrace_stop(waited_pid, pending)
                    elif pending is not None:
                        if not _pending_write_is_unambiguous(
                            pending,
                            file_write_syscalls,
                            fork_syscalls,
                        ):
                            raise RuntimeError(
                                "candidate teardown pending syscall is ambiguous"
                            )
                        reconciliation.held_tasks.add(waited_pid)
                    else:
                        reconciliation.held_tasks.add(waited_pid)
            else:
                raise RuntimeError("candidate teardown reconciliation wait is invalid")
            completed, finalized_ledger = _finalize_teardown_reconciliation(
                proc,
                reconciliation=reconciliation,
                traced=traced,
                awaiting_syscall_exit=awaiting_syscall_exit,
                provisional_children=provisional_children,
                file_write_syscalls=file_write_syscalls,
                fork_syscalls=fork_syscalls,
                teardown_control=teardown_control,
            )
            if completed:
                teardown_reconciliation = None
                teardown_ledger = finalized_ledger
                if finalized_ledger is not None:
                    observed_tasks = _validate_teardown_inventory(finalized_ledger)
                    if _teardown_ledger_is_complete(
                        finalized_ledger,
                        observed_tasks=observed_tasks,
                        traced=traced,
                        awaiting_syscall_exit=awaiting_syscall_exit,
                        provisional_children=provisional_children,
                    ):
                        teardown_ledger = None
            continue
        if waited_pid == 0:
            teardown_control.wait_for_wake(0.001)
            continue
        if os.WIFEXITED(status) or os.WIFSIGNALED(status):
            if os.WIFEXITED(status):
                returncode = os.WEXITSTATUS(status)
            else:
                returncode = -os.WTERMSIG(status)
            total_cpu_usage_us += round((usage.ru_utime + usage.ru_stime) * 1_000_000)
            if waited_pid not in traced:
                raise RuntimeError("unattributed candidate task exited")
            _mark_vfork_child_terminal(
                awaiting_syscall_exit,
                waited_pid,
            )
            pending = awaiting_syscall_exit.get(waited_pid)
            if pending is not None:
                if returncode != -signal.SIGKILL or not _pending_write_is_unambiguous(
                    pending,
                    file_write_syscalls,
                    fork_syscalls,
                ):
                    raise RuntimeError("candidate exited during a traced syscall")
                awaiting_syscall_exit.pop(waited_pid)
            traced.remove(waited_pid)
            if waited_pid == proc.pid:
                main_returncode = returncode
                proc.returncode = returncode
            # Multiple tasks may disappear from procfs before their already
            # queued terminal ptrace waits are consumed.  Every addition is
            # still forbidden; the immutable ``traced`` set retains vanished
            # tasks until each exact wait is drained.
            if not _candidate_task_ids().issubset(traced | provisional_children):
                raise RuntimeError("candidate task inventory drifted after exit")
            if waited_pid == proc.pid and traced:
                teardown_reconciliation = _begin_teardown_quiescence(
                    proc,
                    reason=_TEARDOWN_MAIN_EXIT_CLEANUP,
                    traced=traced,
                    awaiting_syscall_exit=awaiting_syscall_exit,
                    provisional_children=provisional_children,
                    file_write_syscalls=file_write_syscalls,
                    fork_syscalls=fork_syscalls,
                    current_time=current_time,
                )
            continue
        if not os.WIFSTOPPED(status):
            raise RuntimeError("candidate ptrace wait status is invalid")
        # Darwin retains ptrace event bits here while Linux masks them.
        stopped_signal = os.WSTOPSIG(status) & 0xFF
        event = status >> 16
        if event == _PTRACE_EVENT_EXEC and stopped_signal == signal.SIGTRAP:
            _fold_nonleader_exec_state(
                current_tid=waited_pid,
                former_tid=_ptrace_event_message(waited_pid),
                traced=traced,
                awaiting_syscall_exit=awaiting_syscall_exit,
                provisional_children=provisional_children,
                file_write_syscalls=file_write_syscalls,
                fork_syscalls=fork_syscalls,
            )
        if waited_pid not in traced:
            if (
                waited_pid in provisional_children
                or stopped_signal != signal.SIGSTOP
                or event != 0
                or not any(
                    pending.creation_mode is not None and not pending.denied
                    for pending in awaiting_syscall_exit.values()
                )
            ):
                raise RuntimeError("unattributed candidate ptrace stop")
            provisional_children.add(waited_pid)
            if not _candidate_proc_attested(waited_pid, limits):
                raise RuntimeError("provisional candidate task attestation failed")
            candidate_tasks = _candidate_task_ids()
            if candidate_tasks != traced | provisional_children:
                raise RuntimeError("provisional candidate task inventory is incomplete")
            process_peak = max(process_peak, len(candidate_tasks))
            if process_peak > process_limit:
                raise RuntimeError("candidate task count exceeded semantic cap")
            # Keep the task stopped until its parent's creation event names it.
            continue
        if event in {
            _PTRACE_EVENT_FORK,
            _PTRACE_EVENT_VFORK,
            _PTRACE_EVENT_CLONE,
        }:
            pending = awaiting_syscall_exit.get(waited_pid)
            if (
                pending is None
                or pending.creation_mode is None
                or pending.denied
                or pending.expected_event != event
                or pending.event_child is not None
            ):
                raise RuntimeError("candidate creation event is unattributed")
            child = _ptrace_event_message(waited_pid)
            if child <= 1 or child in traced:
                raise RuntimeError("candidate creation event child is invalid")
            was_provisional = child in provisional_children
            provisional_children.discard(child)
            pending.event_child = child
            if event == _PTRACE_EVENT_VFORK:
                _set_vfork_child_initial_phase(
                    pending,
                    held=was_provisional,
                )
            else:
                pending.child_initial_stop_pending = not was_provisional
            traced.add(child)
            if not _candidate_proc_attested(child, limits):
                raise RuntimeError("created candidate task attestation failed")
            candidate_tasks = _candidate_task_ids()
            if candidate_tasks != traced | provisional_children:
                raise RuntimeError("created candidate task inventory is incomplete")
            process_peak = max(process_peak, len(candidate_tasks))
            if process_peak > process_limit:
                raise RuntimeError("candidate task count exceeded semantic cap")
            if was_provisional:
                _resume_after_ptrace_stop(child, None)
            _resume_after_ptrace_stop(waited_pid, pending)
            continue
        initial_stop_owners = [
            candidate_pending
            for candidate_pending in awaiting_syscall_exit.values()
            if candidate_pending.event_child == waited_pid
            and candidate_pending.child_initial_stop_pending
        ]
        if initial_stop_owners:
            if (
                len(initial_stop_owners) != 1
                or stopped_signal != signal.SIGSTOP
                or event != 0
            ):
                raise RuntimeError("created candidate task initial stop is invalid")
            initial_owner = initial_stop_owners[0]
            if initial_owner.expected_event == _PTRACE_EVENT_VFORK:
                _mark_vfork_child_running(initial_owner)
            else:
                initial_owner.child_initial_stop_pending = False
        if event == _PTRACE_EVENT_SECCOMP:
            syscall_number, validated_creation = _seccomp_creation_mode(
                _ptrace_event_message(waited_pid)
            )
            if (
                syscall_number not in fork_syscalls
                and syscall_number not in file_write_syscalls
            ):
                raise RuntimeError("unexpected traced seccomp syscall")
            if waited_pid not in traced or waited_pid in awaiting_syscall_exit:
                raise RuntimeError("candidate seccomp stop is unattributed")
            pending = _PendingSyscall(number=syscall_number)
            if syscall_number in fork_syscalls:
                creation = validated_creation
                if creation is None:
                    creation = _creation_mode(waited_pid, syscall_number)
                if creation is None:
                    pending.denied = True
                    _ptrace_skip_syscall(waited_pid)
                else:
                    pending.creation_mode, pending.expected_event = creation
            awaiting_syscall_exit[waited_pid] = pending
            _resume_after_ptrace_stop(waited_pid, pending)
            continue
        if stopped_signal == (signal.SIGTRAP | 0x80):
            pending = awaiting_syscall_exit.get(waited_pid)
            if pending is None:
                raise RuntimeError("unexpected ptrace syscall stop")
            result = _ptrace_syscall_result(waited_pid)
            if not pending.saw_entry:
                if result != -_ENOSYS:
                    raise RuntimeError("ptrace syscall-entry marker is invalid")
                pending.saw_entry = True
                _resume_after_ptrace_stop(waited_pid, pending)
                continue
            awaiting_syscall_exit.pop(waited_pid)
            if pending.denied:
                if pending.event_child is not None or result != -_ENOSYS:
                    raise RuntimeError("denied clone unexpectedly created a task")
                _ptrace_set_syscall_result(waited_pid, -_EPERM)
            elif pending.number in fork_syscalls:
                _validate_creation_result(
                    pending,
                    result,
                    traced | provisional_children,
                    _candidate_task_ids(),
                )
            if pending.number in fork_syscalls and result == -_EAGAIN:
                candidate_tasks = _candidate_task_ids()
                if candidate_tasks != traced | provisional_children:
                    raise RuntimeError("process-limit task inventory is ambiguous")
                tasks_at_denial = len(candidate_tasks)
                if tasks_at_denial != process_limit:
                    raise RuntimeError(
                        "thread clone EAGAIN did not occur at the exact process cap"
                    )
                process_limit_hit = True
                process_limit_syscall = pending.number
                process_peak = max(process_peak, tasks_at_denial)
            if pending.number in file_write_syscalls and result == -_EFBIG:
                observed_size = _candidate_open_file_size(waited_pid)
                if observed_size != file_size_limit:
                    raise RuntimeError(
                        "write EFBIG did not occur at the exact file-size cap"
                    )
                file_space_limit_source = "guest_monitor_ptrace_write_efbig"
                file_limit_errno = _EFBIG
                file_size_observed_bytes = max(file_size_observed_bytes, observed_size)
            _resume_after_ptrace_stop(waited_pid, None)
            continue
        if stopped_signal == signal.SIGXCPU:
            signo, code = _ptrace_siginfo(waited_pid)
            if signo == signal.SIGXCPU and code == _SI_KERNEL:
                cpu_limit_hit = True
                teardown_reconciliation = _begin_teardown_quiescence(
                    proc,
                    reason=_TEARDOWN_CPU_LIMIT,
                    traced=traced,
                    awaiting_syscall_exit=awaiting_syscall_exit,
                    provisional_children=provisional_children,
                    file_write_syscalls=file_write_syscalls,
                    fork_syscalls=fork_syscalls,
                    current_time=current_time,
                )
                transition = teardown_reconciliation.transitions.get(waited_pid)
                if (
                    transition is not None
                    and not transition.parent_result_seen
                    and transition.parent_terminal is None
                    and not (
                        transition.pending.expected_event == _PTRACE_EVENT_VFORK
                        and transition.pending.event_child is not None
                        and transition.child_release is None
                    )
                ):
                    teardown_reconciliation.held_tasks.discard(waited_pid)
                    _resume_after_ptrace_stop(waited_pid, transition.pending)
                else:
                    teardown_reconciliation.held_tasks.add(waited_pid)
            else:
                _resume_after_ptrace_stop(
                    waited_pid,
                    awaiting_syscall_exit.get(waited_pid),
                    deliver_signal=signal.SIGXCPU,
                )
                continue
        if stopped_signal == signal.SIGXFSZ:
            signo, code = _ptrace_siginfo(waited_pid)
            if signo == signal.SIGXFSZ and code == _SI_KERNEL:
                file_space_limit_source = "guest_monitor_ptrace_siginfo_fsize"
                file_limit_signal = signal.SIGXFSZ
                file_size_observed_bytes = max(
                    file_size_observed_bytes,
                    _candidate_open_file_size(waited_pid),
                )
                teardown_reconciliation = _begin_teardown_quiescence(
                    proc,
                    reason=_TEARDOWN_FILE_SPACE_LIMIT,
                    traced=traced,
                    awaiting_syscall_exit=awaiting_syscall_exit,
                    provisional_children=provisional_children,
                    file_write_syscalls=file_write_syscalls,
                    fork_syscalls=fork_syscalls,
                    current_time=current_time,
                )
                transition = teardown_reconciliation.transitions.get(waited_pid)
                if (
                    transition is not None
                    and not transition.parent_result_seen
                    and transition.parent_terminal is None
                    and not (
                        transition.pending.expected_event == _PTRACE_EVENT_VFORK
                        and transition.pending.event_child is not None
                        and transition.child_release is None
                    )
                ):
                    teardown_reconciliation.held_tasks.discard(waited_pid)
                    _resume_after_ptrace_stop(waited_pid, transition.pending)
                else:
                    teardown_reconciliation.held_tasks.add(waited_pid)
            else:
                _resume_after_ptrace_stop(
                    waited_pid,
                    awaiting_syscall_exit.get(waited_pid),
                    deliver_signal=signal.SIGXFSZ,
                )
                continue
        if teardown_reconciliation is not None:
            completed, finalized_ledger = _finalize_teardown_reconciliation(
                proc,
                reconciliation=teardown_reconciliation,
                traced=traced,
                awaiting_syscall_exit=awaiting_syscall_exit,
                provisional_children=provisional_children,
                file_write_syscalls=file_write_syscalls,
                fork_syscalls=fork_syscalls,
                teardown_control=teardown_control,
            )
            if completed:
                teardown_reconciliation = None
                teardown_ledger = finalized_ledger
            continue
        if event in {
            _PTRACE_EVENT_EXEC,
            _PTRACE_EVENT_EXIT,
        } or stopped_signal in {signal.SIGSTOP, signal.SIGTRAP}:
            if waited_pid not in traced:
                raise RuntimeError("unattributed candidate ptrace stop")
            if event in {
                _PTRACE_EVENT_EXEC,
                _PTRACE_EVENT_EXIT,
            }:
                _mark_vfork_child_released(
                    awaiting_syscall_exit,
                    waited_pid,
                    (
                        _VFORK_RELEASE_EXEC
                        if event == _PTRACE_EVENT_EXEC
                        else _VFORK_RELEASE_EXIT
                    ),
                )
            if event == _PTRACE_EVENT_EXIT:
                _continue_captured_exit_stop(waited_pid)
            else:
                _resume_after_ptrace_stop(
                    waited_pid,
                    awaiting_syscall_exit.get(waited_pid),
                )
            continue
        _resume_after_ptrace_stop(
            waited_pid,
            awaiting_syscall_exit.get(waited_pid),
            deliver_signal=stopped_signal,
        )
    if main_returncode is None:
        raise RuntimeError("candidate main exit status is absent")
    if (
        awaiting_syscall_exit
        or provisional_children
        or teardown_ledger is not None
        or teardown_reconciliation is not None
        or _candidate_task_ids()
    ):
        raise RuntimeError("candidate task teardown is incomplete")
    kill_record = teardown_control.kill_record()
    tracer_killed_main = kill_record is not None and proc.pid in kill_record.targets
    return (
        main_returncode,
        total_cpu_usage_us,
        process_peak,
        file_space_limit_source is not None,
        file_space_limit_source,
        file_limit_signal,
        file_limit_errno,
        file_size_observed_bytes,
        writable_available_bytes,
        cpu_limit_hit,
        process_limit_syscall,
        process_limit_hit,
        tracer_killed_main,
    )


def _kill_candidate_group_once(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        # A captured target may already have vanished from /proc while its
        # terminal wait remains queued.  The ledger still requires that exact
        # wait before the run can be classified.
        pass


def _stop_candidate_group_once(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGSTOP)
    except ProcessLookupError:
        # A terminal wait can be queued after the last group member vanished.
        # The bounded quiescence drain still has to consume that exact wait.
        pass


def _kill_candidate_tree(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (PermissionError, ProcessLookupError):
        pass
    for entry in os.scandir("/proc"):
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name, 10)
        try:
            with open(f"/proc/{pid}/status", encoding="ascii") as handle:
                uid_line = next(line for line in handle if line.startswith("Uid:"))
            if int(uid_line.split()[1], 10) == _CANDIDATE_UID:
                os.kill(pid, signal.SIGKILL)
        except (OSError, StopIteration, ValueError):
            continue


def _run_candidate(
    source: bytes,
    limits: tuple[int, int, int, int, int],
) -> dict[str, Any]:
    code_read = code_write = -1
    ready_read = ready_write = -1
    gate_read = gate_write = -1
    address_space, cpu_seconds, file_size, process_count, open_files = limits
    proc: subprocess.Popen[bytes] | None = None
    trace_completed = False
    failure_site = "pipe_setup"
    try:
        code_read, code_write = os.pipe()
        ready_read, ready_write = os.pipe()
        gate_read, gate_write = os.pipe()
        failure_site = "candidate_spawn"
        proc = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _CANDIDATE_RUNNER,
                str(code_read),
                str(ready_write),
                str(gate_read),
                str(address_space),
                str(cpu_seconds),
                str(file_size),
                str(process_count),
                str(open_files),
            ],
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                "HOME": "/tmp",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "TMPDIR": "/tmp",
            },
            close_fds=True,
            start_new_session=True,
            pass_fds=(code_read, ready_write, gate_read),
        )
        failure_site = "source_delivery"
        os.close(code_read)
        code_read = -1
        os.close(ready_write)
        ready_write = -1
        os.close(gate_read)
        gate_read = -1
        _write_all(code_write, struct.pack("!Q", len(source)) + source)
        os.close(code_write)
        code_write = -1
        failure_site = "setup_readiness"
        # The readiness pipe is the sole owner of this boundary.  In
        # particular, do not call ``Popen.poll()`` here: this is a
        # PTRACE_TRACEME child, so waitpid(WNOHANG) may consume its SIGSTOP as
        # a trace stop and CPython will mis-record it as ``returncode=-19``.
        # The child writes the one-byte trusted marker before raising SIGSTOP;
        # EOF therefore also detects a genuine pre-ready exit without racing
        # the later attestation's exact wait4 ownership.
        _wait_ready(ready_read)
        failure_site = "initial_attestation"
        _attest_initial_trace_stop(proc, limits)
        os.close(ready_read)
        ready_read = -1
        failure_site = "gate_release"
        _write_all(gate_write, b"G")
        os.close(gate_write)
        gate_write = -1
        failure_site = "ptrace_continue"
        _ptrace(_PTRACE_CONT, proc.pid)
        assert proc.stdout is not None and proc.stderr is not None
        teardown_control = _TeardownControl()
        stdout_state: dict[str, Any] = {"bytes": 0, "exceeded": False}
        stderr_state: dict[str, Any] = {"bytes": 0, "exceeded": False}
        stdout_thread = threading.Thread(
            target=_relay,
            args=(
                proc.stdout,
                1,
                _STDOUT_CAP_BYTES,
                stdout_state,
                teardown_control,
            ),
        )
        stderr_thread = threading.Thread(
            target=_relay,
            args=(
                proc.stderr,
                2,
                _STDERR_CAP_BYTES,
                stderr_state,
                teardown_control,
            ),
        )
        failure_site = "relay_start"
        stdout_thread.start()
        stderr_thread.start()
        failure_site = "trace_measure"
        (
            returncode,
            cpu_usage_us,
            process_peak,
            file_space_limit_hit,
            file_space_limit_source,
            file_limit_signal,
            file_limit_errno,
            file_size_observed_bytes,
            writable_available_bytes,
            cpu_limit_hit,
            process_limit_syscall,
            process_limit_hit,
            tracer_killed_main,
        ) = _trace_and_measure(
            proc,
            limits=limits,
            process_limit=process_count,
            file_size_limit=file_size,
            teardown_control=teardown_control,
        )
        # `_trace_and_measure` has consumed the exact terminal wait for every
        # traced task.  Before this point, no `Popen` wait helper may inspect
        # the PTRACE_TRACEME child; on failure the only safe cleanup is an
        # idempotent signal that leaves wait ownership unambiguous.
        trace_completed = True
        failure_site = "relay_drain"
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            raise RuntimeError("candidate output relay did not drain")
        failure_site = "teardown_evidence"
        _validate_teardown_evidence(
            teardown_control,
            stdout_truncated=stdout_state["exceeded"],
            stderr_truncated=stderr_state["exceeded"],
            cpu_limit_hit=cpu_limit_hit,
            file_space_limit_source=file_space_limit_source,
            file_limit_signal=file_limit_signal,
        )
        return {
            "version": 1,
            "candidate_ready_attested": True,
            "returncode": returncode,
            "cpu_usage_us": cpu_usage_us,
            "cpu_limit_us": cpu_seconds * 1_000_000,
            "cpu_limit_hit": cpu_limit_hit,
            "process_peak": process_peak,
            "process_limit": process_count,
            "process_rlimit_nproc": process_count - 1,
            "process_limit_hit": process_limit_hit,
            "process_limit_syscall": process_limit_syscall,
            "tracer_killed_main": tracer_killed_main,
            "stdout_truncated": stdout_state["exceeded"],
            "stderr_truncated": stderr_state["exceeded"],
            "file_space_limit_hit": file_space_limit_hit,
            "file_space_limit_source": file_space_limit_source,
            "file_size_limit_bytes": file_size,
            "writable_limit_bytes": 0,
            "file_limit_signal": file_limit_signal,
            "file_limit_errno": file_limit_errno,
            "file_size_observed_bytes": file_size_observed_bytes,
            "writable_available_bytes": writable_available_bytes,
        }
    except BaseException as exc:
        if isinstance(exc, _MonitorStageError):
            raise
        raise _MonitorStageError(failure_site, exc) from exc
    finally:
        for fd in (
            code_read,
            code_write,
            ready_read,
            ready_write,
            gate_read,
            gate_write,
        ):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if proc is not None and not trace_completed:
            _kill_candidate_tree(proc)


def _emit_status(nonce: str, status: dict[str, Any]) -> None:
    payload = json.dumps(
        status,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    encoded = base64.b64encode(payload)
    _write_all(2, f"{_STATUS_PREFIX}{nonce}:".encode("ascii") + encoded + b"\n")


def main() -> None:
    nonce = ""
    try:
        source, nonce, limits = _prepare_monitor()
    except BaseException as exc:  # noqa: BLE001 - trusted setup boundary
        os.write(2, f"PALAESTRA_EXECUTOR_SETUP_ERROR:{type(exc).__name__}\n".encode())
        raise SystemExit(_INFRA_EXIT) from None

    _write_all(
        2,
        (
            f"{_READY_PREFIX}{nonce}:monitor_uid={os.getuid()}:"
            f"monitor_gid={os.getgid()}\n"
        ).encode("ascii"),
    )
    if _read_exact(0, 1) != b"G":
        raise SystemExit(_INFRA_EXIT)
    try:
        status = _run_candidate(source, limits)
    except BaseException as exc:  # noqa: BLE001 - trusted monitor boundary
        try:
            _emit_status(nonce, _monitor_failure_status(exc))
        finally:
            # PTRACE_O_TRACEEXIT may leave killed tracees stopped with relay
            # pipe descriptors open.  Bypass non-daemon thread shutdown so
            # PTRACE_O_EXITKILL can terminate that tree before the host wall.
            os._exit(_INFRA_EXIT)
    _emit_status(nonce, status)


if __name__ == "__main__":
    main()
