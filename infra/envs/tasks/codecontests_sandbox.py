"""Bootstrap an untrusted CodeContests Python solution.

This file runs *inside* the Linux bubblewrap namespace assembled by
``codecontests.py``.  It deliberately knows nothing about test cases or
expected outputs: its only argument is the candidate source file.  Resource
limits and the seccomp filter are installed before replacing this process
with a fresh isolated Python interpreter for the candidate.

On macOS the same bootstrap supplies process and file-size limits so the real
stdin/stdout verifier remains locally testable.  The production security
boundary is Linux-only and is enforced by the caller.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import resource
import sys


MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
OUTPUT_LIMIT_BYTES = 2 * 1024 * 1024
OPEN_FILE_LIMIT = 64


class _ScmpArgCmp(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_uint),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


def _set_limit(which: int, value: int) -> None:
    resource.setrlimit(which, (value, value))


def _install_resource_limits() -> None:
    """Install irreversible limits before candidate code can run."""
    try:
        _set_limit(resource.RLIMIT_AS, MEMORY_LIMIT_BYTES)
    except (ValueError, resource.error):
        # Darwin refuses to lower RLIMIT_AS below the interpreter's existing
        # mapped address space. It is only the local test path; production
        # Linux must install every limit or fail before candidate execution.
        if sys.platform == "linux":
            raise
    _set_limit(resource.RLIMIT_FSIZE, OUTPUT_LIMIT_BYTES)
    # A crash must not ask the kernel/core_pattern helper to persist candidate
    # memory, and repeated read-only opens must not consume the host fd table.
    _set_limit(resource.RLIMIT_CORE, 0)
    _set_limit(resource.RLIMIT_NOFILE, OPEN_FILE_LIMIT)
    # The Linux sandbox gives every invocation a private user namespace, so
    # this count is not shared by 32 concurrent candidates using uid 65534.
    # bubblewrap's PID-namespace init + candidate main + one worker supports
    # the standard recursion-stack idiom; seccomp restricts clone to
    # same-process threads. On macOS this remains a best-effort local guard.
    try:
        _set_limit(resource.RLIMIT_NPROC, 3)
    except (ValueError, resource.error):
        if sys.platform == "linux":
            raise


def _load_libseccomp() -> ctypes.CDLL:
    # Use the SONAME directly. ctypes.util.find_library may shell out to
    # ldconfig, which is exactly the sort of child-process dependency this
    # bootstrap must avoid.
    try:
        lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    except OSError as exc:
        raise RuntimeError("libseccomp.so.2 is unavailable") from exc

    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_ScmpArgCmp),
    ]
    lib.seccomp_rule_add_array.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    return lib


def _install_linux_seccomp() -> None:
    """Deny side-effect syscall families while permitting normal Python.

    Mount and network namespaces are the first boundary.  Seccomp is the
    independent enforcement layer: if it cannot be installed, bootstrap
    exits before candidate code is executed.
    """
    if os.geteuid() == 0:
        raise RuntimeError("candidate privilege drop failed (euid is 0)")

    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    pr_set_no_new_privs = 38
    if libc.prctl(pr_set_no_new_privs, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise RuntimeError(f"PR_SET_NO_NEW_PRIVS failed: errno {err}")

    lib = _load_libseccomp()
    scmp_act_allow = 0x7FFF0000
    scmp_act_errno = 0x00050000 | errno.EPERM
    scmp_act_enosys = 0x00050000 | errno.ENOSYS
    scmp_cmp_eq = 4
    scmp_cmp_ge = 5
    scmp_cmp_masked_eq = 7
    ctx = lib.seccomp_init(scmp_act_allow)
    if not ctx:
        raise RuntimeError("seccomp_init failed")

    def resolve(name: str, *, required: bool = False) -> int | None:
        number = lib.seccomp_syscall_resolve_name(name.encode("ascii"))
        if number < 0:
            if required:
                raise RuntimeError(f"libseccomp cannot resolve required syscall {name}")
            return None
        return number

    def deny(
        name: str, *, required: bool = False, action: int = scmp_act_errno
    ) -> None:
        number = resolve(name, required=required)
        if number is None:
            return
        rc = lib.seccomp_rule_add_array(ctx, action, number, 0, None)
        if rc != 0:
            raise RuntimeError(f"seccomp rule failed for {name}: rc {rc}")

    def deny_cmp(
        name: str,
        comparisons: list[_ScmpArgCmp],
        *,
        required: bool = False,
        action: int = scmp_act_errno,
    ) -> None:
        number = resolve(name, required=required)
        if number is None:
            return
        array_type = _ScmpArgCmp * len(comparisons)
        array = array_type(*comparisons)
        rc = lib.seccomp_rule_add_array(
            ctx, action, number, len(comparisons), array
        )
        if rc != 0:
            raise RuntimeError(f"seccomp argument rule failed for {name}: rc {rc}")

    try:
        # clone3's flags live behind a pointer seccomp cannot inspect. ENOSYS
        # is deliberate: glibc pthread_create falls back to inspectable clone
        # only for ENOSYS (not EPERM). clone is allowed solely when the kernel
        # dependency chain CLONE_THREAD -> CLONE_SIGHAND -> CLONE_VM is fully
        # present, so the one RLIMIT_NPROC worker shares this process rather
        # than creating a child process.
        deny("clone3", required=True, action=scmp_act_enosys)
        clone_vm = 0x00000100
        clone_sighand = 0x00000800
        clone_thread = 0x00010000
        for required_flag in (clone_vm, clone_sighand, clone_thread):
            deny_cmp(
                "clone",
                [_ScmpArgCmp(0, scmp_cmp_masked_eq, required_flag, 0)],
                required=True,
            )

        # No child processes, cross-process inspection, or signals.
        for name in (
            "fork", "vfork",
            "kill", "tkill", "tgkill", "pidfd_send_signal",
            "rt_sigqueueinfo", "rt_tgsigqueueinfo",
            "ptrace", "process_vm_readv", "process_vm_writev",
            "unshare", "setns",
        ):
            # fork/vfork are absent on architectures such as aarch64; when
            # libc implements os.fork via clone, the clone flag rules above
            # reject it. Resolve the legacy syscalls where present without
            # making their absence a sandbox setup failure.
            deny(name, required=name in {"kill", "ptrace"})

        # No network endpoint can be created.  The private network namespace
        # is also disconnected, so this remains blocked if a future runtime
        # happens to inherit a descriptor unexpectedly.
        for name in (
            "socket", "socketpair", "connect", "bind", "listen",
            "accept", "accept4", "sendto", "sendmmsg", "sendmsg",
            "recvfrom", "recvmmsg", "recvmsg", "shutdown", "socketcall",
        ):
            deny(name, required=name in {"socket", "connect"})

        # Opening for read is needed by Python imports.  Any write/create/
        # truncate mode is denied at the syscall layer; the namespace exposes
        # only read-only binds as an independent backstop.
        write_flags = {
            os.O_WRONLY,
            os.O_RDWR,
            os.O_CREAT,
            os.O_TRUNC,
            os.O_APPEND,
            getattr(os, "O_TMPFILE", 0),
        }
        for name, flags_arg in (("open", 1), ("openat", 2)):
            for flag in sorted(write_flags - {0}):
                deny_cmp(
                    name,
                    [_ScmpArgCmp(flags_arg, scmp_cmp_masked_eq, flag, flag)],
                    required=name == "openat",
                )

        for name in (
            "openat2", "creat", "truncate", "ftruncate", "fallocate",
            "unlink", "unlinkat", "rename", "renameat", "renameat2",
            "mkdir", "mkdirat", "rmdir", "mknod", "mknodat",
            "link", "linkat", "symlink", "symlinkat",
            "chmod", "fchmod", "fchmodat", "chown", "fchown",
            "fchownat", "lchown", "utime", "utimes", "utimensat",
            "setxattr", "lsetxattr", "fsetxattr", "removexattr",
            "lremovexattr", "fremovexattr", "mount", "umount2",
            "pivot_root", "move_mount", "open_tree", "fsopen", "fsmount",
            "fspick", "quotactl", "swapon", "swapoff",
            "pwrite64", "pwritev", "pwritev2", "sendfile", "splice",
            "tee", "vmsplice", "copy_file_range", "memfd_create",
            "io_uring_setup", "io_uring_enter", "io_uring_register",
            "bpf", "perf_event_open", "userfaultfd", "keyctl",
            "add_key", "request_key",
        ):
            deny(name)

        # Candidate output is intentionally limited to the stdio descriptor
        # set (stdin itself is the read end of a pipe). A write to fd >= 3 is
        # rejected even if a future caller accidentally leaks one through
        # exec.
        for name in ("write", "writev"):
            deny_cmp(
                name,
                [_ScmpArgCmp(0, scmp_cmp_ge, 3, 0)],
                required=True,
            )

        # stdout/stderr are append-only capture descriptors. Prevent the
        # candidate from clearing O_APPEND or changing/duplicating inherited
        # descriptor roles after bootstrap wrote its readiness marker.
        for command in (
            fcntl.F_DUPFD,
            fcntl.F_SETFD,
            fcntl.F_SETFL,
            getattr(fcntl, "F_DUPFD_CLOEXEC", -1),
        ):
            if command >= 0:
                deny_cmp(
                    "fcntl",
                    [_ScmpArgCmp(1, scmp_cmp_eq, command, 0)],
                    required=True,
                )

        rc = lib.seccomp_load(ctx)
        if rc != 0:
            raise RuntimeError(f"seccomp_load failed: rc {rc}")
    finally:
        lib.seccomp_release(ctx)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: codecontests_sandbox.py SOLUTION READY_TOKEN")
    solution_path = sys.argv[1]
    ready_token = sys.argv[2]

    _install_resource_limits()
    if sys.platform == "linux":
        _install_linux_seccomp()

    # This unguessable marker lets the trusted parent distinguish a sandbox/
    # bootstrap failure from a candidate that deliberately exits nonzero and
    # writes a fake-looking infrastructure diagnostic. The token is removed
    # from argv by the exec below and /proc is absent in the Linux namespace.
    os.write(2, f"CODECONTESTS_SANDBOX_READY:{ready_token}\n".encode("ascii"))

    # A fresh interpreter removes every bootstrap frame and namespace before
    # untrusted code runs.  The seccomp filter and rlimits survive exec.
    executable = sys.executable
    os.execv(
        executable,
        # Keep the standard `site` module: it defines the commonly generated
        # `exit()`/`quit()` conveniences. -s disables the user site and -P
        # excludes the script directory; the caller supplies a fixed
        # PYTHONPATH=/packages in an otherwise cleared environment. Unlike -I,
        # this deliberately admits that one curated read-only search root.
        [executable, "-s", "-P", solution_path],
    )


if __name__ == "__main__":
    main()
