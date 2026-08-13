#!/usr/bin/env python3
"""Run the hostile pinned-gVisor executor acceptance matrix.

This script is deployment tooling, not part of the seven-file server bundle.
Run it as root inside the hardened transient systemd unit with ``-B``.
``--host-path-probe`` must name a narrow root-owned host file that is absent
from the candidate rootfs; ``--output`` must name a new JSON file in a
root-owned, non-writable-by-others directory.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import platform
import secrets
import signal
import stat
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from codecontests_executor import supervisor as supervisor_module
from codecontests_executor.protocol import (
    FILE_SIZE_CAP_BYTES,
    PID_CAP,
    RUNSC_RELEASE_ARCHIVE_SHA512,
    RUNSC_SHA512,
    STDERR_CAP_BYTES,
    STDOUT_CAP_BYTES,
    make_execute_request,
)
from codecontests_executor.service import (
    SERVER_BUNDLE_FILES,
    _gvisor_processes,
    measured_server_bundle_sha256,
    normalized_rootfs_sha256,
)
from codecontests_executor.supervisor import (
    SandboxExecutorConfig,
    SandboxSupervisor,
    SupervisorConfigurationError,
)

RAW_MEMORY = 4 * 1024**3
LEGACY_CLONE_SYSCALL = 220 if platform.machine() in {"aarch64", "arm64"} else 56
MAX_HOST_PATH_PROBE_BYTES = 1024 * 1024
MAX_HOST_PATH_PROBE_CHARS = 1024
MAX_OUTPUT_PATH_CHARS = 1024


def _descriptor_identity(descriptor: int) -> dict[str, int]:
    metadata = os.fstat(descriptor)
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
    }


@dataclass
class _DescriptorChain:
    path: Path
    components: tuple[str, ...]
    descriptors: tuple[int, ...]
    identities: tuple[dict[str, int], ...]
    leaf_is_directory: bool
    closed: bool = False

    @property
    def leaf_descriptor(self) -> int:
        if self.closed:
            raise RuntimeError("descriptor chain is closed")
        return self.descriptors[-1]

    def revalidate(self) -> None:
        if self.closed or len(self.descriptors) != len(self.identities):
            raise RuntimeError("descriptor chain is unavailable")
        for descriptor, expected in zip(self.descriptors, self.identities, strict=True):
            if _descriptor_identity(descriptor) != expected:
                raise RuntimeError("retained descriptor ancestor changed")
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        reopened = os.dup(self.descriptors[0])
        try:
            if _descriptor_identity(reopened) != self.identities[0]:
                raise RuntimeError("descriptor-chain root binding changed")
            for index, component in enumerate(self.components, start=1):
                component_flags = flags
                if index < len(self.descriptors) - 1 or self.leaf_is_directory:
                    component_flags |= os.O_DIRECTORY
                next_descriptor = os.open(
                    component,
                    component_flags,
                    dir_fd=reopened,
                )
                os.close(reopened)
                reopened = next_descriptor
                if _descriptor_identity(reopened) != self.identities[index]:
                    raise RuntimeError("descriptor ancestor path binding changed")
        finally:
            os.close(reopened)

    def close(self) -> None:
        if self.closed:
            return
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)
        self.closed = True


def _retain_descriptor_chain(
    path: Path,
    *,
    leaf_is_directory: bool,
) -> _DescriptorChain:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("safe descriptor-chain flags are unavailable")
    descriptors: list[int] = []
    identities: list[dict[str, int]] = []
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        root_descriptor = os.open("/", flags | os.O_DIRECTORY)
        descriptors.append(root_descriptor)
        identities.append(_descriptor_identity(root_descriptor))
        for index, component in enumerate(path.parts[1:], start=1):
            component_flags = flags
            if index < len(path.parts) - 1 or leaf_is_directory:
                component_flags |= os.O_DIRECTORY
            descriptor = os.open(
                component,
                component_flags,
                dir_fd=descriptors[-1],
            )
            descriptors.append(descriptor)
            identities.append(_descriptor_identity(descriptor))
        for index, (descriptor, identity) in enumerate(
            zip(descriptors, identities, strict=True)
        ):
            metadata = os.fstat(descriptor)
            is_leaf = index == len(descriptors) - 1
            expected_directory = not is_leaf or leaf_is_directory
            if expected_directory and not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("descriptor-chain ancestor is not a directory")
            if identity["uid"] != 0 or identity["gid"] != 0:
                raise ValueError(
                    "descriptor-chain ancestors must be owned by root:root"
                )
            writable_by_others = identity["mode"] & (stat.S_IWGRP | stat.S_IWOTH)
            sticky_root_directory = (
                expected_directory
                and bool(identity["mode"] & stat.S_ISVTX)
                and identity["uid"] == 0
            )
            if writable_by_others and not sticky_root_directory:
                raise ValueError(
                    "descriptor-chain ancestors may not be group/world writable"
                )
        chain = _DescriptorChain(
            path=path,
            components=tuple(path.parts[1:]),
            descriptors=tuple(descriptors),
            identities=tuple(identities),
            leaf_is_directory=leaf_is_directory,
        )
        chain.revalidate()
        return chain
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retain_host_path_probe(
    value: str | Path,
) -> tuple[dict[str, Any], _DescriptorChain]:
    raw = str(value)
    if not raw or len(raw) > MAX_HOST_PATH_PROBE_CHARS or "\x00" in raw:
        raise ValueError("host-path probe syntax is invalid")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError("host-path probe must be absolute")
    if raw != str(path) or ".." in path.parts:
        raise ValueError("host-path probe must be canonical")
    if path == Path("/") or len(path.parts) < 4:
        raise ValueError("host-path probe is too broad")
    try:
        resolved = Path(os.path.realpath(path, strict=True))
    except OSError as exc:
        raise ValueError("host-path probe does not exist") from exc
    if resolved != path:
        raise ValueError("host-path probe may not traverse symlinks")

    try:
        chain = _retain_descriptor_chain(path, leaf_is_directory=False)
    except OSError as exc:
        raise ValueError("host-path probe cannot be opened safely") from exc
    try:
        return _measure_retained_host_path_probe(chain), chain
    except BaseException:
        chain.close()
        raise


def _measure_retained_host_path_probe(
    chain: _DescriptorChain,
) -> dict[str, Any]:
    if chain.leaf_is_directory:
        raise RuntimeError("retained host-path probe is not a file")
    chain.revalidate()
    descriptor = chain.leaf_descriptor
    metadata = os.fstat(descriptor)
    mode = stat.S_IMODE(metadata.st_mode)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("host-path probe must be a regular file")
    if metadata.st_uid != 0 or metadata.st_gid != 0:
        raise ValueError("host-path probe must be owned by root:root")
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("host-path probe may not be group/world writable")
    if not 1 <= metadata.st_size <= MAX_HOST_PATH_PROBE_BYTES:
        raise ValueError("host-path probe size is outside the safe domain")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    measured_size = 0
    while True:
        chunk = os.read(
            descriptor,
            min(1024 * 1024, MAX_HOST_PATH_PROBE_BYTES + 1 - measured_size),
        )
        if not chunk:
            break
        measured_size += len(chunk)
        if measured_size > MAX_HOST_PATH_PROBE_BYTES:
            raise ValueError("host-path probe size is outside the safe domain")
        digest.update(chunk)
    after = os.fstat(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if measured_size != metadata.st_size or any(
        getattr(metadata, field) != getattr(after, field) for field in stable_fields
    ):
        raise ValueError("host-path probe changed during measurement")
    identity = {
        "path": str(chain.path),
        "size": metadata.st_size,
        "mode": mode,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "nlink": metadata.st_nlink,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "sha256": digest.hexdigest(),
    }
    chain.revalidate()
    return identity


def _assert_retained_host_path_probe(
    expected: dict[str, Any],
    chain: _DescriptorChain,
) -> None:
    if _measure_retained_host_path_probe(chain) != expected:
        raise RuntimeError("retained host-path probe changed")


def _measure_host_path_probe(value: str | Path) -> dict[str, Any]:
    identity, chain = _retain_host_path_probe(value)
    try:
        return identity
    finally:
        chain.close()


def _attest_host_path_absent_from_rootfs(
    identity: dict[str, Any],
    rootfs: str | Path,
) -> dict[str, Any]:
    host_path = Path(identity["path"])
    candidate_path = Path(rootfs) / host_path.relative_to("/")
    if os.path.lexists(candidate_path):
        raise RuntimeError("host-path probe also exists in the candidate rootfs")
    return {**identity, "candidate_rootfs_path_absent": True}


def _retain_output_target(
    value: str | Path,
) -> tuple[Path, dict[str, int], _DescriptorChain]:
    raw = str(value)
    if not raw or len(raw) > MAX_OUTPUT_PATH_CHARS or "\x00" in raw:
        raise ValueError("output path syntax is invalid")
    output = Path(raw)
    if not output.is_absolute():
        raise ValueError("output path must be absolute")
    if raw != str(output) or ".." in output.parts:
        raise ValueError("output path must be canonical")
    if len(output.parts) < 4 or output.suffix != ".json":
        raise ValueError("output path is too broad or has an unsafe suffix")
    parent = output.parent
    try:
        resolved_parent = Path(os.path.realpath(parent, strict=True))
    except OSError as exc:
        raise ValueError("output parent does not exist") from exc
    if resolved_parent != parent:
        raise ValueError("output parent may not traverse symlinks")

    try:
        chain = _retain_descriptor_chain(parent, leaf_is_directory=True)
    except OSError as exc:
        raise ValueError("output parent cannot be opened safely") from exc
    try:
        parent_descriptor = chain.leaf_descriptor
        metadata = os.fstat(parent_descriptor)
        mode = stat.S_IMODE(metadata.st_mode)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("output parent must be a directory")
        if metadata.st_uid != 0 or metadata.st_gid != 0:
            raise ValueError("output parent must be owned by root:root")
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError("output parent may not be group/world writable")
        parent_identity = {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": mode,
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
        }
        try:
            os.stat(
                output.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise ValueError("output target already exists")
        chain.revalidate()
        return output, parent_identity, chain
    except BaseException:
        chain.close()
        raise


def _measure_output_target(
    value: str | Path,
) -> tuple[Path, dict[str, int]]:
    output, parent_identity, chain = _retain_output_target(value)
    try:
        return output, parent_identity
    finally:
        chain.close()


def _publish_output(
    output: Path,
    expected_parent_identity: dict[str, int],
    encoded: bytes,
    *,
    descriptor_chain: _DescriptorChain | None = None,
    post_publish_validator: Callable[[], None] | None = None,
) -> None:
    owns_chain = descriptor_chain is None
    if descriptor_chain is None:
        descriptor_chain = _retain_descriptor_chain(
            output.parent, leaf_is_directory=True
        )
    if descriptor_chain.path != output.parent or not descriptor_chain.leaf_is_directory:
        raise RuntimeError("output descriptor chain targets the wrong parent")
    descriptor_chain.revalidate()
    parent_descriptor = descriptor_chain.leaf_descriptor
    temporary_name = f".{output.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    temporary_created = False
    published_identity: tuple[int, int] | None = None
    output_linked = False
    try:
        metadata = os.fstat(parent_descriptor)
        observed_parent_identity = {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IMODE(metadata.st_mode),
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
        }
        if observed_parent_identity != expected_parent_identity:
            raise RuntimeError("output parent identity changed before publication")
        temporary_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            temporary_flags |= os.O_NOFOLLOW
        descriptor = os.open(
            temporary_name,
            temporary_flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_created = True
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RuntimeError("output artifact write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            descriptor_metadata = os.fstat(descriptor)
            published_identity = (
                descriptor_metadata.st_dev,
                descriptor_metadata.st_ino,
            )
        finally:
            os.close(descriptor)
        os.link(
            temporary_name,
            output.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        output_linked = True
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_created = False
        os.fsync(parent_descriptor)
        descriptor_chain.revalidate()
        if post_publish_validator is not None:
            post_publish_validator()
    except BaseException as exc:
        if output_linked and published_identity is not None:
            try:
                published_metadata = os.stat(
                    output.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    published_metadata.st_dev,
                    published_metadata.st_ino,
                ) == published_identity:
                    os.unlink(output.name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:  # noqa: BLE001 - rejection cleanup
                exc.add_note(
                    "failed to remove rejected output artifact: "
                    f"{type(cleanup_error).__name__}"
                )
        raise
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        if owns_chain:
            descriptor_chain.close()


def _source_inventory(package_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        filename: {
            "size": (package_dir / filename).stat().st_size,
            "sha256": _sha256_file(package_dir / filename),
        }
        for filename in SERVER_BUNDLE_FILES
    }


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    stdout = base64.b64decode(result["stdout_b64"], validate=True)
    stderr = base64.b64decode(result["stderr_b64"], validate=True)
    summarized = {
        key: value
        for key, value in result.items()
        if key not in {"stdout_b64", "stderr_b64"}
    }
    summarized.update(
        {
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "stdout_preview_b64": base64.b64encode(stdout[:4096]).decode("ascii"),
            "stderr_preview_b64": base64.b64encode(stderr[:4096]).decode("ascii"),
        }
    )
    return summarized


def _decoded(result: dict[str, Any], stream: str) -> bytes:
    return base64.b64decode(result[f"{stream}_b64"], validate=True)


def _expect(
    result: dict[str, Any],
    *,
    outcome: str,
    category: str | None,
    resource_event: str | None = None,
) -> None:
    observed = (
        result["outcome"],
        result["category"],
        result["resource_event"],
    )
    expected = (outcome, category, resource_event)
    if observed != expected:
        raise AssertionError(
            f"result mismatch: observed={observed!r} expected={expected!r}"
        )


def _cgroup_snapshot() -> dict[str, Any]:
    unified: str | None = None
    with open("/proc/self/cgroup", encoding="ascii") as handle:
        for line in handle:
            hierarchy, controllers, path = line.rstrip("\n").split(":", 2)
            if hierarchy == "0" and controllers == "":
                unified = path
                break
    if unified is None:
        raise RuntimeError("cgroup v2 path is absent")
    service = Path("/sys/fs/cgroup") / unified.lstrip("/")
    delegated = service.parent

    def read(path: Path) -> str:
        return path.read_text(encoding="ascii").strip()

    return {
        "service_path": str(service),
        "delegated_path": str(delegated),
        "service_members": read(service / "cgroup.procs").split(),
        "delegated_members": read(delegated / "cgroup.procs").split(),
        "delegated_controllers": sorted(read(delegated / "cgroup.controllers").split()),
        "delegated_subtree_control": sorted(
            read(delegated / "cgroup.subtree_control").split()
        ),
        "cpu_max": read(delegated / "cpu.max"),
        "memory_max": read(delegated / "memory.max"),
        "memory_swap_max": read(delegated / "memory.swap.max"),
        "pids_max": read(delegated / "pids.max"),
        "proc_swaps": Path("/proc/swaps").read_text(encoding="ascii"),
    }


def _make_supervisor(package_dir: Path) -> SandboxSupervisor:
    return SandboxSupervisor(
        SandboxExecutorConfig(
            launcher_path=str(package_dir / "sandbox_launcher.py"),
            cgroup_gate_path=str(package_dir / "cgroup_gate.py"),
        )
    )


def _replace_probe_anchor(
    source: str,
    *,
    label: str,
    needle: str,
    replacement: str,
) -> str:
    if source.count(needle) != 1:
        raise RuntimeError(f"forced probe anchor is not exact: {label}")
    return source.replace(needle, replacement, 1)


def _instrument_launcher_source(source_bytes: bytes, probe: str) -> bytes:
    source = source_bytes.decode("utf-8", errors="strict")
    state_anchor = "_TEARDOWN_DRAIN_TIMEOUT_SECONDS = 1.0\n"
    validation_anchor = """) -> None:
    output_limit_hit = stdout_truncated or stderr_truncated
"""

    if probe == "exit_stop_cont_esrch":
        source = _replace_probe_anchor(
            source,
            label="exit-esrch-state",
            needle=state_anchor,
            replacement=(
                state_anchor
                + '_FORCED_PROBE_STATE = {"pending": True, "name": '
                + '"exit_stop_cont_esrch"}\n'
            ),
        )
        source = _replace_probe_anchor(
            source,
            label="exit-esrch-call",
            needle="""def _continue_captured_exit_stop(pid: int) -> None:
    try:
        _ptrace(_PTRACE_CONT, pid)
""",
            replacement="""def _continue_captured_exit_stop(pid: int) -> None:
    try:
        if _FORCED_PROBE_STATE["pending"]:
            _ptrace(_PTRACE_CONT, pid)
            _FORCED_PROBE_STATE["pending"] = False
            raise _PtraceError(_PTRACE_CONT, pid, errno.ESRCH)
        _ptrace(_PTRACE_CONT, pid)
""",
        )
    elif probe == "clone_output_crossing":
        source = _replace_probe_anchor(
            source,
            label="clone-output-state",
            needle=state_anchor,
            replacement=(
                state_anchor
                + '_FORCED_PROBE_STATE = {"pending": True, "name": '
                + '"clone_output_crossing"}\n'
                + "_FORCED_OUTPUT_READY = threading.Event()\n"
                + "_FORCED_CLONE_RELEASE = threading.Event()\n"
            ),
        )
        source = _replace_probe_anchor(
            source,
            label="clone-output-relay",
            needle="""            if state["bytes"] >= cap:
                state["exceeded"] = True
                # This flag is the pre-kill output-limit evidence consumed by
""",
            replacement="""            if state["bytes"] >= cap:
                state["exceeded"] = True
                if _FORCED_PROBE_STATE["pending"]:
                    _FORCED_OUTPUT_READY.set()
                    _FORCED_CLONE_RELEASE.wait(1.0)
                # This flag is the pre-kill output-limit evidence consumed by
""",
        )
        source = _replace_probe_anchor(
            source,
            label="clone-output-seccomp",
            needle="""            awaiting_syscall_exit[waited_pid] = pending
            _resume_after_ptrace_stop(waited_pid, pending)
""",
            replacement="""            awaiting_syscall_exit[waited_pid] = pending
            if (
                _FORCED_PROBE_STATE["pending"]
                and syscall_number in fork_syscalls
            ):
                if not _FORCED_OUTPUT_READY.wait(1.0):
                    raise RuntimeError("forced output overflow was not ready")
                _FORCED_CLONE_RELEASE.set()
                deadline = time.monotonic() + 1.0
                while (
                    not teardown_control.output_limit_requested()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.001)
                if not teardown_control.output_limit_requested():
                    raise RuntimeError("forced output evidence did not cross")
                _FORCED_PROBE_STATE["pending"] = False
            _resume_after_ptrace_stop(waited_pid, pending)
""",
        )
    else:
        raise ValueError(f"unknown forced interleaving probe: {probe}")

    source = _replace_probe_anchor(
        source,
        label=f"{probe}-terminal-proof",
        needle=validation_anchor,
        replacement=(
            """) -> None:
    if _FORCED_PROBE_STATE["pending"]:
        raise RuntimeError("forced interleaving was not observed")
    output_limit_hit = stdout_truncated or stderr_truncated
"""
        ),
    )
    compile(source, f"<forced-launcher-{probe}>", "exec")
    return source.encode("utf-8")


def _pin_negative_probes(package_dir: Path) -> dict[str, str]:
    cases = {
        "wrong_archive_right_binary": (RUNSC_SHA512, "0" * 128),
        "right_archive_wrong_binary": ("0" * 128, RUNSC_RELEASE_ARCHIVE_SHA512),
    }
    results: dict[str, str] = {}
    for name, (binary_digest, archive_digest) in cases.items():
        supervisor = _make_supervisor(package_dir)
        supervisor.validate_host_files()
        try:
            supervisor.freeze_runtime(
                expected_runsc_sha512=binary_digest,
                expected_release_archive_sha512=archive_digest,
            )
        except SupervisorConfigurationError as exc:
            results[name] = f"{type(exc).__name__}:{exc}"
        else:
            raise AssertionError(f"{name} did not fail closed")
    return results


class _Executor(Protocol):
    def execute(self, request_payload: dict[str, Any]) -> dict[str, Any]: ...


class Matrix:
    def __init__(self, supervisor: _Executor):
        self.supervisor = supervisor
        self.cases: dict[str, dict[str, Any]] = {}

    def run(
        self,
        name: str,
        code: str,
        *,
        stdin: str = "",
        seconds: int = 1,
        nanos: int = 0,
        memory: int = RAW_MEMORY,
    ) -> dict[str, Any]:
        request = make_execute_request(
            code=code,
            stdin=stdin,
            raw_limits={
                "time_limit": {"seconds": seconds, "nanos": nanos},
                "memory_limit_bytes": memory,
            },
            identity_digest_value="0" * 64,
            ttl_ns=180_000_000_000,
        )
        result = self.supervisor.execute(request)
        self.cases[name] = _summary(result)
        return result


def _run_missing_cap_kill_regression(
    supervisor: _Executor,
) -> dict[str, Any]:
    original = supervisor_module.TRUSTED_MONITOR_BOOTSTRAP_CAPABILITIES
    without_kill = tuple(
        capability for capability in original if capability != "CAP_KILL"
    )
    if len(without_kill) != len(original) - 1:
        raise AssertionError("trusted monitor capability policy lacks one CAP_KILL")
    try:
        supervisor_module.TRUSTED_MONITOR_BOOTSTRAP_CAPABILITIES = without_kill
        result = Matrix(supervisor).run(
            "missing_cap_kill_output_teardown",
            f"import os; os.write(1,b'x'*{STDOUT_CAP_BYTES})",
        )
    finally:
        supervisor_module.TRUSTED_MONITOR_BOOTSTRAP_CAPABILITIES = original
    if (
        result["outcome"],
        result["category"],
        result["returncode"],
    ) != ("unknown", "LAUNCH_ATTESTATION_MISSING", 125):
        raise AssertionError(
            "removing monitor CAP_KILL did not reproduce rc125/UNKNOWN"
        )
    return _summary(result)


FORK_DENIAL = """\
import os
import time
children = []
for index in range(1):
    try:
        pid = os.fork()
    except OSError as exc:
        print("denied", index, exc.errno, flush=True)
        break
    if pid == 0:
        time.sleep(0.2)
        os._exit(0)
    children.append(pid)
"""

RAW_PROCESS_CLONE_DENIAL = """\
import ctypes
import os
import platform

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
arm = platform.machine() in {"aarch64", "arm64"}
clone_number = 220 if arm else 56
flags = 0x01000000 | 0x00200000 | 17
read_fd, write_fd = os.pipe()
children = []
for index in range(1):
    child_tid = ctypes.c_int()
    arguments = (
        (flags, 0, 0, 0, ctypes.byref(child_tid))
        if arm
        else (flags, 0, 0, ctypes.byref(child_tid), 0)
    )
    ctypes.set_errno(0)
    result = libc.syscall(clone_number, *arguments)
    error = ctypes.get_errno()
    if result == 0:
        os.close(write_fd)
        os.read(read_fd, 1)
        os._exit(0)
    if result < 0:
        print("raw-clone-denied", index, error, flush=True)
        break
    children.append(result)
os.close(read_fd)
os.close(write_fd)
for child in children:
    os.waitpid(child, 0)
"""

CLONE3_PROBE = """\
import ctypes
import os
import signal
libc = ctypes.CDLL(None, use_errno=True)
arguments = (ctypes.c_ulonglong * 11)()
arguments[4] = signal.SIGCHLD
result = libc.syscall(435, ctypes.byref(arguments), ctypes.sizeof(arguments))
if result == 0:
    os._exit(0)
if result > 0:
    os.waitpid(result, 0)
    print("clone3-supported", flush=True)
else:
    print("clone3-error", ctypes.get_errno(), flush=True)
"""

NAMESPACE_FILESYSTEM_ESCAPE_DENIAL = """\
import ctypes
import json
import os
import platform

libc = ctypes.CDLL(None, use_errno=True)
arm = platform.machine() in {"aarch64", "arm64"}
numbers = {
    "unshare": 97 if arm else 272,
    "setns": 268 if arm else 308,
    "mount": 40 if arm else 165,
    "fsopen": 430,
}

def call(name, *arguments):
    ctypes.set_errno(0)
    result = libc.syscall(numbers[name], *arguments)
    return [result, ctypes.get_errno()]

clone_number = 220 if arm else 56
child_tid = ctypes.c_int()
clone_arguments = (
    (0x10000000 | 17, 0, 0, 0, ctypes.byref(child_tid))
    if arm
    else (0x10000000 | 17, 0, 0, ctypes.byref(child_tid), 0)
)
ctypes.set_errno(0)
clone_result = libc.syscall(clone_number, *clone_arguments)
clone_error = ctypes.get_errno()
if clone_result == 0:
    os._exit(99)

status = {}
with open("/proc/self/status", encoding="ascii") as handle:
    for line in handle:
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()
print(json.dumps({
    "capabilities": {
        key: status[key]
        for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
    },
    "clone_newuser": [clone_result, clone_error],
    "fsopen": call("fsopen", ctypes.c_char_p(b"tmpfs"), 0),
    "mount": call("mount", 0, 0, 0, 0, 0),
    "setns": call("setns", -1, 0),
    "unshare": call("unshare", 0x10000000),
}, sort_keys=True), flush=True)
"""

CANDIDATE_CREDENTIAL_CAPABILITY_STATUS = """\
import json
import os

status = {}
with open("/proc/self/status", encoding="ascii") as handle:
    for line in handle:
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()
print(json.dumps({
    "capabilities": {
        key: status[key]
        for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
    },
    "gids": list(os.getresgid()),
    "groups": os.getgroups(),
    "uids": list(os.getresuid()),
}, sort_keys=True), flush=True)
"""

THREAD_LEGAL = """\
import json
import os
import threading

threading.stack_size(1024 * 1024)
release = threading.Event()
started = threading.Event()
thread = threading.Thread(target=lambda: (started.set(), release.wait()))
thread.start()
if not started.wait(1):
    raise RuntimeError("worker did not start")
print(json.dumps({
    "live_proc_tasks": len(os.listdir("/proc/self/task")),
    "live_python_threads": len(threading.enumerate()),
}, sort_keys=True), flush=True)
release.set()
thread.join()
"""

THREAD_LIMIT = """\
import json
import os
import threading

threading.stack_size(1024 * 1024)
release = threading.Event()
started = threading.Event()
thread = threading.Thread(target=lambda: (started.set(), release.wait()))
thread.start()
if not started.wait(1):
    raise RuntimeError("worker did not start")
denial = None
try:
    extra = threading.Thread(target=release.wait)
    extra.start()
except RuntimeError as exc:
    denial = type(exc).__name__
print(json.dumps({
    "denial": denial,
    "live_proc_tasks": len(os.listdir("/proc/self/task")),
    "live_python_threads": len(threading.enumerate()),
}, sort_keys=True), flush=True)
release.set()
thread.join()
"""

THREAD_LIMIT_UNCAUGHT = """\
import threading

threading.stack_size(1024 * 1024)
release = threading.Event()
started = threading.Event()
thread = threading.Thread(target=lambda: (started.set(), release.wait()))
thread.start()
if not started.wait(1):
    raise RuntimeError("worker did not start")
try:
    threading.Thread(target=lambda: None).start()
finally:
    release.set()
    thread.join()
"""

NETWORK_ISOLATION = """\
import json
import socket

def connect_error(address):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(0.25)
    try:
        return client.connect_ex(address)
    finally:
        client.close()

print(json.dumps({
    "external_connect_errno": connect_error(("192.0.2.1", 53)),
    "interface_names": sorted(name for _index, name in socket.if_nameindex()),
    "loopback_executor_connect_errno": connect_error(("127.0.0.1", 8080)),
}, sort_keys=True))
"""

_CREDENTIAL_HOST_PATH_ISOLATION_TEMPLATE = r"""\
import json
import os

host_path_probe = __HOST_PATH_PROBE_JSON__
paths = (
    host_path_probe,
    "/var/lib/codecontests-executor/requests",
    "/run/secrets",
    "/root/.aws/credentials",
    "/root/.config/gcloud",
    "/root/.ssh",
)

def readable(path):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except OSError as exc:
        return {"readable": False, "error": type(exc).__name__}
    try:
        return {"readable": True, "mode": os.fstat(descriptor).st_mode}
    finally:
        os.close(descriptor)

sensitive_fragments = (
    "ANTHROPIC",
    "AWS_",
    "AZURE_",
    "CREDENTIAL",
    "GOOGLE_",
    "HF_TOKEN",
    "OPENAI",
    "PASSWORD",
    "RUNPOD",
    "SECRET",
    "TOKEN",
    "WANDB",
)
mountinfo = open("/proc/self/mountinfo", encoding="ascii").read()
print(json.dumps({
    "environment_keys": sorted(os.environ),
    "host_mount_marker_visible": (
        host_path_probe in mountinfo
        or os.path.dirname(host_path_probe) in mountinfo
    ),
    "host_path_probe": host_path_probe,
    "path_access": {path: readable(path) for path in paths},
    "sensitive_environment_keys": sorted(
        key
        for key in os.environ
        if any(fragment in key.upper() for fragment in sensitive_fragments)
    ),
}, sort_keys=True))
"""


def _credential_host_path_isolation_source(host_path_probe: str) -> str:
    encoded = json.dumps(host_path_probe, ensure_ascii=True)
    return _CREDENTIAL_HOST_PATH_ISOLATION_TEMPLATE.replace(
        "__HOST_PATH_PROBE_JSON__",
        encoded,
    )


NONCE_SCAN = r"""
import json
import os
import re
import sys

pattern = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
frame_matches = []
frame = sys._getframe()
while frame is not None:
    for key, value in list(frame.f_locals.items()):
        if isinstance(value, bytes):
            text = value.decode("ascii", errors="ignore")
        elif isinstance(value, str):
            text = value
        else:
            continue
        if pattern.search(text):
            frame_matches.append(key)
    frame = frame.f_back

def probe(path):
    try:
        with open(path, "rb") as handle:
            value = handle.read(1048576)
    except BaseException as exc:
        return {"readable": False, "error": type(exc).__name__}
    return {
        "readable": True,
        "nonce_name_present": b"PALAESTRA_EXECUTOR_LAUNCH_NONCE" in value,
        "hex64_present": pattern.search(value.decode("ascii", errors="ignore"))
        is not None,
    }

print(json.dumps({
    "candidate_env_nonce": "PALAESTRA_EXECUTOR_LAUNCH_NONCE" in os.environ,
    "frame_hex64_keys": frame_matches,
    "self_environ": probe("/proc/self/environ"),
    "monitor_environ": probe("/proc/1/environ"),
    "monitor_cmdline": probe("/proc/1/cmdline"),
    "monitor_mem": probe("/proc/1/mem"),
}, sort_keys=True))
"""

MARKER_FORGERY = """\
import base64
import json
import os
guess = "0" * 64
fake = base64.b64encode(json.dumps({
    "candidate_ready_attested": True,
    "cpu_limit_hit": True,
    "version": 1,
}, sort_keys=True).encode()).decode()
os.write(2, (
    "PALAESTRA_EXECUTOR_READY:" + guess
    + ":monitor_uid=0:monitor_gid=0\\n"
    + "PALAESTRA_EXECUTOR_STATUS:" + guess + ":" + fake + "\\n"
).encode())
print("forgery-did-not-control-status", flush=True)
"""

CONCURRENT_PROBE = """\
import threading
import time
data = bytearray(32 * 1024 * 1024)
for offset in range(0, len(data), 4096):
    data[offset] = 1
worker = threading.Thread(target=lambda: time.sleep(0.1))
worker.start()
deadline = time.process_time() + 0.1
while time.process_time() < deadline:
    pass
worker.join()
print("concurrent-ok", flush=True)
"""

FORCED_CLONE_OUTPUT_CROSSING = """\
import os
import threading

os.write(1, b"x" * (2 * 1024 * 1024))
worker = threading.Thread(target=lambda: None)
worker.start()
worker.join()
while True:
    pass
"""

SIGKILL_ORIGIN_PROBES = {
    "self_sigkill": ("import os,signal; os.kill(os.getpid(),signal.SIGKILL)"),
    "process_group_sigkill": "import os,signal; os.kill(0,signal.SIGKILL)",
    "pthread_sigkill": (
        "import signal,threading; "
        "signal.pthread_kill(threading.get_ident(),signal.SIGKILL)"
    ),
    "raw_tkill_sigkill": """\
import ctypes
import platform
import signal
import threading

number = 130 if platform.machine() in {"aarch64", "arm64"} else 200
ctypes.CDLL(None).syscall(number, threading.get_native_id(), signal.SIGKILL)
""",
    "pidfd_sigkill": """\
import os
import signal

descriptor = os.pidfd_open(os.getpid())
signal.pidfd_send_signal(descriptor, signal.SIGKILL)
""",
    "thread_issued_sigkill": """\
import os
import signal
import threading

threading.Thread(
    target=lambda: os.kill(os.getpid(), signal.SIGKILL),
).start()
while True:
    pass
""",
}


def _run_sigkill_origin_probes(matrix: Matrix) -> None:
    for name, source in SIGKILL_ORIGIN_PROBES.items():
        result = matrix.run(name, source)
        _expect(
            result,
            outcome="unknown",
            category="AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE",
        )
        if result["returncode"] != -signal.SIGKILL:
            raise AssertionError(f"{name} did not terminate the main candidate")


def _run_forced_interleavings(
    supervisor: SandboxSupervisor,
) -> dict[str, dict[str, Any]]:
    production_launcher = supervisor._launcher_bytes
    production_sha256 = hashlib.sha256(production_launcher).hexdigest()
    cases = (
        (
            "exit_stop_cont_esrch",
            f"import os; os.write(1,b'x'*{STDOUT_CAP_BYTES})",
            "candidate_failure",
            "OUTPUT_LIMIT",
        ),
        (
            "clone_output_crossing",
            FORCED_CLONE_OUTPUT_CROSSING,
            "candidate_failure",
            "OUTPUT_LIMIT",
        ),
    )
    results: dict[str, dict[str, Any]] = {}
    try:
        for name, code, outcome, category in cases:
            instrumented = _instrument_launcher_source(
                production_launcher,
                name,
            )
            supervisor._launcher_bytes = instrumented
            started = time.monotonic()
            result = Matrix(supervisor).run(name, code)
            elapsed = time.monotonic() - started
            _expect(result, outcome=outcome, category=category)
            if name == "clone_output_crossing" and result["guest_process_peak"] < 2:
                raise AssertionError(
                    "clone/output interleaving did not attribute its child"
                )
            results[name] = {
                "production_launcher_sha256": production_sha256,
                "instrumented_launcher_sha256": hashlib.sha256(
                    instrumented
                ).hexdigest(),
                "instrumented_launcher_size": len(instrumented),
                "elapsed_seconds": elapsed,
                "result": _summary(result),
            }
    finally:
        supervisor._launcher_bytes = production_launcher
    if supervisor.launcher_sha256 != production_sha256:
        raise AssertionError("production launcher bytes were not restored")
    return results


def _run_matrix(
    supervisor: SandboxSupervisor,
    *,
    host_path_probe: str,
) -> dict[str, dict[str, Any]]:
    matrix = Matrix(supervisor)

    stdin = "first\x00line\nsecond\r\n終わり"
    normal = matrix.run(
        "normal_exact_stdin",
        (
            "import base64,hashlib,sys\n"
            "data=sys.stdin.buffer.read()\n"
            "print(len(data),hashlib.sha256(data).hexdigest(),"
            "base64.b64encode(data).decode())"
        ),
        stdin=stdin,
    )
    _expect(normal, outcome="executed", category=None)
    expected_stdin = stdin.encode()
    expected_line = (
        f"{len(expected_stdin)} {hashlib.sha256(expected_stdin).hexdigest()} "
        f"{base64.b64encode(expected_stdin).decode()}\n"
    ).encode()
    if _decoded(normal, "stdout") != expected_line:
        raise AssertionError("candidate stdin was not byte exact")

    candidate_status = matrix.run(
        "candidate_credential_capability_status",
        CANDIDATE_CREDENTIAL_CAPABILITY_STATUS,
    )
    _expect(candidate_status, outcome="executed", category=None)
    candidate_status_report = json.loads(_decoded(candidate_status, "stdout"))
    if candidate_status_report != {
        "capabilities": {
            "CapAmb": "0000000000000000",
            "CapBnd": "0000000000000000",
            "CapEff": "0000000000000000",
            "CapInh": "0000000000000000",
            "CapPrm": "0000000000000000",
        },
        "gids": [65534, 65534, 65534],
        "groups": [],
        "uids": [65534, 65534, 65534],
    }:
        raise AssertionError(
            "candidate inherited trusted-monitor identity or capabilities"
        )

    python_parity = matrix.run(
        "contest_python_parity",
        """\
import builtins
import json
import sys
import mpmath
import sympy

site_exit = False
try:
    exit()
except SystemExit:
    site_exit = True
x = sympy.Symbol("x")
print(json.dumps({
    "factor": str(sympy.factor(x**2 - 1)),
    "mpmath": mpmath.__version__,
    "safe_path": "" not in sys.path and "/tmp" not in sys.path,
    "site_exit": site_exit and callable(builtins.quit),
    "sympy": sympy.__version__,
}, sort_keys=True))
""",
        # This case attests package/runtime availability, not the wall policy.
        # Cold SymPy import under gVisor can exceed one second on arm64; the
        # dedicated wall_limit case below independently exercises the exact
        # signed one-second boundary.
        seconds=5,
    )
    _expect(python_parity, outcome="executed", category=None)
    if json.loads(_decoded(python_parity, "stdout")) != {
        "factor": "(x - 1)*(x + 1)",
        "mpmath": "1.3.0",
        "safe_path": True,
        "site_exit": True,
        "sympy": "1.14.0",
    }:
        raise AssertionError("candidate Python/package/site parity drifted")

    syntax = matrix.run("syntax_error", "def broken(:\n    pass")
    _expect(syntax, outcome="candidate_failure", category="RUNTIME_ERROR")
    runtime = matrix.run("runtime_error", "raise RuntimeError('hostile')")
    _expect(runtime, outcome="candidate_failure", category="RUNTIME_ERROR")
    wall = matrix.run(
        "wall_limit",
        "import time; time.sleep(10)",
        seconds=0,
        nanos=100_000_000,
    )
    _expect(wall, outcome="candidate_failure", category="WALL_LIMIT")
    if not 50_000_000 <= wall["execution_ns"] <= 2_000_000_000:
        raise AssertionError("fractional wall limit was not enforced at 1x")

    for seconds in (1, 2):
        cpu = matrix.run(
            f"busy_loop_wall_precedes_cpu_backstop_{seconds}",
            "while True:\n    pass",
            seconds=seconds,
        )
        _expect(
            cpu,
            outcome="candidate_failure",
            category="WALL_LIMIT",
        )
        if cpu["resource_event"] is not None:
            raise AssertionError(
                "valid busy loop crossed a resource backstop before signed wall"
            )

    memory = matrix.run(
        "memory_containment",
        """\
import mmap
try:
    value = mmap.mmap(-1, 5 * 1024**3)
except (MemoryError, OSError, ValueError) as exc:
    print(type(exc).__name__)
else:
    value.close()
    raise RuntimeError("5 GiB mapping escaped RLIMIT_AS")
""",
    )
    _expect(memory, outcome="executed", category=None)
    if (
        memory["host_memory_events_after"]["oom"]
        != memory["host_memory_events_before"]["oom"]
    ):
        raise AssertionError("semantic AS probe reached outer cgroup OOM")

    # This is the production supervisor -> runsc -> trusted monitor ->
    # UID/GID-65534 candidate CAP_KILL regression.  Without monitor CAP_KILL,
    # the exact process-group kill raises PermissionError and the request is
    # classified as an infrastructure UNKNOWN.
    stdout_limit = matrix.run(
        "stdout_limit", f"import os; os.write(1,b'x'*{STDOUT_CAP_BYTES})"
    )
    _expect(stdout_limit, outcome="candidate_failure", category="OUTPUT_LIMIT")
    if (
        not stdout_limit["stdout_truncated"]
        or stdout_limit["stdout_bytes"] != STDOUT_CAP_BYTES
    ):
        raise AssertionError("stdout cap is not exact")

    stderr_limit = matrix.run(
        "stderr_limit", f"import os; os.write(2,b'x'*{STDERR_CAP_BYTES})"
    )
    _expect(stderr_limit, outcome="candidate_failure", category="OUTPUT_LIMIT")
    if (
        not stderr_limit["stderr_truncated"]
        or stderr_limit["stderr_bytes"] != STDERR_CAP_BYTES
    ):
        raise AssertionError("stderr cap is not exact")

    read_only_filesystem = matrix.run(
        "read_only_filesystem",
        """\
import json
import os

evidence = {}
for path in ("/tmp/candidate-write", "/candidate-write", "/dev/candidate-write"):
    try:
        with open(path, "wb") as handle:
            handle.write(b"unsafe")
    except OSError as exc:
        evidence[path] = exc.errno
    else:
        evidence[path] = 0
print(json.dumps(evidence, sort_keys=True))
""",
    )
    _expect(read_only_filesystem, outcome="executed", category=None)
    write_report = json.loads(_decoded(read_only_filesystem, "stdout"))
    if set(write_report) != {
        "/candidate-write",
        "/dev/candidate-write",
        "/tmp/candidate-write",
    } or any(error == 0 for error in write_report.values()):
        raise AssertionError("candidate obtained a writable filesystem path")
    if (
        read_only_filesystem["guest_file_size_limit_bytes"] != FILE_SIZE_CAP_BYTES
        or read_only_filesystem["guest_writable_limit_bytes"] != 0
        or read_only_filesystem["guest_writable_available_bytes"] != 0
    ):
        raise AssertionError("signed read-only/filesize policy evidence drifted")

    self_sigxfsz = matrix.run(
        "self_sigxfsz",
        (
            "import os,signal\n"
            "signal.signal(signal.SIGXFSZ, signal.SIG_DFL)\n"
            "os.kill(os.getpid(),signal.SIGXFSZ)\n"
        ),
    )
    _expect(
        self_sigxfsz,
        outcome="candidate_failure",
        category="RUNTIME_ERROR",
    )
    if (
        self_sigxfsz["returncode"] != -25
        or self_sigxfsz["resource_event"] is not None
        or self_sigxfsz["resource_evidence_source"] is not None
        or self_sigxfsz["guest_file_limit_signal"] is not None
        or self_sigxfsz["guest_file_limit_errno"] is not None
    ):
        raise AssertionError("self SIGXFSZ forged file-limit evidence")

    thread_legal = matrix.run("one_worker_thread_legal", THREAD_LEGAL, seconds=2)
    _expect(thread_legal, outcome="executed", category=None)
    thread_legal_report = json.loads(_decoded(thread_legal, "stdout"))
    if (
        thread_legal["guest_process_peak"],
        thread_legal["guest_process_limit"],
        thread_legal["guest_rlimit_nproc"],
        thread_legal["guest_process_limit_syscall"],
        thread_legal_report["live_proc_tasks"],
        thread_legal_report["live_python_threads"],
    ) != (PID_CAP, PID_CAP, 1, None, 2, 2):
        raise AssertionError("one-worker pthread evidence drifted")

    thread_limited = matrix.run(
        "second_worker_denial_caught",
        THREAD_LIMIT,
        seconds=2,
    )
    _expect(
        thread_limited,
        outcome="executed",
        category=None,
        resource_event="GUEST_PROCESS_LIMIT",
    )
    thread_limit_report = json.loads(_decoded(thread_limited, "stdout"))
    if (
        thread_limited["guest_process_peak"],
        thread_limited["guest_process_limit"],
        thread_limited["guest_rlimit_nproc"],
        thread_limited["guest_process_limit_syscall"],
        thread_limit_report["denial"],
        thread_limit_report["live_proc_tasks"],
        thread_limit_report["live_python_threads"],
    ) != (
        PID_CAP,
        PID_CAP,
        1,
        LEGACY_CLONE_SYSCALL,
        "RuntimeError",
        2,
        2,
    ):
        raise AssertionError("caught second-worker denial evidence is not exact")

    thread_uncaught = matrix.run(
        "second_worker_denial_uncaught",
        THREAD_LIMIT_UNCAUGHT,
        seconds=2,
    )
    _expect(
        thread_uncaught,
        outcome="candidate_failure",
        category="PROCESS_LIMIT",
        resource_event="GUEST_PROCESS_LIMIT",
    )
    if (
        thread_uncaught["guest_process_peak"],
        thread_uncaught["guest_process_limit"],
        thread_uncaught["guest_rlimit_nproc"],
        thread_uncaught["guest_process_limit_syscall"],
    ) != (PID_CAP, PID_CAP, 1, LEGACY_CLONE_SYSCALL):
        raise AssertionError("uncaught second-worker denial evidence is not exact")

    fork_denied = matrix.run("fork_denied_eperm", FORK_DENIAL)
    _expect(fork_denied, outcome="executed", category=None)
    if _decoded(fork_denied, "stdout") != b"denied 0 1\n":
        raise AssertionError("fork was not denied directly with EPERM")

    raw_clone_denied = matrix.run(
        "process_clone_denied_eperm", RAW_PROCESS_CLONE_DENIAL
    )
    _expect(raw_clone_denied, outcome="executed", category=None)
    if _decoded(raw_clone_denied, "stdout") != b"raw-clone-denied 0 1\n":
        raise AssertionError("process-clone layout was not denied with EPERM")

    vfork = matrix.run(
        "vfork_denied_eperm",
        """\
import ctypes
libc = ctypes.CDLL(None, use_errno=True)
ctypes.set_errno(0)
result = libc.vfork()
if result == 0:
    libc._exit(99)
print("vfork", result, ctypes.get_errno(), flush=True)
""",
    )
    _expect(vfork, outcome="executed", category=None)
    if _decoded(vfork, "stdout") != b"vfork -1 1\n":
        raise AssertionError("vfork was not denied directly with EPERM")
    for denied in (fork_denied, raw_clone_denied, vfork):
        if (
            denied["guest_process_peak"] != 1
            or denied["guest_process_limit_syscall"] is not None
            or denied["resource_event"] is not None
            or denied["resource_evidence_source"] is not None
        ):
            raise AssertionError("direct process denial forged task-limit evidence")

    clone3 = matrix.run("clone3_support_probe", CLONE3_PROBE)
    _expect(clone3, outcome="executed", category=None)
    clone3_output = _decoded(clone3, "stdout")
    if clone3_output != b"clone3-error 38\n":
        raise AssertionError("clone3 did not fail with ENOSYS before ptrace")

    escape_denial = matrix.run(
        "namespace_filesystem_escape_denial",
        NAMESPACE_FILESYSTEM_ESCAPE_DENIAL,
    )
    _expect(escape_denial, outcome="executed", category=None)
    escape_report = json.loads(_decoded(escape_denial, "stdout"))
    if set(escape_report["capabilities"].values()) != {"0000000000000000"} or any(
        result != [-1, 1]
        for name, result in escape_report.items()
        if name != "capabilities"
    ):
        raise AssertionError(
            "namespace/filesystem escape syscall was not denied with empty caps"
        )

    self_sigxcpu = matrix.run(
        "self_sigxcpu",
        "import os,signal; os.kill(os.getpid(),signal.SIGXCPU)",
    )
    _expect(
        self_sigxcpu,
        outcome="candidate_failure",
        category="RUNTIME_ERROR",
    )
    if self_sigxcpu["returncode"] != -24:
        raise AssertionError("self SIGXCPU was not delivered to the candidate")

    _run_sigkill_origin_probes(matrix)

    caught_eagain = matrix.run(
        "caught_fake_eagain",
        (
            "import errno\n"
            "try: raise OSError(errno.EAGAIN,'candidate')\n"
            "except OSError as exc: print(exc.errno)\n"
        ),
    )
    _expect(caught_eagain, outcome="executed", category=None)
    uncaught_eagain = matrix.run(
        "uncaught_fake_eagain",
        "import errno; raise OSError(errno.EAGAIN,'candidate')",
    )
    _expect(
        uncaught_eagain,
        outcome="candidate_failure",
        category="RUNTIME_ERROR",
    )
    for result in (caught_eagain, uncaught_eagain):
        if (
            result["resource_event"] is not None
            or result["resource_evidence_source"] is not None
            or result["guest_process_limit_syscall"] is not None
        ):
            raise AssertionError("candidate EAGAIN forged process-limit evidence")

    network = matrix.run("network_isolation", NETWORK_ISOLATION)
    _expect(network, outcome="executed", category=None)
    network_report = json.loads(_decoded(network, "stdout"))
    if (
        network_report["interface_names"] != ["lo"]
        or network_report["external_connect_errno"] == 0
        or network_report["loopback_executor_connect_errno"] == 0
    ):
        raise AssertionError("candidate network namespace is not isolated")

    host_isolation = matrix.run(
        "credential_host_path_isolation",
        _credential_host_path_isolation_source(host_path_probe),
    )
    _expect(host_isolation, outcome="executed", category=None)
    host_isolation_report = json.loads(_decoded(host_isolation, "stdout"))
    expected_environment = [
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "TMPDIR",
    ]
    if (
        host_isolation_report["environment_keys"] != expected_environment
        or host_isolation_report["sensitive_environment_keys"]
        or host_isolation_report["host_mount_marker_visible"]
        or host_isolation_report["host_path_probe"] != host_path_probe
        or any(
            evidence["readable"]
            for evidence in host_isolation_report["path_access"].values()
        )
    ):
        raise AssertionError("candidate can observe credentials or host paths")

    nonce = matrix.run("nonce_scan", NONCE_SCAN)
    _expect(nonce, outcome="executed", category=None)
    nonce_report = json.loads(_decoded(nonce, "stdout"))
    if nonce_report["candidate_env_nonce"] or nonce_report["frame_hex64_keys"]:
        raise AssertionError("candidate recovered nonce through env/frame state")
    if nonce_report["self_environ"].get("nonce_name_present"):
        raise AssertionError("candidate nonce leaked through its proc environment")
    for label in ("monitor_environ", "monitor_mem"):
        if nonce_report[label]["readable"]:
            raise AssertionError(f"candidate could read {label}")
    if nonce_report["monitor_cmdline"].get("hex64_present"):
        raise AssertionError("nonce leaked through monitor cmdline")

    forgery = matrix.run("marker_forgery", MARKER_FORGERY)
    _expect(forgery, outcome="executed", category=None)
    if _decoded(forgery, "stdout") != b"forgery-did-not-control-status\n":
        raise AssertionError("candidate marker changed terminal status")

    return matrix.cases


def _run_concurrency(
    supervisor: SandboxSupervisor,
) -> dict[str, dict[str, Any]]:
    def execute(index: int) -> tuple[str, dict[str, Any]]:
        request = make_execute_request(
            code=CONCURRENT_PROBE,
            stdin=f"concurrent-{index}",
            raw_limits={
                "time_limit": {"seconds": 2, "nanos": 0},
                "memory_limit_bytes": RAW_MEMORY,
            },
            identity_digest_value="0" * 64,
            ttl_ns=180_000_000_000,
        )
        return str(index), supervisor.execute(request)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        completed = dict(pool.map(execute, range(4)))
    for result in completed.values():
        _expect(result, outcome="executed", category=None)
    return {name: _summary(result) for name, result in completed.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--host-path-probe", required=True)
    arguments = parser.parse_args()
    try:
        host_path_probe_before, host_path_descriptor_chain = _retain_host_path_probe(
            arguments.host_path_probe
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        (
            output,
            output_parent_identity,
            output_descriptor_chain,
        ) = _retain_output_target(arguments.output)
    except ValueError as exc:
        host_path_descriptor_chain.close()
        parser.error(str(exc))

    package_dir = REPOSITORY_ROOT / "codecontests_executor"
    if _gvisor_processes():
        raise RuntimeError("gVisor process exists before hostile probe")
    request_root = Path("/var/lib/codecontests-executor/requests")
    if any(request_root.iterdir()):
        raise RuntimeError("request root is not empty before hostile probe")

    source_before = _source_inventory(package_dir)
    bundle_before = measured_server_bundle_sha256()
    cgroup_before = _cgroup_snapshot()
    if cgroup_before["delegated_members"]:
        raise RuntimeError("delegated root is not process-free")
    if cgroup_before["cpu_max"] != "400000 100000":
        raise RuntimeError("service CPUQuota is not exactly 400%")
    if cgroup_before["memory_max"] != "25769803776":
        raise RuntimeError("service memory cap drifted")
    if cgroup_before["memory_swap_max"] != "0":
        raise RuntimeError("service swap cap drifted")
    if cgroup_before["pids_max"] != "768":
        raise RuntimeError("service task cap drifted")
    if len(cgroup_before["proc_swaps"].splitlines()) != 1:
        raise RuntimeError("guest swap is enabled")

    pin_negative_results = _pin_negative_probes(package_dir)
    supervisor = _make_supervisor(package_dir)
    supervisor.validate_host_files()
    supervisor.freeze_runtime(
        expected_runsc_sha512=RUNSC_SHA512,
        expected_release_archive_sha512=RUNSC_RELEASE_ARCHIVE_SHA512,
    )
    rootfs_before = normalized_rootfs_sha256(supervisor.frozen_rootfs_path)
    host_path_probe = _attest_host_path_absent_from_rootfs(
        host_path_probe_before,
        supervisor.frozen_rootfs_path,
    )

    started = time.time()
    cap_kill_negative_regression = _run_missing_cap_kill_regression(supervisor)
    forced_interleavings = _run_forced_interleavings(supervisor)
    cases = _run_matrix(
        supervisor,
        host_path_probe=host_path_probe["path"],
    )
    concurrency = _run_concurrency(supervisor)
    elapsed = time.time() - started

    rootfs_after = normalized_rootfs_sha256(supervisor.frozen_rootfs_path)
    source_after = _source_inventory(package_dir)
    bundle_after = measured_server_bundle_sha256()
    host_path_probe_after = _measure_retained_host_path_probe(
        host_path_descriptor_chain
    )
    cgroup_after = _cgroup_snapshot()
    requests_cgroup = Path(cgroup_after["delegated_path"]) / "requests"
    request_cgroup_children = sorted(
        path.name for path in requests_cgroup.iterdir() if path.is_dir()
    )
    leftovers = _gvisor_processes()
    request_entries = sorted(path.name for path in request_root.iterdir())
    if rootfs_after != rootfs_before:
        raise AssertionError("rootfs manifest changed during hostile probe")
    if source_after != source_before or bundle_after != bundle_before:
        raise AssertionError("executor source changed during hostile probe")
    if host_path_probe_after != host_path_probe_before:
        raise AssertionError("host-path probe identity changed during replay")
    if leftovers or request_entries or request_cgroup_children:
        raise AssertionError(
            "teardown residue:"
            f" gvisor={leftovers} requests={request_entries}"
            f" cgroups={request_cgroup_children}"
        )
    if cgroup_after["delegated_members"]:
        raise AssertionError("delegated root gained direct members")

    artifact = {
        "format": "palaestra.codecontests.executor-live-hostile.v4",
        "elapsed_seconds": elapsed,
        "host_path_probe": host_path_probe,
        "runtime_pins": {
            "archive_sha512": RUNSC_RELEASE_ARCHIVE_SHA512,
            "binary_sha512": RUNSC_SHA512,
            "negative_probes": pin_negative_results,
        },
        "cap_kill_negative_regression": cap_kill_negative_regression,
        "forced_interleavings": forced_interleavings,
        "rootfs_manifest_digest_before": rootfs_before,
        "rootfs_manifest_digest_after": rootfs_after,
        "server_bundle_sha256_before": bundle_before,
        "server_bundle_sha256_after": bundle_after,
        "source_inventory_before": source_before,
        "source_inventory_after": source_after,
        "cgroup_before": cgroup_before,
        "cgroup_after": cgroup_after,
        "cases": cases,
        "concurrency_4": concurrency,
        "teardown": {
            "gvisor_pids": leftovers,
            "request_entries": request_entries,
            "request_cgroup_children": request_cgroup_children,
        },
    }
    encoded = (
        json.dumps(
            artifact,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )

    def validate_terminal_bindings() -> None:
        output_descriptor_chain.revalidate()
        _assert_retained_host_path_probe(
            host_path_probe_before,
            host_path_descriptor_chain,
        )

    _publish_output(
        output,
        output_parent_identity,
        encoded,
        descriptor_chain=output_descriptor_chain,
        post_publish_validator=validate_terminal_bindings,
    )
    host_path_descriptor_chain.close()
    output_descriptor_chain.close()
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "case_count": len(cases),
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
