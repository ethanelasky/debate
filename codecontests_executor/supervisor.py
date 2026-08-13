"""Host-side supervisor for one fresh rootful OCI ``runsc`` sandbox."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import shutil
import signal
import stat
import struct
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from .protocol import (
    FILE_SIZE_CAP_BYTES,
    PID_CAP,
    PINNED_GVISOR_RLIMIT_NPROC,
    READ_ONLY_TMPFS_SIZE_BYTES,
    ROOTFS_PATH,
    RUNSC_PATH,
    RUNSC_RELEASE_ARCHIVE_PATH,
    RUNSC_RELEASE_ARCHIVE_SIZE_BYTES,
    RUNSC_SIZE_BYTES,
    RUNSC_VERSION_OUTPUT,
    STDERR_CAP_BYTES,
    STDOUT_CAP_BYTES,
    TRUSTED_MONITOR_BOOTSTRAP_CAPABILITIES,
    WRITABLE_OVERLAY_CAP_BYTES,
)

NOBODY_UID = 65534
NOBODY_GID = 65534
_CONTAINER_LAUNCHER_PATH = "/opt/palaestra/codecontests_sandbox_launcher.py"
_INFRA_EXIT = 125


class SupervisorConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxExecutorConfig:
    runsc_path: str = RUNSC_PATH
    runsc_release_archive_path: str = RUNSC_RELEASE_ARCHIVE_PATH
    rootfs_path: str = ROOTFS_PATH
    launcher_path: str = str(Path(__file__).with_name("sandbox_launcher.py"))
    cgroup_gate_path: str = str(Path(__file__).with_name("cgroup_gate.py"))
    request_root: str = "/var/lib/codecontests-executor/requests"
    python_path: str = "/usr/bin/python3"
    host_python_path: str = "/usr/bin/python3.12"
    termination_grace_seconds: float = 0.25
    drop_runsc_to_nobody: bool = False
    enforce_request_cgroup: bool = True
    cgroup_root: str | None = None


@dataclass
class _Capture:
    data: bytearray
    exceeded: bool = False
    error: str | None = None


@dataclass(frozen=True)
class _RequestCgroup:
    outer: str
    controller: str
    candidate: str
    oci_path: str
    memory_events_before: dict[str, int]
    pids_events_before: dict[str, int]
    outer_controllers: frozenset[str]


@dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    parent_pid: int
    process_group: int
    session: int
    starttime_ticks: int


class _ResourceEvidence(TypedDict):
    host_cpu_usage_us: int
    host_cpu_before_usage_us: int
    host_cpu_ready_usage_us: int
    host_cpu_cross_usage_us: int
    host_cpu_after_usage_us: int
    host_cpu_budget_us: int
    host_memory_peak_bytes: int
    host_pids_peak: int
    host_memory_events_before: dict[str, int]
    host_memory_events_after: dict[str, int]
    host_pids_events_before: dict[str, int]
    host_pids_events_after: dict[str, int]
    guest_cpu_usage_us: int
    guest_process_peak: int
    guest_process_limit: int
    guest_rlimit_nproc: int
    guest_process_limit_syscall: int | None
    guest_file_size_limit_bytes: int
    guest_writable_limit_bytes: int
    guest_file_limit_signal: int | None
    guest_file_limit_errno: int | None
    guest_file_size_observed_bytes: int
    guest_writable_available_bytes: int
    resource_evidence_source: str | None


_MEMORY_EVENT_KEYS = (
    "low",
    "high",
    "max",
    "oom",
    "oom_kill",
    "oom_group_kill",
)
_PIDS_EVENT_KEYS = ("max",)
_TRUSTED_STATUS_MAX_BYTES = 4096
_RUNSC_LOG_MAX_BYTES = 1024 * 1024


class SandboxSupervisor:
    """Launch and tear down a single isolated candidate execution."""

    def __init__(self, config: SandboxExecutorConfig | None = None):
        self.config = config or SandboxExecutorConfig()
        self._launcher_bytes = self._read_nofollow(
            self.config.launcher_path, max_bytes=256 * 1024
        )
        self._cgroup_gate_bytes = self._read_nofollow(
            self.config.cgroup_gate_path, max_bytes=64 * 1024
        )
        self._runsc_fd: int | None = None
        self._rootfs_fd: int | None = None
        self._rootfs_metadata: os.stat_result | None = None
        self._rootfs_mount_identity: tuple[str, ...] | None = None
        self._rootfs_tree_identity: tuple[tuple[str, tuple[Any, ...]], ...] | None = (
            None
        )
        self._delegated_cgroup: str | None = None

    @staticmethod
    def _read_nofollow(path: str, *, max_bytes: int) -> bytes:
        flags = os.O_RDONLY | os.O_CLOEXEC
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise SupervisorConfigurationError("O_NOFOLLOW is unavailable")
        try:
            fd = os.open(path, flags | nofollow)
        except OSError as exc:
            raise SupervisorConfigurationError(
                "cannot open trusted file without following links"
            ) from exc
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise SupervisorConfigurationError("trusted path is not a file")
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = os.read(fd, min(64 * 1024, max_bytes + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > max_bytes:
                    raise SupervisorConfigurationError(
                        "trusted file exceeds size limit"
                    )
            after = os.fstat(fd)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_uid",
                "st_gid",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if (
                any(
                    getattr(before, field) != getattr(after, field)
                    for field in stable_fields
                )
                or size != before.st_size
            ):
                raise SupervisorConfigurationError(
                    "trusted file changed while being read"
                )
            return b"".join(chunks)
        finally:
            os.close(fd)

    @property
    def launcher_sha256(self) -> str:
        return hashlib.sha256(self._launcher_bytes).hexdigest()

    @property
    def cgroup_gate_sha256(self) -> str:
        return hashlib.sha256(self._cgroup_gate_bytes).hexdigest()

    @property
    def frozen_rootfs_path(self) -> str:
        if self._rootfs_fd is None:
            raise SupervisorConfigurationError("runtime is not frozen")
        return f"/proc/self/fd/{self._rootfs_fd}"

    @property
    def frozen_rootfs_host_path(self) -> str:
        if self._rootfs_fd is None:
            raise SupervisorConfigurationError("runtime is not frozen")
        return f"/proc/{os.getpid()}/fd/{self._rootfs_fd}"

    @property
    def frozen_runsc_path(self) -> str:
        if self._runsc_fd is None:
            raise SupervisorConfigurationError("runtime is not frozen")
        return f"/proc/self/fd/{self._runsc_fd}"

    @property
    def runtime_pass_fds(self) -> tuple[int, int]:
        if self._runsc_fd is None or self._rootfs_fd is None:
            raise SupervisorConfigurationError("runtime is not frozen")
        return self._runsc_fd, self._rootfs_fd

    def freeze_runtime(
        self,
        *,
        expected_runsc_sha512: str,
        expected_release_archive_sha512: str,
        expected_runsc_size_bytes: int = RUNSC_SIZE_BYTES,
        expected_release_archive_size_bytes: int = (RUNSC_RELEASE_ARCHIVE_SIZE_BYTES),
    ) -> None:
        if self._runsc_fd is not None or self._rootfs_fd is not None:
            raise SupervisorConfigurationError("runtime is already frozen")
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise SupervisorConfigurationError("O_NOFOLLOW is unavailable")
        runsc_fd = os.open(
            self.config.runsc_path, os.O_RDONLY | os.O_CLOEXEC | nofollow
        )
        archive_fd: int | None = None
        rootfs_fd: int | None = None
        try:
            archive_fd = os.open(
                self.config.runsc_release_archive_path,
                os.O_RDONLY | os.O_CLOEXEC | nofollow,
            )
            archive_before = os.fstat(archive_fd)
            if (
                not stat.S_ISREG(archive_before.st_mode)
                or archive_before.st_uid != 0
                or archive_before.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                or archive_before.st_size != expected_release_archive_size_bytes
            ):
                raise SupervisorConfigurationError("unsafe runsc release archive")
            archive_digest = hashlib.sha512()
            for chunk in iter(lambda: os.read(archive_fd, 1024 * 1024), b""):
                archive_digest.update(chunk)
            archive_after = os.fstat(archive_fd)
            if archive_digest.hexdigest() != expected_release_archive_sha512 or any(
                getattr(archive_before, field) != getattr(archive_after, field)
                for field in (
                    "st_dev",
                    "st_ino",
                    "st_mode",
                    "st_uid",
                    "st_gid",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            ):
                raise SupervisorConfigurationError(
                    "runsc release archive SHA-512/metadata mismatch"
                )
            before = os.fstat(runsc_fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != 0
                or before.st_size != expected_runsc_size_bytes
            ):
                raise SupervisorConfigurationError("unsafe runsc inode")
            digest = hashlib.sha512()
            while True:
                chunk = os.read(runsc_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            if digest.hexdigest() != expected_runsc_sha512:
                raise SupervisorConfigurationError("runsc SHA-512 mismatch")
            after = os.fstat(runsc_fd)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_uid",
                "st_gid",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if any(
                getattr(before, field) != getattr(after, field)
                for field in stable_fields
            ):
                raise SupervisorConfigurationError("runsc changed while being measured")
            os.lseek(runsc_fd, 0, os.SEEK_SET)
            self._attest_runsc_version(runsc_fd)
            rootfs_fd = os.open(
                self.config.rootfs_path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | nofollow,
            )
            root_metadata = os.fstat(rootfs_fd)
            if not stat.S_ISDIR(root_metadata.st_mode) or root_metadata.st_uid != 0:
                raise SupervisorConfigurationError("unsafe rootfs inode")
            self._runsc_fd = runsc_fd
            self._rootfs_fd = rootfs_fd
            self._rootfs_metadata = root_metadata
            self._rootfs_mount_identity = self._measure_rootfs_mount()
            first_tree = self._measure_rootfs_tree(rootfs_fd)
            second_tree = self._measure_rootfs_tree(rootfs_fd)
            if first_tree != second_tree:
                raise SupervisorConfigurationError(
                    "rootfs topology changed during runtime freeze"
                )
            self._rootfs_tree_identity = first_tree
        except BaseException:
            self._runsc_fd = None
            self._rootfs_fd = None
            self._rootfs_metadata = None
            self._rootfs_mount_identity = None
            self._rootfs_tree_identity = None
            os.close(runsc_fd)
            if archive_fd is not None:
                os.close(archive_fd)
            if rootfs_fd is not None:
                os.close(rootfs_fd)
            raise
        else:
            assert archive_fd is not None
            os.close(archive_fd)

    @staticmethod
    def _attest_runsc_version(runsc_fd: int) -> None:
        try:
            completed = subprocess.run(
                [f"/proc/self/fd/{runsc_fd}", "--version"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                env={
                    "HOME": "/",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                },
                pass_fds=(runsc_fd,),
                close_fds=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SupervisorConfigurationError(
                "runsc version attestation failed"
            ) from exc
        if (
            completed.returncode != 0
            or completed.stdout != RUNSC_VERSION_OUTPUT.encode("ascii")
            or completed.stderr
        ):
            raise SupervisorConfigurationError("runsc version output mismatch")

    def _measure_rootfs_mount(self) -> tuple[str, ...]:
        canonical = os.path.abspath(self.config.rootfs_path)
        if os.path.realpath(canonical) != canonical:
            raise SupervisorConfigurationError("rootfs path is not canonical")
        for ancestor in (Path(canonical), *Path(canonical).parents):
            metadata = ancestor.stat()
            if metadata.st_uid != 0 or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise SupervisorConfigurationError(
                    "rootfs ancestor ownership/mode is unsafe"
                )
            if str(ancestor) == "/":
                break
        matching: list[tuple[str, ...]] = []
        with open("/proc/self/mountinfo", encoding="ascii") as handle:
            for line in handle:
                before, after = line.rstrip("\n").split(" - ", 1)
                fields = before.split()
                mount_point = fields[4].replace("\\040", " ")
                if mount_point != canonical:
                    continue
                after_fields = after.split()
                matching.append(
                    (
                        fields[0],
                        fields[2],
                        fields[3],
                        fields[4],
                        fields[5],
                        after_fields[0],
                        after_fields[1],
                        after_fields[2],
                    )
                )
        if len(matching) != 1:
            raise SupervisorConfigurationError(
                "rootfs must be one exact dedicated bind mount"
            )
        identity = matching[0]
        options = set(identity[4].split(",")) | set(identity[7].split(","))
        if "ro" not in options:
            raise SupervisorConfigurationError("rootfs bind mount is not readonly")
        return identity

    @staticmethod
    def _measure_rootfs_tree(
        root_fd: int,
    ) -> tuple[tuple[str, tuple[Any, ...]], ...]:
        """Measure full topology/metadata below the held read-only root."""

        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise SupervisorConfigurationError("O_NOFOLLOW is unavailable")
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_nlink",
            "st_size",
            "st_rdev",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        records: list[tuple[str, tuple[Any, ...]]] = []

        def stable(metadata: os.stat_result) -> tuple[Any, ...]:
            return tuple(getattr(metadata, field) for field in stable_fields)

        def walk(directory_fd: int, relative_parent: str) -> None:
            directory_before = os.fstat(directory_fd)
            if not stat.S_ISDIR(directory_before.st_mode):
                raise SupervisorConfigurationError(
                    "rootfs directory changed type during topology scan"
                )
            with os.scandir(directory_fd) as iterator:
                names = sorted(entry.name for entry in iterator)
            for name in names:
                relative = f"{relative_parent}/{name}" if relative_parent else name
                before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                target: str | None = None
                kind = stat.S_IFMT(before.st_mode)
                if stat.S_ISLNK(before.st_mode):
                    target = os.readlink(name, dir_fd=directory_fd)
                records.append((relative, (kind, *stable(before), target)))
                if stat.S_ISDIR(before.st_mode):
                    child_fd = os.open(
                        name,
                        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | nofollow,
                        dir_fd=directory_fd,
                    )
                    try:
                        if stable(os.fstat(child_fd)) != stable(before):
                            raise SupervisorConfigurationError(
                                "rootfs directory changed before descent"
                            )
                        walk(child_fd, relative)
                    finally:
                        os.close(child_fd)
                after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stable(after) != stable(before):
                    raise SupervisorConfigurationError(
                        "rootfs entry changed during topology scan"
                    )
                if (
                    target is not None
                    and os.readlink(name, dir_fd=directory_fd) != target
                ):
                    raise SupervisorConfigurationError(
                        "rootfs symlink changed during topology scan"
                    )
            if stable(os.fstat(directory_fd)) != stable(directory_before):
                raise SupervisorConfigurationError(
                    "rootfs directory changed during topology scan"
                )

        retained_root_before = os.fstat(root_fd)
        scan_root_fd = os.open(
            ".",
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | nofollow,
            dir_fd=root_fd,
        )
        try:
            root_before = os.fstat(scan_root_fd)
            if stable(root_before) != stable(retained_root_before):
                raise SupervisorConfigurationError(
                    "rootfs retained/scan descriptors differ"
                )
            records.append((".", (stat.S_IFDIR, *stable(root_before), None)))
            walk(scan_root_fd, "")
            if stable(os.fstat(scan_root_fd)) != stable(root_before):
                raise SupervisorConfigurationError(
                    "rootfs scan root changed during topology scan"
                )
        finally:
            os.close(scan_root_fd)
        if stable(os.fstat(root_fd)) != stable(retained_root_before):
            raise SupervisorConfigurationError(
                "rootfs retained root changed during topology scan"
            )
        records.sort(key=lambda record: record[0])
        return tuple(records)

    def _attest_rootfs_unchanged(self) -> None:
        if (
            self._rootfs_fd is None
            or self._rootfs_metadata is None
            or self._rootfs_mount_identity is None
            or self._rootfs_tree_identity is None
        ):
            raise SupervisorConfigurationError("rootfs attestation is not initialized")
        current_fd = os.fstat(self._rootfs_fd)
        current_path = os.stat(self.config.rootfs_path, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_ctime_ns")
        if any(
            getattr(current_fd, field) != getattr(self._rootfs_metadata, field)
            or getattr(current_path, field) != getattr(self._rootfs_metadata, field)
            for field in fields
        ):
            raise SupervisorConfigurationError("rootfs inode metadata drifted")
        if self._measure_rootfs_mount() != self._rootfs_mount_identity:
            raise SupervisorConfigurationError("rootfs mount identity/options drifted")
        if self._measure_rootfs_tree(self._rootfs_fd) != self._rootfs_tree_identity:
            raise SupervisorConfigurationError(
                "rootfs content/topology metadata drifted"
            )

    @staticmethod
    def _write_cgroup_control(path: str, value: str) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            encoded = value.encode("ascii")
            if os.write(fd, encoded) != len(encoded):
                raise SupervisorConfigurationError("short cgroup control write")
        finally:
            os.close(fd)

    @staticmethod
    def _read_cgroup_control(path: str) -> str:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            return os.read(fd, 64 * 1024).decode("ascii", errors="strict").strip()
        finally:
            os.close(fd)

    def prepare_cgroup_delegation(self) -> None:
        if not self.config.enforce_request_cgroup:
            return
        if self._delegated_cgroup is not None:
            return
        if self.config.cgroup_root is not None:
            delegated = os.path.abspath(self.config.cgroup_root)
        else:
            unified_path: str | None = None
            with open("/proc/self/cgroup", encoding="ascii") as handle:
                for line in handle:
                    hierarchy, controllers, path = line.rstrip("\n").split(":", 2)
                    if hierarchy == "0" and controllers == "":
                        unified_path = path
                        break
            if unified_path is None:
                raise SupervisorConfigurationError("cgroup v2 is unavailable")
            current = os.path.join("/sys/fs/cgroup", unified_path.lstrip("/"))
            if os.path.basename(current) != "service":
                raise SupervisorConfigurationError(
                    "systemd DelegateSubgroup=service is not active"
                )
            delegated = os.path.dirname(current)
        if os.path.commonpath(("/sys/fs/cgroup", delegated)) != "/sys/fs/cgroup":
            raise SupervisorConfigurationError("delegated cgroup escapes cgroupfs")
        required = {"cpu", "memory", "pids"}
        if self._read_cgroup_control(os.path.join(delegated, "cgroup.procs")):
            raise SupervisorConfigurationError(
                "delegated root cgroup must remain process-free"
            )
        controllers = set(
            self._read_cgroup_control(
                os.path.join(delegated, "cgroup.controllers")
            ).split()
        )
        if not required <= controllers:
            raise SupervisorConfigurationError(
                "delegated cgroup lacks cpu/memory/pids controllers"
            )
        subtree_path = os.path.join(delegated, "cgroup.subtree_control")
        enabled = set(self._read_cgroup_control(subtree_path).split())
        if not required <= enabled:
            self._write_cgroup_control(
                subtree_path,
                " ".join(f"+{name}" for name in sorted(required - enabled)),
            )
        if set(self._read_cgroup_control(subtree_path).split()) != required:
            raise SupervisorConfigurationError(
                "delegated subtree controls are not exactly cpu/memory/pids"
            )
        requests = os.path.join(delegated, "requests")
        try:
            os.mkdir(requests, mode=0o700)
        except FileExistsError:
            if not os.path.isdir(requests):
                raise SupervisorConfigurationError(
                    "delegated requests cgroup is not a directory"
                )
        request_subtree = os.path.join(requests, "cgroup.subtree_control")
        self._write_cgroup_control(request_subtree, "+cpu +memory +pids")
        if self._read_cgroup_control(os.path.join(requests, "cgroup.procs")):
            raise SupervisorConfigurationError(
                "requests cgroup must remain process-free"
            )
        if set(self._read_cgroup_control(request_subtree).split()) != required:
            raise SupervisorConfigurationError(
                "requests subtree controls are not exactly cpu/memory/pids"
            )
        self._delegated_cgroup = requests

    def validate_host_files(self) -> None:
        for label, path_string in (
            ("runsc", self.config.runsc_path),
            ("runsc release archive", self.config.runsc_release_archive_path),
            ("rootfs", self.config.rootfs_path),
            ("launcher", self.config.launcher_path),
            ("cgroup gate", self.config.cgroup_gate_path),
            ("host Python", self.config.host_python_path),
        ):
            path = Path(path_string)
            if os.path.realpath(path) != os.path.abspath(path):
                raise SupervisorConfigurationError(f"{label} path uses a symlink")
            if not path.exists():
                raise SupervisorConfigurationError(f"{label} path does not exist")
            mode = path.stat().st_mode
            if path.stat().st_uid != 0:
                raise SupervisorConfigurationError(f"{label} must be owned by root")
            if mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise SupervisorConfigurationError(
                    f"{label} path is group/world writable"
                )
        request_root = Path(self.config.request_root)
        if os.path.realpath(request_root) != os.path.abspath(request_root):
            raise SupervisorConfigurationError("request root uses a symlink")
        request_metadata = request_root.stat()
        if (
            not request_root.is_dir()
            or request_metadata.st_uid != 0
            or request_metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise SupervisorConfigurationError("request root ownership/mode is unsafe")
        launcher = Path(self.config.launcher_path)
        if not launcher.is_file():
            raise SupervisorConfigurationError("launcher path is not a file")
        if not Path(self.config.runsc_path).is_file():
            raise SupervisorConfigurationError("runsc path is not a file")
        if not Path(self.config.rootfs_path).is_dir():
            raise SupervisorConfigurationError("rootfs path is not a directory")
        if self.config.drop_runsc_to_nobody:
            raise SupervisorConfigurationError(
                "rootful OCI runsc must not use the retired host-nobody mode"
            )
        if os.geteuid() != 0:
            raise SupervisorConfigurationError(
                "executor service must start as root to drop runsc to nobody"
            )
        self.prepare_cgroup_delegation()

    def _oci_config(
        self,
        *,
        limits: dict[str, Any],
        nonce: str,
        cgroups_path: str,
    ) -> dict[str, Any]:
        effective = limits["effective"]
        launcher_source = self._launcher_bytes.decode("utf-8", errors="strict")
        rootfs_path = self.config.rootfs_path
        # The trusted monitor needs only enough namespace-local privilege to
        # create and terminate the separately attested UID/GID-65534
        # candidate.  CAP_KILL is required after that credential transition.
        # The fresh candidate drops every bounding capability before its UID
        # transition; no candidate byte runs in this bootstrap process.
        bootstrap_capabilities = list(TRUSTED_MONITOR_BOOTSTRAP_CAPABILITIES)
        capabilities = {
            "bounding": bootstrap_capabilities,
            "effective": bootstrap_capabilities,
            "inheritable": [],
            "permitted": bootstrap_capabilities,
            "ambient": [],
        }
        return {
            "ociVersion": "1.1.0",
            "process": {
                "terminal": False,
                "user": {
                    "uid": 0,
                    "gid": 0,
                    "additionalGids": [],
                },
                "args": [
                    self.config.python_path,
                    "-I",
                    "-B",
                    "-c",
                    launcher_source,
                    str(effective["address_space_bytes"]),
                    str(effective["cpu_seconds"]),
                    str(effective["file_size_bytes"]),
                    str(effective["processes"]),
                    str(effective["open_files"]),
                ],
                "env": [
                    "HOME=/tmp",
                    "LANG=C.UTF-8",
                    "LC_ALL=C.UTF-8",
                    "PATH=/usr/bin:/bin",
                    "PYTHONDONTWRITEBYTECODE=1",
                    "PYTHONHASHSEED=0",
                    "TMPDIR=/tmp",
                    f"PALAESTRA_EXECUTOR_LAUNCH_NONCE={nonce}",
                ],
                "cwd": "/tmp",
                "capabilities": capabilities,
                # Candidate-only semantic rlimits are installed in the fresh
                # child immediately before its privilege/capability drop.
                "rlimits": [{"type": "RLIMIT_CORE", "hard": 0, "soft": 0}],
                "noNewPrivileges": True,
            },
            "root": {"path": rootfs_path, "readonly": True},
            "hostname": "codecontests",
            "mounts": [
                {
                    "destination": "/proc",
                    "type": "proc",
                    "source": "proc",
                    "options": ["nosuid", "noexec", "nodev"],
                },
                {
                    "destination": "/tmp",
                    "type": "tmpfs",
                    "source": "tmpfs",
                    "options": [
                        "nosuid",
                        "nodev",
                        "noexec",
                        "ro",
                        "mode=0555",
                        f"size={READ_ONLY_TMPFS_SIZE_BYTES}",
                    ],
                },
                {
                    "destination": "/dev",
                    "type": "tmpfs",
                    "source": "tmpfs",
                    "options": [
                        "nosuid",
                        "noexec",
                        "mode=0755",
                        "size=65536",
                        "ro",
                    ],
                },
            ],
            "linux": {
                "cgroupsPath": cgroups_path,
                "devices": [
                    {
                        "path": "/dev/null",
                        "type": "c",
                        "major": 1,
                        "minor": 3,
                        "fileMode": 0o666,
                        "uid": 0,
                        "gid": 0,
                    },
                    {
                        "path": "/dev/zero",
                        "type": "c",
                        "major": 1,
                        "minor": 5,
                        "fileMode": 0o666,
                        "uid": 0,
                        "gid": 0,
                    },
                    {
                        "path": "/dev/full",
                        "type": "c",
                        "major": 1,
                        "minor": 7,
                        "fileMode": 0o666,
                        "uid": 0,
                        "gid": 0,
                    },
                    {
                        "path": "/dev/random",
                        "type": "c",
                        "major": 1,
                        "minor": 8,
                        "fileMode": 0o666,
                        "uid": 0,
                        "gid": 0,
                    },
                    {
                        "path": "/dev/urandom",
                        "type": "c",
                        "major": 1,
                        "minor": 9,
                        "fileMode": 0o666,
                        "uid": 0,
                        "gid": 0,
                    },
                ],
                "namespaces": [
                    {"type": "pid"},
                    {"type": "network"},
                    {"type": "ipc"},
                    {"type": "uts"},
                    {"type": "mount"},
                ],
                "maskedPaths": [
                    "/proc/acpi",
                    "/proc/asound",
                    "/proc/kcore",
                    "/proc/keys",
                    "/proc/latency_stats",
                    "/proc/timer_list",
                    "/proc/timer_stats",
                    "/proc/sched_debug",
                    "/sys/firmware",
                ],
                "readonlyPaths": [
                    "/proc/bus",
                    "/proc/fs",
                    "/proc/irq",
                    "/proc/sys",
                    "/proc/sysrq-trigger",
                ],
            },
        }

    def _build_command(
        self,
        state_dir: str,
        limits: dict[str, Any],
        *,
        bundle_dir: str | None = None,
        container_id: str = "codecontests-probe",
    ) -> list[str]:
        del limits
        runsc_path = (
            f"/proc/self/fd/{self._runsc_fd}"
            if self._runsc_fd is not None
            else self.config.runsc_path
        )
        effective_bundle = bundle_dir or os.path.join(
            os.path.dirname(state_dir), "oci-bundle"
        )
        return [
            runsc_path,
            "--network=none",
            f"--root={state_dir}",
            f"--log={os.path.join(state_dir, 'runsc.log')}",
            "--log-format=json",
            "run",
            f"--bundle={effective_bundle}",
            container_id,
        ]

    @staticmethod
    def _sanitized_env(nonce: str) -> dict[str, str]:
        return {
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "TMPDIR": "/tmp",
            "PALAESTRA_EXECUTOR_LAUNCH_NONCE": nonce,
        }

    @classmethod
    def _keyed_cgroup_values(cls, path: str) -> dict[str, int]:
        values: dict[str, int] = {}
        for line in cls._read_cgroup_control(path).splitlines():
            key, raw_value = line.split()
            values[key] = int(raw_value, 10)
        return values

    @classmethod
    def _exact_cgroup_events(
        cls, path: str, expected_keys: tuple[str, ...]
    ) -> dict[str, int]:
        values = cls._keyed_cgroup_values(path)
        if set(values) != set(expected_keys) or any(
            value < 0 for value in values.values()
        ):
            raise SupervisorConfigurationError(
                f"cgroup event schema drifted: {os.path.basename(path)}"
            )
        return {key: values[key] for key in expected_keys}

    def _create_request_cgroup(
        self, nonce: str, effective: dict[str, Any]
    ) -> _RequestCgroup:
        if self._delegated_cgroup is None:
            raise SupervisorConfigurationError("cgroup delegation is not prepared")
        outer = os.path.join(self._delegated_cgroup, f"request-{nonce}")
        controller = os.path.join(outer, "controller")
        candidate = os.path.join(outer, "candidate")
        os.mkdir(outer, mode=0o700)
        try:
            self._write_cgroup_control(
                os.path.join(outer, "memory.max"),
                str(effective["host_cgroup_memory_bytes"]),
            )
            self._write_cgroup_control(os.path.join(outer, "memory.swap.max"), "0")
            self._write_cgroup_control(os.path.join(outer, "memory.oom.group"), "1")
            self._write_cgroup_control(
                os.path.join(outer, "pids.max"),
                str(effective["host_cgroup_pids"]),
            )
            self._write_cgroup_control(
                os.path.join(outer, "cpu.max"),
                (
                    f"{effective['host_cgroup_cpu_quota_us']} "
                    f"{effective['host_cgroup_cpu_period_us']}"
                ),
            )
            self._write_cgroup_control(
                os.path.join(outer, "cgroup.subtree_control"),
                "+cpu +memory +pids",
            )
            os.mkdir(controller, mode=0o700)
            outer_controllers = frozenset(
                self._read_cgroup_control(
                    os.path.join(outer, "cgroup.controllers")
                ).split()
            )
            if not {"cpu", "memory", "pids"} <= outer_controllers:
                raise SupervisorConfigurationError(
                    "request cgroup controller availability drifted"
                )
            relative = os.path.relpath(candidate, "/sys/fs/cgroup")
            if relative.startswith("../"):
                raise SupervisorConfigurationError("request cgroup path escapes")
            return _RequestCgroup(
                outer=outer,
                controller=controller,
                candidate=candidate,
                oci_path="/" + relative,
                memory_events_before=self._exact_cgroup_events(
                    os.path.join(outer, "memory.events"), _MEMORY_EVENT_KEYS
                ),
                pids_events_before=self._exact_cgroup_events(
                    os.path.join(outer, "pids.events"), _PIDS_EVENT_KEYS
                ),
                outer_controllers=outer_controllers,
            )
        except BaseException:
            try:
                os.rmdir(controller)
            except FileNotFoundError:
                pass
            os.rmdir(outer)
            raise

    def _attach_gate_to_cgroup(self, pid: int, request_cgroup: _RequestCgroup) -> None:
        self._write_cgroup_control(
            os.path.join(request_cgroup.controller, "cgroup.procs"), str(pid)
        )
        expected = os.path.relpath(request_cgroup.controller, "/sys/fs/cgroup")
        with open(f"/proc/{pid}/cgroup", encoding="ascii") as handle:
            unified = [
                line.rstrip("\n").split(":", 2)[2].lstrip("/")
                for line in handle
                if line.startswith("0::")
            ]
        if unified != [expected]:
            raise SupervisorConfigurationError(
                "blocked runsc gate cgroup placement failed"
            )
        controller_members = self._subtree_pids(request_cgroup.controller)
        if controller_members != {pid}:
            raise SupervisorConfigurationError(
                "controller cgroup membership is not the blocked runsc gate"
            )
        if self._read_cgroup_control(
            os.path.join(request_cgroup.outer, "cgroup.procs")
        ):
            raise SupervisorConfigurationError(
                "request outer cgroup must remain process-free"
            )

    @staticmethod
    def _subtree_pids(root: str) -> set[int]:
        pids: set[int] = set()
        for directory, _subdirs, files in os.walk(root):
            if "cgroup.procs" not in files:
                continue
            with open(
                os.path.join(directory, "cgroup.procs"), encoding="ascii"
            ) as handle:
                pids.update(int(line.strip(), 10) for line in handle if line.strip())
        return pids

    @staticmethod
    def _read_process_identity(pid: int) -> _ProcessIdentity:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read(4097)
        if len(raw) > 4096 or not raw.endswith(b"\n"):
            raise SupervisorConfigurationError("process stat record is malformed")
        close_paren = raw.rfind(b")")
        if close_paren <= 0 or raw[close_paren : close_paren + 2] != b") ":
            raise SupervisorConfigurationError("process stat comm field is malformed")
        try:
            recorded_pid = int(raw[: raw.find(b" ")], 10)
            fields = raw[close_paren + 2 :].split()
            identity = _ProcessIdentity(
                pid=recorded_pid,
                parent_pid=int(fields[1], 10),
                process_group=int(fields[2], 10),
                session=int(fields[3], 10),
                starttime_ticks=int(fields[19], 10),
            )
        except (IndexError, ValueError) as exc:
            raise SupervisorConfigurationError(
                "process stat fields are malformed"
            ) from exc
        # Processes created during the kernel's first clock tick legitimately
        # report starttime 0 (notably PID 1 and early kernel workers in a fresh
        # VM).  It is still an exact PID/starttime identity token.
        if identity.pid != pid or identity.starttime_ticks < 0:
            raise SupervisorConfigurationError("process identity is inconsistent")
        return identity

    @classmethod
    def _stable_process_identity(cls, pid: int) -> _ProcessIdentity:
        before = cls._read_process_identity(pid)
        after = cls._read_process_identity(pid)
        if before != after:
            raise SupervisorConfigurationError(
                "process identity changed during blocked-gate attestation"
            )
        return before

    @classmethod
    def _snapshot_process_identities(cls) -> dict[int, _ProcessIdentity]:
        identities: dict[int, _ProcessIdentity] = {}
        for entry in os.scandir("/proc"):
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name, 10)
            try:
                identity = cls._read_process_identity(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue
            if pid in identities:
                raise SupervisorConfigurationError("duplicate process identity")
            identities[pid] = identity
        return identities

    @staticmethod
    def _descendant_inventory(
        identities: dict[int, _ProcessIdentity],
        roots: set[int],
    ) -> dict[int, _ProcessIdentity]:
        if not roots <= set(identities):
            raise SupervisorConfigurationError("owned process root disappeared")
        owned = set(roots)
        while True:
            children = {
                pid
                for pid, identity in identities.items()
                if identity.parent_pid in owned
            }
            expanded = owned | children
            if expanded == owned:
                return {pid: identities[pid] for pid in sorted(owned)}
            owned = expanded

    @classmethod
    def _container_process_ids(
        cls,
        identities: dict[int, _ProcessIdentity],
        container_id: str,
    ) -> set[int]:
        marker = container_id.encode("ascii")
        matched: set[int] = set()
        for pid, identity in identities.items():
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as handle:
                    retained = b""
                    contains_marker = False
                    while chunk := handle.read(64 * 1024):
                        window = retained + chunk
                        if marker in window:
                            contains_marker = True
                            break
                        retained = window[-(len(marker) - 1) :]
                if not contains_marker:
                    continue
                if cls._read_process_identity(pid) != identity:
                    raise SupervisorConfigurationError(
                        "container process identity changed during attestation"
                    )
            except (FileNotFoundError, ProcessLookupError):
                continue
            matched.add(pid)
        return matched

    @staticmethod
    def _merge_owned_inventories(
        retained: dict[int, _ProcessIdentity],
        observed: dict[int, _ProcessIdentity],
    ) -> dict[int, _ProcessIdentity]:
        merged = dict(retained)
        for pid, identity in observed.items():
            previous = merged.get(pid)
            if previous is None:
                merged[pid] = identity
                continue
            if previous.starttime_ticks != identity.starttime_ticks:
                raise SupervisorConfigurationError(
                    "owned process PID/starttime identity was reused"
                )
            if previous != identity:
                raise SupervisorConfigurationError(
                    "owned process ancestry/session identity drifted"
                )
        return merged

    @classmethod
    def _stable_owned_inventory(
        cls,
        *,
        proc_pid: int,
        container_id: str,
        initial_inventory: dict[int, _ProcessIdentity] | None = None,
        root_required: bool = True,
    ) -> dict[int, _ProcessIdentity]:
        measured: list[dict[int, _ProcessIdentity]] = []
        for _pass in range(2):
            identities = cls._snapshot_process_identities()
            if initial_inventory is None:
                if proc_pid in identities:
                    roots = {proc_pid}
                elif root_required:
                    raise SupervisorConfigurationError("owned process root disappeared")
                else:
                    roots = set()
            else:
                roots: set[int] = set()
                for pid, expected in initial_inventory.items():
                    observed = identities.get(pid)
                    if observed is None:
                        continue
                    if observed.starttime_ticks != expected.starttime_ticks:
                        continue
                    if observed != expected:
                        raise SupervisorConfigurationError(
                            "owned process ancestry/session identity drifted"
                        )
                    roots.add(pid)
                if not roots:
                    container_ids = cls._container_process_ids(identities, container_id)
                    process_group_ids = {
                        pid
                        for pid, identity in identities.items()
                        if identity.process_group == proc_pid
                    }
                    session_ids = {
                        pid
                        for pid, identity in identities.items()
                        if identity.session == proc_pid
                    }
                    if container_ids or process_group_ids or session_ids:
                        raise SupervisorConfigurationError(
                            "owned process ancestry became ambiguous"
                        )
                    measured.append({})
                    continue
            descendants = cls._descendant_inventory(identities, roots)
            container_ids = cls._container_process_ids(identities, container_id)
            process_group_ids = {
                pid
                for pid, identity in identities.items()
                if identity.process_group == proc_pid
            }
            session_ids = {
                pid
                for pid, identity in identities.items()
                if identity.session == proc_pid
            }
            if not (container_ids | process_group_ids | session_ids) <= set(
                descendants
            ):
                raise SupervisorConfigurationError(
                    "container/process-group/session helper escaped owned ancestry"
                )
            measured.append(descendants)
        if measured[0] != measured[1]:
            raise SupervisorConfigurationError(
                "owned process inventory changed during frozen attestation"
            )
        return measured[0]

    def _attest_runtime_cgroup(
        self,
        *,
        proc_pid: int,
        request_cgroup: _RequestCgroup,
        effective: dict[str, Any],
        container_id: str,
        initial_inventory: dict[int, _ProcessIdentity],
    ) -> dict[int, _ProcessIdentity]:
        if not os.path.isdir(request_cgroup.candidate):
            raise SupervisorConfigurationError("OCI candidate cgroup is absent")
        if (
            self._read_cgroup_control(
                os.path.join(request_cgroup.outer, "cgroup.freeze")
            )
            != "1"
        ):
            raise SupervisorConfigurationError(
                "request cgroup was not frozen for runtime attestation"
            )
        cgroup_events = self._keyed_cgroup_values(
            os.path.join(request_cgroup.outer, "cgroup.events")
        )
        if cgroup_events.get("frozen") != 1:
            raise SupervisorConfigurationError(
                "request cgroup freeze did not reach all runtime processes"
            )
        expected_controls = {
            "memory.max": str(effective["host_cgroup_memory_bytes"]),
            "memory.swap.max": "0",
            "memory.oom.group": "1",
            "pids.max": str(effective["host_cgroup_pids"]),
            "cpu.max": (
                f"{effective['host_cgroup_cpu_quota_us']} "
                f"{effective['host_cgroup_cpu_period_us']}"
            ),
        }
        for filename, expected in expected_controls.items():
            if (
                self._read_cgroup_control(os.path.join(request_cgroup.outer, filename))
                != expected
            ):
                raise SupervisorConfigurationError(f"request cgroup {filename} drifted")
        required = {"cpu", "memory", "pids"}
        outer_subtree_controls = set(
            self._read_cgroup_control(
                os.path.join(request_cgroup.outer, "cgroup.subtree_control")
            ).split()
        )
        runtime_outer_controllers = set(
            self._read_cgroup_control(
                os.path.join(request_cgroup.outer, "cgroup.controllers")
            ).split()
        )
        if (
            not set(request_cgroup.outer_controllers) <= runtime_outer_controllers
            or outer_subtree_controls != runtime_outer_controllers
            or not required <= outer_subtree_controls
        ):
            raise SupervisorConfigurationError(
                "request cgroup subtree controls drifted:"
                + ",".join(sorted(outer_subtree_controls))
            )
        if self._read_cgroup_control(
            os.path.join(request_cgroup.outer, "cgroup.procs")
        ):
            raise SupervisorConfigurationError(
                "request outer cgroup gained direct processes"
            )
        controller_pids = self._subtree_pids(request_cgroup.controller)
        candidate_pids = self._subtree_pids(request_cgroup.candidate)
        if controller_pids != {proc_pid} or not candidate_pids:
            raise SupervisorConfigurationError(
                "controller/candidate cgroup membership is incomplete"
            )
        subtree_pids = self._subtree_pids(request_cgroup.outer)
        if subtree_pids != controller_pids | candidate_pids:
            raise SupervisorConfigurationError("runsc runtime left request cgroup")
        runtime_inventory = self._stable_owned_inventory(
            proc_pid=proc_pid,
            container_id=container_id,
            initial_inventory=initial_inventory,
        )
        if set(runtime_inventory) != subtree_pids:
            raise SupervisorConfigurationError(
                "host descendant/helper inventory differs from request cgroup"
            )
        if not self._container_process_ids(runtime_inventory, container_id):
            raise SupervisorConfigurationError(
                "container-identified host helper inventory is absent"
            )
        return runtime_inventory

    def _attest_later_runtime_cgroup(
        self,
        *,
        proc_pid: int,
        request_cgroup: _RequestCgroup,
        container_id: str,
        initial_inventory: dict[int, _ProcessIdentity],
    ) -> dict[int, _ProcessIdentity]:
        if (
            self._read_cgroup_control(
                os.path.join(request_cgroup.outer, "cgroup.freeze")
            )
            != "1"
        ):
            raise SupervisorConfigurationError(
                "request cgroup was not frozen for later attestation"
            )
        events = self._keyed_cgroup_values(
            os.path.join(request_cgroup.outer, "cgroup.events")
        )
        if events.get("frozen") != 1:
            raise SupervisorConfigurationError(
                "later request freeze did not reach every runtime process"
            )
        controller_pids = self._subtree_pids(request_cgroup.controller)
        candidate_pids = self._subtree_pids(request_cgroup.candidate)
        subtree_pids = self._subtree_pids(request_cgroup.outer)
        if self._read_cgroup_control(
            os.path.join(request_cgroup.outer, "cgroup.procs")
        ):
            raise SupervisorConfigurationError(
                "request outer cgroup gained direct processes"
            )
        if subtree_pids != controller_pids | candidate_pids:
            raise SupervisorConfigurationError(
                "later controller/candidate membership is incomplete"
            )
        runtime_inventory = self._stable_owned_inventory(
            proc_pid=proc_pid,
            container_id=container_id,
            initial_inventory=initial_inventory,
        )
        if set(runtime_inventory) != subtree_pids:
            raise SupervisorConfigurationError(
                "later host descendant/helper inventory differs from request cgroup"
            )
        if any(
            initial_inventory.get(pid) != identity
            for pid, identity in runtime_inventory.items()
            if pid in initial_inventory
        ):
            raise SupervisorConfigurationError(
                "owned process PID/starttime identity was reused"
            )
        return runtime_inventory

    @classmethod
    def _set_cgroup_frozen(cls, outer: str, *, frozen: bool) -> None:
        expected = 1 if frozen else 0
        cls._write_cgroup_control(os.path.join(outer, "cgroup.freeze"), str(expected))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            events = cls._keyed_cgroup_values(os.path.join(outer, "cgroup.events"))
            if events.get("frozen") == expected:
                return
            time.sleep(0.005)
        raise SupervisorConfigurationError(
            f"request cgroup failed to reach frozen={expected}"
        )

    def _cleanup_request_cgroup(
        self,
        request_cgroup: _RequestCgroup,
        *,
        proc_pid: int | None = None,
        container_id: str | None = None,
        initial_inventory: dict[int, _ProcessIdentity] | None = None,
    ) -> None:
        try:
            # Never thaw a task frozen by a failed READY/terminal attestation.
            # cgroup.kill handles concurrent forks/migrations and fatal signals
            # kill frozen tasks, so teardown needs no runnable interval.
            self._write_cgroup_control(
                os.path.join(request_cgroup.outer, "cgroup.kill"), "1"
            )
        except FileNotFoundError:
            pass
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            events = self._keyed_cgroup_values(
                os.path.join(request_cgroup.outer, "cgroup.events")
            )
            if events.get("populated") == 0:
                break
            time.sleep(0.01)
        else:
            raise SupervisorConfigurationError(
                "request cgroup remained populated during cleanup"
            )
        if proc_pid is not None and container_id is not None:
            residual = self._stable_owned_inventory(
                proc_pid=proc_pid,
                container_id=container_id,
                initial_inventory=initial_inventory,
                root_required=False,
            )
            if residual:
                raise SupervisorConfigurationError(
                    "owned/container helper remained after cgroup cleanup"
                )
            if self._subtree_pids(request_cgroup.outer):
                raise SupervisorConfigurationError(
                    "request cgroup gained a process after cleanup"
                )
        for directory, subdirs, _files in os.walk(request_cgroup.outer, topdown=False):
            for subdir in subdirs:
                os.rmdir(os.path.join(directory, subdir))
        os.rmdir(request_cgroup.outer)

    @staticmethod
    def _runsc_log_is_clean(path: str) -> bool:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            return False
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | nofollow)
        except OSError:
            return False
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size < 0
                or before.st_size > _RUNSC_LOG_MAX_BYTES
            ):
                return False
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(64 * 1024, _RUNSC_LOG_MAX_BYTES + 1 - total),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > _RUNSC_LOG_MAX_BYTES:
                    return False
            # The empty read above is the required EOF observation.  Stable
            # inode metadata and exact st_size rule out an append or unseen
            # sparse tail racing that observation.
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
            if total != after.st_size or any(
                getattr(before, field) != getattr(after, field)
                for field in stable_fields
            ):
                return False
            data = b"".join(chunks)
        except OSError:
            return False
        finally:
            os.close(descriptor)
        if not data:
            return True
        lines = data.splitlines(keepends=True)
        if not lines or any(
            not line.endswith(b"\n") or line.endswith(b"\r\n") for line in lines
        ):
            return False
        for framed_line in lines:
            line = framed_line[:-1]
            if not line:
                return False
            try:
                record = json.loads(line)
            except (UnicodeError, ValueError, json.JSONDecodeError):
                return False
            if not isinstance(record, dict):
                return False
            level = record.get("level")
            if isinstance(level, str) and level.lower() in {
                "warn",
                "warning",
                "error",
                "critical",
                "fatal",
                "panic",
            }:
                return False
        return True

    @staticmethod
    def _host_process_is_root(pid: int) -> bool:
        try:
            with open(f"/proc/{pid}/status", encoding="ascii") as status:
                fields = {
                    line.split(":", 1)[0]: line.split(":", 1)[1].split()
                    for line in status
                    if ":" in line
                }
        except (OSError, UnicodeError):
            return False
        uid_values = fields.get("Uid", [])
        gid_values = fields.get("Gid", [])
        groups = fields.get("Groups", [])
        return (
            len(uid_values) >= 3
            and len(gid_values) >= 3
            and all(int(value) == 0 for value in uid_values[:3])
            and all(int(value) == 0 for value in gid_values[:3])
            and all(int(value) == 0 for value in groups)
        )

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen[bytes], sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Darwin can forbid cross-session killpg in the non-production
            # direct-process probe.  The production Ubuntu path must use the
            # process group; this fallback still terminates the probe leader.
            try:
                os.kill(proc.pid, sig)
            except ProcessLookupError:
                pass

    def _terminate_process_group(self, proc: subprocess.Popen[bytes]) -> None:
        self._kill_process_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=self.config.termination_grace_seconds)
        except subprocess.TimeoutExpired:
            self._kill_process_group(proc, signal.SIGKILL)

    @staticmethod
    def _read_capped(
        stream: Any,
        capture: _Capture,
        cap: int,
        proc: subprocess.Popen[bytes],
        stop_event: threading.Event,
        ready_marker: bytes | None = None,
        ready_event: threading.Event | None = None,
        cgroup_freeze_path: str | None = None,
    ) -> None:
        try:
            while True:
                chunk = os.read(stream.fileno(), 64 * 1024)
                if not chunk:
                    return
                remaining = max(0, cap + 1 - len(capture.data))
                capture.data.extend(chunk[:remaining])
                if (
                    ready_marker is not None
                    and ready_event is not None
                    and not ready_event.is_set()
                    and capture.data.startswith(ready_marker)
                ):
                    if cgroup_freeze_path is not None:
                        SandboxSupervisor._set_cgroup_frozen(
                            os.path.dirname(cgroup_freeze_path), frozen=True
                        )
                    ready_event.set()
                if len(capture.data) > cap:
                    capture.exceeded = True
                    stop_event.set()
                    SandboxSupervisor._kill_process_group(proc, signal.SIGTERM)
                    return
        except (OSError, SupervisorConfigurationError) as exc:
            capture.error = type(exc).__name__
            stop_event.set()

    @staticmethod
    def _write_input(
        stream: Any,
        source_frame: bytes,
        candidate_stdin: bytes,
        gate_release: threading.Event,
        capture_error: list[str],
    ) -> None:
        try:
            view = memoryview(source_frame)
            while view:
                written = os.write(stream.fileno(), view[: 64 * 1024])
                view = view[written:]
            if not gate_release.wait(timeout=5):
                capture_error.append("GateReleaseTimeout")
                return
            view = memoryview(b"G" + candidate_stdin)
            while view:
                written = os.write(stream.fileno(), view[: 64 * 1024])
                view = view[written:]
        except (BrokenPipeError, OSError) as exc:
            capture_error.append(type(exc).__name__)
        finally:
            try:
                stream.close()
            except OSError:
                pass

    @staticmethod
    def _trusted_status(
        stderr: bytes,
        *,
        marker: bytes,
        status_marker: bytes,
        effective: dict[str, Any],
    ) -> tuple[bytes, dict[str, Any]]:
        if not stderr.startswith(marker):
            raise SupervisorConfigurationError("trusted READY marker is absent")
        candidate_and_status = stderr[len(marker) :]
        status_offset = candidate_and_status.rfind(status_marker)
        if status_offset < 0:
            raise SupervisorConfigurationError("trusted terminal status is absent")
        encoded = candidate_and_status[status_offset + len(status_marker) :]
        if not encoded.endswith(b"\n") or b"\n" in encoded[:-1]:
            raise SupervisorConfigurationError("trusted terminal status framing failed")
        encoded = encoded[:-1]
        if not encoded or len(encoded) > _TRUSTED_STATUS_MAX_BYTES:
            raise SupervisorConfigurationError(
                "trusted terminal status size is invalid"
            )
        try:
            raw_status = base64.b64decode(encoded, validate=True)
            status = json.loads(raw_status.decode("ascii", errors="strict"))
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise SupervisorConfigurationError(
                "trusted terminal status encoding is invalid"
            ) from exc
        expected_keys = {
            "version",
            "candidate_ready_attested",
            "returncode",
            "cpu_usage_us",
            "cpu_limit_us",
            "cpu_limit_hit",
            "process_peak",
            "process_limit",
            "process_rlimit_nproc",
            "process_limit_hit",
            "process_limit_syscall",
            "tracer_killed_main",
            "stdout_truncated",
            "stderr_truncated",
            "file_space_limit_hit",
            "file_space_limit_source",
            "file_size_limit_bytes",
            "writable_limit_bytes",
            "file_limit_signal",
            "file_limit_errno",
            "file_size_observed_bytes",
            "writable_available_bytes",
        }
        if not isinstance(status, dict) or set(status) != expected_keys:
            raise SupervisorConfigurationError("trusted terminal status schema drifted")
        if status["version"] != 1 or status["candidate_ready_attested"] is not True:
            raise SupervisorConfigurationError("candidate readiness was not attested")
        for key in (
            "returncode",
            "cpu_usage_us",
            "cpu_limit_us",
            "process_peak",
            "process_limit",
            "process_rlimit_nproc",
            "file_size_limit_bytes",
            "writable_limit_bytes",
            "file_size_observed_bytes",
            "writable_available_bytes",
        ):
            if isinstance(status[key], bool) or not isinstance(status[key], int):
                raise SupervisorConfigurationError(
                    "trusted terminal integer evidence is invalid"
                )
        for key in (
            "cpu_limit_hit",
            "process_limit_hit",
            "tracer_killed_main",
            "file_space_limit_hit",
            "stdout_truncated",
            "stderr_truncated",
        ):
            if not isinstance(status[key], bool):
                raise SupervisorConfigurationError(
                    "trusted terminal boolean evidence is invalid"
                )
        expected_cpu_limit = effective["cpu_seconds"] * 1_000_000
        process_limit_syscall = status["process_limit_syscall"]
        if process_limit_syscall is not None and (
            isinstance(process_limit_syscall, bool)
            or not isinstance(process_limit_syscall, int)
            or process_limit_syscall < 0
        ):
            raise SupervisorConfigurationError(
                "trusted process-limit syscall is invalid"
            )
        file_space_limit_source = status["file_space_limit_source"]
        if file_space_limit_source not in {
            None,
            "guest_monitor_ptrace_siginfo_fsize",
            "guest_monitor_ptrace_write_efbig",
        }:
            raise SupervisorConfigurationError("trusted file-limit source is invalid")
        file_limit_signal = status["file_limit_signal"]
        if file_limit_signal is not None and (
            isinstance(file_limit_signal, bool)
            or not isinstance(file_limit_signal, int)
        ):
            raise SupervisorConfigurationError("trusted file-limit signal is invalid")
        file_limit_errno = status["file_limit_errno"]
        if file_limit_errno is not None and (
            isinstance(file_limit_errno, bool) or not isinstance(file_limit_errno, int)
        ):
            raise SupervisorConfigurationError("trusted file-limit errno is invalid")
        if (
            status["cpu_limit_us"] != expected_cpu_limit
            or not 0 <= status["cpu_usage_us"] <= expected_cpu_limit + 2_000_000
            or (status["cpu_limit_hit"] and status["cpu_usage_us"] < expected_cpu_limit)
            or status["process_limit"] != effective["processes"]
            or status["process_rlimit_nproc"] != PINNED_GVISOR_RLIMIT_NPROC
            or status["process_rlimit_nproc"] != status["process_limit"] - 1
            or not 0 <= status["process_peak"] <= effective["processes"]
            or (status["cpu_limit_hit"] and status["returncode"] != -signal.SIGKILL)
            or (
                status["tracer_killed_main"] and status["returncode"] != -signal.SIGKILL
            )
            or (
                status["tracer_killed_main"]
                and not (
                    status["cpu_limit_hit"]
                    or status["stdout_truncated"]
                    or status["stderr_truncated"]
                    or file_space_limit_source == "guest_monitor_ptrace_siginfo_fsize"
                )
            )
            or (status["cpu_limit_hit"] and not status["tracer_killed_main"])
            or (
                file_space_limit_source == "guest_monitor_ptrace_siginfo_fsize"
                and not status["tracer_killed_main"]
            )
            or status["process_limit_hit"] != (process_limit_syscall is not None)
            or status["file_size_limit_bytes"] != effective["file_size_bytes"]
            or status["writable_limit_bytes"] != effective["aggregate_writable_bytes"]
            or not 0
            <= status["file_size_observed_bytes"]
            <= status["file_size_limit_bytes"]
            or not 0
            <= status["writable_available_bytes"]
            <= status["writable_limit_bytes"]
            or status["file_space_limit_hit"] != (file_space_limit_source is not None)
            or (file_space_limit_source == "guest_monitor_ptrace_siginfo_fsize")
            != (file_limit_signal == signal.SIGXFSZ)
            or (file_space_limit_source == "guest_monitor_ptrace_write_efbig")
            != (file_limit_errno == 27)
            or (
                file_space_limit_source != "guest_monitor_ptrace_siginfo_fsize"
                and file_limit_signal is not None
            )
            or (
                file_space_limit_source != "guest_monitor_ptrace_write_efbig"
                and file_limit_errno is not None
            )
            or (
                status["process_limit_hit"]
                and status["process_peak"] != effective["processes"]
            )
        ):
            raise SupervisorConfigurationError(
                "trusted terminal limit evidence is inconsistent"
            )
        return candidate_and_status[:status_offset], status

    def execute(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        task = request_payload["task"]
        limits = task["limits"]
        code_bytes = base64.b64decode(task["code_b64"], validate=True)
        stdin_bytes = base64.b64decode(task["stdin_b64"], validate=True)
        source_frame = struct.pack("!Q", len(code_bytes)) + code_bytes
        nonce = secrets.token_hex(32)
        marker = (
            f"PALAESTRA_EXECUTOR_READY:{nonce}:monitor_uid=0:monitor_gid=0\n"
        ).encode("ascii")
        status_marker = f"PALAESTRA_EXECUTOR_STATUS:{nonce}:".encode("ascii")
        container_id = f"cc-{nonce}"
        started_ns = time.monotonic_ns()
        request_dir: str | None = None
        proc: subprocess.Popen[bytes] | None = None
        request_cgroup: _RequestCgroup | None = None
        gate_read_fd: int | None = None
        gate_write_fd: int | None = None
        primary_error: BaseException | None = None
        stdout = _Capture(bytearray())
        # The trusted marker is transport overhead, not candidate stderr.
        stderr = _Capture(bytearray())
        input_errors: list[str] = []
        stop_event = threading.Event()
        ready_event = threading.Event()
        candidate_gate_release = threading.Event()
        wall_expired = False
        ready_expired = False
        aggregate_cpu_expired = False
        cgroup_attested = False
        owned_inventory: dict[int, _ProcessIdentity] = {}
        cpu_before_usage_us = 0
        cpu_ready_usage_us: int | None = None
        cpu_cross_usage_us = 0
        try:
            if self.config.enforce_request_cgroup:
                self._attest_rootfs_unchanged()
            # The private outer directory contains runsc state only; no
            # candidate source, stdin, secret, or control socket is materialized.
            request_dir = tempfile.mkdtemp(
                prefix="request-", dir=self.config.request_root
            )
            os.chmod(request_dir, 0o700)
            state_dir = os.path.join(request_dir, "runsc-state")
            os.mkdir(state_dir, mode=0o700)
            bundle_dir = os.path.join(request_dir, "oci-bundle")
            os.mkdir(bundle_dir, mode=0o700)
            if self.config.enforce_request_cgroup:
                request_cgroup = self._create_request_cgroup(nonce, limits["effective"])
                cpu_before_usage_us = self._keyed_cgroup_values(
                    os.path.join(request_cgroup.outer, "cpu.stat")
                )["usage_usec"]
                config = self._oci_config(
                    limits=limits,
                    nonce=nonce,
                    cgroups_path=request_cgroup.oci_path,
                )
                config_path = os.path.join(bundle_dir, "config.json")
                config_fd = os.open(
                    config_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                    0o600,
                )
                try:
                    config_bytes = json.dumps(
                        config,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    view = memoryview(config_bytes)
                    while view:
                        view = view[os.write(config_fd, view) :]
                    os.fsync(config_fd)
                finally:
                    os.close(config_fd)
            command = self._build_command(
                state_dir,
                limits,
                bundle_dir=bundle_dir,
                container_id=container_id,
            )
            try:
                pass_fds = tuple(
                    fd for fd in (self._runsc_fd, self._rootfs_fd) if fd is not None
                )
                launch_command = command
                if request_cgroup is not None:
                    gate_read_fd, gate_write_fd = os.pipe()
                    launch_command = [
                        self.config.host_python_path,
                        "-I",
                        "-B",
                        "-c",
                        self._cgroup_gate_bytes.decode("utf-8", errors="strict"),
                        str(gate_read_fd),
                        "--",
                        *command,
                    ]
                    pass_fds = (*pass_fds, gate_read_fd)
                proc = subprocess.Popen(
                    launch_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=self._sanitized_env(nonce),
                    close_fds=True,
                    start_new_session=True,
                    umask=0o077,
                    pass_fds=pass_fds,
                )
                if gate_read_fd is not None:
                    os.close(gate_read_fd)
                    gate_read_fd = None
                if request_cgroup is not None:
                    gate_identity = self._stable_process_identity(proc.pid)
                    owned_inventory = self._merge_owned_inventories(
                        owned_inventory, {proc.pid: gate_identity}
                    )
                    self._attach_gate_to_cgroup(proc.pid, request_cgroup)
                    if not self._host_process_is_root(proc.pid):
                        raise SupervisorConfigurationError(
                            "rootful runsc gate UID/GID attestation failed"
                        )
                    assert gate_write_fd is not None
                    if os.write(gate_write_fd, b"G") != 1:
                        raise SupervisorConfigurationError(
                            "runsc gate release was short"
                        )
                    os.close(gate_write_fd)
                    gate_write_fd = None
            except (OSError, subprocess.SubprocessError) as exc:
                return self._unknown(
                    "CONTROLLER_START_FAILURE",
                    started_ns,
                    controller_error=type(exc).__name__,
                )
            assert proc.stdin is not None
            assert proc.stdout is not None
            assert proc.stderr is not None
            stdout_thread = threading.Thread(
                target=self._read_capped,
                args=(proc.stdout, stdout, STDOUT_CAP_BYTES, proc, stop_event),
                daemon=True,
            )
            # Permit exactly STDERR_CAP_BYTES after the private ready marker.
            stderr_thread = threading.Thread(
                target=self._read_capped,
                args=(
                    proc.stderr,
                    stderr,
                    STDERR_CAP_BYTES
                    + len(marker)
                    + len(status_marker)
                    + _TRUSTED_STATUS_MAX_BYTES,
                    proc,
                    stop_event,
                    marker,
                    ready_event,
                    (
                        os.path.join(request_cgroup.outer, "cgroup.freeze")
                        if request_cgroup is not None
                        else None
                    ),
                ),
                daemon=True,
            )
            input_thread = threading.Thread(
                target=self._write_input,
                args=(
                    proc.stdin,
                    source_frame,
                    stdin_bytes,
                    candidate_gate_release,
                    input_errors,
                ),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            input_thread.start()
            wall_seconds = limits["effective"]["wall_time_ns"] / 1_000_000_000
            ready_deadline = time.monotonic() + 5
            candidate_deadline: float | None = None
            while proc.poll() is None:
                if ready_event.is_set() and not cgroup_attested:
                    if request_cgroup is not None:
                        runtime_inventory = self._attest_runtime_cgroup(
                            proc_pid=proc.pid,
                            request_cgroup=request_cgroup,
                            effective=limits["effective"],
                            container_id=container_id,
                            initial_inventory=owned_inventory,
                        )
                        owned_inventory = self._merge_owned_inventories(
                            owned_inventory,
                            runtime_inventory,
                        )
                        cpu_ready_usage_us = self._keyed_cgroup_values(
                            os.path.join(request_cgroup.outer, "cpu.stat")
                        )["usage_usec"]
                        self._set_cgroup_frozen(request_cgroup.outer, frozen=False)
                    cgroup_attested = True
                    candidate_gate_release.set()
                    candidate_deadline = time.monotonic() + wall_seconds
                if request_cgroup is not None and cpu_ready_usage_us is not None:
                    current_cpu_us = self._keyed_cgroup_values(
                        os.path.join(request_cgroup.outer, "cpu.stat")
                    )["usage_usec"]
                    if (
                        current_cpu_us - cpu_ready_usage_us
                        >= limits["effective"]["host_cgroup_cpu_budget_us"]
                    ):
                        aggregate_cpu_expired = True
                        cpu_cross_usage_us = current_cpu_us
                        self._terminate_process_group(proc)
                        break
                current_time = time.monotonic()
                if not cgroup_attested:
                    remaining = ready_deadline - current_time
                    if remaining <= 0:
                        ready_expired = True
                        self._terminate_process_group(proc)
                        break
                else:
                    assert candidate_deadline is not None
                    remaining = candidate_deadline - current_time
                if remaining <= 0:
                    wall_expired = True
                    self._terminate_process_group(proc)
                    break
                if stop_event.wait(timeout=min(0.01, remaining)):
                    self._terminate_process_group(proc)
                    break

            # At the second boundary, freeze the exact host helper inventory
            # again before teardown.  Killing the whole cgroup avoids relying
            # on a process group that an unexpected helper could have left.
            if request_cgroup is not None and cgroup_attested and owned_inventory:
                self._set_cgroup_frozen(request_cgroup.outer, frozen=True)
                runtime_inventory = self._attest_later_runtime_cgroup(
                    proc_pid=proc.pid,
                    request_cgroup=request_cgroup,
                    container_id=container_id,
                    initial_inventory=owned_inventory,
                )
                owned_inventory = self._merge_owned_inventories(
                    owned_inventory,
                    runtime_inventory,
                )
                self._write_cgroup_control(
                    os.path.join(request_cgroup.outer, "cgroup.kill"), "1"
                )
            # Also kill the original process group as a belt-and-suspenders
            # cleanup for failures before frozen cgroup attestation.
            self._kill_process_group(proc, signal.SIGKILL)
            # A controller that exits before its private READY marker leaves
            # the input writer blocked on this gate.  Release it before the
            # bounded joins so that the writer can observe the closed pipe;
            # otherwise its intentional five-second READY wait is
            # misclassified as a one-second controller-drain failure and
            # masks the real launch-attestation result.
            candidate_gate_release.set()
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            input_thread.join(timeout=1)
            if (
                stdout_thread.is_alive()
                or stderr_thread.is_alive()
                or input_thread.is_alive()
            ):
                return self._unknown("CONTROLLER_DRAIN_FAILURE", started_ns)
            if stdout.error or stderr.error:
                return self._unknown(
                    "CONTROLLER_CAPTURE_FAILURE",
                    started_ns,
                    controller_error=stdout.error or stderr.error,
                )
            if ready_expired:
                return self._unknown("READY_DEADLINE", started_ns)
            if "GateReleaseTimeout" in input_errors:
                return self._unknown(
                    "CONTROLLER_INPUT_FAILURE",
                    started_ns,
                    controller_error="GateReleaseTimeout",
                )

            resource_kwargs: _ResourceEvidence = {
                "host_cpu_usage_us": 0,
                "host_cpu_before_usage_us": 0,
                "host_cpu_ready_usage_us": 0,
                "host_cpu_cross_usage_us": 0,
                "host_cpu_after_usage_us": 0,
                "host_cpu_budget_us": limits["effective"]["host_cgroup_cpu_budget_us"],
                "host_memory_peak_bytes": 0,
                "host_pids_peak": 0,
                "host_memory_events_before": {key: 0 for key in _MEMORY_EVENT_KEYS},
                "host_memory_events_after": {key: 0 for key in _MEMORY_EVENT_KEYS},
                "host_pids_events_before": {key: 0 for key in _PIDS_EVENT_KEYS},
                "host_pids_events_after": {key: 0 for key in _PIDS_EVENT_KEYS},
                "guest_cpu_usage_us": 0,
                "guest_process_peak": 0,
                "guest_process_limit": limits["effective"]["processes"],
                "guest_rlimit_nproc": PINNED_GVISOR_RLIMIT_NPROC,
                "guest_process_limit_syscall": None,
                "guest_file_size_limit_bytes": limits["effective"]["file_size_bytes"],
                "guest_writable_limit_bytes": limits["effective"][
                    "aggregate_writable_bytes"
                ],
                "guest_file_limit_signal": None,
                "guest_file_limit_errno": None,
                "guest_file_size_observed_bytes": 0,
                "guest_writable_available_bytes": 0,
                "resource_evidence_source": None,
            }
            detected_resource_event: str | None = None
            if request_cgroup is not None and cgroup_attested:
                assert cpu_ready_usage_us is not None
                cpu_after_usage_us = self._keyed_cgroup_values(
                    os.path.join(request_cgroup.outer, "cpu.stat")
                )["usage_usec"]
                memory_peak = int(
                    self._read_cgroup_control(
                        os.path.join(request_cgroup.outer, "memory.peak")
                    ),
                    10,
                )
                pids_peak = int(
                    self._read_cgroup_control(
                        os.path.join(request_cgroup.outer, "pids.peak")
                    ),
                    10,
                )
                memory_events_after = self._exact_cgroup_events(
                    os.path.join(request_cgroup.outer, "memory.events"),
                    _MEMORY_EVENT_KEYS,
                )
                pids_events_after = self._exact_cgroup_events(
                    os.path.join(request_cgroup.outer, "pids.events"),
                    _PIDS_EVENT_KEYS,
                )
                memory_hit = any(
                    memory_events_after.get(key, 0)
                    > request_cgroup.memory_events_before.get(key, 0)
                    for key in ("max", "oom", "oom_kill", "oom_group_kill")
                )
                pids_hit = pids_events_after.get(
                    "max", 0
                ) > request_cgroup.pids_events_before.get("max", 0)
                if (
                    not aggregate_cpu_expired
                    and cpu_after_usage_us - cpu_ready_usage_us
                    >= limits["effective"]["host_cgroup_cpu_budget_us"]
                ):
                    # The runtime may exit between polling samples.  Preserve
                    # the final exact crossing as aggregate ambiguity too.
                    aggregate_cpu_expired = True
                    cpu_cross_usage_us = cpu_after_usage_us
                events = [
                    event
                    for event, happened in (
                        ("CGROUP_CPU_BUDGET", aggregate_cpu_expired),
                        ("CGROUP_MEMORY_OOM", memory_hit),
                        ("CGROUP_PIDS_MAX", pids_hit),
                    )
                    if happened
                ]
                resource_kwargs = {
                    "host_cpu_usage_us": (cpu_after_usage_us - cpu_ready_usage_us),
                    "host_cpu_before_usage_us": cpu_before_usage_us,
                    "host_cpu_ready_usage_us": cpu_ready_usage_us,
                    "host_cpu_cross_usage_us": cpu_cross_usage_us,
                    "host_cpu_after_usage_us": cpu_after_usage_us,
                    "host_cpu_budget_us": limits["effective"][
                        "host_cgroup_cpu_budget_us"
                    ],
                    "host_memory_peak_bytes": memory_peak,
                    "host_pids_peak": pids_peak,
                    "host_memory_events_before": dict(
                        request_cgroup.memory_events_before
                    ),
                    "host_memory_events_after": memory_events_after,
                    "host_pids_events_before": dict(request_cgroup.pids_events_before),
                    "host_pids_events_after": pids_events_after,
                    "guest_cpu_usage_us": 0,
                    "guest_process_peak": 0,
                    "guest_process_limit": limits["effective"]["processes"],
                    "guest_rlimit_nproc": PINNED_GVISOR_RLIMIT_NPROC,
                    "guest_process_limit_syscall": None,
                    "guest_file_size_limit_bytes": limits["effective"][
                        "file_size_bytes"
                    ],
                    "guest_writable_limit_bytes": limits["effective"][
                        "aggregate_writable_bytes"
                    ],
                    "guest_file_limit_signal": None,
                    "guest_file_limit_errno": None,
                    "guest_file_size_observed_bytes": 0,
                    "guest_writable_available_bytes": 0,
                    "resource_evidence_source": None,
                }
                print(
                    "executor_request_resources "
                    f"cpu_usage_us={resource_kwargs['host_cpu_usage_us']} "
                    f"memory_peak_bytes={memory_peak} "
                    f"pids_peak={pids_peak} events={','.join(events) or 'none'}",
                    flush=True,
                )
                if len(events) > 1:
                    return self._unknown(
                        "AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE",
                        started_ns,
                        returncode=proc.returncode,
                        **resource_kwargs,
                    )
                detected_resource_event = events[0] if events else None
                if detected_resource_event is not None:
                    resource_kwargs["resource_evidence_source"] = {
                        "CGROUP_CPU_BUDGET": "request_cgroup_cpu_stat",
                        "CGROUP_MEMORY_OOM": "request_cgroup_memory_events",
                        "CGROUP_PIDS_MAX": "request_cgroup_pids_events",
                    }[detected_resource_event]
                if not self._runsc_log_is_clean(os.path.join(state_dir, "runsc.log")):
                    return self._unknown(
                        (
                            "AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE"
                            if detected_resource_event is not None
                            else "RUNSC_WARNING_OR_ERROR"
                        ),
                        started_ns,
                        returncode=proc.returncode,
                        resource_event=detected_resource_event,
                        **resource_kwargs,
                    )
                # Aggregate request-cgroup counters include trusted runsc,
                # gofer, sandbox, and monitor overhead.  They are a fatal
                # ambiguity boundary, never candidate-specific limit proof.
                if detected_resource_event is not None:
                    return self._unknown(
                        "AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE",
                        started_ns,
                        returncode=proc.returncode,
                        resource_event=detected_resource_event,
                        **resource_kwargs,
                    )

            stderr_bytes = bytes(stderr.data)
            if not stderr_bytes.startswith(marker):
                return self._unknown(
                    "LAUNCH_ATTESTATION_MISSING",
                    started_ns,
                    returncode=proc.returncode,
                    stderr=stderr_bytes,
                    **resource_kwargs,
                )
            if wall_expired:
                if (
                    detected_resource_event is not None
                    or stdout.exceeded
                    or stderr.exceeded
                ):
                    return self._unknown(
                        "AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE",
                        started_ns,
                        returncode=proc.returncode,
                        **resource_kwargs,
                    )
                # The host starts this deadline only after the nonce-bound
                # READY marker and complete frozen-cgroup attestation.  A
                # process tree that remains live at that boundary is the
                # semantic wall-time event; SIGTERM necessarily prevents the
                # in-sandbox monitor from appending its terminal record.
                return self._candidate_failure(
                    "WALL_LIMIT",
                    started_ns,
                    proc.returncode,
                    bytes(stdout.data),
                    stderr_bytes[len(marker) :],
                    **resource_kwargs,
                )
            try:
                candidate_stderr, trusted_status = self._trusted_status(
                    stderr_bytes,
                    marker=marker,
                    status_marker=status_marker,
                    effective=limits["effective"],
                )
            except SupervisorConfigurationError:
                return self._unknown(
                    "LAUNCH_ATTESTATION_MISSING",
                    started_ns,
                    returncode=proc.returncode,
                    stderr=stderr_bytes,
                    **resource_kwargs,
                )
            candidate_returncode = trusted_status["returncode"]
            resource_kwargs["guest_cpu_usage_us"] = trusted_status["cpu_usage_us"]
            resource_kwargs["guest_process_peak"] = trusted_status["process_peak"]
            resource_kwargs["guest_process_limit"] = trusted_status["process_limit"]
            resource_kwargs["guest_rlimit_nproc"] = trusted_status[
                "process_rlimit_nproc"
            ]
            resource_kwargs["guest_process_limit_syscall"] = trusted_status[
                "process_limit_syscall"
            ]
            resource_kwargs["guest_file_size_limit_bytes"] = trusted_status[
                "file_size_limit_bytes"
            ]
            resource_kwargs["guest_writable_limit_bytes"] = trusted_status[
                "writable_limit_bytes"
            ]
            resource_kwargs["guest_file_limit_signal"] = trusted_status[
                "file_limit_signal"
            ]
            resource_kwargs["guest_file_limit_errno"] = trusted_status[
                "file_limit_errno"
            ]
            resource_kwargs["guest_file_size_observed_bytes"] = trusted_status[
                "file_size_observed_bytes"
            ]
            resource_kwargs["guest_writable_available_bytes"] = trusted_status[
                "writable_available_bytes"
            ]
            if proc.returncode != 0:
                return self._unknown(
                    "AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE",
                    started_ns,
                    returncode=(
                        proc.returncode
                        if proc.returncode is not None
                        else -signal.SIGKILL
                    ),
                    **resource_kwargs,
                )
            if trusted_status["file_space_limit_hit"]:
                resource_kwargs["resource_evidence_source"] = trusted_status[
                    "file_space_limit_source"
                ]
                return self._candidate_failure(
                    "FILE_SPACE_LIMIT",
                    started_ns,
                    candidate_returncode,
                    bytes(stdout.data[:STDOUT_CAP_BYTES]),
                    candidate_stderr[:STDERR_CAP_BYTES],
                    resource_event="GUEST_FILE_SPACE_LIMIT",
                    **resource_kwargs,
                )
            if (
                trusted_status["process_limit_hit"]
                and candidate_returncode != 0
            ):
                resource_kwargs["resource_evidence_source"] = (
                    "guest_monitor_ptrace_thread_eagain"
                )
                return self._candidate_failure(
                    "PROCESS_LIMIT",
                    started_ns,
                    candidate_returncode,
                    bytes(stdout.data[:STDOUT_CAP_BYTES]),
                    candidate_stderr[:STDERR_CAP_BYTES],
                    resource_event="GUEST_PROCESS_LIMIT",
                    **resource_kwargs,
                )
            handled_process_denial = trusted_status["process_limit_hit"]
            if trusted_status["cpu_limit_hit"]:
                resource_kwargs["resource_evidence_source"] = (
                    "guest_monitor_ptrace_siginfo"
                )
                return self._candidate_failure(
                    "CPU_LIMIT",
                    started_ns,
                    candidate_returncode,
                    bytes(stdout.data[:STDOUT_CAP_BYTES]),
                    candidate_stderr[:STDERR_CAP_BYTES],
                    resource_event="GUEST_CPU_LIMIT",
                    **resource_kwargs,
                )
            stderr_limit = (
                trusted_status["stderr_truncated"]
                or stderr.exceeded
                or len(candidate_stderr) > STDERR_CAP_BYTES
            )
            stdout_limit = trusted_status["stdout_truncated"] or stdout.exceeded
            if stdout_limit or stderr_limit:
                return self._candidate_failure(
                    "OUTPUT_LIMIT",
                    started_ns,
                    candidate_returncode,
                    bytes(stdout.data[:STDOUT_CAP_BYTES]),
                    candidate_stderr[:STDERR_CAP_BYTES],
                    stdout_truncated=stdout_limit,
                    stderr_truncated=stderr_limit,
                    **resource_kwargs,
                )
            if candidate_returncode is None:
                return self._unknown(
                    "CONTROLLER_NO_EXIT_STATUS",
                    started_ns,
                    **resource_kwargs,
                )
            if candidate_returncode == -signal.SIGKILL:
                return self._unknown(
                    "AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE",
                    started_ns,
                    returncode=candidate_returncode,
                    stderr=candidate_stderr,
                    **resource_kwargs,
                )
            if candidate_returncode != 0:
                return self._candidate_failure(
                    "RUNTIME_ERROR",
                    started_ns,
                    candidate_returncode,
                    bytes(stdout.data),
                    candidate_stderr,
                    **resource_kwargs,
                )
            if handled_process_denial:
                resource_kwargs["resource_evidence_source"] = (
                    "guest_monitor_ptrace_thread_eagain"
                )
            return self._executed(
                started_ns,
                candidate_returncode,
                bytes(stdout.data),
                candidate_stderr,
                resource_event=(
                    "GUEST_PROCESS_LIMIT" if handled_process_denial else None
                ),
                **resource_kwargs,
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            candidate_gate_release.set()
            for gate_fd in (gate_read_fd, gate_write_fd):
                if gate_fd is not None:
                    try:
                        os.close(gate_fd)
                    except OSError:
                        pass
            if proc is not None and proc.poll() is None:
                self._terminate_process_group(proc)
                self._kill_process_group(proc, signal.SIGKILL)
            cleanup_errors: list[BaseException] = []
            if request_cgroup is not None:
                try:
                    self._cleanup_request_cgroup(
                        request_cgroup,
                        proc_pid=proc.pid if proc is not None else None,
                        container_id=container_id,
                        initial_inventory=owned_inventory or None,
                    )
                except BaseException as cleanup_error:  # noqa: BLE001 - all-exit cleanup
                    cleanup_errors.append(cleanup_error)
            if request_dir is not None:
                try:
                    shutil.rmtree(request_dir, ignore_errors=False)
                except BaseException as cleanup_error:  # noqa: BLE001 - all-exit cleanup
                    cleanup_errors.append(cleanup_error)
            if self.config.enforce_request_cgroup:
                try:
                    self._attest_rootfs_unchanged()
                except BaseException as cleanup_error:  # noqa: BLE001 - all-exit cleanup
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                if primary_error is not None:
                    for cleanup_error in cleanup_errors:
                        primary_error.add_note(
                            "executor cleanup also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                else:
                    raise cleanup_errors[0]

    @staticmethod
    def _base_result(
        outcome: str,
        category: str | None,
        started_ns: int,
        *,
        returncode: int | None,
        stdout: bytes,
        stderr: bytes,
        stdout_truncated: bool = False,
        stderr_truncated: bool = False,
        controller_error: str | None = None,
        resource_event: str | None = None,
        host_cpu_usage_us: int = 0,
        host_cpu_before_usage_us: int = 0,
        host_cpu_ready_usage_us: int = 0,
        host_cpu_cross_usage_us: int = 0,
        host_cpu_after_usage_us: int = 0,
        host_cpu_budget_us: int = 0,
        host_memory_peak_bytes: int = 0,
        host_pids_peak: int = 0,
        host_memory_events_before: dict[str, int] | None = None,
        host_memory_events_after: dict[str, int] | None = None,
        host_pids_events_before: dict[str, int] | None = None,
        host_pids_events_after: dict[str, int] | None = None,
        guest_cpu_usage_us: int = 0,
        guest_process_peak: int = 0,
        guest_process_limit: int = PID_CAP,
        guest_rlimit_nproc: int = PINNED_GVISOR_RLIMIT_NPROC,
        guest_process_limit_syscall: int | None = None,
        guest_file_size_limit_bytes: int = FILE_SIZE_CAP_BYTES,
        guest_writable_limit_bytes: int = WRITABLE_OVERLAY_CAP_BYTES,
        guest_file_limit_signal: int | None = None,
        guest_file_limit_errno: int | None = None,
        guest_file_size_observed_bytes: int = 0,
        guest_writable_available_bytes: int = 0,
        resource_evidence_source: str | None = None,
    ) -> dict[str, Any]:
        return {
            "outcome": outcome,
            "category": category,
            "retryable": False,
            "stdout_b64": base64.b64encode(stdout).decode("ascii"),
            "stderr_b64": base64.b64encode(stderr).decode("ascii"),
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "returncode": returncode,
            "signal": -returncode
            if returncode is not None and returncode < 0
            else None,
            "controller_error": controller_error,
            "resource_event": resource_event,
            "host_cpu_usage_us": host_cpu_usage_us,
            "host_cpu_before_usage_us": host_cpu_before_usage_us,
            "host_cpu_ready_usage_us": host_cpu_ready_usage_us,
            "host_cpu_cross_usage_us": host_cpu_cross_usage_us,
            "host_cpu_after_usage_us": host_cpu_after_usage_us,
            "host_cpu_budget_us": host_cpu_budget_us,
            "host_memory_peak_bytes": host_memory_peak_bytes,
            "host_pids_peak": host_pids_peak,
            "host_memory_events_before": host_memory_events_before
            or {key: 0 for key in _MEMORY_EVENT_KEYS},
            "host_memory_events_after": host_memory_events_after
            or {key: 0 for key in _MEMORY_EVENT_KEYS},
            "host_pids_events_before": host_pids_events_before
            or {key: 0 for key in _PIDS_EVENT_KEYS},
            "host_pids_events_after": host_pids_events_after
            or {key: 0 for key in _PIDS_EVENT_KEYS},
            "guest_cpu_usage_us": guest_cpu_usage_us,
            "guest_process_peak": guest_process_peak,
            "guest_process_limit": guest_process_limit,
            "guest_rlimit_nproc": guest_rlimit_nproc,
            "guest_process_limit_syscall": guest_process_limit_syscall,
            "guest_file_size_limit_bytes": guest_file_size_limit_bytes,
            "guest_writable_limit_bytes": guest_writable_limit_bytes,
            "guest_file_limit_signal": guest_file_limit_signal,
            "guest_file_limit_errno": guest_file_limit_errno,
            "guest_file_size_observed_bytes": guest_file_size_observed_bytes,
            "guest_writable_available_bytes": guest_writable_available_bytes,
            "resource_evidence_source": resource_evidence_source,
            "execution_ns": time.monotonic_ns() - started_ns,
        }

    @classmethod
    def _executed(
        cls,
        started_ns: int,
        returncode: int,
        stdout: bytes,
        stderr: bytes,
        *,
        resource_event: str | None = None,
        host_cpu_usage_us: int = 0,
        host_cpu_before_usage_us: int = 0,
        host_cpu_ready_usage_us: int = 0,
        host_cpu_cross_usage_us: int = 0,
        host_cpu_after_usage_us: int = 0,
        host_cpu_budget_us: int = 0,
        host_memory_peak_bytes: int = 0,
        host_pids_peak: int = 0,
        host_memory_events_before: dict[str, int] | None = None,
        host_memory_events_after: dict[str, int] | None = None,
        host_pids_events_before: dict[str, int] | None = None,
        host_pids_events_after: dict[str, int] | None = None,
        guest_cpu_usage_us: int = 0,
        guest_process_peak: int = 0,
        guest_process_limit: int = PID_CAP,
        guest_rlimit_nproc: int = PINNED_GVISOR_RLIMIT_NPROC,
        guest_process_limit_syscall: int | None = None,
        guest_file_size_limit_bytes: int = FILE_SIZE_CAP_BYTES,
        guest_writable_limit_bytes: int = WRITABLE_OVERLAY_CAP_BYTES,
        guest_file_limit_signal: int | None = None,
        guest_file_limit_errno: int | None = None,
        guest_file_size_observed_bytes: int = 0,
        guest_writable_available_bytes: int = 0,
        resource_evidence_source: str | None = None,
    ) -> dict[str, Any]:
        return cls._base_result(
            "executed",
            None,
            started_ns,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            resource_event=resource_event,
            host_cpu_usage_us=host_cpu_usage_us,
            host_cpu_before_usage_us=host_cpu_before_usage_us,
            host_cpu_ready_usage_us=host_cpu_ready_usage_us,
            host_cpu_cross_usage_us=host_cpu_cross_usage_us,
            host_cpu_after_usage_us=host_cpu_after_usage_us,
            host_cpu_budget_us=host_cpu_budget_us,
            host_memory_peak_bytes=host_memory_peak_bytes,
            host_pids_peak=host_pids_peak,
            host_memory_events_before=host_memory_events_before,
            host_memory_events_after=host_memory_events_after,
            host_pids_events_before=host_pids_events_before,
            host_pids_events_after=host_pids_events_after,
            guest_cpu_usage_us=guest_cpu_usage_us,
            guest_process_peak=guest_process_peak,
            guest_process_limit=guest_process_limit,
            guest_rlimit_nproc=guest_rlimit_nproc,
            guest_process_limit_syscall=guest_process_limit_syscall,
            guest_file_size_limit_bytes=guest_file_size_limit_bytes,
            guest_writable_limit_bytes=guest_writable_limit_bytes,
            guest_file_limit_signal=guest_file_limit_signal,
            guest_file_limit_errno=guest_file_limit_errno,
            guest_file_size_observed_bytes=guest_file_size_observed_bytes,
            guest_writable_available_bytes=guest_writable_available_bytes,
            resource_evidence_source=resource_evidence_source,
        )

    @classmethod
    def _candidate_failure(
        cls,
        category: str,
        started_ns: int,
        returncode: int | None,
        stdout: bytes,
        stderr: bytes,
        *,
        stdout_truncated: bool = False,
        stderr_truncated: bool = False,
        resource_event: str | None = None,
        host_cpu_usage_us: int = 0,
        host_cpu_before_usage_us: int = 0,
        host_cpu_ready_usage_us: int = 0,
        host_cpu_cross_usage_us: int = 0,
        host_cpu_after_usage_us: int = 0,
        host_cpu_budget_us: int = 0,
        host_memory_peak_bytes: int = 0,
        host_pids_peak: int = 0,
        host_memory_events_before: dict[str, int] | None = None,
        host_memory_events_after: dict[str, int] | None = None,
        host_pids_events_before: dict[str, int] | None = None,
        host_pids_events_after: dict[str, int] | None = None,
        guest_cpu_usage_us: int = 0,
        guest_process_peak: int = 0,
        guest_process_limit: int = PID_CAP,
        guest_rlimit_nproc: int = PINNED_GVISOR_RLIMIT_NPROC,
        guest_process_limit_syscall: int | None = None,
        guest_file_size_limit_bytes: int = FILE_SIZE_CAP_BYTES,
        guest_writable_limit_bytes: int = WRITABLE_OVERLAY_CAP_BYTES,
        guest_file_limit_signal: int | None = None,
        guest_file_limit_errno: int | None = None,
        guest_file_size_observed_bytes: int = 0,
        guest_writable_available_bytes: int = 0,
        resource_evidence_source: str | None = None,
    ) -> dict[str, Any]:
        return cls._base_result(
            "candidate_failure",
            category,
            started_ns,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            resource_event=resource_event,
            host_cpu_usage_us=host_cpu_usage_us,
            host_cpu_before_usage_us=host_cpu_before_usage_us,
            host_cpu_ready_usage_us=host_cpu_ready_usage_us,
            host_cpu_cross_usage_us=host_cpu_cross_usage_us,
            host_cpu_after_usage_us=host_cpu_after_usage_us,
            host_cpu_budget_us=host_cpu_budget_us,
            host_memory_peak_bytes=host_memory_peak_bytes,
            host_pids_peak=host_pids_peak,
            host_memory_events_before=host_memory_events_before,
            host_memory_events_after=host_memory_events_after,
            host_pids_events_before=host_pids_events_before,
            host_pids_events_after=host_pids_events_after,
            guest_cpu_usage_us=guest_cpu_usage_us,
            guest_process_peak=guest_process_peak,
            guest_process_limit=guest_process_limit,
            guest_rlimit_nproc=guest_rlimit_nproc,
            guest_process_limit_syscall=guest_process_limit_syscall,
            guest_file_size_limit_bytes=guest_file_size_limit_bytes,
            guest_writable_limit_bytes=guest_writable_limit_bytes,
            guest_file_limit_signal=guest_file_limit_signal,
            guest_file_limit_errno=guest_file_limit_errno,
            guest_file_size_observed_bytes=guest_file_size_observed_bytes,
            guest_writable_available_bytes=guest_writable_available_bytes,
            resource_evidence_source=resource_evidence_source,
        )

    @classmethod
    def _unknown(
        cls,
        category: str,
        started_ns: int,
        *,
        returncode: int | None = None,
        stderr: bytes = b"",
        controller_error: str | None = None,
        resource_event: str | None = None,
        host_cpu_usage_us: int = 0,
        host_cpu_before_usage_us: int = 0,
        host_cpu_ready_usage_us: int = 0,
        host_cpu_cross_usage_us: int = 0,
        host_cpu_after_usage_us: int = 0,
        host_cpu_budget_us: int = 0,
        host_memory_peak_bytes: int = 0,
        host_pids_peak: int = 0,
        host_memory_events_before: dict[str, int] | None = None,
        host_memory_events_after: dict[str, int] | None = None,
        host_pids_events_before: dict[str, int] | None = None,
        host_pids_events_after: dict[str, int] | None = None,
        guest_cpu_usage_us: int = 0,
        guest_process_peak: int = 0,
        guest_process_limit: int = PID_CAP,
        guest_rlimit_nproc: int = PINNED_GVISOR_RLIMIT_NPROC,
        guest_process_limit_syscall: int | None = None,
        guest_file_size_limit_bytes: int = FILE_SIZE_CAP_BYTES,
        guest_writable_limit_bytes: int = WRITABLE_OVERLAY_CAP_BYTES,
        guest_file_limit_signal: int | None = None,
        guest_file_limit_errno: int | None = None,
        guest_file_size_observed_bytes: int = 0,
        guest_writable_available_bytes: int = 0,
        resource_evidence_source: str | None = None,
    ) -> dict[str, Any]:
        return cls._base_result(
            "unknown",
            category,
            started_ns,
            returncode=returncode,
            stdout=b"",
            stderr=stderr[:STDERR_CAP_BYTES],
            stderr_truncated=len(stderr) > STDERR_CAP_BYTES,
            controller_error=controller_error,
            resource_event=resource_event,
            host_cpu_usage_us=host_cpu_usage_us,
            host_cpu_before_usage_us=host_cpu_before_usage_us,
            host_cpu_ready_usage_us=host_cpu_ready_usage_us,
            host_cpu_cross_usage_us=host_cpu_cross_usage_us,
            host_cpu_after_usage_us=host_cpu_after_usage_us,
            host_cpu_budget_us=host_cpu_budget_us,
            host_memory_peak_bytes=host_memory_peak_bytes,
            host_pids_peak=host_pids_peak,
            host_memory_events_before=host_memory_events_before,
            host_memory_events_after=host_memory_events_after,
            host_pids_events_before=host_pids_events_before,
            host_pids_events_after=host_pids_events_after,
            guest_cpu_usage_us=guest_cpu_usage_us,
            guest_process_peak=guest_process_peak,
            guest_process_limit=guest_process_limit,
            guest_rlimit_nproc=guest_rlimit_nproc,
            guest_process_limit_syscall=guest_process_limit_syscall,
            guest_file_size_limit_bytes=guest_file_size_limit_bytes,
            guest_writable_limit_bytes=guest_writable_limit_bytes,
            guest_file_limit_signal=guest_file_limit_signal,
            guest_file_limit_errno=guest_file_limit_errno,
            guest_file_size_observed_bytes=guest_file_size_observed_bytes,
            guest_writable_available_bytes=guest_writable_available_bytes,
            resource_evidence_source=resource_evidence_source,
        )
