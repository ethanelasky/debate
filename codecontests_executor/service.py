"""Bounded authenticated HTTP service for the CodeContests executor."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import socket
import stat
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .protocol import (
    CONFIGURED_VM_VCPUS,
    FILE_SIZE_CAP_BYTES,
    MAX_REQUEST_BODY_BYTES,
    MAX_RESULT_TIMING_NS,
    MEMORY_EVENT_KEYS,
    MINIMUM_GUEST_MEMORY_BYTES,
    PID_CAP,
    PIDS_EVENT_KEYS,
    PINNED_GVISOR_RLIMIT_NPROC,
    PROTOCOL_VERSION,
    REQUIRED_PYTHON_PACKAGES,
    ROOTFS_SHA256,
    RUNSC_RELEASE_ARCHIVE_SHA512,
    RUNSC_SHA512,
    SEMANTIC_ADDRESS_SPACE_BYTES,
    SERVICE_CPU_AFFINITY_VCPUS,
    SERVICE_MEMORY_MAX_BYTES,
    WRITABLE_OVERLAY_CAP_BYTES,
    ExecutorProtocolError,
    canonical_json,
    encode_envelope,
    make_execute_request,
    payload_digest,
    sign_payload,
    static_identity,
    strict_json_loads,
    validate_execute_request,
    validate_execution_evidence,
    verify_envelope,
)
from .supervisor import SandboxSupervisor

DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
ACTIVE_SANDBOXES = 4
DEFAULT_QUEUE_CAPACITY = 16
DEFAULT_QUEUE_WAIT_NS = 10 * 1_000_000_000
ROOTFS_MANIFEST_PATH = "/var/lib/codecontests-executor/rootfs.manifest.json"
ROOTFS_MANIFEST_FORMAT = "palaestra.codecontests.rootfs-manifest.v1"
SERVER_BUNDLE_FILES = (
    "__init__.py",
    "cgroup_gate.py",
    "client.py",
    "protocol.py",
    "sandbox_launcher.py",
    "service.py",
    "supervisor.py",
)
EXPECTED_HOST_VCPUS = SERVICE_CPU_AFFINITY_VCPUS


def _validated_listener(host: str, port: int) -> tuple[str, int]:
    if host != DEFAULT_BIND_HOST or port != DEFAULT_PORT:
        raise RuntimeError("executor listener must be exactly http://127.0.0.1:8080")
    return host, port


def _read_root_owned_nofollow(
    path_value: str,
    *,
    label: str,
    max_bytes: int,
    secret_mode: bool,
) -> bytes:
    if os.path.realpath(path_value) != os.path.abspath(path_value):
        raise RuntimeError(f"{label} path uses a symlink")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("O_NOFOLLOW is unavailable")
    fd = os.open(path_value, os.O_RDONLY | os.O_CLOEXEC | nofollow)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != 0:
            raise RuntimeError(f"{label} must be a root-owned regular file")
        forbidden = (
            stat.S_IRWXG | stat.S_IRWXO if secret_mode else stat.S_IWGRP | stat.S_IWOTH
        )
        if before.st_mode & forbidden:
            raise RuntimeError(f"{label} has unsafe permissions")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(fd, min(64 * 1024, max_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise RuntimeError(f"{label} exceeds size limit")
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
            raise RuntimeError(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _read_secret(*, file_env: str, value_env: str, label: str) -> bytes:
    file_value = os.environ.get(file_env)
    direct_value = os.environ.get(value_env)
    if bool(file_value) == bool(direct_value):
        raise RuntimeError(f"set exactly one of {file_env} or {value_env} for {label}")
    if file_value:
        value = _read_root_owned_nofollow(
            file_value,
            label=f"{label} secret",
            max_bytes=64 * 1024,
            secret_mode=True,
        ).rstrip(b"\r\n")
    else:
        value = direct_value.encode("utf-8") if direct_value is not None else b""
    if len(value) < 32:
        raise RuntimeError(f"{label} secret must contain at least 32 bytes")
    return value


def _metadata_stable(before: os.stat_result, after: os.stat_result) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return all(getattr(before, field) == getattr(after, field) for field in fields)


def _xattr_record(path: str, *, follow_symlinks: bool = False) -> dict[str, str]:
    try:
        names = sorted(
            os.listxattr(  # pyright: ignore[reportAttributeAccessIssue]
                path, follow_symlinks=follow_symlinks
            )
        )
        return {
            name: base64.b64encode(
                os.getxattr(  # pyright: ignore[reportAttributeAccessIssue]
                    path, name, follow_symlinks=follow_symlinks
                )
            ).decode("ascii")
            for name in names
        }
    except OSError as exc:
        raise RuntimeError(f"cannot measure rootfs xattrs: {path}") from exc


def build_rootfs_manifest(root_path: str) -> dict[str, Any]:
    """Measure every rootfs path, with stable per-file content evidence."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("O_NOFOLLOW is unavailable")
    root_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    # /proc/self/fd/N is already a pinned directory descriptor from the
    # supervisor; ordinary deploy paths must never follow a root symlink.
    if not root_path.startswith("/proc/self/fd/"):
        root_flags |= nofollow
    root_fd = os.open(root_path, root_flags)
    records: list[dict[str, Any]] = []

    def walk(directory_fd: int, relative_parent: str) -> None:
        directory_before = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_before.st_mode):
            raise RuntimeError("rootfs directory descriptor changed type")
        with os.scandir(directory_fd) as iterator:
            names = sorted(entry.name for entry in iterator)
        for name in names:
            relative = f"{relative_parent}/{name}" if relative_parent else name
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            proc_path = f"/proc/self/fd/{directory_fd}/{name}"
            record: dict[str, Any] = {
                "path": relative,
                "mode": stat.S_IMODE(metadata.st_mode),
                "uid": metadata.st_uid,
                "gid": metadata.st_gid,
                "xattrs": _xattr_record(proc_path),
            }
            if stat.S_ISDIR(metadata.st_mode):
                record["type"] = "directory"
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | nofollow,
                    dir_fd=directory_fd,
                )
                try:
                    if not _metadata_stable(metadata, os.fstat(child_fd)):
                        raise RuntimeError("rootfs directory changed before descent")
                    records.append(record)
                    walk(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode):
                record["type"] = "file"
                record["size"] = metadata.st_size
                file_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | nofollow,
                    dir_fd=directory_fd,
                )
                try:
                    opened = os.fstat(file_fd)
                    if not _metadata_stable(metadata, opened):
                        raise RuntimeError("rootfs file changed before measurement")
                    content = hashlib.sha256()
                    for chunk in iter(
                        lambda descriptor=file_fd: os.read(
                            descriptor, 1024 * 1024
                        ),
                        b"",
                    ):
                        content.update(chunk)
                    if not _metadata_stable(opened, os.fstat(file_fd)):
                        raise RuntimeError("rootfs file changed during measurement")
                    record["sha256"] = content.hexdigest()
                finally:
                    os.close(file_fd)
                records.append(record)
            elif stat.S_ISLNK(metadata.st_mode):
                record["type"] = "symlink"
                record["target"] = os.readlink(name, dir_fd=directory_fd)
                records.append(record)
            elif stat.S_ISCHR(metadata.st_mode):
                record["type"] = "character"
                record["rdev"] = metadata.st_rdev
                records.append(record)
            elif stat.S_ISBLK(metadata.st_mode):
                record["type"] = "block"
                record["rdev"] = metadata.st_rdev
                records.append(record)
            elif stat.S_ISFIFO(metadata.st_mode):
                record["type"] = "fifo"
                records.append(record)
            elif stat.S_ISSOCK(metadata.st_mode):
                record["type"] = "socket"
                records.append(record)
            else:
                raise RuntimeError(f"unsupported rootfs entry: {relative}")
            after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not _metadata_stable(metadata, after):
                raise RuntimeError("rootfs entry changed during measurement")
        if not _metadata_stable(directory_before, os.fstat(directory_fd)):
            raise RuntimeError("rootfs directory changed during measurement")

    try:
        root_before = os.fstat(root_fd)
        records.append(
            {
                "path": ".",
                "type": "directory",
                "mode": stat.S_IMODE(root_before.st_mode),
                "uid": root_before.st_uid,
                "gid": root_before.st_gid,
                "xattrs": _xattr_record(
                    f"/proc/self/fd/{root_fd}", follow_symlinks=True
                ),
            }
        )
        walk(root_fd, "")
        if not _metadata_stable(root_before, os.fstat(root_fd)):
            raise RuntimeError("rootfs root changed during measurement")
    finally:
        os.close(root_fd)
    records.sort(key=lambda record: record["path"])
    return {
        "format": ROOTFS_MANIFEST_FORMAT,
        "artifact_sha256": ROOTFS_SHA256,
        "entries": records,
    }


def normalized_rootfs_sha256(root_path: str) -> str:
    return payload_digest(build_rootfs_manifest(root_path))


def measured_server_bundle_sha256() -> str:
    package_dir = Path(__file__).absolute().parent
    actual_entries = tuple(sorted(path.name for path in package_dir.iterdir()))
    if actual_entries != tuple(sorted(SERVER_BUNDLE_FILES)):
        raise RuntimeError("executor package has missing/extra/cache entries")
    digest = hashlib.sha256()
    for filename in SERVER_BUNDLE_FILES:
        content = _read_root_owned_nofollow(
            str(package_dir / filename),
            label=f"server bundle {filename}",
            max_bytes=2 * 1024 * 1024,
            secret_mode=False,
        )
        digest.update(canonical_json({"path": filename, "size": len(content)}))
        digest.update(b"\n")
        digest.update(content)
        digest.update(b"\n")
    return digest.hexdigest()


def verify_host_capacity(
    *,
    memory_max_path: str | None = None,
    memory_swap_max_path: str | None = None,
    pids_max_path: str | None = None,
    cpu_max_path: str | None = None,
    delegated_root_procs_path: str | None = None,
    proc_swaps_path: str = "/proc/swaps",
    userns_clone_path: str = "/proc/sys/kernel/unprivileged_userns_clone",
    max_userns_path: str = "/proc/sys/user/max_user_namespaces",
    apparmor_userns_path: str = (
        "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
    ),
) -> dict[str, Any]:
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("CPU affinity measurement unavailable")
    cpu_affinity = sorted(
        os.sched_getaffinity(0)  # pyright: ignore[reportAttributeAccessIssue]
    )
    if len(cpu_affinity) != EXPECTED_HOST_VCPUS:
        raise RuntimeError("executor CPU capacity is not exactly 4 vCPU")
    if os.cpu_count() != CONFIGURED_VM_VCPUS:
        raise RuntimeError("executor VM is not exactly the configured 16 vCPU")
    if (
        memory_max_path is None
        or memory_swap_max_path is None
        or pids_max_path is None
        or cpu_max_path is None
        or delegated_root_procs_path is None
    ):
        unified_path: str | None = None
        for line in Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines():
            hierarchy, controllers, path = line.split(":", 2)
            if hierarchy == "0" and controllers == "":
                unified_path = path
                break
        if unified_path is None:
            raise RuntimeError("cgroup v2 service path is unavailable")
        current = Path("/sys/fs/cgroup") / unified_path.lstrip("/")
        if current.name != "service":
            raise RuntimeError("DelegateSubgroup=service is not active")
        service_cgroup = current.parent
        memory_max_path = memory_max_path or str(service_cgroup / "memory.max")
        memory_swap_max_path = memory_swap_max_path or str(
            service_cgroup / "memory.swap.max"
        )
        pids_max_path = pids_max_path or str(service_cgroup / "pids.max")
        cpu_max_path = cpu_max_path or str(service_cgroup / "cpu.max")
        delegated_root_procs_path = delegated_root_procs_path or str(
            service_cgroup / "cgroup.procs"
        )
    memory_max = Path(memory_max_path).read_text(encoding="ascii").strip()
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    cgroup_ceiling: int | str = "max"
    if memory_max != "max":
        try:
            cgroup_ceiling = int(memory_max, 10)
        except ValueError as exc:
            raise RuntimeError("invalid cgroup memory ceiling") from exc
    if cgroup_ceiling != SERVICE_MEMORY_MAX_BYTES:
        raise RuntimeError("executor service memory ceiling is not exact 24 GiB")
    try:
        service_swap_max = int(
            Path(memory_swap_max_path).read_text(encoding="ascii").strip(), 10
        )
        service_tasks_max = int(
            Path(pids_max_path).read_text(encoding="ascii").strip(), 10
        )
    except ValueError as exc:
        raise RuntimeError("invalid service cgroup cap") from exc
    if service_swap_max != 0 or service_tasks_max != 768:
        raise RuntimeError("service swap/tasks cap drifted")
    cpu_fields = Path(cpu_max_path).read_text(encoding="ascii").split()
    if cpu_fields != ["400000", "100000"]:
        raise RuntimeError("service CPUQuota is not exactly 400%")
    delegated_root_procs_empty = (
        Path(delegated_root_procs_path).read_text(encoding="ascii").strip() == ""
    )
    if not delegated_root_procs_empty:
        raise RuntimeError("delegated root cgroup is not process-free")
    swaps_lines = Path(proc_swaps_path).read_text(encoding="ascii").splitlines()
    guest_swap_enabled = len(swaps_lines) > 1
    if guest_swap_enabled:
        raise RuntimeError("guest swap must be disabled")
    if physical < MINIMUM_GUEST_MEMORY_BYTES:
        raise RuntimeError("executor guest-visible memory is below minimum")
    try:
        userns_clone = int(
            Path(userns_clone_path).read_text(encoding="ascii").strip(), 10
        )
    except ValueError as exc:
        raise RuntimeError("invalid unprivileged-userns sysctl") from exc
    if userns_clone != 1:
        raise RuntimeError("unprivileged user namespaces are disabled")
    try:
        max_userns = int(Path(max_userns_path).read_text(encoding="ascii").strip(), 10)
    except ValueError as exc:
        raise RuntimeError("invalid user namespace sysctl") from exc
    if max_userns < ACTIVE_SANDBOXES:
        raise RuntimeError("insufficient user namespace capacity")
    try:
        apparmor_restrict = int(
            Path(apparmor_userns_path).read_text(encoding="ascii").strip(), 10
        )
    except ValueError as exc:
        raise RuntimeError("invalid AppArmor userns sysctl") from exc
    if apparmor_restrict != 0:
        raise RuntimeError("AppArmor restricts unprivileged user namespaces")
    return {
        "cpu_affinity_count": len(cpu_affinity),
        "cpu_affinity_cpus": cpu_affinity,
        "cgroup_memory_ceiling_bytes": cgroup_ceiling,
        "guest_visible_memory_bytes": physical,
        "unprivileged_userns_clone": userns_clone,
        "max_user_namespaces": max_userns,
        "apparmor_restrict_unprivileged_userns": apparmor_restrict,
        "service_memory_max_bytes": cgroup_ceiling,
        "service_memory_swap_max_bytes": service_swap_max,
        "service_tasks_max": service_tasks_max,
        "service_cpu_quota_us": 400_000,
        "service_cpu_period_us": 100_000,
        "delegated_root_cgroup_procs_empty": delegated_root_procs_empty,
        "guest_swap_enabled": guest_swap_enabled,
    }


def _verify_runsc_version(supervisor: SandboxSupervisor) -> None:
    completed = subprocess.run(
        [supervisor.frozen_runsc_path, "--version"],
        capture_output=True,
        check=False,
        timeout=5,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        pass_fds=supervisor.runtime_pass_fds,
        text=True,
    )
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    if completed.returncode != 0 or first_line != "runsc version release-20260721.0":
        raise RuntimeError("pinned runsc release mismatch")


def _gvisor_processes() -> list[int]:
    leftovers: list[int] = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        if any(marker in cmdline for marker in (b"runsc", b"gofer", b"sandbox")):
            leftovers.append(int(entry.name))
    return leftovers


def live_sandbox_self_test(
    supervisor: SandboxSupervisor, *, rootfs_manifest_sha256: str
) -> None:
    request_root = Path(supervisor.config.request_root)
    if any(request_root.iterdir()):
        raise RuntimeError("executor request root is not empty before self-test")
    if _gvisor_processes():
        raise RuntimeError("gVisor process exists before self-test")
    probe_code = """\
import ctypes
import json
import os
import socket
import sys
import mpmath
import sympy
write_blocked = False
exit_works = False
try:
    with open("/tmp/executor-self-test-write", "w", encoding="ascii") as handle:
        handle.write("unsafe")
except OSError:
    write_blocked = True
try:
    exit()
except SystemExit:
    exit_works = True
print(json.dumps({
    "python": list(sys.version_info[:3]),
    "uid": list(os.getresuid()),
    "gid": list(os.getresgid()),
    "groups": os.getgroups(),
    "interfaces": sorted(name for _index, name in socket.if_nameindex()),
    "no_new_privs": ctypes.CDLL(None).prctl(39, 0, 0, 0, 0),
    "packages": {"mpmath": mpmath.__version__, "sympy": sympy.__version__},
    "site_exit_works": exit_works,
    "safe_import_path": "" not in sys.path and "/tmp" not in sys.path,
    "tmp_write_blocked": write_blocked,
}, sort_keys=True))
"""
    request = make_execute_request(
        code=probe_code,
        stdin="",
        raw_limits={
            "time_limit": {"seconds": 2, "nanos": 0},
            "memory_limit_bytes": SEMANTIC_ADDRESS_SPACE_BYTES,
        },
        identity_digest_value="0" * 64,
        ttl_ns=30 * 1_000_000_000,
    )
    result = supervisor.execute(request)
    if result.get("outcome") != "executed":
        try:
            failure_stderr = base64.b64decode(
                result.get("stderr_b64", ""), validate=True
            )
        except (ValueError, TypeError):
            failure_stderr = b"<invalid-stderr-evidence>"
        bounded_stderr = failure_stderr[:4096].decode(
            "utf-8", errors="backslashreplace"
        )
        raise RuntimeError(
            "live sandbox self-test failed: "
            f"category={result.get('category')} "
            f"returncode={result.get('returncode')} "
            f"controller_error={result.get('controller_error')} "
            f"resource_event={result.get('resource_event')} "
            f"stderr={bounded_stderr!r}"
        )
    try:
        stdout = base64.b64decode(result["stdout_b64"], validate=True)
        evidence = json.loads(stdout)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid live sandbox self-test evidence") from exc
    if evidence != {
        "python": [3, 12, 3],
        "uid": [65534, 65534, 65534],
        "gid": [65534, 65534, 65534],
        "groups": [],
        "interfaces": ["lo"],
        "no_new_privs": 1,
        "packages": REQUIRED_PYTHON_PACKAGES,
        "site_exit_works": True,
        "safe_import_path": True,
        "tmp_write_blocked": True,
    }:
        raise RuntimeError("live sandbox identity/version/network mismatch")
    if any(request_root.iterdir()) or _gvisor_processes():
        raise RuntimeError("sandbox teardown left process/state remnants")
    if (
        normalized_rootfs_sha256(supervisor.frozen_rootfs_path)
        != rootfs_manifest_sha256
    ):
        raise RuntimeError("rootfs changed during live self-test")


def verify_pinned_runtime(
    supervisor: SandboxSupervisor,
    *,
    rootfs_manifest_path: str = ROOTFS_MANIFEST_PATH,
    expected_manifest_file_sha256: str,
) -> tuple[str, str]:
    supervisor.validate_host_files()
    supervisor.freeze_runtime(
        expected_runsc_sha512=RUNSC_SHA512,
        expected_release_archive_sha512=RUNSC_RELEASE_ARCHIVE_SHA512,
    )
    manifest_bytes = _read_root_owned_nofollow(
        rootfs_manifest_path,
        label="rootfs manifest",
        max_bytes=256 * 1024 * 1024,
        secret_mode=False,
    )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_file_sha256):
        raise RuntimeError("expected manifest-file SHA-256 is invalid")
    manifest_file_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if not hmac.compare_digest(manifest_file_sha256, expected_manifest_file_sha256):
        raise RuntimeError("deploy-pinned manifest-file SHA-256 mismatch")
    try:
        expected_manifest = strict_json_loads(manifest_bytes)
    except ExecutorProtocolError as exc:
        raise RuntimeError("invalid rootfs manifest JSON") from exc
    if (
        not isinstance(expected_manifest, dict)
        or set(expected_manifest) != {"format", "artifact_sha256", "entries"}
        or expected_manifest["format"] != ROOTFS_MANIFEST_FORMAT
        or not isinstance(expected_manifest["entries"], list)
    ):
        raise RuntimeError("rootfs manifest fields mismatch")
    if expected_manifest["artifact_sha256"] != ROOTFS_SHA256:
        raise RuntimeError("pinned rootfs SHA-256 mismatch")
    if manifest_bytes != canonical_json(expected_manifest) + b"\n":
        raise RuntimeError("rootfs manifest is not exact canonical JSON")
    first_measurement = build_rootfs_manifest(supervisor.frozen_rootfs_path)
    second_measurement = build_rootfs_manifest(supervisor.frozen_rootfs_path)
    if first_measurement != second_measurement:
        raise RuntimeError("rootfs changed between stable measurement passes")
    if first_measurement != expected_manifest:
        raise RuntimeError("rootfs per-file manifest mismatch")
    return payload_digest(expected_manifest), manifest_file_sha256


class ExecutorApplication:
    """Protocol/auth/admission layer, separated for direct behavior probes."""

    def __init__(
        self,
        *,
        bearer_token: bytes,
        hmac_key: bytes,
        identity: dict[str, Any],
        supervisor: SandboxSupervisor,
        queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
        queue_wait_ns: int = DEFAULT_QUEUE_WAIT_NS,
    ):
        if len(bearer_token) < 32 or len(hmac_key) < 32:
            raise ValueError("application secrets must contain at least 32 bytes")
        try:
            bearer_token.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("bearer token must be ASCII") from exc
        self._bearer_token = bearer_token
        self._hmac_key = hmac_key
        self.identity = identity
        self.identity_digest = payload_digest(identity)
        self.supervisor = supervisor
        self.queue_wait_ns = queue_wait_ns
        self._active = threading.BoundedSemaphore(ACTIVE_SANDBOXES)
        self._admitted = threading.BoundedSemaphore(ACTIVE_SANDBOXES + queue_capacity)

    def authorized(self, authorization: str | None) -> bool:
        if not isinstance(authorization, str) or not authorization.isascii():
            return False
        supplied = authorization
        expected = "Bearer " + self._bearer_token.decode("ascii", errors="strict")
        try:
            return hmac.compare_digest(supplied, expected)
        except TypeError:
            return False

    def signed_identity(self) -> bytes:
        return encode_envelope(sign_payload(self.identity, self._hmac_key))

    def _result_payload(
        self,
        request: dict[str, Any],
        request_digest: str,
        execution: dict[str, Any],
        *,
        queued_ns: int,
        total_ns: int,
    ) -> dict[str, Any]:
        execution_copy = dict(execution)
        execution_ns = int(execution_copy.pop("execution_ns", 0))
        return {
            "kind": "execute_result",
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request["request_id"],
            "request_digest": request_digest,
            "identity_digest": self.identity_digest,
            "limits": request["task"]["limits"],
            "outcome": execution_copy.pop("outcome"),
            "category": execution_copy.pop("category"),
            "retryable": execution_copy.pop("retryable"),
            "timing": {
                "queue_ns": queued_ns,
                "execution_ns": execution_ns,
                "total_ns": total_ns,
            },
            "evidence": execution_copy,
        }

    def _signed_protocol_error(self, request_digest: str) -> bytes:
        return encode_envelope(
            sign_payload(
                {
                    "kind": "protocol_error",
                    "protocol_version": PROTOCOL_VERSION,
                    "request_digest": request_digest,
                    "identity_digest": self.identity_digest,
                    "category": "PROTOCOL_REJECTED",
                },
                self._hmac_key,
            )
        )

    @staticmethod
    def _normalize_execution(
        execution: Any, expected_limits: dict[str, Any]
    ) -> dict[str, Any]:
        expected = {
            "outcome",
            "category",
            "retryable",
            "stdout_b64",
            "stderr_b64",
            "stdout_bytes",
            "stderr_bytes",
            "stdout_truncated",
            "stderr_truncated",
            "returncode",
            "signal",
            "controller_error",
            "resource_event",
            "host_cpu_usage_us",
            "host_cpu_before_usage_us",
            "host_cpu_ready_usage_us",
            "host_cpu_cross_usage_us",
            "host_cpu_after_usage_us",
            "host_cpu_budget_us",
            "host_memory_peak_bytes",
            "host_pids_peak",
            "host_memory_events_before",
            "host_memory_events_after",
            "host_pids_events_before",
            "host_pids_events_after",
            "guest_cpu_usage_us",
            "guest_process_peak",
            "guest_process_limit",
            "guest_rlimit_nproc",
            "guest_process_limit_syscall",
            "guest_file_size_limit_bytes",
            "guest_writable_limit_bytes",
            "guest_file_limit_signal",
            "guest_file_limit_errno",
            "guest_file_size_observed_bytes",
            "guest_writable_available_bytes",
            "resource_evidence_source",
            "execution_ns",
        }
        try:
            if not isinstance(execution, dict) or set(execution) != expected:
                raise ExecutorProtocolError("controller result fields mismatch")
            if (
                isinstance(execution["execution_ns"], bool)
                or not isinstance(execution["execution_ns"], int)
                or execution["execution_ns"] < 0
                or execution["execution_ns"] > MAX_RESULT_TIMING_NS
            ):
                raise ExecutorProtocolError("invalid controller execution timing")
            validate_execution_evidence(
                outcome=execution["outcome"],
                category=execution["category"],
                retryable=execution["retryable"],
                evidence={
                    key: execution[key]
                    for key in expected
                    if key
                    not in {
                        "outcome",
                        "category",
                        "retryable",
                        "execution_ns",
                    }
                },
                expected_limits=expected_limits,
            )
        except (ExecutorProtocolError, KeyError, TypeError, ValueError):
            return ExecutorApplication._controller_unknown(
                "CONTROLLER_RESULT_INVALID", "schema", expected_limits
            )
        return dict(execution)

    def _signed_result(
        self,
        request: dict[str, Any],
        request_digest: str,
        execution: dict[str, Any],
        *,
        queued_ns: int,
        started_ns: int,
    ) -> bytes:
        payload = self._result_payload(
            request,
            request_digest,
            execution,
            queued_ns=queued_ns,
            total_ns=time.monotonic_ns() - started_ns,
        )
        return encode_envelope(sign_payload(payload, self._hmac_key))

    @staticmethod
    def _overload(
        category: str, expected_limits: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        cpu_budget = (
            expected_limits["effective"]["host_cgroup_cpu_budget_us"]
            if expected_limits is not None
            else 0
        )
        return {
            "outcome": "unknown",
            "category": category,
            "retryable": True,
            "stdout_b64": "",
            "stderr_b64": "",
            "stdout_bytes": 0,
            "stderr_bytes": 0,
            "stdout_truncated": False,
            "stderr_truncated": False,
            "returncode": None,
            "signal": None,
            "controller_error": None,
            "resource_event": None,
            "host_cpu_usage_us": 0,
            "host_cpu_before_usage_us": 0,
            "host_cpu_ready_usage_us": 0,
            "host_cpu_cross_usage_us": 0,
            "host_cpu_after_usage_us": 0,
            "host_cpu_budget_us": cpu_budget,
            "host_memory_peak_bytes": 0,
            "host_pids_peak": 0,
            "host_memory_events_before": {key: 0 for key in MEMORY_EVENT_KEYS},
            "host_memory_events_after": {key: 0 for key in MEMORY_EVENT_KEYS},
            "host_pids_events_before": {key: 0 for key in PIDS_EVENT_KEYS},
            "host_pids_events_after": {key: 0 for key in PIDS_EVENT_KEYS},
            "guest_cpu_usage_us": 0,
            "guest_process_peak": 0,
            "guest_process_limit": PID_CAP,
            "guest_rlimit_nproc": PINNED_GVISOR_RLIMIT_NPROC,
            "guest_process_limit_syscall": None,
            "guest_file_size_limit_bytes": expected_limits["effective"][
                "file_size_bytes"
            ]
            if expected_limits is not None
            else FILE_SIZE_CAP_BYTES,
            "guest_writable_limit_bytes": expected_limits["effective"][
                "aggregate_writable_bytes"
            ]
            if expected_limits is not None
            else WRITABLE_OVERLAY_CAP_BYTES,
            "guest_file_limit_signal": None,
            "guest_file_limit_errno": None,
            "guest_file_size_observed_bytes": 0,
            "guest_writable_available_bytes": 0,
            "resource_evidence_source": None,
            "execution_ns": 0,
        }

    @staticmethod
    def _controller_unknown(
        category: str,
        error: str,
        expected_limits: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = ExecutorApplication._overload(category, expected_limits)
        result["retryable"] = False
        result["controller_error"] = error
        return result

    def execute(self, body: bytes) -> tuple[int, bytes]:
        started_ns = time.monotonic_ns()
        try:
            envelope = strict_json_loads(body)
            request = verify_envelope(
                envelope, self._hmac_key, expected_kind="execute_request"
            )
        except ExecutorProtocolError:
            return HTTPStatus.UNAUTHORIZED, b'{"error":"unauthorized"}'

        request_digest = payload_digest(request)
        try:
            request = validate_execute_request(
                request,
                expected_identity_digest=self.identity_digest,
                expected_client_provenance=self.identity["expected_client_provenance"],
            )
        except ExecutorProtocolError:
            return (
                HTTPStatus.BAD_REQUEST,
                self._signed_protocol_error(request_digest),
            )

        if not self._admitted.acquire(blocking=False):
            return HTTPStatus.SERVICE_UNAVAILABLE, self._signed_result(
                request,
                request_digest,
                self._overload("OVERLOADED", request["task"]["limits"]),
                queued_ns=0,
                started_ns=started_ns,
            )
        queue_started_ns = time.monotonic_ns()
        acquired_active = False
        try:
            remaining_request_ns = max(
                0, request["expires_at_unix_ns"] - time.time_ns()
            )
            wait_ns = min(self.queue_wait_ns, remaining_request_ns)
            acquired_active = self._active.acquire(timeout=wait_ns / 1_000_000_000)
            queued_ns = time.monotonic_ns() - queue_started_ns
            if not acquired_active:
                return HTTPStatus.SERVICE_UNAVAILABLE, self._signed_result(
                    request,
                    request_digest,
                    self._overload("QUEUE_DEADLINE", request["task"]["limits"]),
                    queued_ns=queued_ns,
                    started_ns=started_ns,
                )
            try:
                execution = self.supervisor.execute(request)
            except Exception as exc:  # noqa: BLE001 - controller trust boundary
                execution = self._controller_unknown(
                    "CONTROLLER_EXCEPTION",
                    type(exc).__name__,
                    request["task"]["limits"],
                )
            execution = self._normalize_execution(
                execution, request["task"]["limits"]
            )
            return HTTPStatus.OK, self._signed_result(
                request,
                request_digest,
                execution,
                queued_ns=queued_ns,
                started_ns=started_ns,
            )
        finally:
            if acquired_active:
                self._active.release()
            self._admitted.release()


class ExecutorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    # The authenticated loopback service is deliberately restarted after the
    # one-shot hostile probe.  That probe's live identity checks leave bounded
    # TCP TIME_WAIT sockets on guest port 8080; SO_REUSEADDR permits the exact
    # same loopback listener to return immediately without permitting a second
    # concurrent bind.
    allow_reuse_address = True
    request_queue_size = ACTIVE_SANDBOXES + DEFAULT_QUEUE_CAPACITY

    def __init__(self, address: tuple[str, int], application: ExecutorApplication):
        _validated_listener(*address)
        self.application = application
        self._connection_slots = threading.BoundedSemaphore(
            ACTIVE_SANDBOXES + DEFAULT_QUEUE_CAPACITY
        )
        super().__init__(address, ExecutorRequestHandler)

    def process_request(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, request: socket.socket, client_address: tuple[str, int]
    ) -> None:
        if not self._connection_slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connection_slots.release()
            raise

    def process_request_thread(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, request: socket.socket, client_address: tuple[str, int]
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_slots.release()


class ExecutorRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CodeContestsExecutor/1"
    sys_version = ""

    @property
    def app(self) -> ExecutorApplication:
        return self.server.application  # type: ignore[attr-defined,no-any-return]

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(5.0)

    def log_message(self, format: str, *args: Any) -> None:
        # Never log headers, bodies, code, stdin, signatures, or secrets.
        message = format % args
        print(f"executor_http peer={self.client_address[0]} {message}", flush=True)

    def _write(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _authenticate(self) -> bool:
        if self.app.authorized(self.headers.get("Authorization")):
            return True
        self._write(HTTPStatus.UNAUTHORIZED, b'{"error":"unauthorized"}')
        return False

    def do_GET(self) -> None:
        if not self._authenticate():
            return
        if self.path != "/v1/identity":
            self._write(HTTPStatus.NOT_FOUND, b'{"error":"not_found"}')
            return
        self._write(HTTPStatus.OK, self.app.signed_identity())

    def do_POST(self) -> None:
        if not self._authenticate():
            return
        if self.path != "/v1/execute":
            self._write(HTTPStatus.NOT_FOUND, b'{"error":"not_found"}')
            return
        if self.headers.get("Transfer-Encoding") is not None:
            self._write(
                HTTPStatus.BAD_REQUEST, b'{"error":"transfer_encoding_forbidden"}'
            )
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""), 10)
        except ValueError:
            self._write(HTTPStatus.BAD_REQUEST, b'{"error":"content_length"}')
            return
        if not 0 < content_length <= MAX_REQUEST_BODY_BYTES:
            self._write(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, b'{"error":"body_size"}')
            return
        try:
            body = self.rfile.read(content_length)
        except (OSError, TimeoutError):
            self.close_connection = True
            return
        if len(body) != content_length:
            self._write(HTTPStatus.BAD_REQUEST, b'{"error":"short_body"}')
            return
        status, response = self.app.execute(body)
        self._write(status, response)


def build_application_from_env() -> ExecutorApplication:
    bearer = _read_secret(
        file_env="CODECONTESTS_EXECUTOR_BEARER_FILE",
        value_env="CODECONTESTS_EXECUTOR_BEARER",
        label="bearer",
    )
    hmac_key = _read_secret(
        file_env="CODECONTESTS_EXECUTOR_HMAC_KEY_FILE",
        value_env="CODECONTESTS_EXECUTOR_HMAC_KEY",
        label="HMAC",
    )
    service_id = os.environ.get("CODECONTESTS_EXECUTOR_SERVICE_ID", "")
    client_provenance_path = os.environ.get(
        "CODECONTESTS_EXECUTOR_CLIENT_PROVENANCE_FILE", ""
    )
    if not client_provenance_path:
        raise RuntimeError("expected client-provenance file is absent")
    try:
        expected_client_provenance = strict_json_loads(
            _read_root_owned_nofollow(
                client_provenance_path,
                label="expected client provenance",
                max_bytes=64 * 1024,
                secret_mode=False,
            )
        )
    except ExecutorProtocolError as exc:
        raise RuntimeError("invalid expected client provenance") from exc
    if not isinstance(expected_client_provenance, dict):
        raise RuntimeError(  # noqa: TRY004 - malformed root-owned startup state
            "expected client provenance must be an object"
        )
    supervisor = SandboxSupervisor()
    expected_manifest_file_sha256 = os.environ.get(
        "CODECONTESTS_EXECUTOR_ROOTFS_MANIFEST_SHA256", ""
    )
    (
        rootfs_manifest_digest,
        rootfs_manifest_file_digest,
    ) = verify_pinned_runtime(
        supervisor,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
    )
    host_policy_measurement = verify_host_capacity()
    _verify_runsc_version(supervisor)
    live_sandbox_self_test(supervisor, rootfs_manifest_sha256=rootfs_manifest_digest)
    server_bundle_digest = measured_server_bundle_sha256()
    expected_server_bundle = os.environ.get(
        "CODECONTESTS_EXECUTOR_SERVER_BUNDLE_SHA256", ""
    )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_server_bundle):
        raise RuntimeError("expected server-bundle SHA-256 is invalid")
    if not hmac.compare_digest(server_bundle_digest, expected_server_bundle):
        raise RuntimeError("deploy-pinned server-bundle SHA-256 mismatch")
    identity = static_identity(
        service_id=service_id,
        launcher_sha256=supervisor.launcher_sha256,
        cgroup_gate_sha256=supervisor.cgroup_gate_sha256,
        rootfs_manifest_sha256=rootfs_manifest_digest,
        rootfs_manifest_file_sha256=rootfs_manifest_file_digest,
        server_bundle_sha256=server_bundle_digest,
        measured_guest_memory_bytes=host_policy_measurement[
            "guest_visible_memory_bytes"
        ],
        host_policy_measurement=host_policy_measurement,
        expected_client_provenance=expected_client_provenance,
        active_sandboxes=ACTIVE_SANDBOXES,
        queue_capacity=DEFAULT_QUEUE_CAPACITY,
    )
    return ExecutorApplication(
        bearer_token=bearer,
        hmac_key=hmac_key,
        identity=identity,
        supervisor=supervisor,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default=DEFAULT_BIND_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--print-identity",
        action="store_true",
        help="print unsigned static identity after pinned-runtime verification",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    bind_host, port = _validated_listener(args.bind, args.port)
    application = build_application_from_env()
    if args.print_identity:
        print(json.dumps(application.identity, sort_keys=True, indent=2))
        return
    server = ExecutorHTTPServer((bind_host, port), application)
    print(
        "executor_ready "
        f"bind={args.bind}:{args.port} identity={application.identity_digest}",
        flush=True,
    )
    server.serve_forever(poll_interval=0.25)


if __name__ == "__main__":
    main()
