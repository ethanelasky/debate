"""Canonical wire protocol and immutable CodeContests execution policy.

No repository or third-party imports are allowed here.  Both the credential
bearing client and the credential-free Ubuntu executor import this module.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import re
import time
import uuid
from collections.abc import Mapping
from typing import Any

PROTOCOL_VERSION = "palaestra.codecontests.executor.v2"
IMPLEMENTATION_VERSION = "4"

DEEP_MIND_SOURCE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/code_contests/"
    "fa7a4f8139aab08362503f3344778eb86901709a/execution/tester_sandboxer.cc"
)
DEEP_MIND_SOURCE_SHA256 = (
    "692660d2fc64f19d780bce3400af81eeb3f823a7bb7b33d8a3467ee930821580"
)
LIMIT_POLICY_VERSION = "palaestra-codecontests-local-parity-v1"

RUNSC_PATH = "/opt/gvisor/20260721.0/runsc"
RUNSC_RELEASE_ARCHIVE_PATH = "/opt/gvisor/20260721.0/gvisor.tar.bz2"
RUNSC_RELEASE = "20260721.0"
RUNSC_RELEASE_ARCH = "aarch64"
RUNSC_RELEASE_URL = (
    "https://storage.googleapis.com/gvisor/releases/release/"
    "20260721.0/aarch64/gvisor.tar.bz2"
)
RUNSC_RELEASE_ARCHIVE_SIZE_BYTES = 131_637_510
RUNSC_RELEASE_ARCHIVE_SHA512 = (
    "27a6d5103c36ef11c7e8c6158b7039ac43623d77147227d4e6d083835d0cd20f"
    "e3b100680ffeca9e6691ee7ba31de19b613c10ef860771e861e7971b24bc2947"
)
RUNSC_SIZE_BYTES = 96_196_105
RUNSC_SHA512 = (
    "23465f6a5d7c1da2c31ac25af95e0db783e2776f0fb2afb3a3c421b8928c51d"
    "7d4d3a680c555ff821d12504d51af985a226855f56d22dc258d3058b537995734"
)
RUNSC_SPEC_VERSION = "1.2.1"
RUNSC_VERSION_OUTPUT = "runsc version release-20260721.0\nspec: 1.2.1\n"
ROOTFS_PATH = "/var/lib/codecontests-executor/rootfs"
# Deterministic digest measured twice after offline installation of the exact
# package pins, root:root ownership normalization, and removal of every
# setuid/setgid bit.  The artifact builder independently rechecks imports,
# versions, ownership, archive metadata, and this digest before deployment.
ROOTFS_SHA256 = "83e694da5d1e0b94700da2a195d760527ce609ea631f7302ec930666bae136d0"
PYTHON_VERSION = "3.12.3"
REQUIRED_PYTHON_PACKAGES = {
    "mpmath": "1.3.0",
    "sympy": "1.14.0",
}
TRUSTED_MONITOR_BOOTSTRAP_CAPABILITIES = (
    "CAP_KILL",
    "CAP_SETGID",
    "CAP_SETPCAP",
    "CAP_SETUID",
    "CAP_SYS_PTRACE",
)

MIB = 1024 * 1024
SEMANTIC_ADDRESS_SPACE_BYTES = 4 * 1024**3
ADDRESS_SPACE_OVERHEAD_BYTES = 0
STDOUT_CAP_BYTES = 2 * MIB
STDERR_CAP_BYTES = 2 * MIB
FILE_SIZE_CAP_BYTES = 2 * MIB
WRITABLE_OVERLAY_CAP_BYTES = 0
READ_ONLY_TMPFS_SIZE_BYTES = 64 * 1024
# Candidate main plus one concurrent same-process pthread.  Pinned gVisor does
# not charge the pre-UID-transition main task to RLIMIT_NPROC, hence raw=1.
PID_CAP = 2
PINNED_GVISOR_RLIMIT_NPROC = PID_CAP - 1
NOFILE_CAP = 64
MIN_SOURCE_MEMORY_BYTES = SEMANTIC_ADDRESS_SPACE_BYTES
MAX_SOURCE_MEMORY_BYTES = SEMANTIC_ADDRESS_SPACE_BYTES
HOST_REQUEST_PID_CAP = 128
RUNSC_INFRA_MEMORY_HEADROOM_BYTES = 384 * MIB
SERVICE_MEMORY_MAX_BYTES = 24 * 1024**3
CONFIGURED_VM_MEMORY_BYTES = 32 * 1024**3
MINIMUM_GUEST_MEMORY_BYTES = 30_000_000_000
CONFIGURED_VM_VCPUS = 16
SERVICE_CPU_AFFINITY_VCPUS = 4
HOST_CPU_PERIOD_US = 100_000
HOST_CPU_QUOTA_US = 100_000
HOST_CPU_FAILSAFE_MULTIPLIER = 2
HOST_CPU_FAILSAFE_FIXED_US = 1_000_000
HOST_CPU_CALIBRATION_SHA256 = (
    "72c48e00d27b13bc50aeb682d7bf01ab15d4a1a63e68d8abaca85125962cc411"
)
WALL_CEILING_NS = 120 * 1_000_000_000
MAX_SOURCE_DURATION_NS = WALL_CEILING_NS
# Transport-only allowance: the service may wait up to 10 seconds for an
# active slot, followed by a 5-second response-delivery margin.  This is not
# added to the signed candidate wall-time limit enforced inside the sandbox.
CLIENT_HTTP_OVERHEAD_SECONDS = 15
# The reverse tunnel is supervised independently from the verifier client.  A
# request whose response is lost must remain replayable while that supervisor
# rides out a transient host-network outage.  This is deliberately finite and
# is part of the frozen protocol policy rather than an open-ended client retry.
TRANSPORT_RECOVERY_WINDOW_SECONDS = 180
# A retry that begins just before the recovery deadline still needs bounded
# time to re-attest and deliver the cached signed response.  Without this
# explicit margin, max-wall execution plus max clock skew lands exactly on the
# request-expiry/prune boundary.
REPLAY_DELIVERY_MARGIN_SECONDS = 15
MAX_INT64 = (1 << 63) - 1
MAX_UINT64 = (1 << 64) - 1

MAX_CODE_BYTES = 1 * MIB
# Frozen-campaign evidence has a maximum stdin below 500 KiB.  Keep one
# additional factor of two while preventing the twenty pre-admission
# connections from retaining multi-gigabyte request bodies on the 8 GiB VM.
MAX_STDIN_BYTES = 1 * MIB
MAX_REQUEST_BODY_BYTES = (
    4 * ((MAX_CODE_BYTES + 2) // 3) + 4 * ((MAX_STDIN_BYTES + 2) // 3) + (256 * 1024)
)
MAX_RESPONSE_BODY_BYTES = 6 * MIB
MAX_CLOCK_SKEW_NS = 30 * 1_000_000_000
# A request is created only after initial identity attestation.  Its validity
# covers one maximum execution, HTTP/queue overhead, a full tunnel-recovery
# window after response loss, final replay delivery, and the permitted
# client/server clock skew.
DEFAULT_EXECUTE_REQUEST_TTL_NS = (
    (
        WALL_CEILING_NS // 1_000_000_000
        + CLIENT_HTTP_OVERHEAD_SECONDS
        + TRANSPORT_RECOVERY_WINDOW_SECONDS
        + REPLAY_DELIVERY_MARGIN_SECONDS
    )
    * 1_000_000_000
    + MAX_CLOCK_SKEW_NS
)
MAX_REQUEST_TTL_NS = DEFAULT_EXECUTE_REQUEST_TTL_NS
# Completed responses remain available briefly beyond signed request expiry.
# New work is never admitted from an expired request, but an exact authenticated
# replay can still recover the response to work that already ran.
REPLAY_CACHE_GRACE_NS = TRANSPORT_RECOVERY_WINDOW_SECONDS * 1_000_000_000
REPLAY_CACHE_CAPACITY = 8192
REPLAY_CACHE_BYTES = 256 * MIB

CANDIDATE_FAILURE_CATEGORIES = frozenset(
    {
        "CPU_LIMIT",
        "FILE_SPACE_LIMIT",
        "OUTPUT_LIMIT",
        "PROCESS_LIMIT",
        "RUNTIME_ERROR",
        "WALL_LIMIT",
    }
)
UNKNOWN_CATEGORIES = frozenset(
    {
        "AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE",
        "CONTROLLER_CAPTURE_FAILURE",
        "CONTROLLER_DRAIN_FAILURE",
        "CONTROLLER_EXCEPTION",
        "CONTROLLER_INPUT_FAILURE",
        "CONTROLLER_NO_EXIT_STATUS",
        "CONTROLLER_RESULT_INVALID",
        "CONTROLLER_START_FAILURE",
        "HOST_PRIVILEGE_DROP_ATTESTATION_FAILED",
        "LAUNCH_ATTESTATION_MISSING",
        "OVERLOADED",
        "QUEUE_DEADLINE",
        "READY_DEADLINE",
        "RUNSC_WARNING_OR_ERROR",
    }
)
RETRYABLE_UNKNOWN_CATEGORIES = frozenset({"OVERLOADED", "QUEUE_DEADLINE"})
CONTROLLER_ERROR_CATEGORIES = frozenset(
    {
        "CONTROLLER_CAPTURE_FAILURE",
        "CONTROLLER_EXCEPTION",
        "CONTROLLER_INPUT_FAILURE",
        "CONTROLLER_RESULT_INVALID",
        "CONTROLLER_START_FAILURE",
    }
)
MAX_CONTROLLER_ERROR_CHARS = 128
MAX_RESULT_TIMING_NS = MAX_REQUEST_TTL_NS + WALL_CEILING_NS
MEMORY_EVENT_KEYS = (
    "low",
    "high",
    "max",
    "oom",
    "oom_kill",
    "oom_group_kill",
)
PIDS_EVENT_KEYS = ("max",)
RESOURCE_EVIDENCE_SOURCES = frozenset(
    {
        "guest_monitor_ptrace_siginfo",
        "guest_monitor_ptrace_siginfo_fsize",
        "guest_monitor_ptrace_write_efbig",
        "guest_monitor_ptrace_thread_eagain",
        "request_cgroup_cpu_stat",
        "request_cgroup_memory_events",
        "request_cgroup_pids_events",
    }
)

_ASCII_WHITESPACE = frozenset(" \n\t\r\v")
_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_DECIMAL_DOUBLE = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    r"|inf(?:inity)?|nan)\Z"
)


class ExecutorProtocolError(ValueError):
    """A closed protocol/validation error, never a candidate failure."""


def _reject_bool_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExecutorProtocolError(f"{label} must be an integer")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ExecutorProtocolError(
            f"{label} keys mismatch (missing={missing}, extra={extra})"
        )


def derive_limits(raw_limits: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the exact signed local-verifier-parity execution limits.

    Wall time is the byte-exact seconds/nanoseconds duration supplied by the
    trusted driver. Requests above the explicit service maximum and any memory
    value other than the semantic 4 GiB RLIMIT_AS are rejected, never clamped.
    """

    if not isinstance(raw_limits, Mapping):
        raise ExecutorProtocolError("limits must be an object")
    _require_exact_keys(raw_limits, {"time_limit", "memory_limit_bytes"}, "limits")
    raw_time = raw_limits["time_limit"]
    if not isinstance(raw_time, Mapping):
        raise ExecutorProtocolError("limits.time_limit must be an object")
    _require_exact_keys(raw_time, {"seconds", "nanos"}, "limits.time_limit")

    seconds = _reject_bool_int(raw_time["seconds"], "limits.time_limit.seconds")
    nanos = _reject_bool_int(raw_time["nanos"], "limits.time_limit.nanos")
    memory = _reject_bool_int(
        raw_limits["memory_limit_bytes"], "limits.memory_limit_bytes"
    )
    if seconds < 0:
        raise ExecutorProtocolError("limits.time_limit.seconds must be nonnegative")
    if not 0 <= nanos < 1_000_000_000:
        raise ExecutorProtocolError(
            "limits.time_limit.nanos must be in [0, 1000000000)"
        )
    if memory != SEMANTIC_ADDRESS_SPACE_BYTES:
        raise ExecutorProtocolError(
            "limits.memory_limit_bytes must equal the 4 GiB semantic RLIMIT_AS"
        )
    if memory > MAX_UINT64 - ADDRESS_SPACE_OVERHEAD_BYTES:
        raise ExecutorProtocolError("address-space limit overflows uint64")

    source_duration_ns = seconds * 1_000_000_000 + nanos
    if source_duration_ns <= 0:
        raise ExecutorProtocolError("source time limit must be positive")
    if source_duration_ns > MAX_INT64:
        raise ExecutorProtocolError("source time limit overflows int64 nanoseconds")
    if source_duration_ns > MAX_SOURCE_DURATION_NS:
        raise ExecutorProtocolError("source time limit exceeds service wall maximum")

    effective_wall_ns = source_duration_ns
    # The candidate cgroup is restricted to one CPU.  One additional whole
    # second keeps the integer RLIMIT_CPU backstop from firing before the exact
    # (possibly fractional) wall deadline, while remaining tightly bounded.
    effective_cpu_seconds = (
        (source_duration_ns + 1_000_000_000 - 1) // 1_000_000_000 + 1
    )
    raw = {
        "time_limit": {"seconds": seconds, "nanos": nanos},
        "memory_limit_bytes": memory,
    }
    effective = {
        "address_space_bytes": memory,
        "cpu_seconds": effective_cpu_seconds,
        "wall_time_ns": effective_wall_ns,
        "wall_time_was_clamped": False,
        "stdout_bytes": STDOUT_CAP_BYTES,
        "stderr_bytes": STDERR_CAP_BYTES,
        "file_size_bytes": FILE_SIZE_CAP_BYTES,
        "aggregate_writable_bytes": WRITABLE_OVERLAY_CAP_BYTES,
        "host_cgroup_memory_bytes": (
            memory
            + ADDRESS_SPACE_OVERHEAD_BYTES
            + WRITABLE_OVERLAY_CAP_BYTES
            + RUNSC_INFRA_MEMORY_HEADROOM_BYTES
        ),
        "host_cgroup_pids": HOST_REQUEST_PID_CAP,
        "host_cgroup_cpu_quota_us": HOST_CPU_QUOTA_US,
        "host_cgroup_cpu_period_us": HOST_CPU_PERIOD_US,
        "host_cgroup_cpu_budget_us": (
            HOST_CPU_FAILSAFE_MULTIPLIER * effective_cpu_seconds * 1_000_000
            + HOST_CPU_FAILSAFE_FIXED_US
        ),
        "processes": PID_CAP,
        "open_files": NOFILE_CAP,
    }
    return {
        "raw": raw,
        "effective": effective,
        "transform": limit_policy_identity(),
    }


def limit_policy_identity() -> dict[str, Any]:
    return {
        "version": LIMIT_POLICY_VERSION,
        "source_url": DEEP_MIND_SOURCE_URL,
        "source_sha256": DEEP_MIND_SOURCE_SHA256,
        "address_space_overhead_bytes": ADDRESS_SPACE_OVERHEAD_BYTES,
        "semantic_address_space_bytes": SEMANTIC_ADDRESS_SPACE_BYTES,
        "cpu_rule": "ceil(raw_duration_seconds)+1_with_one_cpu_cgroup",
        "cpu_semantics": (
            "RLIMIT_CPU_is_unreachable_backstop_for_valid_request_because_"
            "exact_1x_wall_precedes_ceil(raw)+1_at_one_CPU"
        ),
        "wall_multiplier": 1,
        "wall_ceiling_ns": WALL_CEILING_NS,
        "wall_clamp": "none; requests_above_ceiling_rejected",
        "client_http_overhead_seconds": CLIENT_HTTP_OVERHEAD_SECONDS,
        "transport_recovery_window_seconds": TRANSPORT_RECOVERY_WINDOW_SECONDS,
        "replay_delivery_margin_seconds": REPLAY_DELIVERY_MARGIN_SECONDS,
        "execute_request_ttl_ns": DEFAULT_EXECUTE_REQUEST_TTL_NS,
        "replay_cache_grace_ns": REPLAY_CACHE_GRACE_NS,
        "replay_cache_capacity": REPLAY_CACHE_CAPACITY,
        "replay_cache_bytes": REPLAY_CACHE_BYTES,
        "stdout_cap_bytes": STDOUT_CAP_BYTES,
        "stderr_cap_bytes": STDERR_CAP_BYTES,
        "file_size_cap_bytes": FILE_SIZE_CAP_BYTES,
        "aggregate_writable_cap_bytes": WRITABLE_OVERLAY_CAP_BYTES,
        "read_only_tmpfs_size_bytes": READ_ONLY_TMPFS_SIZE_BYTES,
        "process_cap": PID_CAP,
        "process_policy": "candidate-main+one-exact-pthread;process-clones-denied",
        "pinned_gvisor_rlimit_nproc": PINNED_GVISOR_RLIMIT_NPROC,
        "pinned_gvisor_rlimit_nproc_accounting": (
            "pre-uid-transition-main-uncharged;one-pthread-slot"
        ),
        "source_memory_min_bytes": MIN_SOURCE_MEMORY_BYTES,
        "source_memory_max_bytes": MAX_SOURCE_MEMORY_BYTES,
        "host_request_process_cap": HOST_REQUEST_PID_CAP,
        "runsc_infra_memory_headroom_bytes": RUNSC_INFRA_MEMORY_HEADROOM_BYTES,
        "host_cpu_quota_us": HOST_CPU_QUOTA_US,
        "host_cpu_period_us": HOST_CPU_PERIOD_US,
        "host_cpu_budget_rule": "2*effective_cpu_seconds*1000000+1000000",
        "host_cpu_failsafe_multiplier": HOST_CPU_FAILSAFE_MULTIPLIER,
        "host_cpu_failsafe_fixed_us": HOST_CPU_FAILSAFE_FIXED_US,
        "host_cpu_calibration_sha256": HOST_CPU_CALIBRATION_SHA256,
        "host_cpu_calibration_outer_guest_ratio_range": "1.1677-1.2336",
        "host_cpu_calibration_max_guest_outer_seconds": "29.5/35.433",
        "open_file_cap": NOFILE_CAP,
        "request_body_cap_bytes": MAX_REQUEST_BODY_BYTES,
        "candidate_code_cap_bytes": MAX_CODE_BYTES,
        "stdin_cap_bytes": MAX_STDIN_BYTES,
    }


def static_identity(
    *,
    service_id: str,
    launcher_sha256: str,
    cgroup_gate_sha256: str = "0" * 64,
    rootfs_manifest_sha256: str = "0" * 64,
    rootfs_manifest_file_sha256: str = "0" * 64,
    server_bundle_sha256: str = "0" * 64,
    measured_guest_memory_bytes: int = CONFIGURED_VM_MEMORY_BYTES,
    host_policy_measurement: Mapping[str, Any] | None = None,
    expected_client_provenance: Mapping[str, Any] | None = None,
    active_sandboxes: int = 4,
    queue_capacity: int = 16,
) -> dict[str, Any]:
    if not service_id or len(service_id) > 128:
        raise ExecutorProtocolError("service_id must contain 1..128 characters")
    if not re.fullmatch(r"[A-Za-z0-9._:-]+", service_id):
        raise ExecutorProtocolError("service_id contains unsupported characters")
    if not re.fullmatch(r"[0-9a-f]{64}", launcher_sha256):
        raise ExecutorProtocolError("launcher_sha256 must be lowercase SHA-256")
    if not re.fullmatch(r"[0-9a-f]{64}", cgroup_gate_sha256):
        raise ExecutorProtocolError("cgroup_gate_sha256 must be lowercase SHA-256")
    if not re.fullmatch(r"[0-9a-f]{64}", rootfs_manifest_sha256):
        raise ExecutorProtocolError("rootfs_manifest_sha256 must be lowercase SHA-256")
    if not re.fullmatch(r"[0-9a-f]{64}", rootfs_manifest_file_sha256):
        raise ExecutorProtocolError(
            "rootfs_manifest_file_sha256 must be lowercase SHA-256"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", server_bundle_sha256):
        raise ExecutorProtocolError("server_bundle_sha256 must be lowercase SHA-256")
    if (
        isinstance(measured_guest_memory_bytes, bool)
        or not isinstance(measured_guest_memory_bytes, int)
        or measured_guest_memory_bytes < MINIMUM_GUEST_MEMORY_BYTES
    ):
        raise ExecutorProtocolError("measured guest memory is below minimum")
    if expected_client_provenance is None:
        expected_client_provenance = {
            "format": "palaestra.codecontests.client-provenance.v1",
            "client_sha256": "0" * 64,
            "protocol_sha256": "0" * 64,
            "verifier_sha256": "0" * 64,
        }
    expected_client = dict(expected_client_provenance)
    _require_exact_keys(
        expected_client,
        {
            "format",
            "client_sha256",
            "protocol_sha256",
            "verifier_sha256",
        },
        "expected_client_provenance",
    )
    if expected_client["format"] != "palaestra.codecontests.client-provenance.v1":
        raise ExecutorProtocolError("client provenance format mismatch")
    for key in ("client_sha256", "protocol_sha256", "verifier_sha256"):
        if not isinstance(expected_client[key], str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_client[key]
        ):
            raise ExecutorProtocolError(f"invalid client provenance {key}")
    if active_sandboxes != 4:
        raise ExecutorProtocolError("production active_sandboxes must equal 4")
    if queue_capacity < 1:
        raise ExecutorProtocolError("queue_capacity must be positive")
    if host_policy_measurement is None:
        host_policy_measurement = {
            "cpu_affinity_count": 4,
            "cpu_affinity_cpus": [0, 1, 2, 3],
            "cgroup_memory_ceiling_bytes": SERVICE_MEMORY_MAX_BYTES,
            "guest_visible_memory_bytes": measured_guest_memory_bytes,
            "unprivileged_userns_clone": 1,
            "max_user_namespaces": 4,
            "apparmor_restrict_unprivileged_userns": 0,
            "service_memory_max_bytes": SERVICE_MEMORY_MAX_BYTES,
            "service_memory_swap_max_bytes": 0,
            "service_tasks_max": 768,
            "service_cpu_quota_us": 400_000,
            "service_cpu_period_us": 100_000,
            "delegated_root_cgroup_procs_empty": True,
            "guest_swap_enabled": False,
        }
    host_policy = dict(host_policy_measurement)
    _require_exact_keys(
        host_policy,
        {
            "cpu_affinity_count",
            "cpu_affinity_cpus",
            "cgroup_memory_ceiling_bytes",
            "guest_visible_memory_bytes",
            "unprivileged_userns_clone",
            "max_user_namespaces",
            "apparmor_restrict_unprivileged_userns",
            "service_memory_max_bytes",
            "service_memory_swap_max_bytes",
            "service_tasks_max",
            "service_cpu_quota_us",
            "service_cpu_period_us",
            "delegated_root_cgroup_procs_empty",
            "guest_swap_enabled",
        },
        "host_policy_measurement",
    )
    affinity = host_policy["cpu_affinity_cpus"]
    if (
        host_policy["cpu_affinity_count"] != 4
        or isinstance(host_policy["cpu_affinity_count"], bool)
        or not isinstance(affinity, list)
        or len(affinity) != 4
        or any(
            isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
            for cpu in affinity
        )
        or affinity != sorted(set(affinity))
    ):
        raise ExecutorProtocolError("host CPU affinity measurement is invalid")
    cgroup_memory = host_policy["cgroup_memory_ceiling_bytes"]
    if (
        isinstance(cgroup_memory, bool)
        or not isinstance(cgroup_memory, int)
        or cgroup_memory != SERVICE_MEMORY_MAX_BYTES
    ):
        raise ExecutorProtocolError("host cgroup memory measurement is invalid")
    guest_memory = host_policy["guest_visible_memory_bytes"]
    if (
        isinstance(guest_memory, bool)
        or not isinstance(guest_memory, int)
        or guest_memory < MINIMUM_GUEST_MEMORY_BYTES
        or guest_memory != measured_guest_memory_bytes
    ):
        raise ExecutorProtocolError("guest memory measurements disagree")
    for key, exact in (
        ("unprivileged_userns_clone", 1),
        ("apparmor_restrict_unprivileged_userns", 0),
    ):
        if host_policy[key] != exact or isinstance(host_policy[key], bool):
            raise ExecutorProtocolError(f"host policy measurement {key} is unsafe")
    max_user_namespaces = host_policy["max_user_namespaces"]
    if (
        isinstance(max_user_namespaces, bool)
        or not isinstance(max_user_namespaces, int)
        or max_user_namespaces < 4
    ):
        raise ExecutorProtocolError("host user-namespace capacity is invalid")
    if (
        host_policy["service_memory_max_bytes"] != SERVICE_MEMORY_MAX_BYTES
        or host_policy["service_memory_swap_max_bytes"] != 0
        or host_policy["service_tasks_max"] != 768
        or host_policy["service_cpu_quota_us"] != 400_000
        or host_policy["service_cpu_period_us"] != 100_000
        or host_policy["delegated_root_cgroup_procs_empty"] is not True
        or host_policy["guest_swap_enabled"] is not False
    ):
        raise ExecutorProtocolError("service cgroup/swap policy measurement is unsafe")
    return {
        "kind": "identity",
        "protocol_version": PROTOCOL_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "service_id": service_id,
        "listener": {
            "scheme": "http",
            "host": "127.0.0.1",
            "port": 8080,
        },
        "server_bundle_sha256": server_bundle_sha256,
        "expected_client_provenance": expected_client,
        "runtime": {
            "runsc_path": RUNSC_PATH,
            "runsc_release_archive_path": RUNSC_RELEASE_ARCHIVE_PATH,
            "runsc_release": RUNSC_RELEASE,
            "runsc_release_arch": RUNSC_RELEASE_ARCH,
            "runsc_release_url": RUNSC_RELEASE_URL,
            "runsc_release_archive_size_bytes": (RUNSC_RELEASE_ARCHIVE_SIZE_BYTES),
            "runsc_sha512": RUNSC_SHA512,
            "runsc_release_archive_sha512": RUNSC_RELEASE_ARCHIVE_SHA512,
            "runsc_size_bytes": RUNSC_SIZE_BYTES,
            "runsc_spec_version": RUNSC_SPEC_VERSION,
            "runsc_version_output": RUNSC_VERSION_OUTPUT,
            "runsc_derivation": (
                "verified-official-tar.bz2:extract-runsc-member:"
                "root-owned-immutable-install"
            ),
            "rootfs_path": ROOTFS_PATH,
            "rootfs_sha256": ROOTFS_SHA256,
            "rootfs_manifest_sha256": rootfs_manifest_sha256,
            "rootfs_manifest_file_sha256": rootfs_manifest_file_sha256,
            "rootfs_measurement": "pinned-per-file-manifest-v1",
            "rootfs_access": (
                "canonical-readonly-bind+held-rootfd+per-request-tree-metadata-v2"
            ),
            "rootfs_proc_fd_rejected_by_pinned_gofer": True,
            "python_version": PYTHON_VERSION,
            "required_python_packages": REQUIRED_PYTHON_PACKAGES,
            "candidate_python_flags": ["-I", "-B", "-c"],
            "candidate_import_path": "isolated-safe-path-system-site-no-cwd",
            "candidate_site_conveniences": ["exit", "quit"],
            "network": "none",
            "no_new_privs": True,
            "runsc_invocation": "rootful-oci-run-dedicated-credential-free-vm",
            "configured_runsc_host_uid": 0,
            "configured_runsc_host_gid": 0,
            "candidate_uid": 65534,
            "candidate_gid": 65534,
            "candidate_delivery": (
                "fresh-nonce-free-traced-interpreter-framed-stdin-code-object"
            ),
            "trusted_monitor_uid": 0,
            "trusted_monitor_bootstrap_capabilities": list(
                TRUSTED_MONITOR_BOOTSTRAP_CAPABILITIES
            ),
            "candidate_capability_sets": "all-empty-before-candidate-gate",
            "candidate_cpu_attribution": (
                "ptrace-kernel-SIGXCPU-soft-limit-seccomp-procfs-attested-rlimits"
            ),
            "candidate_file_attribution": (
                "ptrace-kernel-SIGXFSZ-or-write-EFBIG;filesystem-readonly"
            ),
            "writable_overlay": {
                "driver": "none;root-and-tmp-readonly",
                "aggregate_bytes": WRITABLE_OVERLAY_CAP_BYTES,
            },
            "tmp_mount": {
                "driver": "readonly-tmpfs",
                "bytes": READ_ONLY_TMPFS_SIZE_BYTES,
            },
            "root_readonly": True,
            "dev_mount": "tmpfs-65536-bytes-readonly",
            "request_cgroup_topology": "outer/controller+candidate",
            "host_request_pids_max": HOST_REQUEST_PID_CAP,
            "host_request_memory_rule": (
                "semantic_as_RSS_ceiling+runsc_headroom384MiB;"
                "RLIMIT_AS_and_cgroup_RSS_are_distinct_currencies"
            ),
            "host_request_cpu_max": (f"{HOST_CPU_QUOTA_US} {HOST_CPU_PERIOD_US}"),
            "cgroup_gate_sha256": cgroup_gate_sha256,
            "launcher_sha256": launcher_sha256,
        },
        "capacity": {
            "configured_vm_vcpus": CONFIGURED_VM_VCPUS,
            "service_cpu_affinity_vcpus": SERVICE_CPU_AFFINITY_VCPUS,
            "configured_vm_memory_bytes": CONFIGURED_VM_MEMORY_BYTES,
            "service_memory_max_bytes": SERVICE_MEMORY_MAX_BYTES,
            "measured_guest_memory_bytes": measured_guest_memory_bytes,
            "minimum_guest_memory_bytes": MINIMUM_GUEST_MEMORY_BYTES,
            "active_sandboxes": active_sandboxes,
            "queue_capacity": queue_capacity,
            "host_policy_measurement": host_policy,
        },
        "limits": limit_policy_identity(),
    }


def _validate_json_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise ExecutorProtocolError(f"{path}: floating-point JSON is forbidden")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExecutorProtocolError(f"{path}: object key is not a string")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ExecutorProtocolError(f"{path}: unsupported JSON type")


def canonical_json(value: Any) -> bytes:
    try:
        _validate_json_value(value)
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except ExecutorProtocolError:
        raise
    except (
        RecursionError,
        TypeError,
        ValueError,
        OverflowError,
        UnicodeEncodeError,
    ) as exc:
        raise ExecutorProtocolError("value cannot be canonically encoded") from exc


def payload_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(payload))).hexdigest()


def sign_payload(payload: Mapping[str, Any], key: bytes) -> dict[str, Any]:
    if not key:
        raise ExecutorProtocolError("HMAC key is empty")
    normalized = dict(payload)
    signature = hmac.new(key, canonical_json(normalized), hashlib.sha256).hexdigest()
    return {"payload": normalized, "signature": f"hmac-sha256:{signature}"}


def verify_envelope(envelope: Any, key: bytes, *, expected_kind: str) -> dict[str, Any]:
    if not isinstance(envelope, dict):
        raise ExecutorProtocolError("signed envelope must be an object")
    _require_exact_keys(envelope, {"payload", "signature"}, "signed envelope")
    payload = envelope["payload"]
    signature = envelope["signature"]
    if (
        not isinstance(payload, dict)
        or not isinstance(signature, str)
        or re.fullmatch(r"hmac-sha256:[0-9a-f]{64}", signature) is None
    ):
        raise ExecutorProtocolError("malformed signed envelope")
    expected = sign_payload(payload, key)["signature"]
    if not hmac.compare_digest(expected, signature):
        raise ExecutorProtocolError("signature mismatch")
    if payload.get("kind") != expected_kind:
        raise ExecutorProtocolError("signed payload kind mismatch")
    return payload


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExecutorProtocolError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def strict_json_loads(data: bytes) -> Any:
    try:
        decoded = data.decode("utf-8", errors="strict")
        value = json.loads(decoded, object_pairs_hook=_pairs_no_duplicates)
    except ExecutorProtocolError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise ExecutorProtocolError("invalid JSON body") from exc
    try:
        _validate_json_value(value)
    except ExecutorProtocolError:
        raise
    except RecursionError as exc:
        raise ExecutorProtocolError("JSON nesting is too deep") from exc
    return value


def encode_envelope(envelope: Mapping[str, Any]) -> bytes:
    return canonical_json(dict(envelope))


def validate_execution_evidence(
    *,
    outcome: Any,
    category: Any,
    retryable: Any,
    evidence: Any,
    expected_limits: Mapping[str, Any],
) -> tuple[bytes, bytes]:
    """Validate the complete signed controller-result state machine.

    This function is shared by the service trust boundary and the driver
    client so a malformed controller dictionary can never reach canonical JSON
    and a signed result cannot exploit validation drift.
    """

    if not isinstance(outcome, str) or outcome not in {
        "executed",
        "candidate_failure",
        "unknown",
    }:
        raise ExecutorProtocolError("invalid execution outcome")
    if not isinstance(retryable, bool):
        raise ExecutorProtocolError("invalid retryable flag")
    if not isinstance(evidence, dict):
        raise ExecutorProtocolError("result evidence must be an object")
    _require_exact_keys(
        evidence,
        {
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
        },
        "result evidence",
    )

    encoded_stdout = evidence["stdout_b64"]
    encoded_stderr = evidence["stderr_b64"]
    if not isinstance(encoded_stdout, str) or not isinstance(encoded_stderr, str):
        raise ExecutorProtocolError("encoded output evidence must be strings")
    if len(encoded_stdout) > 4 * ((STDOUT_CAP_BYTES + 2) // 3):
        raise ExecutorProtocolError("encoded stdout exceeds policy cap")
    if len(encoded_stderr) > 4 * ((STDERR_CAP_BYTES + 2) // 3):
        raise ExecutorProtocolError("encoded stderr exceeds policy cap")
    try:
        stdout = base64.b64decode(encoded_stdout, validate=True)
        stderr = base64.b64decode(encoded_stderr, validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise ExecutorProtocolError("invalid encoded output evidence") from exc
    if len(stdout) > STDOUT_CAP_BYTES or len(stderr) > STDERR_CAP_BYTES:
        raise ExecutorProtocolError("signed output exceeds policy cap")

    stdout_count = _reject_bool_int(evidence["stdout_bytes"], "evidence.stdout_bytes")
    stderr_count = _reject_bool_int(evidence["stderr_bytes"], "evidence.stderr_bytes")
    if stdout_count < 0 or stdout_count != len(stdout):
        raise ExecutorProtocolError("stdout byte count mismatch")
    if stderr_count < 0 or stderr_count != len(stderr):
        raise ExecutorProtocolError("stderr byte count mismatch")

    stdout_truncated = evidence["stdout_truncated"]
    stderr_truncated = evidence["stderr_truncated"]
    if not isinstance(stdout_truncated, bool) or not isinstance(stderr_truncated, bool):
        raise ExecutorProtocolError("output truncation flags must be booleans")

    returncode = evidence["returncode"]
    if returncode is not None:
        returncode = _reject_bool_int(returncode, "evidence.returncode")
        if not -255 <= returncode <= 255:
            raise ExecutorProtocolError("return code is outside the signed domain")
    result_signal = evidence["signal"]
    if result_signal is not None:
        result_signal = _reject_bool_int(result_signal, "evidence.signal")
        if not 1 <= result_signal <= 255:
            raise ExecutorProtocolError("signal is outside the signed domain")
    expected_signal = -returncode if returncode is not None and returncode < 0 else None
    if result_signal != expected_signal:
        raise ExecutorProtocolError("signal contradicts return code")

    controller_error = evidence["controller_error"]
    if controller_error is not None and (
            not isinstance(controller_error, str)
            or not 1 <= len(controller_error) <= MAX_CONTROLLER_ERROR_CHARS
            or not controller_error.isascii()
            or re.fullmatch(r"[A-Za-z0-9_.:-]+", controller_error) is None
    ):
        raise ExecutorProtocolError("invalid bounded controller error")
    resource_event = evidence["resource_event"]
    if resource_event not in {
        None,
        "CGROUP_CPU_BUDGET",
        "CGROUP_MEMORY_OOM",
        "CGROUP_PIDS_MAX",
        "GUEST_CPU_LIMIT",
        "GUEST_FILE_SPACE_LIMIT",
        "GUEST_PROCESS_LIMIT",
    }:
        raise ExecutorProtocolError("invalid resource-event evidence")
    for key, maximum in (
        ("host_cpu_usage_us", MAX_RESULT_TIMING_NS // 1_000),
        ("host_cpu_before_usage_us", MAX_RESULT_TIMING_NS // 1_000),
        ("host_cpu_ready_usage_us", MAX_RESULT_TIMING_NS // 1_000),
        ("host_cpu_cross_usage_us", MAX_RESULT_TIMING_NS // 1_000),
        ("host_cpu_after_usage_us", MAX_RESULT_TIMING_NS // 1_000),
        ("host_cpu_budget_us", MAX_RESULT_TIMING_NS // 1_000),
        (
            "host_memory_peak_bytes",
            MAX_SOURCE_MEMORY_BYTES
            + ADDRESS_SPACE_OVERHEAD_BYTES
            + WRITABLE_OVERLAY_CAP_BYTES
            + RUNSC_INFRA_MEMORY_HEADROOM_BYTES,
        ),
        ("host_pids_peak", HOST_REQUEST_PID_CAP),
        ("guest_cpu_usage_us", MAX_RESULT_TIMING_NS // 1_000),
        ("guest_process_peak", PID_CAP),
        ("guest_process_limit", PID_CAP),
        ("guest_rlimit_nproc", PID_CAP),
        ("guest_file_size_limit_bytes", FILE_SIZE_CAP_BYTES),
        ("guest_writable_limit_bytes", WRITABLE_OVERLAY_CAP_BYTES),
        ("guest_file_size_observed_bytes", FILE_SIZE_CAP_BYTES),
        ("guest_writable_available_bytes", WRITABLE_OVERLAY_CAP_BYTES),
    ):
        measured = _reject_bool_int(evidence[key], f"evidence.{key}")
        if not 0 <= measured <= maximum:
            raise ExecutorProtocolError(f"{key} is outside the signed domain")
    before_cpu = evidence["host_cpu_before_usage_us"]
    ready_cpu = evidence["host_cpu_ready_usage_us"]
    cross_cpu = evidence["host_cpu_cross_usage_us"]
    after_cpu = evidence["host_cpu_after_usage_us"]
    cpu_budget = evidence["host_cpu_budget_us"]
    try:
        expected_cpu_budget = expected_limits["effective"]["host_cgroup_cpu_budget_us"]
        expected_process_limit = expected_limits["effective"]["processes"]
        expected_file_size_limit = expected_limits["effective"]["file_size_bytes"]
        expected_writable_limit = expected_limits["effective"][
            "aggregate_writable_bytes"
        ]
    except (KeyError, TypeError) as exc:
        raise ExecutorProtocolError("expected limit evidence is malformed") from exc
    if cpu_budget != expected_cpu_budget:
        raise ExecutorProtocolError("signed host CPU budget mismatches request")
    if (
        evidence["guest_process_limit"] != expected_process_limit
        or evidence["guest_process_limit"] != PID_CAP
        or evidence["guest_rlimit_nproc"] != PINNED_GVISOR_RLIMIT_NPROC
        or evidence["guest_rlimit_nproc"] != evidence["guest_process_limit"] - 1
    ):
        raise ExecutorProtocolError(
            "guest semantic/raw process-limit evidence mismatches policy"
        )
    if (
        evidence["guest_file_size_limit_bytes"] != expected_file_size_limit
        or evidence["guest_file_size_limit_bytes"] != FILE_SIZE_CAP_BYTES
        or evidence["guest_writable_limit_bytes"] != expected_writable_limit
        or evidence["guest_writable_limit_bytes"] != WRITABLE_OVERLAY_CAP_BYTES
    ):
        raise ExecutorProtocolError("guest file-limit evidence mismatches policy")
    file_limit_signal = evidence["guest_file_limit_signal"]
    if file_limit_signal is not None:
        file_limit_signal = _reject_bool_int(
            file_limit_signal, "evidence.guest_file_limit_signal"
        )
        if file_limit_signal != 25:
            raise ExecutorProtocolError(
                "guest file-limit signal is outside the pinned domain"
            )
    file_limit_errno = evidence["guest_file_limit_errno"]
    if file_limit_errno is not None:
        file_limit_errno = _reject_bool_int(
            file_limit_errno, "evidence.guest_file_limit_errno"
        )
        if file_limit_errno != 27:
            raise ExecutorProtocolError(
                "guest file-limit errno is outside the pinned domain"
            )
    process_limit_syscall = evidence["guest_process_limit_syscall"]
    if process_limit_syscall is not None:
        process_limit_syscall = _reject_bool_int(
            process_limit_syscall, "evidence.guest_process_limit_syscall"
        )
        if process_limit_syscall not in {56, 57, 58, 220}:
            raise ExecutorProtocolError(
                "guest process-limit syscall is outside the pinned domain"
            )
    if (
        ready_cpu < before_cpu
        or after_cpu < ready_cpu
        or evidence["host_cpu_usage_us"] != (after_cpu - ready_cpu)
    ):
        raise ExecutorProtocolError("host CPU accounting is inconsistent")
    cpu_hit = cpu_budget > 0 and after_cpu - ready_cpu >= cpu_budget
    if cpu_hit:
        if (
            cross_cpu == 0
            or cross_cpu - ready_cpu < cpu_budget
            or after_cpu < cross_cpu
        ):
            raise ExecutorProtocolError("CPU budget crossing evidence is inconsistent")
    elif cross_cpu != 0:
        raise ExecutorProtocolError("unexpected CPU crossing evidence")

    def event_map(value: Any, keys: tuple[str, ...], label: str) -> dict[str, int]:
        if not isinstance(value, dict) or set(value) != set(keys):
            raise ExecutorProtocolError(f"{label} event fields mismatch")
        normalized: dict[str, int] = {}
        for key in keys:
            measured = _reject_bool_int(value[key], f"{label}.{key}")
            if not 0 <= measured <= MAX_INT64:
                raise ExecutorProtocolError(f"{label}.{key} is outside domain")
            normalized[key] = measured
        return normalized

    memory_before = event_map(
        evidence["host_memory_events_before"],
        MEMORY_EVENT_KEYS,
        "host_memory_events_before",
    )
    memory_after = event_map(
        evidence["host_memory_events_after"],
        MEMORY_EVENT_KEYS,
        "host_memory_events_after",
    )
    pids_before = event_map(
        evidence["host_pids_events_before"],
        PIDS_EVENT_KEYS,
        "host_pids_events_before",
    )
    pids_after = event_map(
        evidence["host_pids_events_after"],
        PIDS_EVENT_KEYS,
        "host_pids_events_after",
    )
    if any(memory_after[key] < memory_before[key] for key in MEMORY_EVENT_KEYS):
        raise ExecutorProtocolError("memory event counters regressed")
    if any(pids_after[key] < pids_before[key] for key in PIDS_EVENT_KEYS):
        raise ExecutorProtocolError("pids event counters regressed")
    memory_hit = any(
        memory_after[key] > memory_before[key]
        for key in ("max", "oom", "oom_kill", "oom_group_kill")
    )
    pids_hit = pids_after["max"] > pids_before["max"]
    source = evidence["resource_evidence_source"]
    if source is not None and source not in RESOURCE_EVIDENCE_SOURCES:
        raise ExecutorProtocolError("resource evidence source is invalid")
    outer_events = [
        event
        for event, happened in (
            ("CGROUP_CPU_BUDGET", cpu_hit),
            ("CGROUP_MEMORY_OOM", memory_hit),
            ("CGROUP_PIDS_MAX", pids_hit),
        )
        if happened
    ]
    outer_sources = {
        "CGROUP_CPU_BUDGET": "request_cgroup_cpu_stat",
        "CGROUP_MEMORY_OOM": "request_cgroup_memory_events",
        "CGROUP_PIDS_MAX": "request_cgroup_pids_events",
    }
    if len(outer_events) == 1:
        if (
            resource_event != outer_events[0]
            or source != outer_sources[outer_events[0]]
        ):
            raise ExecutorProtocolError(
                "single outer resource delta is not encoded exactly"
            )
    elif len(outer_events) > 1:
        if resource_event is not None or source is not None:
            raise ExecutorProtocolError(
                "multiple outer resource deltas must remain unclassified"
            )
    elif resource_event in outer_sources:
        raise ExecutorProtocolError("outer resource event lacks its exact delta")
    if outer_events and (
        outcome != "unknown" or category != "AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE"
    ):
        raise ExecutorProtocolError(
            "outer aggregate resource evidence is never candidate-specific"
        )
    expected_source = {
        **outer_sources,
        "GUEST_CPU_LIMIT": "guest_monitor_ptrace_siginfo",
        "GUEST_PROCESS_LIMIT": "guest_monitor_ptrace_thread_eagain",
    }.get(resource_event)
    if resource_event == "GUEST_FILE_SPACE_LIMIT":
        if source not in {
            "guest_monitor_ptrace_siginfo_fsize",
            "guest_monitor_ptrace_write_efbig",
        }:
            raise ExecutorProtocolError("resource event/evidence source mismatch")
    elif source != expected_source:
        raise ExecutorProtocolError("resource event/evidence source mismatch")
    if (resource_event == "GUEST_PROCESS_LIMIT") != (process_limit_syscall is not None):
        raise ExecutorProtocolError(
            "guest process-limit syscall provenance is inconsistent"
        )
    if (source == "guest_monitor_ptrace_siginfo_fsize") != (file_limit_signal == 25):
        raise ExecutorProtocolError(
            "guest file-limit signal provenance is inconsistent"
        )
    if source != "guest_monitor_ptrace_siginfo_fsize" and file_limit_signal is not None:
        raise ExecutorProtocolError(
            "guest file-limit signal appears without kernel provenance"
        )
    if (source == "guest_monitor_ptrace_write_efbig") != (file_limit_errno == 27):
        raise ExecutorProtocolError("guest file-limit errno provenance is inconsistent")
    if source != "guest_monitor_ptrace_write_efbig" and file_limit_errno is not None:
        raise ExecutorProtocolError(
            "guest file-limit errno appears without traced-write provenance"
        )
    if (
        source == "guest_monitor_ptrace_write_efbig"
        and evidence["guest_file_size_observed_bytes"]
        != evidence["guest_file_size_limit_bytes"]
    ):
        raise ExecutorProtocolError(
            "traced-write file-limit evidence lacks the exact file size"
        )
    if outcome == "executed":
        if (
            category is not None
            or retryable
            or returncode != 0
            or result_signal is not None
            or stdout_truncated
            or stderr_truncated
            or controller_error is not None
            or resource_event not in {None, "GUEST_PROCESS_LIMIT"}
            or cpu_hit
            or memory_hit
            or pids_hit
        ):
            raise ExecutorProtocolError("inconsistent executed result")
        return stdout, stderr

    if not isinstance(category, str):
        raise ExecutorProtocolError("result category must be a string")
    if len(category) > 64 or re.fullmatch(r"[A-Z0-9_]+", category) is None:
        raise ExecutorProtocolError("invalid result category syntax")

    if outcome == "candidate_failure":
        if (
            category not in CANDIDATE_FAILURE_CATEGORIES
            or retryable
            or controller_error is not None
        ):
            raise ExecutorProtocolError("inconsistent candidate failure")
        if category == "OUTPUT_LIMIT":
            if not (stdout_truncated or stderr_truncated):
                raise ExecutorProtocolError(
                    "output-limit result lacks truncation evidence"
                )
        elif stdout_truncated or stderr_truncated:
            raise ExecutorProtocolError(
                "non-output candidate failure reports truncation"
            )
        if category == "RUNTIME_ERROR" and returncode in {None, 0}:
            raise ExecutorProtocolError("runtime error lacks failing return code")
        if category == "CPU_LIMIT":
            if resource_event != "GUEST_CPU_LIMIT":
                raise ExecutorProtocolError("CPU-limit resource event is inconsistent")
            if (returncode, result_signal) != (-9, 9):
                raise ExecutorProtocolError(
                    "trusted guest CPU-limit exit evidence is inconsistent"
                )
            expected_guest_limit = (
                expected_limits["effective"]["cpu_seconds"] * 1_000_000
            )
            if not (
                expected_guest_limit
                <= evidence["guest_cpu_usage_us"]
                <= expected_guest_limit + 2_000_000
            ):
                raise ExecutorProtocolError(
                    "guest CPU usage does not cross the attested limit"
                )
        elif category == "PROCESS_LIMIT":
            if resource_event != "GUEST_PROCESS_LIMIT":
                raise ExecutorProtocolError(
                    "process-limit resource event is inconsistent"
                )
            if returncode in {None, 0}:
                raise ExecutorProtocolError(
                    "process-limit failure lacks a failing return code"
                )
            if evidence["guest_process_peak"] != PID_CAP:
                raise ExecutorProtocolError(
                    "guest process-limit denial lacks exact task-count evidence"
                )
        elif category == "FILE_SPACE_LIMIT":
            if resource_event != "GUEST_FILE_SPACE_LIMIT":
                raise ExecutorProtocolError(
                    "file-space limit lacks trusted monitor evidence"
                )
        elif resource_event is not None:
            raise ExecutorProtocolError(
                "candidate category/resource event is inconsistent"
            )
        return stdout, stderr

    if category not in UNKNOWN_CATEGORIES:
        raise ExecutorProtocolError("unknown result category is not allowed")
    if retryable != (category in RETRYABLE_UNKNOWN_CATEGORIES):
        raise ExecutorProtocolError("unknown retryability is inconsistent")
    if controller_error is not None and category not in CONTROLLER_ERROR_CATEGORIES:
        raise ExecutorProtocolError("controller error contradicts category")
    if category in CONTROLLER_ERROR_CATEGORIES and controller_error is None:
        raise ExecutorProtocolError("controller category lacks bounded error")
    if stdout_truncated or (
        stderr_truncated and category != "LAUNCH_ATTESTATION_MISSING"
    ):
        raise ExecutorProtocolError("unknown result has contradictory truncation")
    if resource_event is not None and resource_event not in outer_sources:
        raise ExecutorProtocolError("unknown result contains a guest resource event")
    if category in RETRYABLE_UNKNOWN_CATEGORIES and (
        stdout
        or stderr
        or returncode is not None
        or result_signal is not None
        or controller_error is not None
        or resource_event is not None
    ):
        raise ExecutorProtocolError("retryable overload contains execution evidence")
    if (
        category == "AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE"
        and not outer_events
        and returncode in {None, 0}
    ):
        raise ExecutorProtocolError("ambiguous failure lacks a failing exit status")
    return stdout, stderr


def validate_execute_request(
    payload: Mapping[str, Any],
    *,
    expected_identity_digest: str,
    expected_client_provenance: Mapping[str, Any] | None = None,
    now_ns: int | None = None,
) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        {
            "kind",
            "protocol_version",
            "request_id",
            "issued_at_unix_ns",
            "expires_at_unix_ns",
            "identity_digest",
            "client_provenance",
            "task",
        },
        "execute request",
    )
    if payload["kind"] != "execute_request":
        raise ExecutorProtocolError("request kind mismatch")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise ExecutorProtocolError("protocol version mismatch")
    request_id = payload["request_id"]
    if not isinstance(request_id, str):
        raise ExecutorProtocolError("request_id must be a string")
    try:
        if str(uuid.UUID(request_id)) != request_id:
            raise ValueError
    except ValueError as exc:
        raise ExecutorProtocolError("request_id must be canonical UUID") from exc
    issued = _reject_bool_int(payload["issued_at_unix_ns"], "issued_at_unix_ns")
    expires = _reject_bool_int(payload["expires_at_unix_ns"], "expires_at_unix_ns")
    current = time.time_ns() if now_ns is None else now_ns
    if issued > current + MAX_CLOCK_SKEW_NS:
        raise ExecutorProtocolError("request issued too far in the future")
    if expires <= current:
        raise ExecutorProtocolError("request expired")
    if expires <= issued or expires - issued > MAX_REQUEST_TTL_NS:
        raise ExecutorProtocolError("invalid request validity interval")
    supplied_identity_digest = payload["identity_digest"]
    if (
        not isinstance(supplied_identity_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", supplied_identity_digest) is None
        or re.fullmatch(r"[0-9a-f]{64}", expected_identity_digest) is None
        or not hmac.compare_digest(supplied_identity_digest, expected_identity_digest)
    ):
        raise ExecutorProtocolError("request identity mismatch")
    client_provenance = payload["client_provenance"]
    if not isinstance(client_provenance, Mapping):
        raise ExecutorProtocolError("client provenance must be an object")
    _require_exact_keys(
        client_provenance,
        {
            "format",
            "client_sha256",
            "protocol_sha256",
            "verifier_sha256",
        },
        "client_provenance",
    )
    if client_provenance["format"] != ("palaestra.codecontests.client-provenance.v1"):
        raise ExecutorProtocolError("client provenance format mismatch")
    for key in ("client_sha256", "protocol_sha256", "verifier_sha256"):
        if not isinstance(client_provenance[key], str) or not re.fullmatch(
            r"[0-9a-f]{64}", client_provenance[key]
        ):
            raise ExecutorProtocolError("client provenance digest is invalid")
    if expected_client_provenance is not None and dict(client_provenance) != dict(
        expected_client_provenance
    ):
        raise ExecutorProtocolError("client/verifier provenance mismatch")

    task = payload["task"]
    if not isinstance(task, Mapping):
        raise ExecutorProtocolError("task must be an object")
    _require_exact_keys(task, {"language", "code_b64", "stdin_b64", "limits"}, "task")
    if task["language"] != "python":
        raise ExecutorProtocolError("only Python candidates are supported")
    code_b64 = task["code_b64"]
    stdin_b64 = task["stdin_b64"]
    if not isinstance(code_b64, str) or not isinstance(stdin_b64, str):
        raise ExecutorProtocolError("encoded code and stdin must be strings")
    try:
        code_bytes = base64.b64decode(code_b64, validate=True)
        stdin_bytes = base64.b64decode(stdin_b64, validate=True)
        code_bytes.decode("utf-8", errors="strict")
        stdin_bytes.decode("utf-8", errors="strict")
    except (
        binascii.Error,
        UnicodeDecodeError,
        UnicodeEncodeError,
        ValueError,
    ) as exc:
        raise ExecutorProtocolError("code/stdin encoding is invalid") from exc
    if len(code_bytes) > MAX_CODE_BYTES:
        raise ExecutorProtocolError("candidate code exceeds byte limit")
    if len(stdin_bytes) > MAX_STDIN_BYTES:
        raise ExecutorProtocolError("stdin exceeds byte limit")
    supplied_limits = task["limits"]
    if not isinstance(supplied_limits, Mapping):
        raise ExecutorProtocolError("task.limits must be an object")
    raw = supplied_limits.get("raw")
    if not isinstance(raw, Mapping):
        raise ExecutorProtocolError("task.limits.raw must be an object")
    expected_limits = derive_limits(raw)
    if dict(supplied_limits) != expected_limits:
        raise ExecutorProtocolError("effective limits or provenance mismatch")
    return dict(payload)


def make_execute_request(
    *,
    code: str,
    stdin: str,
    raw_limits: Mapping[str, Any],
    identity_digest_value: str,
    client_provenance: Mapping[str, Any] | None = None,
    ttl_ns: int,
    request_id: str | None = None,
    now_ns: int | None = None,
) -> dict[str, Any]:
    current = time.time_ns() if now_ns is None else now_ns
    if ttl_ns <= 0 or ttl_ns > MAX_REQUEST_TTL_NS:
        raise ExecutorProtocolError("invalid request TTL")
    provenance = (
        {
            "format": "palaestra.codecontests.client-provenance.v1",
            "client_sha256": "0" * 64,
            "protocol_sha256": "0" * 64,
            "verifier_sha256": "0" * 64,
        }
        if client_provenance is None
        else dict(client_provenance)
    )
    return {
        "kind": "execute_request",
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id or str(uuid.uuid4()),
        "issued_at_unix_ns": current,
        "expires_at_unix_ns": current + ttl_ns,
        "identity_digest": identity_digest_value,
        "client_provenance": provenance,
        "task": {
            "language": "python",
            "code_b64": base64.b64encode(code.encode("utf-8")).decode("ascii"),
            "stdin_b64": base64.b64encode(stdin.encode("utf-8")).decode("ascii"),
            "limits": derive_limits(raw_limits),
        },
    }


def _split_and_lower(value: str) -> list[str]:
    tokens: list[str] = []
    start: int | None = None
    for index, character in enumerate(value):
        if character in _ASCII_WHITESPACE:
            if start is not None:
                tokens.append(value[start:index].translate(_ASCII_LOWER))
                start = None
        elif start is None:
            start = index
    if start is not None:
        tokens.append(value[start:].translate(_ASCII_LOWER))
    return tokens


def _typed_token(token: str) -> tuple[str, int | float | str]:
    unsigned = token[1:] if token[:1] in {"+", "-"} else token
    if unsigned and unsigned.isascii() and unsigned.isdecimal():
        value = int(token, 10)
        if -(2**31) <= value <= 2**31 - 1:
            return "int", value
    if token.isascii() and _DECIMAL_DOUBLE.fullmatch(token) is not None:
        return "double", float(token)
    return "string", token


def outputs_match(actual: str, expected: str) -> bool:
    """Match DeepMind ``OutputsMatch`` token/case/numeric semantics."""

    actual_tokens = _split_and_lower(actual)
    expected_tokens = _split_and_lower(expected)
    if actual_tokens == expected_tokens:
        return True
    if len(actual_tokens) != len(expected_tokens):
        return False
    for actual_token, expected_token in zip(actual_tokens, expected_tokens):
        actual_kind, actual_value = _typed_token(actual_token)
        expected_kind, expected_value = _typed_token(expected_token)
        if "string" in (actual_kind, expected_kind):
            if actual_token != expected_token:
                return False
        elif "double" in (actual_kind, expected_kind):
            difference = math.fabs(float(actual_value) - float(expected_value))
            if not difference < 1e-5:
                return False
        elif actual_value != expected_value:
            return False
    return True
