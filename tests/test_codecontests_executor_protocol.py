"""Protocol/policy regressions for the production CodeContests executor."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from codecontests_executor.protocol import (
    ADDRESS_SPACE_OVERHEAD_BYTES,
    CONFIGURED_VM_MEMORY_BYTES,
    FILE_SIZE_CAP_BYTES,
    MAX_INT64,
    MAX_UINT64,
    MEMORY_EVENT_KEYS,
    PID_CAP,
    PIDS_EVENT_KEYS,
    PINNED_GVISOR_RLIMIT_NPROC,
    REQUIRED_PYTHON_PACKAGES,
    ROOTFS_SHA256,
    RUNSC_RELEASE_ARCHIVE_PATH,
    RUNSC_RELEASE_ARCHIVE_SHA512,
    RUNSC_RELEASE_ARCHIVE_SIZE_BYTES,
    RUNSC_SHA512,
    RUNSC_SIZE_BYTES,
    SERVICE_MEMORY_MAX_BYTES,
    STDERR_CAP_BYTES,
    STDOUT_CAP_BYTES,
    WRITABLE_OVERLAY_CAP_BYTES,
    ExecutorProtocolError,
    canonical_json,
    derive_limits,
    make_execute_request,
    outputs_match,
    payload_digest,
    sign_payload,
    static_identity,
    strict_json_loads,
    validate_execution_evidence,
    verify_envelope,
)

RAW_MEMORY = 4 * 1024**3


def _raw(seconds: int, nanos: int = 0, memory: int = RAW_MEMORY):
    return {
        "time_limit": {"seconds": seconds, "nanos": nanos},
        "memory_limit_bytes": memory,
    }


@pytest.mark.parametrize(
    ("seconds", "nanos", "cpu", "wall_ns", "clamped"),
    [
        (0, 400_000_000, 2, 400_000_000, False),
        (1, 0, 2, 1_000_000_000, False),
        (30, 0, 31, 30_000_000_000, False),
        (120, 0, 121, 120_000_000_000, False),
    ],
)
def test_limit_transform_boundaries(
    seconds: int, nanos: int, cpu: int, wall_ns: int, clamped: bool
):
    limits = derive_limits(_raw(seconds, nanos))
    assert limits["raw"] == _raw(seconds, nanos)
    assert limits["effective"]["cpu_seconds"] == cpu
    assert limits["effective"]["wall_time_ns"] == wall_ns
    assert limits["effective"]["wall_time_was_clamped"] is clamped
    assert limits["effective"]["address_space_bytes"] == (
        RAW_MEMORY + ADDRESS_SPACE_OVERHEAD_BYTES
    )
    assert limits["effective"]["stdout_bytes"] == STDOUT_CAP_BYTES
    assert limits["effective"]["stderr_bytes"] == STDERR_CAP_BYTES


@pytest.mark.parametrize(
    "raw",
    [
        _raw(0, 0),
        _raw(-1),
        _raw(1, -1),
        _raw(1, 1_000_000_000),
        _raw(1, memory=0),
        _raw(1, memory=MAX_UINT64),
        _raw(120, 1),
        _raw(MAX_INT64 // 1_000_000_000 + 1),
        {"time_limit": {"seconds": True, "nanos": 0}, "memory_limit_bytes": 1},
        {"time_limit": {"seconds": 1, "nanos": 0}, "memory_limit_bytes": 1, "x": 2},
    ],
)
def test_limit_transform_rejects_invalid_or_overflow(raw):
    with pytest.raises(ExecutorProtocolError):
        derive_limits(raw)


def test_canonical_signing_and_tamper_detection():
    key = b"k" * 32
    payload = {"kind": "identity", "z": 1, "a": "é"}
    envelope = sign_payload(payload, key)
    assert canonical_json(payload) == '{"a":"é","kind":"identity","z":1}'.encode()
    assert verify_envelope(envelope, key, expected_kind="identity") == payload

    tampered = copy.deepcopy(envelope)
    tampered["payload"]["z"] = 2
    with pytest.raises(ExecutorProtocolError, match="signature"):
        verify_envelope(tampered, key, expected_kind="identity")


def test_strict_json_rejects_duplicate_keys_and_floats():
    with pytest.raises(ExecutorProtocolError, match="duplicate"):
        strict_json_loads(b'{"x":1,"x":2}')
    with pytest.raises(ExecutorProtocolError, match="floating"):
        strict_json_loads(b'{"x":1.2}')


def test_request_signs_raw_effective_limits_and_provenance():
    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    request = make_execute_request(
        code="print(input())",
        stdin="hello\n",
        raw_limits=_raw(1),
        identity_digest_value=payload_digest(identity),
        ttl_ns=10_000_000_000,
        request_id="00000000-0000-0000-0000-000000000001",
        now_ns=1_000_000_000,
    )
    limits = request["task"]["limits"]
    assert limits["raw"] == _raw(1)
    assert limits["transform"]["source_sha256"]
    assert limits["effective"]["stdout_bytes"] == 2 * 1024 * 1024


def test_static_identity_binds_all_runtime_and_safety_pins():
    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    assert identity["runtime"]["runsc_sha512"] == RUNSC_SHA512
    assert (
        identity["runtime"]["runsc_release_archive_path"] == RUNSC_RELEASE_ARCHIVE_PATH
    )
    assert (
        identity["runtime"]["runsc_release_archive_sha512"]
        == RUNSC_RELEASE_ARCHIVE_SHA512
    )
    assert (
        identity["runtime"]["runsc_release_archive_size_bytes"]
        == RUNSC_RELEASE_ARCHIVE_SIZE_BYTES
    )
    assert identity["runtime"]["runsc_size_bytes"] == RUNSC_SIZE_BYTES
    assert identity["runtime"]["runsc_spec_version"] == "1.2.1"
    assert identity["runtime"]["runsc_derivation"].startswith(
        "verified-official-tar.bz2"
    )
    assert identity["runtime"]["rootfs_sha256"] == ROOTFS_SHA256
    assert identity["runtime"]["candidate_uid"] == 65534
    assert identity["runtime"]["candidate_gid"] == 65534
    assert identity["runtime"]["trusted_monitor_bootstrap_capabilities"] == [
        "CAP_KILL",
        "CAP_SETGID",
        "CAP_SETPCAP",
        "CAP_SETUID",
        "CAP_SYS_PTRACE",
    ]
    assert identity["runtime"]["candidate_capability_sets"] == (
        "all-empty-before-candidate-gate"
    )
    assert identity["capacity"]["active_sandboxes"] == 4
    assert identity["capacity"]["configured_vm_memory_bytes"] == 32 * 1024**3
    assert identity["capacity"]["measured_guest_memory_bytes"] == (
        CONFIGURED_VM_MEMORY_BYTES
    )
    assert identity["capacity"]["service_memory_max_bytes"] == (
        SERVICE_MEMORY_MAX_BYTES
    )
    assert identity["capacity"]["configured_vm_vcpus"] == 16
    assert identity["capacity"]["service_cpu_affinity_vcpus"] == 4
    assert identity["limits"]["process_cap"] == 2
    assert identity["limits"]["wall_multiplier"] == 1
    assert identity["limits"]["cpu_semantics"] == (
        "RLIMIT_CPU_is_unreachable_backstop_for_valid_request_because_"
        "exact_1x_wall_precedes_ceil(raw)+1_at_one_CPU"
    )
    assert identity["limits"]["aggregate_writable_cap_bytes"] == 0
    assert identity["runtime"]["required_python_packages"] == (
        REQUIRED_PYTHON_PACKAGES
    )
    assert identity["runtime"]["candidate_python_flags"] == ["-I", "-B", "-c"]
    assert identity["runtime"]["candidate_import_path"] == (
        "isolated-safe-path-system-site-no-cwd"
    )
    assert identity["runtime"]["candidate_site_conveniences"] == ["exit", "quit"]
    assert identity["runtime"]["runsc_invocation"].startswith("rootful-oci")
    assert identity["runtime"]["no_new_privs"] is True
    assert identity["listener"] == {
        "scheme": "http",
        "host": "127.0.0.1",
        "port": 8080,
    }
    assert identity["limits"]["candidate_code_cap_bytes"] == 1024 * 1024
    assert identity["limits"]["host_cpu_budget_rule"] == (
        "2*effective_cpu_seconds*1000000+1000000"
    )
    assert identity["limits"]["host_cpu_calibration_sha256"] == (
        "72c48e00d27b13bc50aeb682d7bf01ab15d4a1a63e68d8abaca85125962cc411"
    )
    assert (
        derive_limits(_raw(30))["effective"]["host_cgroup_cpu_budget_us"] == 63_000_000
    )


@pytest.mark.parametrize(
    ("actual", "expected", "matches"),
    [
        ("YES\n", "yes ", True),
        ("1 2\n3", "1\t2\r\n3", True),
        ("3.500000", "3.5", True),
        ("1", "1.000000001", True),
        ("2147483648", "2147483648.0", True),
        ("1.00001", "1.0", False),
        ("0x10", "16", False),
        ("nan", "+nan", False),
    ],
)
def test_local_comparator_deepmind_parity(actual, expected, matches):
    assert outputs_match(actual, expected) is matches


def test_expected_output_is_not_a_protocol_field():
    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    marker = "EXPECTED_MUST_NEVER_LEAVE_DRIVER"
    request = make_execute_request(
        code="print('ok')",
        stdin="input only",
        raw_limits=_raw(1),
        identity_digest_value=payload_digest(identity),
        ttl_ns=10_000_000_000,
    )
    serialized = json.dumps(sign_payload(request, b"k" * 32))
    assert marker not in serialized
    assert set(request["task"]) == {
        "language",
        "code_b64",
        "stdin_b64",
        "limits",
    }


def test_signed_cgroup_cpu_event_is_unknown_and_requires_exact_crossing():
    limits = derive_limits(_raw(1))
    evidence = {
        "stdout_b64": "",
        "stderr_b64": "",
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "returncode": -15,
        "signal": 15,
        "controller_error": None,
        "resource_event": "CGROUP_CPU_BUDGET",
        "host_cpu_usage_us": limits["effective"]["host_cgroup_cpu_budget_us"],
        "host_cpu_before_usage_us": 10,
        "host_cpu_ready_usage_us": 100,
        "host_cpu_cross_usage_us": limits["effective"]["host_cgroup_cpu_budget_us"] + 100,
        "host_cpu_after_usage_us": limits["effective"]["host_cgroup_cpu_budget_us"] + 100,
        "host_cpu_budget_us": limits["effective"]["host_cgroup_cpu_budget_us"],
        "host_memory_peak_bytes": 64 * 1024 * 1024,
        "host_pids_peak": 30,
        "host_memory_events_before": {key: 0 for key in MEMORY_EVENT_KEYS},
        "host_memory_events_after": {key: 0 for key in MEMORY_EVENT_KEYS},
        "host_pids_events_before": {key: 0 for key in PIDS_EVENT_KEYS},
        "host_pids_events_after": {key: 0 for key in PIDS_EVENT_KEYS},
        "guest_cpu_usage_us": 0,
        "guest_process_peak": 0,
        "guest_process_limit": PID_CAP,
        "guest_rlimit_nproc": PINNED_GVISOR_RLIMIT_NPROC,
        "guest_process_limit_syscall": None,
        "guest_file_size_limit_bytes": FILE_SIZE_CAP_BYTES,
        "guest_writable_limit_bytes": WRITABLE_OVERLAY_CAP_BYTES,
        "guest_file_limit_signal": None,
        "guest_file_limit_errno": None,
        "guest_file_size_observed_bytes": 0,
        "guest_writable_available_bytes": 0,
        "resource_evidence_source": "request_cgroup_cpu_stat",
    }
    validate_execution_evidence(
        outcome="unknown",
        category="AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE",
        retryable=False,
        evidence=evidence,
        expected_limits=limits,
    )
    with pytest.raises(ExecutorProtocolError, match="never candidate-specific"):
        validate_execution_evidence(
            outcome="candidate_failure",
            category="CPU_LIMIT",
            retryable=False,
            evidence=evidence,
            expected_limits=limits,
        )
    evidence["host_cpu_cross_usage_us"] = (
        limits["effective"]["host_cgroup_cpu_budget_us"] + 99
    )
    with pytest.raises(ExecutorProtocolError, match="crossing"):
        validate_execution_evidence(
            outcome="unknown",
            category="AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE",
            retryable=False,
            evidence=evidence,
            expected_limits=limits,
        )

    guest_evidence = copy.deepcopy(evidence)
    guest_evidence.update(
        {
            "returncode": -9,
            "signal": 9,
            "resource_event": "GUEST_CPU_LIMIT",
            "guest_cpu_usage_us": 2_000_000,
            "resource_evidence_source": "guest_monitor_ptrace_siginfo",
            "host_cpu_usage_us": 0,
            "host_cpu_before_usage_us": 0,
            "host_cpu_ready_usage_us": 0,
            "host_cpu_cross_usage_us": 0,
            "host_cpu_after_usage_us": 0,
        }
    )
    validate_execution_evidence(
        outcome="candidate_failure",
        category="CPU_LIMIT",
        retryable=False,
        evidence=guest_evidence,
        expected_limits=limits,
    )
    guest_evidence["guest_cpu_usage_us"] = 1_999_999
    with pytest.raises(ExecutorProtocolError, match="cross"):
        validate_execution_evidence(
            outcome="candidate_failure",
            category="CPU_LIMIT",
            retryable=False,
            evidence=guest_evidence,
            expected_limits=limits,
        )
    guest_evidence["guest_cpu_usage_us"] = 4_000_001
    with pytest.raises(ExecutorProtocolError, match="cross"):
        validate_execution_evidence(
            outcome="candidate_failure",
            category="CPU_LIMIT",
            retryable=False,
            evidence=guest_evidence,
            expected_limits=limits,
        )


def _outer_resource_evidence(limits: dict[str, Any]) -> dict[str, Any]:
    return {
        "stdout_b64": "",
        "stderr_b64": "",
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "returncode": 0,
        "signal": None,
        "controller_error": None,
        "resource_event": None,
        "host_cpu_usage_us": 0,
        "host_cpu_before_usage_us": 100,
        "host_cpu_ready_usage_us": 100,
        "host_cpu_cross_usage_us": 0,
        "host_cpu_after_usage_us": 100,
        "host_cpu_budget_us": limits["effective"]["host_cgroup_cpu_budget_us"],
        "host_memory_peak_bytes": 64 * 1024 * 1024,
        "host_pids_peak": 30,
        "host_memory_events_before": {key: 0 for key in MEMORY_EVENT_KEYS},
        "host_memory_events_after": {key: 0 for key in MEMORY_EVENT_KEYS},
        "host_pids_events_before": {key: 0 for key in PIDS_EVENT_KEYS},
        "host_pids_events_after": {key: 0 for key in PIDS_EVENT_KEYS},
        "guest_cpu_usage_us": 0,
        "guest_process_peak": 0,
        "guest_process_limit": PID_CAP,
        "guest_rlimit_nproc": PINNED_GVISOR_RLIMIT_NPROC,
        "guest_process_limit_syscall": None,
        "guest_file_size_limit_bytes": FILE_SIZE_CAP_BYTES,
        "guest_writable_limit_bytes": WRITABLE_OVERLAY_CAP_BYTES,
        "guest_file_limit_signal": None,
        "guest_file_limit_errno": None,
        "guest_file_size_observed_bytes": 0,
        "guest_writable_available_bytes": 0,
        "resource_evidence_source": None,
    }


@pytest.mark.parametrize(
    ("event", "source", "counter_group", "counter_key"),
    [
        (
            "CGROUP_MEMORY_OOM",
            "request_cgroup_memory_events",
            "host_memory_events_after",
            "oom",
        ),
        (
            "CGROUP_PIDS_MAX",
            "request_cgroup_pids_events",
            "host_pids_events_after",
            "max",
        ),
    ],
)
def test_outer_memory_and_pids_events_can_only_sign_unknown(
    event: str,
    source: str,
    counter_group: str,
    counter_key: str,
) -> None:
    limits = derive_limits(_raw(1))
    evidence = _outer_resource_evidence(limits)
    evidence[counter_group][counter_key] = 1
    evidence["resource_event"] = event
    evidence["resource_evidence_source"] = source
    validate_execution_evidence(
        outcome="unknown",
        category="AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE",
        retryable=False,
        evidence=evidence,
        expected_limits=limits,
    )
    with pytest.raises(ExecutorProtocolError, match="never candidate-specific"):
        validate_execution_evidence(
            outcome="candidate_failure",
            category="PROCESS_LIMIT",
            retryable=False,
            evidence=evidence,
            expected_limits=limits,
        )


def test_multiple_outer_events_are_signed_raw_but_never_singly_classified() -> None:
    limits = derive_limits(_raw(1))
    evidence = _outer_resource_evidence(limits)
    evidence["host_memory_events_after"]["oom"] = 1
    evidence["host_pids_events_after"]["max"] = 1
    validate_execution_evidence(
        outcome="unknown",
        category="AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE",
        retryable=False,
        evidence=evidence,
        expected_limits=limits,
    )

    evidence["resource_event"] = "CGROUP_MEMORY_OOM"
    evidence["resource_evidence_source"] = "request_cgroup_memory_events"
    with pytest.raises(ExecutorProtocolError, match="multiple outer"):
        validate_execution_evidence(
            outcome="unknown",
            category="AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE",
            retryable=False,
            evidence=evidence,
            expected_limits=limits,
        )


def test_candidate_memory_limit_category_is_not_signable_from_outer_oom() -> None:
    limits = derive_limits(_raw(1))
    evidence = _outer_resource_evidence(limits)
    evidence["host_memory_events_after"]["oom_kill"] = 1
    evidence["resource_event"] = "CGROUP_MEMORY_OOM"
    evidence["resource_evidence_source"] = "request_cgroup_memory_events"
    with pytest.raises(ExecutorProtocolError):
        validate_execution_evidence(
            outcome="candidate_failure",
            category="MEMORY_LIMIT",
            retryable=False,
            evidence=evidence,
            expected_limits=limits,
        )


def test_guest_process_limit_binds_semantic_raw_and_syscall_evidence():
    limits = derive_limits(_raw(1))
    evidence = {
        "stdout_b64": "",
        "stderr_b64": "",
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "returncode": 1,
        "signal": None,
        "controller_error": None,
        "resource_event": "GUEST_PROCESS_LIMIT",
        "host_cpu_usage_us": 0,
        "host_cpu_before_usage_us": 0,
        "host_cpu_ready_usage_us": 0,
        "host_cpu_cross_usage_us": 0,
        "host_cpu_after_usage_us": 0,
        "host_cpu_budget_us": limits["effective"]["host_cgroup_cpu_budget_us"],
        "host_memory_peak_bytes": 64 * 1024 * 1024,
        "host_pids_peak": 30,
        "host_memory_events_before": {key: 0 for key in MEMORY_EVENT_KEYS},
        "host_memory_events_after": {key: 0 for key in MEMORY_EVENT_KEYS},
        "host_pids_events_before": {key: 0 for key in PIDS_EVENT_KEYS},
        "host_pids_events_after": {key: 0 for key in PIDS_EVENT_KEYS},
        "guest_cpu_usage_us": 0,
        "guest_process_peak": PID_CAP,
        "guest_process_limit": PID_CAP,
        "guest_rlimit_nproc": PINNED_GVISOR_RLIMIT_NPROC,
        "guest_process_limit_syscall": 220,
        "guest_file_size_limit_bytes": FILE_SIZE_CAP_BYTES,
        "guest_writable_limit_bytes": WRITABLE_OVERLAY_CAP_BYTES,
        "guest_file_limit_signal": None,
        "guest_file_limit_errno": None,
        "guest_file_size_observed_bytes": 0,
        "guest_writable_available_bytes": 0,
        "resource_evidence_source": "guest_monitor_ptrace_thread_eagain",
    }
    validate_execution_evidence(
        outcome="candidate_failure",
        category="PROCESS_LIMIT",
        retryable=False,
        evidence=evidence,
        expected_limits=limits,
    )

    handled_denial = copy.deepcopy(evidence)
    handled_denial["returncode"] = 0
    validate_execution_evidence(
        outcome="executed",
        category=None,
        retryable=False,
        evidence=handled_denial,
        expected_limits=limits,
    )

    missing_syscall = copy.deepcopy(evidence)
    missing_syscall["guest_process_limit_syscall"] = None
    with pytest.raises(ExecutorProtocolError, match="syscall provenance"):
        validate_execution_evidence(
            outcome="candidate_failure",
            category="PROCESS_LIMIT",
            retryable=False,
            evidence=missing_syscall,
            expected_limits=limits,
        )

    wrong_raw_limit = copy.deepcopy(evidence)
    wrong_raw_limit["guest_rlimit_nproc"] = 2
    with pytest.raises(ExecutorProtocolError, match="semantic/raw"):
        validate_execution_evidence(
            outcome="candidate_failure",
            category="PROCESS_LIMIT",
            retryable=False,
            evidence=wrong_raw_limit,
            expected_limits=limits,
        )

    legal_peak_only = copy.deepcopy(evidence)
    legal_peak_only["resource_event"] = None
    legal_peak_only["resource_evidence_source"] = None
    legal_peak_only["guest_process_limit_syscall"] = None
    legal_peak_only["returncode"] = 0
    validate_execution_evidence(
        outcome="executed",
        category=None,
        retryable=False,
        evidence=legal_peak_only,
        expected_limits=limits,
    )


def test_guest_file_limit_binds_traced_errno_and_exact_size():
    limits = derive_limits(_raw(1))
    evidence = {
        "stdout_b64": "",
        "stderr_b64": "",
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "returncode": 1,
        "signal": None,
        "controller_error": None,
        "resource_event": "GUEST_FILE_SPACE_LIMIT",
        "host_cpu_usage_us": 0,
        "host_cpu_before_usage_us": 0,
        "host_cpu_ready_usage_us": 0,
        "host_cpu_cross_usage_us": 0,
        "host_cpu_after_usage_us": 0,
        "host_cpu_budget_us": limits["effective"]["host_cgroup_cpu_budget_us"],
        "host_memory_peak_bytes": 64 * 1024 * 1024,
        "host_pids_peak": 30,
        "host_memory_events_before": {key: 0 for key in MEMORY_EVENT_KEYS},
        "host_memory_events_after": {key: 0 for key in MEMORY_EVENT_KEYS},
        "host_pids_events_before": {key: 0 for key in PIDS_EVENT_KEYS},
        "host_pids_events_after": {key: 0 for key in PIDS_EVENT_KEYS},
        "guest_cpu_usage_us": 0,
        "guest_process_peak": 1,
        "guest_process_limit": PID_CAP,
        "guest_rlimit_nproc": PINNED_GVISOR_RLIMIT_NPROC,
        "guest_process_limit_syscall": None,
        "guest_file_size_limit_bytes": FILE_SIZE_CAP_BYTES,
        "guest_writable_limit_bytes": WRITABLE_OVERLAY_CAP_BYTES,
        "guest_file_limit_signal": None,
        "guest_file_limit_errno": 27,
        "guest_file_size_observed_bytes": FILE_SIZE_CAP_BYTES,
        "guest_writable_available_bytes": 0,
        "resource_evidence_source": "guest_monitor_ptrace_write_efbig",
    }
    validate_execution_evidence(
        outcome="candidate_failure",
        category="FILE_SPACE_LIMIT",
        retryable=False,
        evidence=evidence,
        expected_limits=limits,
    )

    missing_errno = copy.deepcopy(evidence)
    missing_errno["guest_file_limit_errno"] = None
    with pytest.raises(ExecutorProtocolError, match="errno provenance"):
        validate_execution_evidence(
            outcome="candidate_failure",
            category="FILE_SPACE_LIMIT",
            retryable=False,
            evidence=missing_errno,
            expected_limits=limits,
        )

    forged_signal = copy.deepcopy(evidence)
    forged_signal["guest_file_limit_signal"] = 25
    with pytest.raises(ExecutorProtocolError, match="signal provenance"):
        validate_execution_evidence(
            outcome="candidate_failure",
            category="FILE_SPACE_LIMIT",
            retryable=False,
            evidence=forged_signal,
            expected_limits=limits,
        )

    wrong_size = copy.deepcopy(evidence)
    wrong_size["guest_file_size_observed_bytes"] -= 1
    with pytest.raises(ExecutorProtocolError, match="exact file size"):
        validate_execution_evidence(
            outcome="candidate_failure",
            category="FILE_SPACE_LIMIT",
            retryable=False,
            evidence=wrong_size,
            expected_limits=limits,
        )
