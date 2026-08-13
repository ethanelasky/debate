"""Adversarial trust-boundary tests for the production executor."""

from __future__ import annotations

import base64
import http.client
import importlib.util
import io
import json
import os
import shutil
import threading
import urllib.request
from pathlib import Path

import pytest

from codecontests_executor import sandbox_launcher
from codecontests_executor.client import (
    UrllibTransport,
    _RejectRedirectHandler,
)
from codecontests_executor.protocol import (
    MAX_CODE_BYTES,
    MAX_REQUEST_BODY_BYTES,
    MAX_STDIN_BYTES,
    MEMORY_EVENT_KEYS,
    PIDS_EVENT_KEYS,
    ExecutorProtocolError,
    derive_limits,
    encode_envelope,
    make_execute_request,
    payload_digest,
    sign_payload,
    static_identity,
    validate_execute_request,
    verify_envelope,
)
from codecontests_executor.service import (
    ACTIVE_SANDBOXES,
    DEFAULT_QUEUE_CAPACITY,
    ExecutorApplication,
    _validated_listener,
    build_rootfs_manifest,
    verify_host_capacity,
)
from codecontests_executor.supervisor import SandboxSupervisor

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_twenty_max_request_bodies_have_tight_pre_admission_bound():
    assert MAX_STDIN_BYTES == 1024 * 1024
    assert (
        MAX_REQUEST_BODY_BYTES * (ACTIVE_SANDBOXES + DEFAULT_QUEUE_CAPACITY)
        < 64 * 1024 * 1024
    )
    identity = static_identity(service_id="body-cap-test", launcher_sha256="a" * 64)
    request = make_execute_request(
        code="x" * MAX_CODE_BYTES,
        stdin="y" * MAX_STDIN_BYTES,
        raw_limits={
            "time_limit": {"seconds": 1, "nanos": 0},
            "memory_limit_bytes": 4 * 1024**3,
        },
        identity_digest_value=payload_digest(identity),
        ttl_ns=1_000_000_000,
    )
    assert len(encode_envelope(sign_payload(request, b"k" * 32))) <= (
        MAX_REQUEST_BODY_BYTES
    )


@pytest.mark.parametrize("signature", ["é", "hmac-sha256:" + "é" * 64])
def test_non_ascii_signature_is_bounded_protocol_error(signature):
    with pytest.raises(ExecutorProtocolError):
        verify_envelope(
            {"payload": {"kind": "identity"}, "signature": signature},
            b"k" * 32,
            expected_kind="identity",
        )


def test_non_ascii_request_identity_digest_is_bounded_protocol_error():
    identity = static_identity(service_id="unicode-test", launcher_sha256="a" * 64)
    request = make_execute_request(
        code="pass",
        stdin="",
        raw_limits={
            "time_limit": {"seconds": 1, "nanos": 0},
            "memory_limit_bytes": 4 * 1024**3,
        },
        identity_digest_value=payload_digest(identity),
        ttl_ns=1_000_000_000,
        now_ns=10,
    )
    request["identity_digest"] = "é"
    with pytest.raises(ExecutorProtocolError):
        validate_execute_request(
            request,
            expected_identity_digest=payload_digest(identity),
            now_ns=11,
        )


def test_non_ascii_authorization_is_rejected_without_compare_digest_crash():
    identity = static_identity(service_id="auth-unicode", launcher_sha256="a" * 64)
    application = ExecutorApplication(
        bearer_token=b"b" * 32,
        hmac_key=b"h" * 32,
        identity=identity,
        supervisor=object(),  # type: ignore[arg-type]
    )
    assert application.authorized("Bearer é") is False


def _valid_controller_result():
    limits = derive_limits(
        {
            "time_limit": {"seconds": 1, "nanos": 0},
            "memory_limit_bytes": 4 * 1024**3,
        }
    )
    return {
        "outcome": "executed",
        "category": None,
        "retryable": False,
        "stdout_b64": base64.b64encode(b"ok").decode("ascii"),
        "stderr_b64": "",
        "stdout_bytes": 2,
        "stderr_bytes": 0,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "returncode": 0,
        "signal": None,
        "controller_error": None,
        "resource_event": None,
        "host_cpu_usage_us": 0,
        "host_cpu_before_usage_us": 0,
        "host_cpu_ready_usage_us": 0,
        "host_cpu_cross_usage_us": 0,
        "host_cpu_after_usage_us": 0,
        "host_cpu_budget_us": limits["effective"]["host_cgroup_cpu_budget_us"],
        "host_memory_peak_bytes": 0,
        "host_pids_peak": 0,
        "host_memory_events_before": {key: 0 for key in MEMORY_EVENT_KEYS},
        "host_memory_events_after": {key: 0 for key in MEMORY_EVENT_KEYS},
        "host_pids_events_before": {key: 0 for key in PIDS_EVENT_KEYS},
        "host_pids_events_after": {key: 0 for key in PIDS_EVENT_KEYS},
        "guest_cpu_usage_us": 0,
        "guest_process_peak": 0,
        "guest_process_limit": 2,
        "guest_rlimit_nproc": 1,
        "guest_process_limit_syscall": None,
        "guest_file_size_limit_bytes": 2 * 1024 * 1024,
        "guest_writable_limit_bytes": 0,
        "guest_file_limit_signal": None,
        "guest_file_limit_errno": None,
        "guest_file_size_observed_bytes": 0,
        "guest_writable_available_bytes": 0,
        "resource_evidence_source": None,
        "execution_ns": 1,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("outcome", []),
        ("stdout_b64", b"b2s="),
        ("stdout_bytes", True),
        ("controller_error", "x" * 10_000),
        ("category", "NOT_A_CATEGORY"),
        ("stdout_truncated", True),
        ("returncode", True),
        ("signal", 9),
    ],
)
def test_malformed_controller_fields_become_bounded_signed_unknown(field, value):
    result = _valid_controller_result()
    result[field] = value
    normalized = ExecutorApplication._normalize_execution(
        result,
        derive_limits(
            {
                "time_limit": {"seconds": 1, "nanos": 0},
                "memory_limit_bytes": 4 * 1024**3,
            }
        ),
    )
    assert normalized["outcome"] == "unknown"
    assert normalized["category"] == "CONTROLLER_RESULT_INVALID"
    assert normalized["retryable"] is False
    assert normalized["controller_error"] == "schema"


@pytest.mark.parametrize(
    "host,port",
    [
        ("0.0.0.0", 8080),
        ("::1", 8080),
        ("127.0.0.1", 8787),
    ],
)
def test_plaintext_listener_rejects_identity_drift(host, port):
    with pytest.raises(RuntimeError, match="127.0.0.1:8080"):
        _validated_listener(host, port)


@pytest.mark.parametrize(
    "destination",
    [
        "https://executor.example/v1/execute-elsewhere",
        "http://127.0.0.1:9999/stolen",
    ],
)
def test_redirect_handler_rejects_before_constructing_second_request(destination):
    handler = _RejectRedirectHandler()
    original = urllib.request.Request(
        "https://executor.example/v1/execute",
        headers={"Authorization": "Bearer secret"},
    )
    with pytest.raises(ExecutorProtocolError, match="redirect"):
        handler.redirect_request(
            original,
            io.BytesIO(),
            302,
            "Found",
            http.client.HTTPMessage(),
            destination,
        )


class _MismatchedResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def geturl(self):
        return "https://other.example/v1/identity"

    def read(self, _limit):
        raise AssertionError("body must not be read from a mismatched endpoint")


class _MismatchedOpener:
    def open(self, _request, timeout):
        del timeout
        return _MismatchedResponse()


def test_transport_rejects_response_endpoint_mismatch_before_body_read():
    with pytest.raises(ExecutorProtocolError, match="endpoint"):
        UrllibTransport(opener=_MismatchedOpener()).request(
            method="GET",
            url="https://executor.example/v1/identity",
            headers={"Authorization": "Bearer secret"},
            body=None,
            timeout_seconds=1,
        )


def test_two_phase_input_writer_releases_byte_exact_stdin_only_after_gate():
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "rb", buffering=0)
    writer = os.fdopen(write_fd, "wb", buffering=0)
    release = threading.Event()
    errors: list[str] = []
    thread = threading.Thread(
        target=SandboxSupervisor._write_input,
        args=(writer, b"source-frame", b"\x00stdin\n", release, errors),
    )
    thread.start()
    assert reader.read(len(b"source-frame")) == b"source-frame"
    release.set()
    assert reader.read() == b"G\x00stdin\n"
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert errors == []


def test_no_new_privs_requires_set_and_verified_get(monkeypatch):
    class Libc:
        def __init__(self, get_value):
            self.get_value = get_value

        def prctl(self, operation, *_args):
            return 0 if operation == 38 else self.get_value

    monkeypatch.setattr(
        sandbox_launcher.ctypes, "CDLL", lambda *_args, **_kwargs: Libc(1)
    )
    sandbox_launcher._set_no_new_privs()
    monkeypatch.setattr(
        sandbox_launcher.ctypes, "CDLL", lambda *_args, **_kwargs: Libc(0)
    )
    with pytest.raises(RuntimeError, match="verification"):
        sandbox_launcher._set_no_new_privs()


def test_capacity_measurement_binds_sysctls_service_caps_and_swap(
    monkeypatch, tmp_path
):
    values = {
        "memory.max": "25769803776\n",
        "memory.swap.max": "0\n",
        "pids.max": "768\n",
        "cpu.max": "400000 100000\n",
        "cgroup.procs": "\n",
        "userns": "1\n",
        "max_userns": "128\n",
        "apparmor": "0\n",
        "swaps": "Filename Type Size Used Priority\n",
    }
    paths = {}
    for name, value in values.items():
        path = tmp_path / name
        path.write_text(value)
        paths[name] = str(path)
    monkeypatch.setattr(
        os,
        "sched_getaffinity",
        lambda _pid: {0, 1, 2, 3},
        raising=False,
    )
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    original_sysconf = os.sysconf
    monkeypatch.setattr(
        os,
        "sysconf",
        lambda key: (
            8_100_000
            if key == "SC_PHYS_PAGES"
            else 4096
            if key == "SC_PAGE_SIZE"
            else original_sysconf(key)
        ),
    )
    measured = verify_host_capacity(
        memory_max_path=paths["memory.max"],
        memory_swap_max_path=paths["memory.swap.max"],
        pids_max_path=paths["pids.max"],
        cpu_max_path=paths["cpu.max"],
        delegated_root_procs_path=paths["cgroup.procs"],
        proc_swaps_path=paths["swaps"],
        userns_clone_path=paths["userns"],
        max_userns_path=paths["max_userns"],
        apparmor_userns_path=paths["apparmor"],
    )
    assert measured["service_tasks_max"] == 768
    assert measured["service_memory_swap_max_bytes"] == 0
    assert measured["guest_swap_enabled"] is False
    assert measured["cpu_affinity_cpus"] == [0, 1, 2, 3]
    assert measured["service_cpu_quota_us"] == 400_000
    assert measured["delegated_root_cgroup_procs_empty"] is True


def test_preimport_inventory_rejects_tamper_and_extra_cache(tmp_path):
    measure = _load_file_module(
        "measure_executor_bundle_test",
        REPO_ROOT / "scripts/measure_codecontests_executor_bundle.py",
    )
    verifier = _load_file_module(
        "verify_staged_executor_test",
        REPO_ROOT / "deploy/codecontests-executor/verify_staged_executor.py",
    )
    package = tmp_path / "package"
    package.mkdir()
    for filename in measure.SERVER_BUNDLE_FILES:
        shutil.copyfile(
            REPO_ROOT / "codecontests_executor" / filename, package / filename
        )
    inventory = measure.build_inventory(package)
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_bytes(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    required_uid = os.geteuid()
    assert (
        verifier.verify_staged_executor(
            package_dir=str(package),
            inventory_path=str(inventory_path),
            expected_bundle_sha256=inventory["bundle_sha256"],
            required_uid=required_uid,
        )
        == inventory["bundle_sha256"]
    )
    (package / "client.py").write_text("# tampered\n")
    with pytest.raises(RuntimeError, match="checksum"):
        verifier.verify_staged_executor(
            package_dir=str(package),
            inventory_path=str(inventory_path),
            expected_bundle_sha256=inventory["bundle_sha256"],
            required_uid=required_uid,
        )
    shutil.copyfile(
        REPO_ROOT / "codecontests_executor/client.py", package / "client.py"
    )
    (package / "__pycache__").mkdir()
    with pytest.raises(RuntimeError, match="extra/cache"):
        verifier.verify_staged_executor(
            package_dir=str(package),
            inventory_path=str(inventory_path),
            expected_bundle_sha256=inventory["bundle_sha256"],
            required_uid=required_uid,
        )


def test_systemd_asset_pins_preimport_gate_listener_and_service_caps():
    unit = (
        REPO_ROOT / "deploy/codecontests-executor/codecontests-executor.service"
    ).read_text()
    assert "ExecStartPre=" in unit
    assert "--bind 127.0.0.1 --port 8080" in unit
    assert "MemoryMax=25769803776" in unit
    assert "MemorySwapMax=0" in unit
    assert "TasksMax=768" in unit
    assert "NoNewPrivileges=no" in unit
    assert "MemoryDenyWriteExecute=no" in unit
    assert "Delegate=cpu memory pids" in unit
    assert "DelegateSubgroup=service" in unit
    assert "BindReadOnlyPaths=/var/lib/codecontests-executor/rootfs" in unit
    assert "--rootless" not in unit


def test_rootfs_builder_pins_packages_digest_ownership_and_privilege_bits():
    builder = (
        REPO_ROOT / "scripts/build_codecontests_rootfs_artifact.sh"
    ).read_text()
    assert (
        'EXPECTED_SHA256="83e694da5d1e0b94700da2a195d760527ce609ea631f7302ec930666bae136d0"'
        in builder
    )
    assert "sympy=1.14.0" in builder and "mpmath=1.3.0" in builder
    assert "! -user root -o ! -group root" in builder
    assert "-perm /6000" in builder
    assert "/usr/bin/python3 -I -B" in builder


@pytest.mark.skipif(
    not Path("/proc/self/fd").exists(), reason="fd-relative Linux manifest walker"
)
def test_rootfs_manifest_binds_xattrs_and_extra_paths(tmp_path):
    root = tmp_path / "rootfs"
    root.mkdir()
    source = root / "file"
    source.write_bytes(b"original")
    setxattr = getattr(os, "setxattr", None)
    if setxattr is None:
        pytest.skip("host Python lacks os.setxattr")
    try:
        setxattr(source, "user.palaestra", b"bound")
    except OSError:
        pytest.skip("test filesystem lacks user xattrs")
    first = build_rootfs_manifest(str(root))
    source.write_bytes(b"changed")
    second = build_rootfs_manifest(str(root))
    assert first != second
    source.write_bytes(b"original")
    (root / "extra").write_bytes(b"x")
    third = build_rootfs_manifest(str(root))
    assert len(third["entries"]) == len(first["entries"]) + 1
