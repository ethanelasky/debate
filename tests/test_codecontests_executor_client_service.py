"""Direct protocol/service/client behavior probes (no gVisor fake oracle)."""

from __future__ import annotations

import base64
import copy
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any

import pytest

from codecontests_executor.client import RemoteExecutorClient
from codecontests_executor.protocol import (
    MEMORY_EVENT_KEYS,
    PIDS_EVENT_KEYS,
    ExecutorProtocolError,
    derive_limits,
    encode_envelope,
    payload_digest,
    sign_payload,
    static_identity,
    strict_json_loads,
    verify_envelope,
)
from codecontests_executor.service import ExecutorApplication, ExecutorHTTPServer

KEY = b"h" * 32
BEARER = b"b" * 32
RAW_LIMITS = {
    "time_limit": {"seconds": 1, "nanos": 0},
    "memory_limit_bytes": 4 * 1024**3,
}


def test_loopback_server_can_rebind_after_authenticated_probe_time_wait() -> None:
    assert ExecutorHTTPServer.allow_reuse_address is True


def _execution(
    stdout: bytes = b"ok\n",
    *,
    outcome: str = "executed",
    category: str | None = None,
) -> dict[str, Any]:
    limits = derive_limits(RAW_LIMITS)
    return {
        "outcome": outcome,
        "category": category,
        "retryable": False,
        "stdout_b64": base64.b64encode(stdout).decode(),
        "stderr_b64": "",
        "stdout_bytes": len(stdout),
        "stderr_bytes": 0,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "returncode": 0 if outcome == "executed" else 1,
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
        "execution_ns": 123,
    }


class StubSupervisor:
    launcher_sha256 = "a" * 64

    def __init__(self, result: dict[str, Any] | None = None):
        self.result = result or _execution()
        self.requests: list[dict[str, Any]] = []

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(copy.deepcopy(request))
        return copy.deepcopy(self.result)


class ApplicationTransport:
    def __init__(
        self,
        app: ExecutorApplication,
        *,
        bearer: bytes = BEARER,
    ):
        self.app = app
        self.bearer = bearer
        self.posts: list[bytes] = []

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> tuple[int, bytes]:
        del url, timeout_seconds
        if not self.app.authorized(headers.get("Authorization")):
            return HTTPStatus.UNAUTHORIZED, b'{"error":"unauthorized"}'
        if method == "GET":
            return HTTPStatus.OK, self.app.signed_identity()
        assert body is not None
        self.posts.append(body)
        return self.app.execute(body)


def _stack(
    *,
    result: dict[str, Any] | None = None,
    expected_identity: dict[str, Any] | None = None,
    bearer: bytes = BEARER,
):
    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    supervisor = StubSupervisor(result)
    app = ExecutorApplication(
        bearer_token=BEARER,
        hmac_key=KEY,
        identity=identity,
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    transport = ApplicationTransport(app, bearer=bearer)
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=bearer,
        hmac_key=KEY,
        expected_identity=expected_identity or identity,
        transport=transport,
        sleep=lambda _seconds: None,
    )
    return identity, supervisor, app, transport, client


def test_bearer_is_required_before_protocol_handling():
    _identity, _supervisor, app, _transport, _client = _stack()
    assert app.authorized(None) is False
    assert app.authorized("Bearer wrong") is False
    assert app.authorized("Bearer " + BEARER.decode()) is True


def test_client_verifies_identity_and_result_and_never_sends_expected_output():
    _identity, supervisor, _app, transport, client = _stack(
        result=_execution(b"x" * 8192)
    )
    expected_marker = "EXPECTED_OUTPUT_PRIVATE_7ef5"
    result = client.execute(
        code="print('x')",
        stdin="public stdin",
        raw_limits=RAW_LIMITS,
    )
    assert result.outcome == "executed"
    assert result.stdout == b"x" * 8192
    assert len(supervisor.requests) == 1
    serialized = transport.posts[0].decode()
    assert expected_marker not in serialized
    task = supervisor.requests[0]["task"]
    assert set(task) == {"language", "code_b64", "stdin_b64", "limits"}


def test_identity_mismatch_is_unknown_and_never_executes():
    expected = static_identity(
        service_id="wrong-executor",
        launcher_sha256="a" * 64,
    )
    _identity, supervisor, _app, _transport, client = _stack(expected_identity=expected)
    result = client.execute(code="pass", stdin="", raw_limits=RAW_LIMITS)
    assert result.outcome == "unknown"
    assert result.category == "TRANSPORT_OR_ATTESTATION"
    assert supervisor.requests == []


class TamperTransport(ApplicationTransport):
    def request(self, **kwargs):  # type: ignore[no-untyped-def]
        status, body = super().request(**kwargs)
        if kwargs["method"] == "POST":
            envelope = strict_json_loads(body)
            envelope["payload"]["outcome"] = "candidate_failure"
            body = encode_envelope(envelope)
        return status, body


def test_tampered_signed_result_is_unknown():
    identity, supervisor, app, _transport, _client = _stack()
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=TamperTransport(app),
    )
    result = client.execute(code="pass", stdin="", raw_limits=RAW_LIMITS)
    assert result.outcome == "unknown"
    assert result.category == "TRANSPORT_OR_ATTESTATION"
    assert len(supervisor.requests) == 1


class RebindTransport(ApplicationTransport):
    def request(self, **kwargs):  # type: ignore[no-untyped-def]
        status, body = super().request(**kwargs)
        if kwargs["method"] == "POST":
            envelope = strict_json_loads(body)
            payload = envelope["payload"]
            payload["request_digest"] = "0" * 64
            body = encode_envelope(sign_payload(payload, KEY))
        return status, body


def test_validly_signed_result_for_wrong_request_is_unknown():
    identity, _supervisor, app, _transport, _client = _stack()
    client = RemoteExecutorClient(
        base_url="https://executor.invalid",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=RebindTransport(app),
    )
    result = client.execute(code="pass", stdin="", raw_limits=RAW_LIMITS)
    assert result.outcome == "unknown"
    assert result.category == "TRANSPORT_OR_ATTESTATION"


class DeadTransport:
    def request(self, **kwargs):  # type: ignore[no-untyped-def]
        raise ConnectionResetError("service died")


def test_service_death_is_unknown():
    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    client = RemoteExecutorClient(
        base_url="https://executor.invalid",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=DeadTransport(),
    )
    result = client.execute(code="pass", stdin="", raw_limits=RAW_LIMITS)
    assert result.outcome == "unknown"
    assert result.category == "TRANSPORT_OR_ATTESTATION"


def test_overload_is_signed_retryable_unknown():
    identity, _supervisor, app, _transport, client = _stack()
    client.verify_identity()
    # Saturate active+queue admission directly.
    for _ in range(20):
        assert app._admitted.acquire(blocking=False)
    try:
        from codecontests_executor.protocol import make_execute_request

        request = make_execute_request(
            code="pass",
            stdin="",
            raw_limits=RAW_LIMITS,
            identity_digest_value=payload_digest(identity),
            ttl_ns=10_000_000_000,
        )
        status, body = app.execute(encode_envelope(sign_payload(request, KEY)))
        assert status == HTTPStatus.SERVICE_UNAVAILABLE
        result = verify_envelope(
            strict_json_loads(body), KEY, expected_kind="execute_result"
        )
        assert result["outcome"] == "unknown"
        assert result["category"] == "OVERLOADED"
        assert result["retryable"] is True
        assert set(result["timing"]) == {"queue_ns", "execution_ns", "total_ns"}
    finally:
        for _ in range(20):
            app._admitted.release()


def test_validly_signed_malformed_request_is_bounded_protocol_error():
    _identity, supervisor, app, _transport, _client = _stack()
    malformed = {
        "kind": "execute_request",
        "protocol_version": "palaestra.codecontests.executor.v1",
    }
    status, body = app.execute(encode_envelope(sign_payload(malformed, KEY)))
    assert status == HTTPStatus.BAD_REQUEST
    error = verify_envelope(
        strict_json_loads(body), KEY, expected_kind="protocol_error"
    )
    assert error["request_digest"] == payload_digest(malformed)
    assert supervisor.requests == []


@pytest.mark.parametrize("field", ["code_b64", "stdin_b64"])
def test_valid_hmac_non_ascii_base64_is_bounded_signed_protocol_error(field):
    identity, supervisor, app, _transport, _client = _stack()
    from codecontests_executor.protocol import make_execute_request

    malformed = make_execute_request(
        code="pass",
        stdin="",
        raw_limits=RAW_LIMITS,
        identity_digest_value=payload_digest(identity),
        ttl_ns=10_000_000_000,
    )
    malformed["task"][field] = "é"
    status, body = app.execute(encode_envelope(sign_payload(malformed, KEY)))
    assert status == HTTPStatus.BAD_REQUEST
    error = verify_envelope(
        strict_json_loads(body), KEY, expected_kind="protocol_error"
    )
    assert error["request_digest"] == payload_digest(malformed)
    assert supervisor.requests == []


def test_malformed_controller_result_becomes_signed_unknown():
    identity, _supervisor, _app, transport, _client = _stack(result={"bad": True})
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=transport,
    )
    result = client.execute(code="pass", stdin="", raw_limits=RAW_LIMITS)
    assert result.outcome == "unknown"
    assert result.category == "CONTROLLER_RESULT_INVALID"
    assert result.error == "schema"


def test_controller_exception_class_survives_signed_client_boundary(monkeypatch):
    _identity, supervisor, _app, _transport, client = _stack()

    def fail(_request):
        raise FileNotFoundError("private server-side detail")

    monkeypatch.setattr(supervisor, "execute", fail)
    result = client.execute(code="pass", stdin="", raw_limits=RAW_LIMITS)

    assert result.outcome == "unknown"
    assert result.category == "CONTROLLER_EXCEPTION"
    assert result.error == "FileNotFoundError"
    assert result.result_payload["evidence"]["controller_error"] == "FileNotFoundError"


def test_signed_executed_result_with_nonzero_returncode_is_unknown():
    inconsistent = _execution()
    inconsistent["returncode"] = 1
    identity, _supervisor, _app, transport, _client = _stack(result=inconsistent)
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=transport,
    )
    result = client.execute(code="pass", stdin="", raw_limits=RAW_LIMITS)
    assert result.outcome == "unknown"
    assert result.category == "CONTROLLER_RESULT_INVALID"


class OverloadOnceTransport(ApplicationTransport):
    def __init__(self, app: ExecutorApplication):
        super().__init__(app)
        self.execute_attempts = 0

    def request(self, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs["method"] == "POST":
            self.execute_attempts += 1
            if self.execute_attempts == 1:
                for _ in range(20):
                    assert self.app._admitted.acquire(blocking=False)
                try:
                    return super().request(**kwargs)
                finally:
                    for _ in range(20):
                        self.app._admitted.release()
        return super().request(**kwargs)


def test_client_retries_only_signed_retryable_overload():
    identity, supervisor, app, _transport, _client = _stack()
    transport = OverloadOnceTransport(app)
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=transport,
        sleep=lambda _seconds: None,
    )
    result = client.execute(code="print('ok')", stdin="", raw_limits=RAW_LIMITS)
    assert result.outcome == "executed"
    assert transport.execute_attempts == 2
    assert len(supervisor.requests) == 1


class CountingSupervisor(StubSupervisor):
    def __init__(self):
        super().__init__(_execution(b"ok\n"))
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            return super().execute(request)
        finally:
            with self._lock:
                self.active -= 1


def test_driver_admission_gate_completes_thirty_two_executions_without_overload():
    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    supervisor = CountingSupervisor()
    app = ExecutorApplication(
        bearer_token=BEARER,
        hmac_key=KEY,
        identity=identity,
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=ApplicationTransport(app),
    )
    client.verify_identity()
    admission = threading.Semaphore(8)

    def execute(index: int):  # type: ignore[no-untyped-def]
        with admission:
            return client.execute(
                code=f"print('ok') # {index}",
                stdin="input",
                raw_limits=RAW_LIMITS,
            )

    with ThreadPoolExecutor(max_workers=32) as pool:
        executions = list(pool.map(execute, range(32)))
    assert all(execution.outcome == "executed" for execution in executions)
    assert all(execution.stdout == b"ok\n" for execution in executions)
    assert supervisor.max_active <= 4
    assert len(supervisor.requests) == 32


def test_url_without_frozen_identity_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("CODECONTESTS_EXECUTOR_URL", "https://executor.invalid")
    monkeypatch.delenv("CODECONTESTS_EXECUTOR_IDENTITY_FILE", raising=False)
    with pytest.raises(ExecutorProtocolError, match="identity"):
        RemoteExecutorClient.from_env()


@pytest.mark.parametrize(
    "url",
    [
        "http://executor.example",
        "http://10.0.0.2:8787",
        "http://127.0.0.2:8787",
        "http://localhost:8787",
    ],
)
def test_plain_http_is_rejected_off_exact_loopback(url):
    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    with pytest.raises(ExecutorProtocolError, match="HTTPS"):
        RemoteExecutorClient(
            base_url=url,
            bearer_token=BEARER,
            hmac_key=KEY,
            expected_identity=identity,
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8787",
        "http://[::1]:8787",
        "https://executor.example",
    ],
)
def test_https_or_exact_loopback_transport_is_accepted(url):
    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    RemoteExecutorClient(
        base_url=url,
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
    )
