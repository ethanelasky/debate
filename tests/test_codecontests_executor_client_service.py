"""Direct protocol/service/client behavior probes (no gVisor fake oracle)."""

from __future__ import annotations

import base64
import copy
import http.client
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http import HTTPStatus
from typing import Any

import pytest

from codecontests_executor.client import RemoteExecutorClient, UrllibTransport
from codecontests_executor.protocol import (
    CLIENT_HTTP_OVERHEAD_SECONDS,
    DEFAULT_EXECUTE_REQUEST_TTL_NS,
    MAX_CLOCK_SKEW_NS,
    MAX_RESPONSE_BODY_BYTES,
    MEMORY_EVENT_KEYS,
    PIDS_EVENT_KEYS,
    REPLAY_CACHE_GRACE_NS,
    REPLAY_DELIVERY_MARGIN_SECONDS,
    TRANSPORT_RECOVERY_WINDOW_SECONDS,
    WALL_CEILING_NS,
    ExecutorProtocolError,
    derive_limits,
    encode_envelope,
    make_execute_request,
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
    transport = TamperTransport(app)
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=transport,
    )
    result = client.execute(code="pass", stdin="", raw_limits=RAW_LIMITS)
    assert result.outcome == "unknown"
    assert result.category == "TRANSPORT_OR_ATTESTATION"
    assert len(supervisor.requests) == 1
    assert len(transport.posts) == 1


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
    def __init__(self):
        self.calls = 0

    def request(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise ConnectionResetError("service died")


def test_service_death_is_unknown():
    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    transport = DeadTransport()
    sleeps = []
    client = RemoteExecutorClient(
        base_url="https://executor.invalid",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=transport,
        transport_retries=2,
        sleep=sleeps.append,
    )
    result = client.execute(code="pass", stdin="", raw_limits=RAW_LIMITS)
    assert result.outcome == "unknown"
    assert result.category == "TRANSPORT_OR_ATTESTATION"
    assert transport.calls == 3
    assert sleeps == [1.0, 2.0]


class LostResponseTransport(ApplicationTransport):
    def __init__(self, app: ExecutorApplication):
        super().__init__(app)
        self.methods: list[str] = []
        self.lost_response: tuple[int, bytes] | None = None

    def request(self, **kwargs):  # type: ignore[no-untyped-def]
        self.methods.append(kwargs["method"])
        response = super().request(**kwargs)
        if kwargs["method"] == "POST" and self.lost_response is None:
            self.lost_response = response
            raise ConnectionResetError("response lost after execution")
        return response


def test_response_loss_replays_identical_signed_result_without_reexecution():
    identity, supervisor, app, _transport, _client = _stack()
    transport = LostResponseTransport(app)
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=transport,
        sleep=lambda _seconds: None,
    )

    result = client.execute(code="print('once')", stdin="", raw_limits=RAW_LIMITS)

    assert result.outcome == "executed"
    assert transport.methods == ["GET", "POST", "GET", "POST"]
    assert len(transport.posts) == 2
    assert transport.posts[0] == transport.posts[1]
    assert len(supervisor.requests) == 1
    assert transport.lost_response is not None
    assert result.result_payload == verify_envelope(
        strict_json_loads(transport.lost_response[1]),
        KEY,
        expected_kind="execute_result",
    )


class _FramedResponse:
    def __init__(
        self,
        *,
        url: str,
        status: int,
        body: bytes,
        declared_length: int,
        incomplete_read: bool = False,
    ):
        self._url = url
        self.status = status
        self._body = body
        self._offset = 0
        self._incomplete_read = incomplete_read
        self._read_attempted = False
        self.headers = Message()
        self.headers["Content-Length"] = str(declared_length)

    def __enter__(self):
        return self

    def __exit__(self, *_args):  # type: ignore[no-untyped-def]
        return False

    def geturl(self) -> str:
        return self._url

    def read(self, amount: int) -> bytes:
        if self._incomplete_read and not self._read_attempted:
            self._read_attempted = True
            raise http.client.IncompleteRead(
                partial=self._body[: max(1, len(self._body) // 2)],
                expected=len(self._body),
            )
        chunk = self._body[self._offset : self._offset + amount]
        self._offset += len(chunk)
        return chunk


class _FramingLossOpener:
    def __init__(self, app: ExecutorApplication, failure: str):
        self.app = app
        self.failure = failure
        self.methods: list[str] = []
        self.posts: list[bytes] = []
        self._failed = False

    def open(self, request, timeout):  # type: ignore[no-untyped-def]
        del timeout
        method = request.get_method()
        self.methods.append(method)
        if method == "GET":
            status, body = HTTPStatus.OK, self.app.signed_identity()
        else:
            assert request.data is not None
            self.posts.append(request.data)
            status, body = self.app.execute(request.data)

        if method == "POST" and not self._failed:
            self._failed = True
            if self.failure == "short_content_length":
                delivered = body[: max(1, len(body) // 2)]
                return _FramedResponse(
                    url=request.full_url,
                    status=status,
                    body=delivered,
                    declared_length=len(body),
                )
            assert self.failure == "incomplete_read"
            return _FramedResponse(
                url=request.full_url,
                status=status,
                body=body,
                declared_length=len(body),
                incomplete_read=True,
            )

        return _FramedResponse(
            url=request.full_url,
            status=status,
            body=body,
            declared_length=len(body),
        )


@pytest.mark.parametrize("failure", ["short_content_length", "incomplete_read"])
def test_urllib_framing_loss_reattests_and_replays_exact_body_once(failure: str):
    identity, supervisor, app, _transport, _client = _stack()
    opener = _FramingLossOpener(app, failure)
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=UrllibTransport(opener),
        sleep=lambda _seconds: None,
    )

    result = client.execute(code="print('once')", stdin="", raw_limits=RAW_LIMITS)

    assert result.outcome == "executed"
    assert opener.methods == ["GET", "POST", "GET", "POST"]
    assert len(opener.posts) == 2
    assert opener.posts[0] == opener.posts[1]
    assert len(supervisor.requests) == 1


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class BlockingIdentityTransport:
    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.timeouts: list[float] = []

    def request(self, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["method"] == "GET"
        timeout = kwargs["timeout_seconds"]
        self.timeouts.append(timeout)
        self.clock.now += timeout
        raise TimeoutError("identity read reached its socket timeout")


def test_initial_attestation_calls_cannot_overrun_recovery_deadline():
    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    clock = FakeClock()
    transport = BlockingIdentityTransport(clock)
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    result = client.execute(code="pass", stdin="", raw_limits=RAW_LIMITS)

    assert result.outcome == "unknown"
    assert result.category == "TRANSPORT_OR_ATTESTATION"
    assert clock.now == pytest.approx(180.0)
    assert transport.timeouts
    assert all(0 < timeout <= 5.0 for timeout in transport.timeouts)


def test_public_identity_preflight_retries_with_hard_recovery_deadline():
    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    clock = FakeClock()
    transport = BlockingIdentityTransport(clock)
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    with pytest.raises(TimeoutError, match="recovery deadline expired"):
        client.verify_identity()

    assert clock.now == pytest.approx(180.0)
    assert transport.timeouts
    assert all(0 < timeout <= 5.0 for timeout in transport.timeouts)


def test_public_identity_preflight_does_not_retry_protocol_mismatch():
    actual_identity, _supervisor, app, _transport, _client = _stack()
    wrong_identity = {**actual_identity, "service_id": "wrong-executor"}
    transport = ApplicationTransport(app)
    sleeps: list[float] = []
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=wrong_identity,
        client_provenance=actual_identity["expected_client_provenance"],
        transport=transport,
        sleep=sleeps.append,
    )

    with pytest.raises(ExecutorProtocolError, match="identity mismatch"):
        client.verify_identity()

    assert sleeps == []


class LostThenBlockingReplayTransport(ApplicationTransport):
    def __init__(self, app: ExecutorApplication, clock: FakeClock):
        super().__init__(app)
        self.clock = clock
        self.post_timeouts: list[float] = []

    def request(self, **kwargs):  # type: ignore[no-untyped-def]
        response = super().request(**kwargs)
        if kwargs["method"] != "POST":
            return response
        timeout = kwargs["timeout_seconds"]
        self.post_timeouts.append(timeout)
        if len(self.post_timeouts) == 1:
            raise ConnectionResetError("initial response was lost")
        self.clock.now += timeout
        raise TimeoutError("replay response reached its socket timeout")


def test_replay_posts_are_clamped_without_shortening_initial_candidate_timeout():
    identity, supervisor, app, _transport, _client = _stack()
    clock = FakeClock()
    transport = LostThenBlockingReplayTransport(app, clock)
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    result = client.execute(code="print('once')", stdin="", raw_limits=RAW_LIMITS)

    expected_candidate_timeout = (
        derive_limits(RAW_LIMITS)["effective"]["wall_time_ns"] / 1_000_000_000
        + CLIENT_HTTP_OVERHEAD_SECONDS
    )
    assert result.outcome == "unknown"
    assert result.category == "TRANSPORT_OR_ATTESTATION"
    assert transport.post_timeouts[0] == expected_candidate_timeout
    assert all(
        0 < timeout <= expected_candidate_timeout
        for timeout in transport.post_timeouts[1:]
    )
    assert clock.now == pytest.approx(180.0)
    assert len(transport.posts) >= 2
    assert len(set(transport.posts)) == 1
    assert len(supervisor.requests) == 1


class LostThenOutageTransport(ApplicationTransport):
    def __init__(self, app: ExecutorApplication, clock: FakeClock):
        super().__init__(app)
        self.clock = clock
        self.executed_then_lost = False

    def request(self, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs["method"] == "POST" and not self.executed_then_lost:
            self.executed_then_lost = True
            super().request(**kwargs)
            raise ConnectionResetError("response lost as tunnel went down")
        if self.executed_then_lost and self.clock.now < 156.0:
            raise ConnectionResetError("observed host-network outage")
        return super().request(**kwargs)


def test_client_recovers_exact_replay_after_156_second_outage():
    identity, supervisor, app, _transport, _client = _stack()
    clock = FakeClock()
    transport = LostThenOutageTransport(app, clock)
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    result = client.execute(code="print('once')", stdin="", raw_limits=RAW_LIMITS)

    assert result.outcome == "executed"
    assert 156.0 <= clock.now <= 180.0
    assert max(clock.sleeps) <= 15.0
    assert len(supervisor.requests) == 1


def test_client_transport_recovery_window_is_strictly_bounded():
    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    clock = FakeClock()
    client = RemoteExecutorClient(
        base_url="https://executor.invalid",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=DeadTransport(),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    result = client.execute(code="pass", stdin="", raw_limits=RAW_LIMITS)

    assert result.outcome == "unknown"
    assert result.category == "TRANSPORT_OR_ATTESTATION"
    assert clock.now == 180.0
    assert sum(clock.sleeps) == 180.0


class DisconnectOnceTransport(ApplicationTransport):
    def __init__(self, app: ExecutorApplication):
        super().__init__(app)
        self.methods = []
        self.disconnected = False

    def request(self, **kwargs):  # type: ignore[no-untyped-def]
        self.methods.append(kwargs["method"])
        if kwargs["method"] == "POST" and not self.disconnected:
            self.disconnected = True
            raise ConnectionResetError("tunnel restarted")
        return super().request(**kwargs)


def test_transport_reconnect_reverifies_signed_identity_before_retry():
    identity, supervisor, app, _transport, _client = _stack()
    transport = DisconnectOnceTransport(app)
    sleeps = []
    client = RemoteExecutorClient(
        base_url="http://127.0.0.1:8787",
        bearer_token=BEARER,
        hmac_key=KEY,
        expected_identity=identity,
        transport=transport,
        transport_retries=1,
        sleep=sleeps.append,
    )

    result = client.execute(code="print('ok')", stdin="", raw_limits=RAW_LIMITS)

    assert result.outcome == "executed"
    assert transport.methods == ["GET", "POST", "GET", "POST"]
    assert sleeps == [1.0]
    assert len(supervisor.requests) == 1


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


def _signed_request_body(
    identity: dict[str, Any],
    *,
    code: str = "print('ok')",
    request_id: str | None = None,
    now_ns: int | None = None,
    ttl_ns: int = 10_000_000_000,
) -> tuple[dict[str, Any], bytes]:
    request = make_execute_request(
        code=code,
        stdin="",
        raw_limits=RAW_LIMITS,
        identity_digest_value=payload_digest(identity),
        client_provenance=identity["expected_client_provenance"],
        ttl_ns=ttl_ns,
        request_id=request_id,
        now_ns=now_ns,
    )
    return request, encode_envelope(sign_payload(request, KEY))


class BlockingSupervisor(StubSupervisor):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(copy.deepcopy(request))
        self.started.set()
        assert self.release.wait(timeout=5)
        return copy.deepcopy(self.result)


def test_concurrent_duplicate_requests_are_single_flight_and_byte_identical():
    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    supervisor = BlockingSupervisor()
    app = ExecutorApplication(
        bearer_token=BEARER,
        hmac_key=KEY,
        identity=identity,
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    _request, body = _signed_request_body(identity)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(app.execute, body) for _ in range(8)]
        assert supervisor.started.wait(timeout=2)
        supervisor.release.set()
        responses = [future.result(timeout=5) for future in futures]

    assert len(supervisor.requests) == 1
    assert all(status == HTTPStatus.OK for status, _body in responses)
    assert len({response_body for _status, response_body in responses}) == 1


def test_request_id_reuse_with_different_digest_fails_closed():
    identity, supervisor, app, _transport, _client = _stack()
    request, body = _signed_request_body(identity)
    status, _response = app.execute(body)
    assert status == HTTPStatus.OK

    changed = copy.deepcopy(request)
    changed["task"]["code_b64"] = base64.b64encode(b"print('different')").decode()
    changed_body = encode_envelope(sign_payload(changed, KEY))
    status, response = app.execute(changed_body)

    assert status == HTTPStatus.BAD_REQUEST
    error = verify_envelope(
        strict_json_loads(response), KEY, expected_kind="protocol_error"
    )
    assert error["request_digest"] == payload_digest(changed)
    assert len(supervisor.requests) == 1


def test_concurrent_cache_byte_reservation_prevents_untracked_execution():
    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    supervisor = BlockingSupervisor()
    app = ExecutorApplication(
        bearer_token=BEARER,
        hmac_key=KEY,
        identity=identity,
        supervisor=supervisor,  # type: ignore[arg-type]
        replay_cache_bytes=MAX_RESPONSE_BODY_BYTES,
    )
    _first, first_body = _signed_request_body(identity, code="print('first')")
    _second, second_body = _signed_request_body(identity, code="print('second')")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(app.execute, first_body)
        assert supervisor.started.wait(timeout=2)
        second_status, second_response = app.execute(second_body)
        assert len(supervisor.requests) == 1
        supervisor.release.set()
        first_status, _first_response = first_future.result(timeout=5)

    second_result = verify_envelope(
        strict_json_loads(second_response), KEY, expected_kind="execute_result"
    )
    assert first_status == HTTPStatus.OK
    assert second_status == HTTPStatus.SERVICE_UNAVAILABLE
    assert second_result["category"] == "OVERLOADED"
    assert second_result["retryable"] is True
    assert app._replay_cache_bytes + app._replay_reserved_bytes <= (
        MAX_RESPONSE_BODY_BYTES
    )


def test_completed_replay_survives_expiry_grace_then_is_pruned(
    monkeypatch: pytest.MonkeyPatch,
):
    import codecontests_executor.service as service_module

    now_ns = [1_000_000_000_000]
    monkeypatch.setattr(service_module.time, "time_ns", lambda: now_ns[0])
    identity, supervisor, app, _transport, _client = _stack()
    request, body = _signed_request_body(
        identity,
        now_ns=now_ns[0],
        ttl_ns=10_000_000_000,
    )
    first = app.execute(body)
    assert first[0] == HTTPStatus.OK

    now_ns[0] = request["expires_at_unix_ns"] + 1
    assert app.execute(body) == first
    assert len(supervisor.requests) == 1

    now_ns[0] = request["expires_at_unix_ns"] + REPLAY_CACHE_GRACE_NS + 1
    status, response = app.execute(body)
    assert status == HTTPStatus.BAD_REQUEST
    verify_envelope(strict_json_loads(response), KEY, expected_kind="protocol_error")
    assert len(supervisor.requests) == 1
    assert app._replay_entries == {}
    assert app._replay_cache_bytes == 0


def test_max_wall_replay_has_explicit_final_delivery_margin(
    monkeypatch: pytest.MonkeyPatch,
):
    """Max clock skew + max execution cannot prune at recovery boundary."""
    import codecontests_executor.service as service_module

    issued_ns = 1_000_000_000_000
    now_ns = [issued_ns + MAX_CLOCK_SKEW_NS]
    monkeypatch.setattr(service_module.time, "time_ns", lambda: now_ns[0])

    class MaxDurationSupervisor(StubSupervisor):
        def execute(self, request: dict[str, Any]) -> dict[str, Any]:
            self.requests.append(copy.deepcopy(request))
            now_ns[0] += WALL_CEILING_NS + CLIENT_HTTP_OVERHEAD_SECONDS * 1_000_000_000
            return copy.deepcopy(self.result)

    identity = static_identity(
        service_id="executor-test",
        launcher_sha256="a" * 64,
    )
    supervisor = MaxDurationSupervisor()
    app = ExecutorApplication(
        bearer_token=BEARER,
        hmac_key=KEY,
        identity=identity,
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    request, body = _signed_request_body(
        identity,
        now_ns=issued_ns,
        ttl_ns=DEFAULT_EXECUTE_REQUEST_TTL_NS,
    )
    first = app.execute(body)
    assert first[0] == HTTPStatus.OK

    now_ns[0] = (
        issued_ns
        + MAX_CLOCK_SKEW_NS
        + WALL_CEILING_NS
        + CLIENT_HTTP_OVERHEAD_SECONDS * 1_000_000_000
        + TRANSPORT_RECOVERY_WINDOW_SECONDS * 1_000_000_000
        + (REPLAY_DELIVERY_MARGIN_SECONDS - 1) * 1_000_000_000
    )
    assert now_ns[0] < request["expires_at_unix_ns"]
    assert app.execute(body) == first
    assert len(supervisor.requests) == 1

    now_ns[0] = request["expires_at_unix_ns"]
    status, response = app.execute(body)
    assert status == HTTPStatus.BAD_REQUEST
    verify_envelope(strict_json_loads(response), KEY, expected_kind="protocol_error")
    assert len(supervisor.requests) == 1


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
