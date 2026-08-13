"""Fail-closed client for the authenticated CodeContests executor."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .protocol import (
    CLIENT_HTTP_OVERHEAD_SECONDS,
    DEFAULT_EXECUTE_REQUEST_TTL_NS,
    MAX_REQUEST_BODY_BYTES,
    MAX_RESPONSE_BODY_BYTES,
    MAX_RESULT_TIMING_NS,
    PROTOCOL_VERSION,
    TRANSPORT_RECOVERY_WINDOW_SECONDS,
    ExecutorProtocolError,
    encode_envelope,
    make_execute_request,
    payload_digest,
    sign_payload,
    strict_json_loads,
    validate_execution_evidence,
    verify_envelope,
)


class Transport(Protocol):
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> tuple[int, bytes]: ...


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(  # type: ignore[no-untyped-def]
        self, req, fp, code, msg, headers, newurl
    ):
        del req, fp, code, msg, headers, newurl
        raise ExecutorProtocolError("executor redirects are forbidden")


class _RetryableHTTPTransportError(OSError):
    """HTTP connection/framing failure that is safe to retry by request ID."""


class UrllibTransport:
    def __init__(self, opener: Any | None = None):
        self._opener = opener or urllib.request.build_opener(_RejectRedirectHandler())

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> tuple[int, bytes]:
        request = urllib.request.Request(
            url=url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        try:
            response = self._opener.open(request, timeout=timeout_seconds)
        except urllib.error.HTTPError as exc:
            response = exc
        except http.client.HTTPException as exc:
            # urllib does not consistently wrap failures raised while parsing
            # the status line/headers in URLError.  Keep those failures on the
            # transport-recovery path rather than treating them as a signed
            # executor protocol response.
            raise _RetryableHTTPTransportError(
                "executor HTTP response framing failed"
            ) from exc
        with response:
            if response.geturl() != url:
                raise ExecutorProtocolError(
                    "executor response endpoint does not match request"
                )
            content_length = self._content_length(response)
            if (
                content_length is not None
                and content_length > MAX_RESPONSE_BODY_BYTES
            ):
                raise ExecutorProtocolError("executor response exceeds byte limit")
            try:
                data = self._read_bounded(response)
            except http.client.IncompleteRead as exc:
                raise _RetryableHTTPTransportError(
                    "executor HTTP response body was truncated"
                ) from exc
            except http.client.HTTPException as exc:
                raise _RetryableHTTPTransportError(
                    "executor HTTP response body framing failed"
                ) from exc
            if len(data) > MAX_RESPONSE_BODY_BYTES:
                raise ExecutorProtocolError("executor response exceeds byte limit")
            if content_length is not None and len(data) != content_length:
                # HTTPResponse.read(amount) can return a short body without
                # raising IncompleteRead.  A signed response may already have
                # been produced server-side, so this must enter the exact-body
                # replay path instead of becoming a terminal JSON/HMAC error.
                raise _RetryableHTTPTransportError(
                    "executor HTTP response body length mismatch"
                )
            status = response.status
            if isinstance(status, bool) or not isinstance(status, int):
                raise ExecutorProtocolError("executor response status is invalid")
            return status, data

    @staticmethod
    def _content_length(response: Any) -> int | None:
        headers = getattr(response, "headers", None)
        values = headers.get_all("Content-Length") if headers is not None else None
        if not values:
            return None
        tokens = [part.strip() for value in values for part in value.split(",")]
        if not tokens or any(
            not token or any(character not in "0123456789" for character in token)
            for token in tokens
        ):
            raise _RetryableHTTPTransportError(
                "executor HTTP Content-Length is invalid"
            )
        normalized = [token.lstrip("0") or "0" for token in tokens]
        max_length_digits = len(str(MAX_RESPONSE_BODY_BYTES))
        if any(len(token) > max_length_digits for token in normalized):
            raise ExecutorProtocolError("executor response exceeds byte limit")
        lengths = {int(token) for token in normalized}
        if len(lengths) != 1:
            raise _RetryableHTTPTransportError(
                "executor HTTP Content-Length values conflict"
            )
        return lengths.pop()

    @staticmethod
    def _read_bounded(response: Any) -> bytes:
        chunks: list[bytes] = []
        size = 0
        while size <= MAX_RESPONSE_BODY_BYTES:
            chunk = response.read(
                min(64 * 1024, MAX_RESPONSE_BODY_BYTES + 1 - size)
            )
            if not isinstance(chunk, bytes):
                raise ExecutorProtocolError("executor response body is invalid")
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        return b"".join(chunks)


@dataclass(frozen=True)
class RemoteExecution:
    outcome: str
    category: str | None
    retryable: bool
    stdout: bytes
    stderr: bytes
    result_payload: dict[str, Any] | None
    error: str | None = None
    # Authenticated service-side wall time for this exact request.  The
    # CodeContests driver uses it to keep tunnel/reconnect time out of the
    # candidate's shared solution budget while still charging queue, sandbox,
    # and execution time.  UNKNOWN results deliberately carry no credit.
    attested_service_total_ns: int | None = None

    @classmethod
    def unknown(cls, category: str, error: str | None = None) -> RemoteExecution:
        return cls(
            outcome="unknown",
            category=category,
            retryable=False,
            stdout=b"",
            stderr=b"",
            result_payload=None,
            error=error,
            attested_service_total_ns=None,
        )


def _read_client_secret(file_env: str, value_env: str, label: str) -> bytes:
    file_value = os.environ.get(file_env)
    direct_value = os.environ.get(value_env)
    if bool(file_value) == bool(direct_value):
        raise ExecutorProtocolError(
            f"set exactly one {label} file or environment value"
        )
    value = (
        _read_root_owned_nofollow(
            file_value, label=label, max_bytes=64 * 1024, secret_mode=True
        ).rstrip(b"\r\n")
        if file_value
        else (direct_value or "").encode()
    )
    if len(value) < 32:
        raise ExecutorProtocolError(f"{label} must contain at least 32 bytes")
    return value


def _read_root_owned_nofollow(
    path_value: str,
    *,
    label: str,
    max_bytes: int,
    secret_mode: bool,
) -> bytes:
    if os.path.realpath(path_value) != os.path.abspath(path_value):
        raise ExecutorProtocolError(f"{label} file path uses a symlink")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ExecutorProtocolError("O_NOFOLLOW is unavailable")
    try:
        fd = os.open(path_value, os.O_RDONLY | os.O_CLOEXEC | nofollow)
    except OSError as exc:
        raise ExecutorProtocolError(f"cannot open {label} file safely") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != 0:
            raise ExecutorProtocolError(f"{label} file must be root-owned/regular")
        forbidden = (
            stat.S_IRWXG | stat.S_IRWXO if secret_mode else stat.S_IWGRP | stat.S_IWOTH
        )
        if before.st_mode & forbidden:
            raise ExecutorProtocolError(f"{label} file permissions are unsafe")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(fd, min(64 * 1024, max_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise ExecutorProtocolError(f"{label} file exceeds size limit")
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
            raise ExecutorProtocolError(f"{label} file changed while being read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _validated_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ExecutorProtocolError("executor URL must use http or https")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ExecutorProtocolError("executor URL authority is invalid")
    try:
        _validated_port = parsed.port
    except ValueError as exc:
        raise ExecutorProtocolError("executor URL port is invalid") from exc
    if parsed.scheme == "http" and parsed.hostname.lower() not in {
        "127.0.0.1",
        "::1",
    }:
        raise ExecutorProtocolError("non-loopback executor transport requires HTTPS")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ExecutorProtocolError("executor URL must not include path/query/fragment")
    return value.rstrip("/")


def measure_client_provenance(verifier_path: str) -> dict[str, str]:
    client_bytes = _read_root_owned_nofollow(
        str(Path(__file__).absolute()),
        label="executor client module",
        max_bytes=2 * 1024 * 1024,
        secret_mode=False,
    )
    protocol_bytes = _read_root_owned_nofollow(
        str(Path(__file__).with_name("protocol.py").absolute()),
        label="executor protocol module",
        max_bytes=2 * 1024 * 1024,
        secret_mode=False,
    )
    verifier_bytes = _read_root_owned_nofollow(
        str(Path(verifier_path).absolute()),
        label="CodeContests verifier module",
        max_bytes=4 * 1024 * 1024,
        secret_mode=False,
    )
    return {
        "format": "palaestra.codecontests.client-provenance.v1",
        "client_sha256": hashlib.sha256(client_bytes).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "verifier_sha256": hashlib.sha256(verifier_bytes).hexdigest(),
    }


class RemoteExecutorClient:
    def __init__(
        self,
        *,
        base_url: str,
        bearer_token: bytes,
        hmac_key: bytes,
        expected_identity: Mapping[str, Any],
        client_provenance: Mapping[str, Any] | None = None,
        transport: Transport | None = None,
        overload_retries: int = 2,
        transport_retries: int = 32,
        transport_recovery_window_seconds: float = (
            TRANSPORT_RECOVERY_WINDOW_SECONDS
        ),
        transport_retry_max_delay_seconds: float = 15.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        if len(bearer_token) < 32 or len(hmac_key) < 32:
            raise ExecutorProtocolError("executor secrets must contain 32+ bytes")
        try:
            bearer_token.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise ExecutorProtocolError("bearer token must be ASCII") from exc
        identity = dict(expected_identity)
        if identity.get("kind") != "identity":
            raise ExecutorProtocolError("expected executor identity is absent")
        if identity.get("protocol_version") != PROTOCOL_VERSION:
            raise ExecutorProtocolError("expected identity protocol mismatch")
        self.base_url = _validated_base_url(base_url)
        self._bearer = bearer_token
        self._hmac_key = hmac_key
        self.expected_identity = identity
        expected_client = identity.get("expected_client_provenance")
        if not isinstance(expected_client, dict):
            raise ExecutorProtocolError("expected client/verifier provenance is absent")
        self.client_provenance = (
            dict(expected_client)
            if client_provenance is None
            else dict(client_provenance)
        )
        if self.client_provenance != expected_client:
            raise ExecutorProtocolError("local client/verifier provenance mismatch")
        if (
            isinstance(overload_retries, bool)
            or not isinstance(overload_retries, int)
            or overload_retries < 0
            or isinstance(transport_retries, bool)
            or not isinstance(transport_retries, int)
            or transport_retries < 0
        ):
            raise ExecutorProtocolError("executor retry counts are invalid")
        for value in (
            transport_recovery_window_seconds,
            transport_retry_max_delay_seconds,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ExecutorProtocolError("transport retry timing is invalid")
        self.identity_digest = payload_digest(identity)
        self.transport = transport or UrllibTransport()
        self.overload_retries = overload_retries
        self.transport_retries = transport_retries
        self.transport_recovery_window_seconds = float(
            transport_recovery_window_seconds
        )
        self.transport_retry_max_delay_seconds = float(
            transport_retry_max_delay_seconds
        )
        self.sleep = sleep
        self.monotonic = monotonic
        self._identity_verified = False
        self._identity_lock = threading.Lock()

    @classmethod
    def from_env(cls, *, verifier_path: str | None = None) -> RemoteExecutorClient:
        base_url = os.environ.get("CODECONTESTS_EXECUTOR_URL", "")
        identity_file = os.environ.get("CODECONTESTS_EXECUTOR_IDENTITY_FILE", "")
        if not base_url:
            raise ExecutorProtocolError("CODECONTESTS_EXECUTOR_URL is absent")
        if not identity_file:
            raise ExecutorProtocolError(
                "transport URL without frozen executor identity/provenance"
            )
        effective_verifier_path = verifier_path or os.environ.get(
            "CODECONTESTS_EXECUTOR_VERIFIER_PATH", ""
        )
        if not effective_verifier_path:
            raise ExecutorProtocolError("verifier provenance path is absent")
        try:
            identity_value = json.loads(
                _read_root_owned_nofollow(
                    identity_file,
                    label="executor identity",
                    max_bytes=256 * 1024,
                    secret_mode=False,
                ).decode("utf-8", errors="strict")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ExecutorProtocolError("invalid executor identity file") from exc
        if not isinstance(identity_value, dict):
            raise ExecutorProtocolError("executor identity file must be an object")
        return cls(
            base_url=base_url,
            bearer_token=_read_client_secret(
                "CODECONTESTS_EXECUTOR_BEARER_FILE",
                "CODECONTESTS_EXECUTOR_BEARER",
                "bearer token",
            ),
            hmac_key=_read_client_secret(
                "CODECONTESTS_EXECUTOR_HMAC_KEY_FILE",
                "CODECONTESTS_EXECUTOR_HMAC_KEY",
                "HMAC key",
            ),
            expected_identity=identity_value,
            client_provenance=measure_client_provenance(effective_verifier_path),
        )

    def _headers(self, *, json_body: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": (
                "Bearer " + self._bearer.decode("ascii", errors="strict")
            ),
            "Accept": "application/json",
            "Connection": "close",
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _verify_identity_unlocked(self, *, timeout_seconds: float = 5.0) -> None:
        # Never retain a stale success if an explicit or reconnect-time
        # attestation attempt fails partway through.
        self._identity_verified = False
        status, data = self.transport.request(
            method="GET",
            url=f"{self.base_url}/v1/identity",
            headers=self._headers(),
            body=None,
            timeout_seconds=timeout_seconds,
        )
        if status != 200:
            raise ExecutorProtocolError("identity endpoint rejected request")
        envelope = strict_json_loads(data)
        identity = verify_envelope(envelope, self._hmac_key, expected_kind="identity")
        if identity != self.expected_identity:
            raise ExecutorProtocolError("executor identity mismatch")
        self._identity_verified = True

    def verify_identity(self, *, timeout_seconds: float = 5.0) -> None:
        """Verify frozen identity with bounded transport recovery.

        Signed/protocol failures are terminal.  Only connection-level failures
        use the client's configured retry count, exponential backoff, and hard
        recovery deadline; this is the public preflight used before the grader
        caches a remote client.
        """
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ExecutorProtocolError("identity timeout is invalid")
        per_attempt_timeout = float(timeout_seconds)
        deadline = self.monotonic() + self.transport_recovery_window_seconds
        transport_failures = 0

        while True:
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                raise TimeoutError("executor identity recovery deadline expired")
            attempt_timeout = min(per_attempt_timeout, remaining)
            started = self.monotonic()
            if not self._identity_lock.acquire(timeout=attempt_timeout):
                error: Exception = TimeoutError("executor identity lock timed out")
            else:
                try:
                    request_timeout = attempt_timeout - (
                        self.monotonic() - started
                    )
                    if request_timeout <= 0:
                        error = TimeoutError("executor identity deadline expired")
                    else:
                        try:
                            self._verify_identity_unlocked(
                                timeout_seconds=request_timeout
                            )
                        except (
                            TimeoutError,
                            OSError,
                            urllib.error.URLError,
                        ) as exc:
                            error = exc
                        else:
                            return
                finally:
                    self._identity_lock.release()

            # ExecutorProtocolError intentionally bypasses this block: a
            # reachable endpoint with invalid HMAC/provenance is not made safe
            # by reconnecting or retrying.
            if transport_failures >= self.transport_retries:
                raise error
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "executor identity recovery deadline expired"
                ) from error
            delay = min(
                float(1 << min(transport_failures, 4)),
                self.transport_retry_max_delay_seconds,
                remaining,
            )
            transport_failures += 1
            self.sleep(delay)

    def _ensure_identity(self, *, timeout_seconds: float = 5.0) -> None:
        if self._identity_verified:
            return
        started = self.monotonic()
        if not self._identity_lock.acquire(timeout=timeout_seconds):
            raise TimeoutError("executor identity lock timed out")
        try:
            if not self._identity_verified:
                remaining = timeout_seconds - (self.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("executor identity deadline expired")
                self._verify_identity_unlocked(timeout_seconds=remaining)
        finally:
            self._identity_lock.release()

    def _forget_identity(self, *, timeout_seconds: float | None = None) -> None:
        # A lost connection may come back bound to a different service. Force a
        # fresh signed identity check before any retry is allowed to execute.
        if timeout_seconds is None:
            self._identity_lock.acquire()
        elif not self._identity_lock.acquire(timeout=max(0.0, timeout_seconds)):
            raise TimeoutError("executor identity lock timed out")
        try:
            self._identity_verified = False
        finally:
            self._identity_lock.release()

    @staticmethod
    def _validate_result(
        payload: dict[str, Any],
        *,
        request: dict[str, Any],
        request_digest_value: str,
        identity_digest_value: str,
    ) -> tuple[bytes, bytes]:
        required = {
            "kind",
            "protocol_version",
            "request_id",
            "request_digest",
            "identity_digest",
            "limits",
            "outcome",
            "category",
            "retryable",
            "timing",
            "evidence",
        }
        if set(payload) != required:
            raise ExecutorProtocolError("result fields mismatch")
        if payload["protocol_version"] != PROTOCOL_VERSION:
            raise ExecutorProtocolError("result protocol mismatch")
        if payload["request_id"] != request["request_id"]:
            raise ExecutorProtocolError("result request ID mismatch")
        if payload["request_digest"] != request_digest_value:
            raise ExecutorProtocolError("result request digest mismatch")
        if payload["identity_digest"] != identity_digest_value:
            raise ExecutorProtocolError("result identity mismatch")
        if payload["limits"] != request["task"]["limits"]:
            raise ExecutorProtocolError("result limits/provenance mismatch")
        timing = payload["timing"]
        if not isinstance(timing, dict) or set(timing) != {
            "queue_ns",
            "execution_ns",
            "total_ns",
        }:
            raise ExecutorProtocolError("invalid timing evidence")
        for value in timing.values():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > MAX_RESULT_TIMING_NS
            ):
                raise ExecutorProtocolError("invalid timing value")
        if timing["total_ns"] < (timing["queue_ns"] + timing["execution_ns"]):
            raise ExecutorProtocolError("inconsistent timing evidence")

        evidence = payload["evidence"]
        return validate_execution_evidence(
            outcome=payload["outcome"],
            category=payload["category"],
            retryable=payload["retryable"],
            evidence=evidence,
            expected_limits=request["task"]["limits"],
        )

    def execute(
        self,
        *,
        code: str,
        stdin: str,
        raw_limits: Mapping[str, Any],
    ) -> RemoteExecution:
        transport_failures = 0
        # Initial attestation has the same hard recovery budget as a later
        # reconnect. Start its deadline before the first potentially blocking
        # network call rather than after the first timeout returns.
        recovery_deadline: float | None = (
            self.monotonic() + self.transport_recovery_window_seconds
        )

        def remaining_recovery_time() -> float | None:
            if recovery_deadline is None:
                return None
            return recovery_deadline - self.monotonic()

        def recovery_exhausted() -> RemoteExecution:
            return RemoteExecution.unknown("TRANSPORT_OR_ATTESTATION", "TimeoutError")

        def retry_transport(exc: BaseException) -> RemoteExecution | None:
            nonlocal recovery_deadline, transport_failures
            if transport_failures >= self.transport_retries:
                return RemoteExecution.unknown(
                    "TRANSPORT_OR_ATTESTATION", type(exc).__name__
                )
            now = self.monotonic()
            if recovery_deadline is None:
                recovery_deadline = now + self.transport_recovery_window_seconds
            remaining = recovery_deadline - now
            if remaining <= 0:
                return RemoteExecution.unknown(
                    "TRANSPORT_OR_ATTESTATION", type(exc).__name__
                )
            delay = min(
                float(1 << min(transport_failures, 4)),
                self.transport_retry_max_delay_seconds,
                remaining,
            )
            transport_failures += 1
            # Give the external tunnel supervisor a bounded window to restore
            # connectivity. The next loop iteration must re-attest identity.
            self.sleep(delay)
            return None

        # Attest before minting the request validity interval.  An initial
        # 156-second host-network outage therefore does not consume the time
        # needed for a maximum-duration candidate or an idempotent replay.
        while not self._identity_verified:
            remaining = remaining_recovery_time()
            if remaining is None or remaining <= 0:
                return recovery_exhausted()
            try:
                self._ensure_identity(timeout_seconds=min(5.0, remaining))
            except (TimeoutError, OSError, urllib.error.URLError) as exc:
                unknown = retry_transport(exc)
                if unknown is not None:
                    return unknown
            except ExecutorProtocolError as exc:
                return RemoteExecution.unknown(
                    "TRANSPORT_OR_ATTESTATION", type(exc).__name__
                )

        # A later response-loss episode receives its own bounded recovery
        # window; the just-completed initial attestation cannot consume it.
        transport_failures = 0
        recovery_deadline = None
        try:
            # The maximum source-derived wall is 120s.  The remaining validity
            # interval bounds queue and response handling without changing the
            # candidate wall limit.
            request = make_execute_request(
                code=code,
                stdin=stdin,
                raw_limits=raw_limits,
                identity_digest_value=self.identity_digest,
                client_provenance=self.client_provenance,
                ttl_ns=DEFAULT_EXECUTE_REQUEST_TTL_NS,
            )
            request_digest_value = payload_digest(request)
            body = encode_envelope(sign_payload(request, self._hmac_key))
            if len(body) > MAX_REQUEST_BODY_BYTES:
                raise ExecutorProtocolError("encoded request exceeds body cap")
        except (ExecutorProtocolError, OSError, UnicodeError) as exc:
            return RemoteExecution.unknown("CLIENT_CONFIGURATION", type(exc).__name__)

        timeout = (
            request["task"]["limits"]["effective"]["wall_time_ns"]
            / 1_000_000_000
            + CLIENT_HTTP_OVERHEAD_SECONDS
        )
        overload_attempt = 0
        while True:
            if not self._identity_verified:
                remaining = remaining_recovery_time()
                if remaining is None or remaining <= 0:
                    return recovery_exhausted()
                try:
                    self._ensure_identity(timeout_seconds=min(5.0, remaining))
                except (TimeoutError, OSError, urllib.error.URLError) as exc:
                    unknown = retry_transport(exc)
                    if unknown is not None:
                        return unknown
                    continue
                except ExecutorProtocolError as exc:
                    # A reachable endpoint with the wrong/invalid signed
                    # identity is not a reconnect race and is never retried.
                    return RemoteExecution.unknown(
                        "TRANSPORT_OR_ATTESTATION", type(exc).__name__
                    )
            request_timeout = timeout
            remaining = remaining_recovery_time()
            if remaining is not None:
                if remaining <= 0:
                    return recovery_exhausted()
                request_timeout = min(request_timeout, remaining)
            try:
                _status, data = self.transport.request(
                    method="POST",
                    url=f"{self.base_url}/v1/execute",
                    headers=self._headers(json_body=True),
                    body=body,
                    timeout_seconds=request_timeout,
                )
                envelope = strict_json_loads(data)
                result = verify_envelope(
                    envelope, self._hmac_key, expected_kind="execute_result"
                )
                stdout, stderr = self._validate_result(
                    result,
                    request=request,
                    request_digest_value=request_digest_value,
                    identity_digest_value=self.identity_digest,
                )
            except (TimeoutError, OSError, urllib.error.URLError) as exc:
                # Start the recovery deadline at the transport failure, before
                # even waiting for another client's attestation lock.
                if recovery_deadline is None:
                    recovery_deadline = (
                        self.monotonic() + self.transport_recovery_window_seconds
                    )
                remaining = remaining_recovery_time()
                if remaining is None or remaining <= 0:
                    return recovery_exhausted()
                try:
                    self._forget_identity(timeout_seconds=remaining)
                except TimeoutError:
                    return recovery_exhausted()
                unknown = retry_transport(exc)
                if unknown is not None:
                    return unknown
                continue
            except ExecutorProtocolError as exc:
                # Invalid signatures, request binding, or attestation are never
                # made acceptable by retrying the same response.
                return RemoteExecution.unknown(
                    "TRANSPORT_OR_ATTESTATION", type(exc).__name__
                )
            execution = RemoteExecution(
                outcome=result["outcome"],
                category=result["category"],
                retryable=result["retryable"],
                stdout=stdout,
                stderr=stderr,
                result_payload=result,
                # This value is inside the authenticated and schema-validated
                # evidence. Preserve it for fail-closed operator diagnostics;
                # it never changes the verdict or retry policy here.
                error=result["evidence"].get("controller_error"),
                attested_service_total_ns=result["timing"]["total_ns"],
            )
            if not execution.retryable:
                return execution
            if overload_attempt >= self.overload_retries:
                return RemoteExecution.unknown("OVERLOAD_RETRIES_EXHAUSTED")
            overload_attempt += 1
            self.sleep(0.05 * overload_attempt)
