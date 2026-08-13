"""Fail-closed client for the authenticated CodeContests executor."""

from __future__ import annotations

import hashlib
import json
import os
import stat
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
    MAX_REQUEST_BODY_BYTES,
    MAX_RESPONSE_BODY_BYTES,
    MAX_RESULT_TIMING_NS,
    PROTOCOL_VERSION,
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
        with response:
            if response.geturl() != url:
                raise ExecutorProtocolError(
                    "executor response endpoint does not match request"
                )
            data = response.read(MAX_RESPONSE_BODY_BYTES + 1)
            if len(data) > MAX_RESPONSE_BODY_BYTES:
                raise ExecutorProtocolError("executor response exceeds byte limit")
            status = response.status
            if isinstance(status, bool) or not isinstance(status, int):
                raise ExecutorProtocolError("executor response status is invalid")
            return status, data


@dataclass(frozen=True)
class RemoteExecution:
    outcome: str
    category: str | None
    retryable: bool
    stdout: bytes
    stderr: bytes
    result_payload: dict[str, Any] | None
    error: str | None = None

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
        sleep: Callable[[float], None] = time.sleep,
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
        self.identity_digest = payload_digest(identity)
        self.transport = transport or UrllibTransport()
        self.overload_retries = overload_retries
        self.sleep = sleep
        self._identity_verified = False

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

    def verify_identity(self, *, timeout_seconds: float = 5.0) -> None:
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
        if not self._identity_verified:
            try:
                self.verify_identity()
            except (TimeoutError, ExecutorProtocolError, OSError, urllib.error.URLError) as exc:
                return RemoteExecution.unknown(
                    "TRANSPORT_OR_ATTESTATION", type(exc).__name__
                )
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
                ttl_ns=150 * 1_000_000_000,
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
        for attempt in range(self.overload_retries + 1):
            try:
                _status, data = self.transport.request(
                    method="POST",
                    url=f"{self.base_url}/v1/execute",
                    headers=self._headers(json_body=True),
                    body=body,
                    timeout_seconds=timeout,
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
            except (TimeoutError, ExecutorProtocolError, OSError, urllib.error.URLError) as exc:
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
            )
            if not execution.retryable:
                return execution
            if attempt < self.overload_retries:
                self.sleep(0.05 * (attempt + 1))
        return RemoteExecution.unknown("OVERLOAD_RETRIES_EXHAUSTED")
