#!/usr/bin/env python3
"""Fetch, authenticate, and atomically freeze a live executor identity twice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from codecontests_executor.protocol import (  # noqa: E402
    ExecutorProtocolError,
    canonical_json,
    strict_json_loads,
    verify_envelope,
)

MAX_IDENTITY_BYTES = 256 * 1024


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(  # type: ignore[no-untyped-def]
        self, request, file_pointer, code, message, headers, new_url
    ):
        del request, file_pointer, code, message, headers, new_url
        raise ExecutorProtocolError("identity endpoint redirects are forbidden")


def _read_secret(path: Path, label: str) -> bytes:
    if path.is_symlink() or path.resolve() != path.absolute():
        raise RuntimeError(f"{label} path is not canonical/non-symlink")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid not in {0, os.geteuid()}
            or before.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
        ):
            raise RuntimeError(f"{label} ownership/mode is unsafe")
        content = os.read(descriptor, 64 * 1024 + 1)
        after = os.fstat(descriptor)
        stable = (
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
            len(content) > 64 * 1024
            or len(content) != before.st_size
            or any(getattr(before, key) != getattr(after, key) for key in stable)
        ):
            raise RuntimeError(f"{label} changed or exceeds its size limit")
    finally:
        os.close(descriptor)
    value = content.rstrip(b"\r\n")
    if len(value) < 32:
        raise RuntimeError(f"{label} must contain at least 32 bytes")
    return value


def _identity_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("capture URL must be a loopback HTTP origin")
    try:
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("capture URL port is invalid") from exc
    if port is None:
        raise RuntimeError("capture URL must include an explicit port")
    return base_url.rstrip("/") + "/v1/identity"


def _fetch(url: str, bearer: bytes, hmac_key: bytes, timeout: float) -> tuple[bytes, dict]:
    try:
        authorization = "Bearer " + bearer.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError("bearer token is not ASCII") from exc
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": authorization,
            "Connection": "close",
        },
        method="GET",
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _RejectRedirects()
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200 or response.geturl() != url:
                raise RuntimeError("identity endpoint did not return exact HTTP 200")
            body = response.read(MAX_IDENTITY_BYTES + 1)
    except urllib.error.URLError as exc:
        raise RuntimeError("identity endpoint request failed") from exc
    if len(body) > MAX_IDENTITY_BYTES:
        raise RuntimeError("signed identity exceeds byte limit")
    envelope = strict_json_loads(body)
    payload = verify_envelope(envelope, hmac_key, expected_kind="identity")
    if body != canonical_json(envelope):
        raise RuntimeError("signed identity is not exact canonical JSON")
    return body, payload


def _write_new(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace identity artifact: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError("identity output parent is unsafe/missing")
    mode = stat.S_IMODE(path.parent.stat().st_mode)
    if path.parent.stat().st_uid != os.geteuid() or mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise RuntimeError("identity output parent must be private and caller-owned")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        view = memoryview(content)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    finally:
        os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--bearer-file", required=True, type=Path)
    parser.add_argument("--hmac-key-file", required=True, type=Path)
    parser.add_argument("--identity-output", required=True, type=Path)
    parser.add_argument("--envelope-output", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= 60:
        parser.error("timeout must be in (0, 60]")
    if args.identity_output.absolute() == args.envelope_output.absolute():
        parser.error("identity and envelope outputs must differ")
    for output in (args.identity_output, args.envelope_output):
        if output.exists() or output.is_symlink():
            parser.error(f"output already exists: {output}")

    bearer = _read_secret(args.bearer_file.absolute(), "bearer")
    hmac_key = _read_secret(args.hmac_key_file.absolute(), "HMAC key")
    url = _identity_url(args.url)
    first_body, first_payload = _fetch(url, bearer, hmac_key, args.timeout_seconds)
    second_body, second_payload = _fetch(url, bearer, hmac_key, args.timeout_seconds)
    if first_body != second_body or first_payload != second_payload:
        raise RuntimeError("executor identity changed between consecutive fetches")

    _write_new(args.envelope_output.absolute(), first_body + b"\n")
    _write_new(args.identity_output.absolute(), canonical_json(first_payload) + b"\n")
    summary = {
        "identity_sha256": hashlib.sha256(canonical_json(first_payload)).hexdigest(),
        "protocol_version": first_payload.get("protocol_version"),
        "server_bundle_sha256": first_payload.get("server_bundle_sha256"),
        "service_id": first_payload.get("service_id"),
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
