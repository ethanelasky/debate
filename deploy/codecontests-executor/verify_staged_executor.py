#!/usr/bin/env python3
"""Pre-import integrity gate for the staged executor source package.

This file deliberately has no imports from ``codecontests_executor``.  It
must finish before systemd starts Python's package importer or the service
reads credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import stat
from typing import Any

BUNDLE_FORMAT = "palaestra.codecontests.server-bundle-inventory.v1"
EXPECTED_FILES = (
    "__init__.py",
    "cgroup_gate.py",
    "client.py",
    "protocol.py",
    "sandbox_launcher.py",
    "service.py",
    "supervisor.py",
)
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_INVENTORY_BYTES = 64 * 1024


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError("bundle inventory has duplicate keys")
        result[key] = value
    return result


def _read_fd_stable(fd: int, *, label: str, max_bytes: int, required_uid: int) -> bytes:
    before = os.fstat(fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != required_uid
        or before.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise RuntimeError(f"{label} ownership/type/mode is unsafe")
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = os.read(fd, min(64 * 1024, max_bytes + 1 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > max_bytes:
            raise RuntimeError(f"{label} exceeds byte limit")
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
        any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or size != before.st_size
    ):
        raise RuntimeError(f"{label} changed during verification")
    return b"".join(chunks)


def _open_nofollow(path: str, *, directory: bool = False) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("O_NOFOLLOW is unavailable")
    flags = os.O_RDONLY | os.O_CLOEXEC | nofollow
    if directory:
        flags |= os.O_DIRECTORY
    if os.path.realpath(path) != os.path.abspath(path):
        raise RuntimeError("trusted path uses a symlink")
    return os.open(path, flags)


def verify_staged_executor(
    *,
    package_dir: str,
    inventory_path: str,
    expected_bundle_sha256: str,
    required_uid: int = 0,
) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", expected_bundle_sha256) is None:
        raise RuntimeError("expected bundle SHA-256 is invalid")

    inventory_fd = _open_nofollow(inventory_path)
    try:
        inventory_bytes = _read_fd_stable(
            inventory_fd,
            label="bundle inventory",
            max_bytes=MAX_INVENTORY_BYTES,
            required_uid=required_uid,
        )
    finally:
        os.close(inventory_fd)
    try:
        inventory = json.loads(
            inventory_bytes.decode("utf-8", errors="strict"),
            object_pairs_hook=_no_duplicate_pairs,
            parse_float=lambda _value: (_ for _ in ()).throw(
                RuntimeError("floating-point inventory value")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid bundle inventory JSON") from exc
    if not isinstance(inventory, dict) or set(inventory) != {
        "format",
        "bundle_sha256",
        "files",
    }:
        raise RuntimeError("bundle inventory schema mismatch")
    if inventory["format"] != BUNDLE_FORMAT:
        raise RuntimeError("bundle inventory format mismatch")
    inventory_digest = inventory["bundle_sha256"]
    if (
        not isinstance(inventory_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", inventory_digest) is None
        or not hmac.compare_digest(inventory_digest, expected_bundle_sha256)
    ):
        raise RuntimeError("bundle inventory digest does not match deploy pin")
    records = inventory["files"]
    if not isinstance(records, list) or len(records) != len(EXPECTED_FILES):
        raise RuntimeError("bundle inventory file count mismatch")

    package_fd = _open_nofollow(package_dir, directory=True)
    try:
        package_metadata = os.fstat(package_fd)
        if package_metadata.st_uid != required_uid or package_metadata.st_mode & (
            stat.S_IWGRP | stat.S_IWOTH
        ):
            raise RuntimeError("executor package directory mode is unsafe")
        actual_entries = sorted(os.listdir(package_fd))
        if actual_entries != sorted(EXPECTED_FILES):
            raise RuntimeError("executor package contains missing/extra/cache entries")

        digest = hashlib.sha256()
        for filename, record in zip(EXPECTED_FILES, records, strict=True):
            if not isinstance(record, dict) or set(record) != {
                "path",
                "sha256",
                "size",
            }:
                raise RuntimeError("bundle inventory file record mismatch")
            if record["path"] != filename:
                raise RuntimeError("bundle inventory file order/path mismatch")
            if (
                isinstance(record["size"], bool)
                or not isinstance(record["size"], int)
                or not 0 <= record["size"] <= MAX_SOURCE_BYTES
                or not isinstance(record["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
            ):
                raise RuntimeError("bundle inventory file metadata is invalid")
            source_fd = os.open(
                filename,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=package_fd,
            )
            try:
                content = _read_fd_stable(
                    source_fd,
                    label=f"executor source {filename}",
                    max_bytes=MAX_SOURCE_BYTES,
                    required_uid=required_uid,
                )
            finally:
                os.close(source_fd)
            if (
                len(content) != record["size"]
                or hashlib.sha256(content).hexdigest() != record["sha256"]
            ):
                raise RuntimeError(f"executor source checksum mismatch: {filename}")
            digest.update(_canonical_json({"path": filename, "size": len(content)}))
            digest.update(b"\n")
            digest.update(content)
            digest.update(b"\n")
        measured = digest.hexdigest()
        if not hmac.compare_digest(measured, inventory_digest):
            raise RuntimeError("measured executor bundle digest mismatch")
        return measured
    finally:
        os.close(package_fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", default="/opt/palaestra/codecontests_executor")
    parser.add_argument(
        "--inventory",
        default="/etc/codecontests-executor/server-bundle.inventory.json",
    )
    args = parser.parse_args()
    expected = os.environ.get("CODECONTESTS_EXECUTOR_SERVER_BUNDLE_SHA256", "")
    measured = verify_staged_executor(
        package_dir=args.package_dir,
        inventory_path=args.inventory,
        expected_bundle_sha256=expected,
    )
    print(f"executor_preimport_integrity_ok bundle_sha256={measured}", flush=True)


if __name__ == "__main__":
    main()
