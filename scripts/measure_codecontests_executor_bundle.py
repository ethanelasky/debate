#!/usr/bin/env python3
"""Measure sources and optionally create the pre-import checksum inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_FORMAT = "palaestra.codecontests.server-bundle-inventory.v1"
SERVER_BUNDLE_FILES = (
    "__init__.py",
    "cgroup_gate.py",
    "client.py",
    "protocol.py",
    "sandbox_launcher.py",
    "service.py",
    "supervisor.py",
)
MAX_SOURCE_BYTES = 2 * 1024 * 1024


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_source(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"unsafe/missing bundle file: {path.name}")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("O_NOFOLLOW is unavailable")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | nofollow)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_mode & (
            stat.S_IWGRP | stat.S_IWOTH
        ):
            raise RuntimeError(f"unsafe source mode: {path.name}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(fd, min(64 * 1024, MAX_SOURCE_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_SOURCE_BYTES:
                raise RuntimeError(f"source exceeds byte limit: {path.name}")
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
            raise RuntimeError(f"source changed during measurement: {path.name}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def build_inventory(package_dir: Path) -> dict[str, Any]:
    if package_dir.is_symlink() or not package_dir.is_dir():
        raise RuntimeError("executor package must be a non-symlink directory")
    entries = sorted(path.name for path in package_dir.iterdir())
    if entries != sorted(SERVER_BUNDLE_FILES):
        raise RuntimeError("executor package contains missing/extra/cache entries")
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for filename in SERVER_BUNDLE_FILES:
        content = _read_source(package_dir / filename)
        records.append(
            {
                "path": filename,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
        digest.update(_canonical_json({"path": filename, "size": len(content)}))
        digest.update(b"\n")
        digest.update(content)
        digest.update(b"\n")
    return {
        "format": BUNDLE_FORMAT,
        "bundle_sha256": digest.hexdigest(),
        "files": records,
    }


def _write_atomic(path: Path, content: bytes, *, force: bool) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() and not force:
        raise RuntimeError("inventory exists; pass --force to replace it")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        try:
            view = memoryview(content)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        if force:
            os.replace(temporary, path)
        else:
            os.link(temporary, path, follow_symlinks=False)
            os.unlink(temporary)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir", type=Path, default=REPO_ROOT / "codecontests_executor"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="explicitly replace an existing inventory",
    )
    args = parser.parse_args()
    inventory = build_inventory(args.package_dir.absolute())
    if args.output is not None:
        _write_atomic(
            args.output.absolute(),
            _canonical_json(inventory) + b"\n",
            force=args.force,
        )
    print(inventory["bundle_sha256"])


if __name__ == "__main__":
    main()
