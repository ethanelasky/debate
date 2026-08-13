#!/usr/bin/env python3
"""Build the deploy-pinned, exhaustive CodeContests rootfs manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codecontests_executor.protocol import (
    ROOTFS_SHA256,
    canonical_json,
    payload_digest,
)
from codecontests_executor.service import (
    build_rootfs_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootfs", required=True, type=Path)
    parser.add_argument(
        "--rootfs-artifact",
        required=True,
        type=Path,
        help="the immutable archive/image whose pinned SHA-256 is ROOTFS_SHA256",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="explicitly replace an existing manifest",
    )
    args = parser.parse_args()

    rootfs = args.rootfs.absolute()
    artifact = args.rootfs_artifact.absolute()
    output = args.output.absolute()
    if os.geteuid() != 0:
        raise SystemExit("manifest generation must run as root")
    if not rootfs.is_dir() or rootfs.is_symlink():
        raise SystemExit("rootfs must be a non-symlink directory")
    if not artifact.is_file() or artifact.is_symlink():
        raise SystemExit("rootfs artifact must be a non-symlink regular file")
    if os.path.commonpath((str(rootfs), str(output))) == str(rootfs):
        raise SystemExit("manifest output must live outside the rootfs")
    if _sha256(artifact) != ROOTFS_SHA256:
        raise SystemExit("rootfs artifact SHA-256 does not match the pinned digest")
    if os.path.lexists(output) and not args.force:
        raise SystemExit("manifest already exists; pass --force to replace it")

    first = build_rootfs_manifest(str(rootfs))
    second = build_rootfs_manifest(str(rootfs))
    if first != second:
        raise SystemExit("rootfs changed between manifest measurement passes")
    encoded = canonical_json(first) + b"\n"

    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_metadata = output.parent.stat()
    if parent_metadata.st_uid != 0 or parent_metadata.st_mode & (
        stat.S_IWGRP | stat.S_IWOTH
    ):
        raise SystemExit("manifest parent must be root-owned and non-writable")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        try:
            view = memoryview(encoded)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        if args.force:
            os.replace(temporary, output)
        else:
            os.link(temporary, output, follow_symlinks=False)
            os.unlink(temporary)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
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
    print(
        f"rootfs_manifest={output} entries={len(first['entries'])} "
        f"file_sha256={hashlib.sha256(encoded).hexdigest()} "
        f"payload_sha256={payload_digest(first)}"
    )


if __name__ == "__main__":
    main()
