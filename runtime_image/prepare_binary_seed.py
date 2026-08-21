#!/usr/bin/env python3
"""Deterministically relocate the pinned B200 portable environment offline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Mapping


_HERE = Path(__file__).resolve(strict=True).parent
_GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "debate_runtime_metadata", _HERE / "generate_runtime_metadata.py"
)
if _GENERATOR_SPEC is None or _GENERATOR_SPEC.loader is None:
    raise RuntimeError("cannot load the tracked runtime metadata generator")
metadata = importlib.util.module_from_spec(_GENERATOR_SPEC)
_GENERATOR_SPEC.loader.exec_module(metadata)
contract = metadata.contract

RUNTIME_ROOT = contract.RUNTIME_ROOT
RUNTIME_PYTHON = contract.RUNTIME_PYTHON
MAX_FILES = contract.MAX_FILES
MAX_FILE_SIZE = contract.MAX_FILE_SIZE
WORKSPACE_PREFIX = b"/workspace/"


class Refusal(RuntimeError):
    """The binary seed cannot be transformed by the closed policy."""


class _HashingReader:
    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        payload = self.source.read(size)
        self.digest.update(payload)
        self.size += len(payload)
        return payload


def _write_once(path: Path, source: BinaryIO, *, mode: int) -> None:
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            view = memoryview(block)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short binary-seed write")
                view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, mode)
        os.utime(fd, ns=(contract.RUNTIME_MTIME_NS, contract.RUNTIME_MTIME_NS))
    finally:
        os.close(fd)


def _write_bytes_once(path: Path, payload: bytes, *, mode: int) -> None:
    _write_once(path, io.BytesIO(payload), mode=mode)


def _relative_member(name: str, prefix: str) -> str | None:
    if "\\" in name or "\x00" in name:
        raise Refusal("binary-seed archive member path is unsafe")
    normalized = PurePosixPath(name)
    if (
        normalized.is_absolute()
        or normalized.as_posix() != name.rstrip("/")
        or any(part in ("", ".", "..") for part in normalized.parts)
    ):
        raise Refusal("binary-seed archive member path is not normalized")
    if name.rstrip("/") == prefix:
        return None
    required = prefix + "/"
    if not name.startswith(required):
        raise Refusal("binary-seed archive member escapes the fixed prefix")
    return metadata._relative_path(name[len(required) :].rstrip("/"), "seed member")


def _read_bounded(member: tarfile.TarInfo, source: BinaryIO, maximum: int) -> bytes:
    if member.size > maximum:
        raise Refusal("binary-seed control file is oversized")
    payload = source.read(maximum + 1)
    if len(payload) != member.size:
        raise Refusal("binary-seed member size differs")
    return payload


def _rewrite_record(payload: bytes) -> bytes:
    try:
        text = payload.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise Refusal("binary-seed RECORD is malformed") from exc
    retained: list[list[str]] = []
    for row in rows:
        if len(row) != 3:
            raise Refusal("binary-seed RECORD has a non-canonical row")
        record_path = PurePosixPath(row[0])
        if row[0].endswith(".pyc") or "__pycache__" in record_path.parts:
            continue
        retained.append(row)
    result = io.StringIO(newline="")
    writer = csv.writer(result, lineterminator="\r\n")
    writer.writerows(retained)
    return result.getvalue().encode("utf-8")


def _check_direct_origin(relative: str, payload: bytes) -> None:
    if not relative.endswith(".dist-info/direct_url.json"):
        return
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise Refusal("binary-seed direct origin is malformed") from exc
    if not isinstance(document, dict):
        raise Refusal("binary-seed direct origin is not an object")
    directory = document.get("dir_info")
    if (
        isinstance(directory, dict)
        and directory.get("editable") is True
    ) or (
        isinstance(document.get("url"), str)
        and document["url"].lower().startswith("file:")
    ):
        raise Refusal("editable or local direct origin in binary seed")


def _contains_bytes(path: Path, needle: bytes) -> bool:
    overlap = b""
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            combined = overlap + block
            if needle in combined:
                return True
            overlap = combined[-(len(needle) - 1) :]
    return False


def _safe_new_directory(path: Path) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        path.mkdir(mode=0o755)
        info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise Refusal("binary-seed destination ancestor is unsafe")


def _safe_destination(staging_root: Path) -> Path:
    if not staging_root.is_absolute():
        raise Refusal("binary-seed staging root must be absolute")
    staging_info = os.lstat(staging_root)
    if (
        not stat.S_ISDIR(staging_info.st_mode)
        or stat.S_ISLNK(staging_info.st_mode)
        or staging_root.resolve(strict=True) != staging_root.absolute()
    ):
        raise Refusal("binary-seed staging root is unsafe")
    current = staging_root
    for part in PurePosixPath(RUNTIME_ROOT).parts[1:-1]:
        current = current / part
        _safe_new_directory(current)
        try:
            current.resolve(strict=True).relative_to(staging_root)
        except ValueError as exc:
            raise Refusal("binary-seed destination escapes staging root") from exc
    destination = staging_root / RUNTIME_ROOT.removeprefix("/")
    if os.path.lexists(destination):
        raise Refusal("binary-seed runtime destination already exists")
    return destination


def _runtime_records(
    destination: Path, *, allowed_workspace_files: Mapping[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = []
    for current, names, filenames in os.walk(destination, followlinks=False):
        current_path = Path(current)
        for name in names:
            entry = current_path / name
            info = os.lstat(entry)
            if (
                not stat.S_ISDIR(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or stat.S_IMODE(info.st_mode) != contract.RUNTIME_DIRECTORY_MODE
                or info.st_mtime_ns != contract.RUNTIME_MTIME_NS
            ):
                raise Refusal("transformed binary seed contains an unsafe directory")
            directories.append(
                {
                    "mode": contract.RUNTIME_DIRECTORY_MODE,
                    "mtime_ns": contract.RUNTIME_MTIME_NS,
                    "path": entry.relative_to(destination).as_posix(),
                }
            )
        for name in filenames:
            entry = current_path / name
            info = os.lstat(entry)
            mode = stat.S_IMODE(info.st_mode)
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_nlink != 1
                or mode not in (
                    contract.RUNTIME_FILE_MODE,
                    contract.RUNTIME_EXECUTABLE_MODE,
                )
                or info.st_mtime_ns != contract.RUNTIME_MTIME_NS
            ):
                raise Refusal("transformed binary seed contains an unsafe file")
            relative = entry.relative_to(destination).as_posix()
            sha256 = metadata.digest_file(entry)
            if _contains_bytes(entry, b"/workspace/envs/verl-b200"):
                raise Refusal("transformed binary seed retains its mutable source prefix")
            has_workspace = _contains_bytes(entry, WORKSPACE_PREFIX)
            if has_workspace and allowed_workspace_files.get(relative) != sha256:
                raise Refusal("transformed binary seed has an unreviewed /workspace occurrence")
            if not has_workspace and relative in allowed_workspace_files:
                raise Refusal("declared workspace occurrence is absent")
            files.append(
                {
                    "mode": mode,
                    "mtime_ns": contract.RUNTIME_MTIME_NS,
                    "path": relative,
                    "sha256": sha256,
                    "size": info.st_size,
                }
            )
    files.sort(key=lambda item: item["path"])
    directories.sort(key=lambda item: item["path"])
    observed_workspace = {
        item["path"]
        for item in files
        if item["path"] in allowed_workspace_files
    }
    if observed_workspace != set(allowed_workspace_files):
        raise Refusal("binary-seed workspace occurrence set differs")
    return files, directories


def extract_seed_stream(
    archive_stream: BinaryIO,
    *,
    destination: Path,
    lock_document: Mapping[str, Any],
    base_python: BinaryIO,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract one validated raw tar stream and return its exact final tree."""
    if os.path.lexists(destination):
        raise Refusal("binary-seed runtime destination already exists")
    _safe_new_directory(destination.parent)
    destination.mkdir(mode=0o755)
    archive_destination = destination / "python"
    archive_destination.mkdir(mode=0o755)
    seed = lock_document["seed"]
    transformation = lock_document["transformation"]
    prefix = str(seed["prefix"])
    removed_paths = set(transformation["removed_paths"])
    removed_links = {
        item["path"]: item["target"] for item in transformation["removed_links"]
    }
    allowed_pth = {
        item["path"]: item["sha256"] for item in transformation["allowed_pth"]
    }
    allowed_workspace_files = {
        item["path"]: item["sha256"]
        for item in transformation["allowed_workspace_files"]
    }
    rewrite_paths = set(transformation["rewrite_prefix_paths"])
    ignored_members = set(transformation["ignored_archive_members"])
    observed_removed: set[str] = set()
    observed_links: set[str] = set()
    observed_pth: set[str] = set()
    observed_rewrites: set[str] = set()
    observed_members: set[str] = set()
    observed_ignored: set[str] = set()
    source_shebang = transformation["source_shebang"].encode("ascii")
    target_shebang = transformation["target_shebang"].encode("ascii")
    source_prefix = transformation["source_prefix"].encode("ascii")
    target_prefix = transformation["target_prefix"].encode("ascii")
    pyvenv_target = transformation["target_pyvenv"].encode("ascii")

    try:
        hashed_stream = _HashingReader(archive_stream)
        archive = tarfile.open(fileobj=hashed_stream, mode="r|")
        with archive:
            for member in archive:
                archive_name = member.name.rstrip("/")
                if archive_name in ignored_members:
                    if not member.isdir() or member.size != 0:
                        raise Refusal("ignored binary-seed archive member differs")
                    observed_ignored.add(archive_name)
                    continue
                relative = _relative_member(member.name, prefix)
                if relative is None:
                    if not member.isdir():
                        raise Refusal("binary-seed prefix is not a directory")
                    continue
                if relative in observed_members:
                    raise Refusal("binary-seed archive member is duplicated")
                observed_members.add(relative)
                if len(observed_members) > MAX_FILES:
                    raise Refusal("binary-seed archive has too many members")
                if member.size < 0 or member.size > MAX_FILE_SIZE:
                    raise Refusal("binary-seed archive member is oversized")

                if member.islnk():
                    raise Refusal("binary-seed archive contains a hard link")
                if member.issym():
                    expected_target = removed_links.get(relative)
                    if expected_target is None or member.linkname != expected_target:
                        raise Refusal("binary-seed archive contains an unsafe link")
                    observed_links.add(relative)
                    continue
                if not member.isdir() and not member.isfile():
                    raise Refusal("binary-seed archive contains a special file")

                removed = relative in removed_paths or any(
                    relative.startswith(path + "/") for path in removed_paths
                )
                if removed:
                    observed_removed.add(relative)
                    continue
                relative_path = PurePosixPath(relative)
                if (
                    transformation["remove_bytecode"] is True
                    and (relative.endswith(".pyc") or "__pycache__" in relative_path.parts)
                ):
                    continue
                if "__editable__" in relative_path.name or relative.endswith(".egg-link"):
                    raise Refusal("unexpected editable hook in binary seed")

                output = archive_destination / relative
                if output.parent != archive_destination:
                    output.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                if member.isdir():
                    output.mkdir(exist_ok=True, mode=0o755)
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise Refusal("binary-seed regular member cannot be read")
                mode = (
                    contract.RUNTIME_EXECUTABLE_MODE
                    if member.mode & 0o111
                    else contract.RUNTIME_FILE_MODE
                )
                if relative == "pyvenv.cfg":
                    payload = _read_bounded(member, source, 4096)
                    if metadata.digest_bytes(payload) != transformation[
                        "source_pyvenv_sha256"
                    ]:
                        raise Refusal("binary-seed pyvenv source differs")
                    _write_bytes_once(output, pyvenv_target, mode=contract.RUNTIME_FILE_MODE)
                elif relative.startswith("bin/") and mode == contract.RUNTIME_EXECUTABLE_MODE:
                    payload = _read_bounded(member, source, 64 * 1024 * 1024)
                    first, separator, rest = payload.partition(b"\n")
                    first_line = first + separator
                    if first_line == source_shebang:
                        payload = target_shebang + rest
                    elif source_prefix in first_line or WORKSPACE_PREFIX in first_line:
                        raise Refusal("binary-seed shebang rewrite is not deterministic")
                    _write_bytes_once(output, payload, mode=mode)
                elif relative in rewrite_paths:
                    payload = _read_bounded(member, source, 64 * 1024 * 1024)
                    if source_prefix not in payload:
                        raise Refusal("binary-seed prefix rewrite source differs")
                    payload = payload.replace(source_prefix, target_prefix)
                    observed_rewrites.add(relative)
                    _write_bytes_once(output, payload, mode=mode)
                elif relative.endswith(".dist-info/RECORD"):
                    payload = _read_bounded(member, source, 64 * 1024 * 1024)
                    _write_bytes_once(output, _rewrite_record(payload), mode=mode)
                elif relative.endswith(".pth"):
                    payload = _read_bounded(member, source, 1024 * 1024)
                    if (
                        relative not in allowed_pth
                        or metadata.digest_bytes(payload) != allowed_pth[relative]
                    ):
                        raise Refusal("binary-seed PTH is absent from the exact allowlist")
                    observed_pth.add(relative)
                    _write_bytes_once(output, payload, mode=mode)
                elif relative.endswith(".dist-info/direct_url.json"):
                    payload = _read_bounded(member, source, 1024 * 1024)
                    _check_direct_origin(relative, payload)
                    _write_bytes_once(output, payload, mode=mode)
                else:
                    _write_once(output, source, mode=mode)
    except tarfile.TarError as exc:
        raise Refusal("binary-seed tar stream is malformed") from exc
    for _ in iter(lambda: hashed_stream.read(1024 * 1024), b""):
        pass
    if (
        hashed_stream.size != seed["uncompressed_size"]
        or "sha256:" + hashed_stream.digest.hexdigest()
        != seed["uncompressed_sha256"]
    ):
        raise Refusal("binary-seed uncompressed tar bytes differ")
    if set(removed_links) != observed_links:
        raise Refusal("binary-seed declared link set differs")
    if set(allowed_pth) != observed_pth:
        raise Refusal("binary-seed declared PTH set differs")
    if ignored_members != observed_ignored:
        raise Refusal("binary-seed ignored archive member set differs")
    if rewrite_paths != observed_rewrites:
        raise Refusal("binary-seed declared prefix rewrite set differs")
    missing_removals = {
        path
        for path in removed_paths
        if path not in observed_removed
        and not any(item.startswith(path + "/") for item in observed_removed)
    }
    if missing_removals:
        raise Refusal("binary-seed declared removal set differs")

    python_target = destination / RUNTIME_PYTHON.removeprefix(RUNTIME_ROOT + "/")
    if os.path.lexists(python_target):
        raise Refusal("binary-seed Python replacement destination exists")
    python_target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    base_python.seek(0)
    _write_once(python_target, base_python, mode=contract.RUNTIME_EXECUTABLE_MODE)
    if metadata.digest_file(python_target) != lock_document["base_image"]["python"]["sha256"]:
        raise Refusal("copied base Python bytes differ")

    for current, names, _ in os.walk(destination, topdown=False, followlinks=False):
        for name in names:
            entry = Path(current) / name
            os.chmod(entry, contract.RUNTIME_DIRECTORY_MODE, follow_symlinks=False)
            os.utime(
                entry,
                ns=(contract.RUNTIME_MTIME_NS, contract.RUNTIME_MTIME_NS),
                follow_symlinks=False,
            )
    return _runtime_records(
        destination, allowed_workspace_files=allowed_workspace_files
    )


def prepare(
    *,
    lock_path: Path,
    archive_path: Path,
    base_python: Path,
    staging_root: Path,
    archive_format: str,
) -> Path:
    if archive_format != "tar":
        raise Refusal("only the locked decoder-free raw tar is accepted")
    lock_payload, _, _ = metadata.validate_lock(
        lock_path, uncompressed_seed_path=archive_path
    )
    lock_document = json.loads(lock_payload.decode("ascii"))
    base_identity = lock_document["base_image"]["python"]
    if str(base_python) != base_identity["path"] or not base_python.is_absolute():
        raise Refusal("pinned base Python path differs")
    try:
        base_lstat = os.lstat(base_python)
        base_fd = os.open(base_python, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except (FileNotFoundError, OSError) as exc:
        raise Refusal("pinned base Python is absent or unsafe") from exc
    try:
        base_fstat = os.fstat(base_fd)
        if (
            not stat.S_ISREG(base_lstat.st_mode)
            or stat.S_ISLNK(base_lstat.st_mode)
            or base_lstat.st_nlink != 1
            or base_fstat.st_dev != base_lstat.st_dev
            or base_fstat.st_ino != base_lstat.st_ino
            or base_fstat.st_nlink != 1
            or metadata.digest_fd(base_fd) != base_identity["sha256"]
        ):
            raise Refusal("pinned base Python bytes differ")
        destination = _safe_destination(staging_root)
        archive_fd = os.open(archive_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            archive_info = os.fstat(archive_fd)
            seed = lock_document["seed"]
            if (
                not stat.S_ISREG(archive_info.st_mode)
                or archive_info.st_nlink != 1
                or archive_info.st_size != seed["uncompressed_size"]
                or metadata.digest_fd(archive_fd) != seed["uncompressed_sha256"]
            ):
                raise Refusal("binary-seed raw tar bytes differ")
            with os.fdopen(os.dup(archive_fd), "rb") as source, os.fdopen(
                os.dup(base_fd), "rb"
            ) as python_source:
                files, directories = extract_seed_stream(
                    source,
                    destination=destination,
                    lock_document=lock_document,
                    base_python=python_source,
                )
        finally:
            os.close(archive_fd)
    finally:
        os.close(base_fd)
    receipt = {
        "base_python_sha256": base_identity["sha256"],
        "dependency_lock_sha256": metadata.digest_bytes(lock_payload),
        "directories": directories,
        "files": files,
        "schema": contract.SEED_TRANSFORMATION_SCHEMA,
        "seed_sha256": lock_document["seed"]["sha256"],
        "uncompressed_seed_sha256": lock_document["seed"]["uncompressed_sha256"],
    }
    receipt_path = destination / metadata.SEED_TRANSFORMATION_NAME
    _write_bytes_once(receipt_path, metadata.canonical(receipt), mode=contract.RUNTIME_FILE_MODE)
    return receipt_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument("--binary-seed", required=True, type=Path)
    parser.add_argument("--base-python", required=True, type=Path)
    parser.add_argument("--staging-root", required=True, type=Path)
    parser.add_argument("--archive-format", choices=("tar",), required=True)
    args = parser.parse_args(argv)
    try:
        prepare(
            lock_path=args.dependency_lock,
            archive_path=args.binary_seed,
            base_python=args.base_python,
            staging_root=args.staging_root,
            archive_format=args.archive_format,
        )
    except (OSError, Refusal, metadata.Refusal) as exc:
        print(f"prepare_binary_seed: REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
