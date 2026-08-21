#!/usr/bin/env python3
"""Generate the closed D043 runtime evidence objects without network access.

This does not resolve or install dependencies.  It accepts only an already
complete, artifact-hashed lock and an already installed runtime tree, then
cross-checks the two before publishing canonical metadata.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import email.parser
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import os
import re
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


_CONTRACT_PATH = Path(__file__).resolve(strict=True).with_name("runtime_contract.py")
_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "debate_runtime_contract", _CONTRACT_PATH
)
if _CONTRACT_SPEC is None or _CONTRACT_SPEC.loader is None:
    raise RuntimeError("cannot load the tracked D043 runtime contract")
contract = importlib.util.module_from_spec(_CONTRACT_SPEC)
_CONTRACT_SPEC.loader.exec_module(contract)

RUNTIME_ROOT = contract.RUNTIME_ROOT
RUNTIME_PYTHON = contract.RUNTIME_PYTHON
MANIFEST_SCHEMA = contract.MANIFEST_SCHEMA
LOCK_SCHEMA = contract.LOCK_SCHEMA
SPEC_SCHEMA = contract.SPEC_SCHEMA
INVENTORY_SCHEMA = contract.INVENTORY_SCHEMA
BINARY_SEED_MANIFEST_SCHEMA = contract.BINARY_SEED_MANIFEST_SCHEMA
BINARY_SEED_LOCK_SCHEMA = contract.BINARY_SEED_LOCK_SCHEMA
BINARY_SEED_SPEC_SCHEMA = contract.BINARY_SEED_SPEC_SCHEMA
BINARY_SEED_INVENTORY_SCHEMA = contract.BINARY_SEED_INVENTORY_SCHEMA
SEED_TRANSFORMATION_SCHEMA = contract.SEED_TRANSFORMATION_SCHEMA
IMPORT_PROBE_SCHEMA = contract.IMPORT_PROBE_SCHEMA
CUDA_PROBE_SCHEMA = contract.CUDA_PROBE_SCHEMA
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
NAME_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,126}[a-z0-9])?")
VERSION_RE = re.compile(r"[!-~]{1,256}")
MODULE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){0,31}")
MAX_DISTRIBUTIONS = contract.MAX_DISTRIBUTIONS
MAX_FILES = contract.MAX_FILES
MAX_FILE_SIZE = contract.MAX_FILE_SIZE
FIXED_IMPORT_SOURCE = contract.FIXED_IMPORT_SOURCE
FIXED_CUDA_SOURCE = contract.FIXED_CUDA_SOURCE
SEED_TRANSFORMATION_NAME = "seed-transformation.json"


class Refusal(RuntimeError):
    """An input cannot establish the immutable runtime contract."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def canonical(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def digest_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    for block in iter(lambda: os.read(fd, 1024 * 1024), b""):
        digest.update(block)
    os.lseek(fd, 0, os.SEEK_SET)
    return "sha256:" + digest.hexdigest()


def _read_canonical(path: Path, *, maximum: int) -> tuple[bytes, dict[str, Any]]:
    info = os.lstat(path)
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or info.st_size < 2
        or info.st_size > maximum
    ):
        raise Refusal(f"unsafe or oversized canonical input: {path}")
    payload = path.read_bytes()
    try:
        document = json.loads(
            payload.decode("ascii"), object_pairs_hook=_strict_object
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise Refusal(f"malformed canonical input: {path}") from exc
    if not isinstance(document, dict) or canonical(document) != payload:
        raise Refusal(f"input is not canonical compact ASCII JSON: {path}")
    return payload, document


def _exact_keys(document: Mapping[str, Any], expected: set[str], field: str) -> None:
    if set(document) != expected:
        raise Refusal(f"{field} has missing or unknown keys")


def _name(value: object, field: str) -> str:
    if not isinstance(value, str) or NAME_RE.fullmatch(value) is None:
        raise Refusal(f"{field} must be a normalized distribution name")
    if re.sub(r"[-_.]+", "-", value).lower() != value:
        raise Refusal(f"{field} must be PEP 503 normalized")
    return value


def _version(value: object, field: str) -> str:
    if not isinstance(value, str) or VERSION_RE.fullmatch(value) is None:
        raise Refusal(f"{field} must be exact printable ASCII")
    if any(operator in value for operator in ("*", "<", ">", "=", "~")):
        raise Refusal(f"{field} must not contain a version operator or wildcard")
    return value


def _module(value: object, field: str) -> str:
    if not isinstance(value, str) or MODULE_RE.fullmatch(value) is None:
        raise Refusal(f"{field} is not a canonical import name")
    return value


def _relative_path(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8", "surrogatepass")) > contract.MAX_PATH_BYTES
        or "\\" in value
    ):
        raise Refusal(f"{field} must be a bounded POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise Refusal(f"{field} must be normalized below its declared root")
    if path.as_posix() != value or any(ord(char) < 32 or ord(char) > 126 for char in value):
        raise Refusal(f"{field} must be normalized printable ASCII")
    return value


def _inspect_locked_wheel(
    path: Path, *, expected_name: str, expected_version: str
) -> dict[str, str]:
    """Validate wheel identity/RECORD and return every source-file digest."""
    if path.suffix != ".whl":
        raise Refusal("every locked runtime artifact must be a wheel")
    try:
        with zipfile.ZipFile(path) as wheel:
            infos = [item for item in wheel.infolist() if not item.is_dir()]
            if not infos or len(infos) > MAX_FILES:
                raise Refusal("locked wheel file count is invalid")
            archive_files: dict[str, zipfile.ZipInfo] = {}
            for info in infos:
                name = _relative_path(info.filename, "locked wheel member")
                mode = (info.external_attr >> 16) & 0xFFFF
                if (
                    name in archive_files
                    or info.file_size > MAX_FILE_SIZE
                    or (mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG))
                ):
                    raise Refusal("locked wheel contains an unsafe member")
                archive_files[name] = info
            metadata_files = [
                name for name in archive_files if name.endswith(".dist-info/METADATA")
            ]
            record_files = [
                name for name in archive_files if name.endswith(".dist-info/RECORD")
            ]
            if len(metadata_files) != 1 or len(record_files) != 1:
                raise Refusal("locked wheel must contain one METADATA and RECORD")
            metadata_path = metadata_files[0]
            record_path = record_files[0]
            if metadata_path.rsplit("/", 1)[0] != record_path.rsplit("/", 1)[0]:
                raise Refusal("locked wheel METADATA and RECORD disagree")
            try:
                metadata = email.parser.Parser().parsestr(
                    wheel.read(metadata_path).decode("utf-8")
                )
            except (UnicodeDecodeError, ValueError) as exc:
                raise Refusal("locked wheel METADATA is malformed") from exc
            observed_name = _name(
                re.sub(r"[-_.]+", "-", metadata.get("Name", "")).lower(),
                "locked wheel METADATA name",
            )
            if observed_name != expected_name or metadata.get("Version") != expected_version:
                raise Refusal("locked wheel identity differs from dependency lock")
            try:
                rows = list(
                    csv.reader(io.StringIO(wheel.read(record_path).decode("utf-8")))
                )
            except (UnicodeDecodeError, csv.Error) as exc:
                raise Refusal("locked wheel RECORD is malformed") from exc
            source_hashes: dict[str, str] = {}
            for row in rows:
                if len(row) != 3:
                    raise Refusal("locked wheel RECORD row is malformed")
                member = _relative_path(row[0], "locked wheel RECORD member")
                if member not in archive_files or member in source_hashes:
                    raise Refusal("locked wheel RECORD is incomplete or duplicated")
                if member == record_path:
                    if row[1] or row[2]:
                        raise Refusal("locked wheel RECORD must not self-hash")
                    source_hashes[member] = digest_bytes(wheel.read(member))
                    continue
                if not row[1].startswith("sha256=") or not row[2].isdigit():
                    raise Refusal("locked wheel member lacks an exact SHA-256 and size")
                try:
                    encoded = row[1].removeprefix("sha256=")
                    if re.fullmatch(r"[A-Za-z0-9_-]{43}", encoded) is None:
                        raise ValueError("noncanonical SHA-256 encoding")
                    raw_digest = base64.urlsafe_b64decode(
                        encoded + "=" * (-len(encoded) % 4)
                    )
                except (ValueError, binascii.Error) as exc:
                    raise Refusal("locked wheel RECORD digest is malformed") from exc
                payload = wheel.read(member)
                if (
                    len(raw_digest) != 32
                    or int(row[2]) != len(payload)
                    or hashlib.sha256(payload).digest() != raw_digest
                ):
                    raise Refusal("locked wheel member differs from its RECORD")
                source_hashes[member] = "sha256:" + raw_digest.hex()
            if set(source_hashes) != set(archive_files):
                raise Refusal("locked wheel RECORD does not cover the archive")
            return source_hashes
    except zipfile.BadZipFile as exc:
        raise Refusal("locked runtime artifact is not a valid wheel ZIP") from exc


def _validate_binary_seed_lock(
    path: Path,
    payload: bytes,
    document: dict[str, Any],
    seed_path: Path | None,
    uncompressed_seed_path: Path | None,
) -> tuple[bytes, dict[str, tuple[str, dict[str, object]]], str]:
    _exact_keys(document, contract.BINARY_SEED_LOCK_KEYS, "binary-seed lock")
    if (
        document["schema"] != BINARY_SEED_LOCK_SCHEMA
        or document["provenance_tier"] != contract.PROVENANCE_TIER_BINARY_SEED
        or document["runtime_root"] != RUNTIME_ROOT
    ):
        raise Refusal("binary-seed lock identity differs")
    python_version = _version(document["python_version"], "binary-seed Python version")

    base = document["base_image"]
    if not isinstance(base, dict):
        raise Refusal("binary-seed base image must be an object")
    _exact_keys(base, contract.BINARY_SEED_BASE_IMAGE_KEYS, "binary-seed base image")
    reference = base["reference"]
    if (
        not isinstance(reference, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9./_-]{1,254}@sha256:[0-9a-f]{64}", reference)
        is None
        or reference != contract.B200_BASE_IMAGE_REFERENCE
    ):
        raise Refusal("binary-seed base image must be digest-pinned")
    base_python = base["python"]
    if not isinstance(base_python, dict):
        raise Refusal("binary-seed base Python must be an object")
    _exact_keys(
        base_python, contract.BINARY_SEED_BASE_PYTHON_KEYS, "binary-seed base Python"
    )
    if base_python["path"] != contract.B200_BASE_PYTHON_PATH or not isinstance(
        base_python["sha256"], str
    ) or DIGEST_RE.fullmatch(base_python["sha256"]) is None or base_python[
        "sha256"
    ] != contract.B200_BASE_PYTHON_SHA256:
        raise Refusal("binary-seed base Python identity differs")

    seed = document["seed"]
    if not isinstance(seed, dict):
        raise Refusal("binary-seed source must be an object")
    _exact_keys(seed, contract.BINARY_SEED_KEYS, "binary-seed source")
    if (
        not isinstance(seed["repository"], str)
        or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}/[A-Za-z0-9_.-]{1,128}", seed["repository"])
        is None
        or not isinstance(seed["revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", seed["revision"]) is None
        or _relative_path(seed["path"], "binary-seed artifact path")
        != seed["path"]
        or "/" in seed["path"]
        or type(seed["size"]) is not int
        or seed["size"] <= 0
        or not isinstance(seed["sha256"], str)
        or DIGEST_RE.fullmatch(seed["sha256"]) is None
        or _relative_path(seed["uncompressed_path"], "binary-seed raw tar path")
        != seed["uncompressed_path"]
        or "/" in seed["uncompressed_path"]
        or seed["format"] != "tar+zstd"
        or _relative_path(seed["prefix"], "binary-seed archive prefix")
        != seed["prefix"]
        or seed["repository"] != contract.B200_SEED_REPOSITORY
        or seed["revision"] != contract.B200_SEED_REVISION
        or seed["path"] != contract.B200_SEED_PATH
        or seed["size"] != contract.B200_SEED_SIZE
        or seed["sha256"] != contract.B200_SEED_SHA256
        or type(seed["uncompressed_size"]) is not int
        or seed["uncompressed_size"] != contract.B200_SEED_UNCOMPRESSED_SIZE
        or not isinstance(seed["uncompressed_sha256"], str)
        or DIGEST_RE.fullmatch(seed["uncompressed_sha256"]) is None
        or seed["uncompressed_sha256"]
        != contract.B200_SEED_UNCOMPRESSED_SHA256
        or seed["uncompressed_path"] != contract.B200_SEED_UNCOMPRESSED_PATH
        or seed["prefix"] != contract.B200_SEED_PREFIX
    ):
        raise Refusal("binary-seed source identity is malformed")
    if (seed_path is None) == (uncompressed_seed_path is None):
        raise Refusal("exactly one compressed or uncompressed binary seed is required")
    selected = seed_path if seed_path is not None else uncompressed_seed_path
    assert selected is not None
    expected_path = seed["path"] if seed_path is not None else seed["uncompressed_path"]
    expected_size = seed["size"] if seed_path is not None else seed["uncompressed_size"]
    expected_sha256 = (
        seed["sha256"] if seed_path is not None else seed["uncompressed_sha256"]
    )
    compressed_blob_name = str(expected_sha256).removeprefix("sha256:")
    allowed_names = (
        {str(expected_path), compressed_blob_name}
        if seed_path is not None
        else {str(expected_path)}
    )
    if selected.name not in allowed_names or not selected.is_absolute():
        raise Refusal("content-addressed binary seed path differs")
    try:
        seed_info = os.lstat(selected)
        resolved_seed = selected.resolve(strict=True)
        if resolved_seed != selected.absolute():
            raise Refusal("content-addressed binary seed path is redirected")
        fd = os.open(selected, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError as exc:
        raise Refusal("content-addressed binary seed is absent") from exc
    except OSError as exc:
        raise Refusal("content-addressed binary seed is unsafe") from exc
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(seed_info.st_mode)
            or stat.S_ISLNK(seed_info.st_mode)
            or seed_info.st_nlink != 1
            or opened.st_dev != seed_info.st_dev
            or opened.st_ino != seed_info.st_ino
            or opened.st_nlink != 1
            or opened.st_size != expected_size
            or digest_fd(fd) != expected_sha256
        ):
            raise Refusal("content-addressed binary seed bytes differ")
    finally:
        os.close(fd)

    transformation = document["transformation"]
    if not isinstance(transformation, dict):
        raise Refusal("binary-seed transformation must be an object")
    _exact_keys(
        transformation,
        contract.BINARY_SEED_TRANSFORMATION_KEYS,
        "binary-seed transformation",
    )
    removed_paths = transformation["removed_paths"]
    if not isinstance(removed_paths, list) or not removed_paths:
        raise Refusal("binary-seed removed paths are malformed")
    prior_path = ""
    for raw in removed_paths:
        item = _relative_path(raw, "binary-seed removed path")
        if item <= prior_path:
            raise Refusal("binary-seed removed paths must be unique and sorted")
        prior_path = item
    removed_links = transformation["removed_links"]
    if not isinstance(removed_links, list) or not removed_links:
        raise Refusal("binary-seed removed links are malformed")
    prior_link = ""
    for raw in removed_links:
        if not isinstance(raw, dict):
            raise Refusal("binary-seed removed link must be an object")
        _exact_keys(raw, contract.BINARY_SEED_REMOVED_LINK_KEYS, "removed link")
        link_path = _relative_path(raw["path"], "removed link path")
        if link_path <= prior_link or not isinstance(raw["target"], str) or not raw["target"]:
            raise Refusal("binary-seed removed links must be unique and sorted")
        prior_link = link_path
    allowed_pth = transformation["allowed_pth"]
    if not isinstance(allowed_pth, list) or not allowed_pth:
        raise Refusal("binary-seed PTH allowlist is malformed")
    prior_pth = ""
    for raw in allowed_pth:
        if not isinstance(raw, dict):
            raise Refusal("binary-seed PTH allowlist entry must be an object")
        _exact_keys(raw, contract.BINARY_SEED_PTH_KEYS, "allowed PTH")
        pth_path = _relative_path(raw["path"], "allowed PTH path")
        if (
            pth_path <= prior_pth
            or not pth_path.endswith(".pth")
            or not isinstance(raw["sha256"], str)
            or DIGEST_RE.fullmatch(raw["sha256"]) is None
        ):
            raise Refusal("binary-seed PTH allowlist must be unique and sorted")
        prior_pth = pth_path
    allowed_workspace = transformation["allowed_workspace_files"]
    if not isinstance(allowed_workspace, list):
        raise Refusal("binary-seed workspace occurrence allowlist is malformed")
    prior_workspace = ""
    for raw in allowed_workspace:
        if not isinstance(raw, dict):
            raise Refusal("workspace occurrence allowlist entry must be an object")
        _exact_keys(
            raw,
            contract.BINARY_SEED_WORKSPACE_FILE_KEYS,
            "allowed workspace occurrence",
        )
        workspace_path = _relative_path(raw["path"], "allowed workspace file path")
        if (
            workspace_path <= prior_workspace
            or not isinstance(raw["sha256"], str)
            or DIGEST_RE.fullmatch(raw["sha256"]) is None
        ):
            raise Refusal("workspace occurrence allowlist must be unique and sorted")
        prior_workspace = workspace_path
    ignored_members = transformation["ignored_archive_members"]
    if (
        not isinstance(ignored_members, list)
        or ignored_members != list(contract.B200_IGNORED_ARCHIVE_MEMBERS)
    ):
        raise Refusal("binary-seed ignored archive members differ")
    prior_ignored = ""
    for raw in ignored_members:
        item = _relative_path(raw, "ignored archive member")
        if item <= prior_ignored or item.startswith(seed["prefix"] + "/"):
            raise Refusal("ignored archive members must be unique and outside the seed")
        prior_ignored = item
    rewrite_paths = transformation["rewrite_prefix_paths"]
    if not isinstance(rewrite_paths, list) or not rewrite_paths:
        raise Refusal("binary-seed prefix rewrite paths are malformed")
    prior_rewrite = ""
    for raw in rewrite_paths:
        item = _relative_path(raw, "binary-seed prefix rewrite path")
        if item <= prior_rewrite:
            raise Refusal("binary-seed prefix rewrite paths must be unique and sorted")
        prior_rewrite = item
    if (
        transformation["remove_bytecode"] is not True
        or transformation["source_prefix"] != "/workspace/envs/verl-b200"
        or transformation["target_prefix"] != RUNTIME_ROOT + "/python"
        or
        transformation["source_shebang"]
        != "#!/workspace/envs/verl-b200/bin/python\n"
        or transformation["target_shebang"] != "#!" + RUNTIME_PYTHON + "\n"
        or not isinstance(transformation["source_pyvenv_sha256"], str)
        or DIGEST_RE.fullmatch(transformation["source_pyvenv_sha256"]) is None
        or not isinstance(transformation["target_pyvenv"], str)
        or transformation["target_pyvenv"]
        != "home = /usr/bin\nimplementation = CPython\nuv = 0.12.0\nversion_info = 3.12.3\ninclude-system-site-packages = false\nseed = true\n"
    ):
        raise Refusal("binary-seed deterministic rewrite policy differs")

    distributions = document["distributions"]
    if not isinstance(distributions, list) or not 1 <= len(distributions) <= MAX_DISTRIBUTIONS:
        raise Refusal("binary-seed distribution count is invalid")
    locked: dict[str, tuple[str, dict[str, object]]] = {}
    prior_name = ""
    for index, raw in enumerate(distributions):
        if not isinstance(raw, dict):
            raise Refusal("binary-seed distribution must be an object")
        _exact_keys(
            raw, contract.BINARY_SEED_DISTRIBUTION_KEYS, "binary-seed distribution"
        )
        name = _name(raw["name"], f"distributions[{index}].name")
        version = _version(raw["version"], f"distributions[{index}].version")
        if name <= prior_name or name in locked or name == "debate":
            raise Refusal("binary-seed distributions must be unique, sorted, and non-editable")
        locked[name] = (
            version,
            {
                "provenance_tier": contract.PROVENANCE_TIER_BINARY_SEED,
                "seed_sha256": seed["sha256"],
            },
        )
        prior_name = name
    return payload, locked, python_version


def validate_lock(
    path: Path,
    *,
    seed_path: Path | None = None,
    uncompressed_seed_path: Path | None = None,
) -> tuple[bytes, dict[str, tuple[str, dict[str, object]]], str]:
    payload, document = _read_canonical(path, maximum=16 * 1024 * 1024)
    if document.get("schema") == BINARY_SEED_LOCK_SCHEMA:
        return _validate_binary_seed_lock(
            path, payload, document, seed_path, uncompressed_seed_path
        )
    _exact_keys(
        document,
        contract.LOCK_KEYS,
        "dependency lock",
    )
    if document["schema"] != LOCK_SCHEMA or document["runtime_root"] != RUNTIME_ROOT:
        raise Refusal("dependency lock schema or runtime root differs")
    python_version = _version(
        document["python_version"], "dependency lock python_version"
    )
    distributions = document["distributions"]
    if not isinstance(distributions, list) or not 1 <= len(distributions) <= MAX_DISTRIBUTIONS:
        raise Refusal("dependency lock distributions count is invalid")
    result: dict[str, tuple[str, dict[str, object]]] = {}
    artifact_paths: set[str] = set()
    previous = ""
    for index, raw in enumerate(distributions):
        if not isinstance(raw, dict):
            raise Refusal("dependency lock distribution must be an object")
        _exact_keys(raw, contract.LOCK_DISTRIBUTION_KEYS, "lock distribution")
        name = _name(raw["name"], f"distributions[{index}].name")
        version = _version(raw["version"], f"distributions[{index}].version")
        if name <= previous or name in result:
            raise Refusal("dependency lock distributions must be unique and name-sorted")
        previous = name
        artifact = raw["artifact"]
        if not isinstance(artifact, dict):
            raise Refusal("lock artifact must be an object")
        _exact_keys(artifact, contract.LOCK_ARTIFACT_KEYS, "lock artifact")
        artifact_path = _relative_path(artifact["path"], "lock artifact path")
        if artifact_path in artifact_paths:
            raise Refusal("dependency lock artifact paths must be unique")
        artifact_paths.add(artifact_path)
        artifact_digest = artifact["sha256"]
        if not isinstance(artifact_digest, str) or DIGEST_RE.fullmatch(artifact_digest) is None:
            raise Refusal("lock artifact sha256 is malformed")
        source = path.parent / artifact_path
        try:
            info = os.lstat(source)
            resolved_source = source.resolve(strict=True)
        except FileNotFoundError as exc:
            raise Refusal(f"locked artifact is absent: {artifact_path}") from exc
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1
            or info.st_size > MAX_FILE_SIZE
            or resolved_source != source.absolute()
        ):
            raise Refusal(f"locked artifact is unsafe: {artifact_path}")
        if digest_file(source) != artifact_digest:
            raise Refusal(f"locked artifact digest differs: {artifact_path}")
        wheel_files = _inspect_locked_wheel(
            source, expected_name=name, expected_version=version
        )
        result[name] = (
            version,
            {
                "path": artifact_path,
                "sha256": artifact_digest,
                "wheel_files": wheel_files,
            },
        )
    return payload, result, python_version


def validate_spec(
    path: Path,
    locked: Mapping[str, tuple[str, object]],
    locked_python_version: str,
) -> dict[str, Any]:
    _, document = _read_canonical(path, maximum=1024 * 1024)
    seed_mode = bool(locked) and all(
        isinstance(item[1], dict)
        and item[1].get("provenance_tier") == contract.PROVENANCE_TIER_BINARY_SEED
        for item in locked.values()
    )
    _exact_keys(
        document,
        contract.SPEC_KEYS,
        "build spec",
    )
    expected_schema = BINARY_SEED_SPEC_SCHEMA if seed_mode else SPEC_SCHEMA
    if document["schema"] != expected_schema or document["runtime_root"] != RUNTIME_ROOT:
        raise Refusal("build spec schema or runtime root differs")
    python = document["python"]
    if not isinstance(python, dict):
        raise Refusal("build spec python must be an object")
    _exact_keys(python, contract.PYTHON_KEYS, "build spec python")
    if python["path"] != RUNTIME_PYTHON:
        raise Refusal("build spec Python path is not fixed")
    python_version = _version(python["version"], "build spec Python version")
    if python_version != locked_python_version:
        raise Refusal("build spec Python version differs from dependency lock")
    site_packages = document["site_packages_path"]
    expected_prefix = RUNTIME_ROOT + "/python/lib/python3.12/"
    if (
        not isinstance(site_packages, str)
        or not site_packages.startswith(expected_prefix)
        or not site_packages.endswith("/site-packages")
        or PurePosixPath(site_packages).as_posix() != site_packages
    ):
        raise Refusal("site_packages_path must be the fixed runtime Python tree")
    imports = document["required_imports"]
    if not isinstance(imports, list) or not 1 <= len(imports) <= contract.MAX_IMPORTS:
        raise Refusal("required_imports count is invalid")
    previous_module = ""
    for index, item in enumerate(imports):
        if not isinstance(item, dict):
            raise Refusal("required import must be an object")
        _exact_keys(item, contract.REQUIRED_IMPORT_KEYS, "required import")
        module = _module(item["module"], f"required_imports[{index}].module")
        distribution = _name(item["distribution"], "required import distribution")
        version = _version(item["version"], "required import version")
        if module <= previous_module:
            raise Refusal("required imports must be unique and module-sorted")
        previous_module = module
        if distribution not in locked or locked[distribution][0] != version:
            raise Refusal("required import does not match the exact dependency lock")
    cuda = document["cuda"]
    if not isinstance(cuda, dict):
        raise Refusal("cuda build spec must be an object")
    _exact_keys(
        cuda,
        contract.BINARY_SEED_CUDA_SPEC_KEYS if seed_mode else contract.CUDA_SPEC_KEYS,
        "cuda build spec",
    )
    torch_distribution = _name(cuda["torch_distribution"], "torch distribution")
    torch_version = _version(cuda["torch_version"], "torch version")
    torch_distribution_version = (
        _version(cuda["torch_distribution_version"], "torch distribution version")
        if seed_mode
        else torch_version
    )
    if seed_mode and (
        torch_distribution_version != contract.B200_TORCH_DISTRIBUTION_VERSION
        or torch_version != contract.B200_TORCH_LIVE_VERSION
    ):
        raise Refusal("B200 seed requires the exact Torch distribution/live-version pair")
    torch_cuda_version = _version(cuda["torch_cuda_version"], "torch CUDA version")
    if (
        torch_distribution not in locked
        or locked[torch_distribution][0] != torch_distribution_version
    ):
        raise Refusal("CUDA torch identity does not match the dependency lock")
    if torch_cuda_version != contract.EXACT_TORCH_CUDA_VERSION:
        raise Refusal("D043 B200 runtime requires exact torch CUDA version 13.0")
    device_count = cuda["device_count"]
    if type(device_count) is not int or device_count != contract.EXACT_DEVICE_COUNT:
        raise Refusal("D043 B200 runtime requires exact device_count 2")
    device_name = cuda["device_name"]
    if device_name != contract.EXACT_DEVICE_NAME:
        raise Refusal("D043 B200 runtime requires exact device name NVIDIA B200")
    capability = cuda["compute_capability"]
    if (
        not isinstance(capability, list)
        or capability != contract.EXACT_COMPUTE_CAPABILITY
        or any(type(component) is not int for component in capability)
    ):
        raise Refusal("D043 B200 runtime requires exact compute capability [10,0]")
    minimum_driver = cuda["minimum_host_driver_version"]
    if (
        type(minimum_driver) is not int
        or minimum_driver != contract.MINIMUM_HOST_DRIVER_VERSION
    ):
        raise Refusal("D043 B200 runtime requires host driver version >=580")
    nvlink_pairs = cuda["nvlink_pairs"]
    if (
        not isinstance(nvlink_pairs, list)
        or nvlink_pairs != contract.EXACT_NVLINK_PAIRS
        or any(
            not isinstance(pair, list)
            or any(type(index) is not int for index in pair)
            for pair in nvlink_pairs
        )
    ):
        raise Refusal("D043 B200 runtime requires exact peer NVLink pair [[0,1]]")
    if cuda["nvlink_link_label"] != contract.EXACT_NVLINK_LINK_LABEL:
        raise Refusal("D043 B200 runtime requires exact peer NVLink label NV18")
    extensions = cuda["compiled_extensions"]
    if not isinstance(extensions, list) or not 1 <= len(extensions) <= contract.MAX_EXTENSIONS:
        raise Refusal("compiled_extensions count is invalid")
    previous_extension = ""
    for index, item in enumerate(extensions):
        if not isinstance(item, dict):
            raise Refusal("compiled extension must be an object")
        _exact_keys(item, contract.COMPILED_EXTENSION_SPEC_KEYS, "compiled extension")
        module = _module(item["module"], f"compiled_extensions[{index}].module")
        if module <= previous_extension:
            raise Refusal("compiled extensions must be unique and module-sorted")
        previous_extension = module
    if document["python"]["version"] != python_version:
        raise Refusal("invalid exact build spec")
    return document


def _physical(staging_root: Path, logical: str) -> Path:
    return staging_root / logical.removeprefix("/")


def _safe_runtime_file(path: Path, physical_root: Path) -> tuple[str, int, str]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(physical_root.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise Refusal(f"installed distribution file escapes runtime root: {path}") from exc
    info = os.lstat(path)
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or info.st_size > MAX_FILE_SIZE
    ):
        raise Refusal(f"installed distribution file is unsafe: {path}")
    return _relative_path(relative, "installed file path"), info.st_size, digest_file(path)


def inventory(
    staging_root: Path,
    spec: Mapping[str, Any],
    locked: Mapping[str, tuple[str, object]],
    *,
    runtime_files: list[dict[str, Any]] | None = None,
    runtime_directories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    seed_mode = runtime_files is not None or runtime_directories is not None
    if (runtime_files is None) != (runtime_directories is None):
        raise Refusal("binary-seed runtime tree inventories must be provided together")
    physical_root = _physical(staging_root, RUNTIME_ROOT)
    site_packages = _physical(staging_root, str(spec["site_packages_path"]))
    if not site_packages.is_dir() or site_packages.is_symlink():
        raise Refusal("fixed site-packages directory is absent or redirected")
    if list(site_packages.glob("*.egg-info")) or list(site_packages.glob("*.egg-link")):
        raise Refusal("editable or legacy installed distributions are forbidden")
    site_relative = site_packages.resolve(strict=True).relative_to(
        physical_root.resolve(strict=True)
    ).as_posix()
    distributions: list[dict[str, Any]] = []
    all_paths: set[str] = set()
    claims_by_path: dict[str, set[str]] = {}
    dist_infos = sorted(site_packages.glob("*.dist-info"))
    for dist_info in dist_infos:
        if (
            not dist_info.is_dir()
            or dist_info.is_symlink()
            or not (dist_info / "METADATA").is_file()
            or not (dist_info / "RECORD").is_file()
        ):
            raise Refusal(f"incomplete or redirected dist-info directory: {dist_info}")
    for dist_info in dist_infos:
        metadata_file = dist_info / "METADATA"
        distribution = importlib.metadata.PathDistribution(metadata_file.parent)
        name = _name(
            re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower(),
            "installed distribution name",
        )
        version = _version(distribution.version, "installed distribution version")
        files = distribution.files
        if not files:
            raise Refusal(f"installed distribution has no RECORD inventory: {name}")
        file_records: list[dict[str, Any]] = []
        for package_path in files:
            located = Path(distribution.locate_file(package_path))
            relative, size, sha256 = _safe_runtime_file(located, physical_root)
            claimants = claims_by_path.setdefault(relative, set())
            if name in claimants:
                raise Refusal(f"installed file is duplicated within RECORD: {relative}")
            claimants.add(name)
            all_paths.add(relative)
            file_records.append({"path": relative, "sha256": sha256, "size": size})
        file_records.sort(key=lambda item: item["path"])
        required_metadata_files = {
            (metadata_file.parent / "METADATA").resolve(strict=True)
            .relative_to(physical_root.resolve(strict=True))
            .as_posix(),
            (metadata_file.parent / "RECORD").resolve(strict=True)
            .relative_to(physical_root.resolve(strict=True))
            .as_posix(),
        }
        recorded_paths = {str(item["path"]) for item in file_records}
        if not required_metadata_files <= recorded_paths:
            raise Refusal(f"RECORD omits METADATA or itself for {name}")
        if name not in locked:
            raise Refusal(f"installed distribution is absent from dependency lock: {name}")
        locked_artifact = locked[name][1]
        if not isinstance(locked_artifact, dict):
            raise Refusal("internal locked artifact identity is malformed")
        wheel_files = locked_artifact.get("wheel_files")
        if not seed_mode:
            if not isinstance(wheel_files, dict):
                raise Refusal("locked wheel source inventory is unavailable")
            record_digest_by_path = {
                str(item["path"]): str(item["sha256"]) for item in file_records
            }
            for wheel_path, wheel_digest in wheel_files.items():
                if wheel_path.endswith(".dist-info/RECORD") or ".data/" in wheel_path:
                    continue
                installed_path = site_relative + "/" + wheel_path
                if record_digest_by_path.get(installed_path) != wheel_digest:
                    raise Refusal(
                        f"installed bytes differ from locked wheel source for {name}: "
                        f"{wheel_path}"
                    )
        direct_path = metadata_file.parent / "direct_url.json"
        direct_origin: object = None
        if direct_path.exists():
            relative, _, direct_sha256 = _safe_runtime_file(direct_path, physical_root)
            if relative not in recorded_paths:
                raise Refusal(f"RECORD omits direct_url.json for {name}")
            raw = direct_path.read_bytes()
            try:
                parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
            except (UnicodeDecodeError, ValueError) as exc:
                raise Refusal(f"malformed direct_url.json for {name}") from exc
            if not isinstance(parsed, dict):
                raise Refusal(f"direct_url.json is not an object for {name}")
            dir_info = parsed.get("dir_info")
            if isinstance(dir_info, dict) and dir_info.get("editable") is True:
                raise Refusal(f"editable direct_url.json is forbidden for {name}")
            if seed_mode and isinstance(parsed.get("url"), str) and parsed[
                "url"
            ].lower().startswith("file:"):
                raise Refusal(f"local direct_url.json is forbidden for {name}")
            direct_origin = {
                "canonical_json": canonical(parsed).decode("ascii"),
                "path": relative,
                "sha256": direct_sha256,
            }
        metadata_relative = metadata_file.parent.resolve(strict=True).relative_to(
            physical_root.resolve(strict=True)
        ).as_posix()
        distributions.append(
            {
                "direct_origin": direct_origin,
                "files": file_records,
                "metadata_path": _relative_path(metadata_relative, "metadata path"),
                "name": name,
                "version": version,
            }
        )
    distributions.sort(key=lambda item: (item["name"], item["version"], item["metadata_path"]))
    if not distributions or len(distributions) > MAX_DISTRIBUTIONS:
        raise Refusal("installed distribution inventory count is invalid")
    if len(all_paths) > MAX_FILES:
        raise Refusal("installed distribution file inventory is oversized")
    installed = {item["name"]: item["version"] for item in distributions}
    if len(installed) != len(distributions):
        raise Refusal("duplicate normalized installed distribution names")
    expected = {name: value[0] for name, value in locked.items()}
    if installed != expected:
        missing = sorted(set(expected) - set(installed))
        extra = sorted(set(installed) - set(expected))
        mismatched = sorted(
            name for name in set(installed) & set(expected) if installed[name] != expected[name]
        )
        raise Refusal(
            "installed distributions differ from dependency lock "
            f"(missing={missing}, extra={extra}, version={mismatched})"
        )
    actual_site_files: set[str] = set()
    for current, directories, files in os.walk(site_packages, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            entry = current_path / directory
            info = os.lstat(entry)
            if (
                not stat.S_ISDIR(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or directory.endswith(".egg-info")
            ):
                raise Refusal(f"site-packages contains an unsafe directory: {entry}")
        for filename in files:
            entry = current_path / filename
            info = os.lstat(entry)
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_nlink != 1
                or filename.endswith(".egg-link")
            ):
                raise Refusal(f"site-packages contains an unsafe file: {entry}")
            actual_site_files.add(
                entry.resolve(strict=True)
                .relative_to(physical_root.resolve(strict=True))
                .as_posix()
            )
            if len(actual_site_files) > MAX_FILES:
                raise Refusal("site-packages file inventory is oversized")
    claimed_site_files = {
        path for path in all_paths if path.startswith(site_relative + "/")
    }
    if seed_mode:
        if (
            claimed_site_files - actual_site_files
            or actual_site_files - claimed_site_files
            != set(contract.B200_ALLOWED_UNCLAIMED_SITE_FILES)
        ):
            raise Refusal("binary-seed site-packages unclaimed file set differs")
    elif actual_site_files != claimed_site_files:
        raise Refusal("installed inventory is not complete for site-packages")
    result = {
        "distributions": distributions,
        "python_path": RUNTIME_PYTHON,
        "runtime_root": RUNTIME_ROOT,
        "schema": BINARY_SEED_INVENTORY_SCHEMA if seed_mode else INVENTORY_SCHEMA,
    }
    if runtime_files is not None:
        result["runtime_files"] = runtime_files
        result["runtime_directories"] = runtime_directories
    return result


def probe_documents(spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    python = dict(spec["python"])
    import_expected = {"imports": spec["required_imports"], "python": python}
    import_probe = {
        "argv": [
            RUNTIME_PYTHON,
            "-I",
            "-c",
            FIXED_IMPORT_SOURCE,
            canonical(import_expected).decode("ascii"),
        ],
        "expected_result": import_expected,
        "schema": IMPORT_PROBE_SCHEMA,
    }
    cuda = spec["cuda"]
    extension_results = [
        {"loaded": True, "module": item["module"]}
        for item in cuda["compiled_extensions"]
    ]
    cuda_expected = {
        "compiled_extensions": extension_results,
        "compute": {
            "bf16_matmul": [
                {"device_index": index, "passed": True}
                for index in range(cuda["device_count"])
            ],
        },
        "driver": {
            "compatible": True,
            "minimum_version": cuda["minimum_host_driver_version"],
        },
        "gpu": {
            "available": True,
            "compute_capabilities": [
                list(cuda["compute_capability"]) for _ in range(cuda["device_count"])
            ],
            "count": cuda["device_count"],
            "device_names": [
                cuda["device_name"] for _ in range(cuda["device_count"])
            ],
        },
        "python": python,
        "topology": {
            "compatible": True,
            "link_label": cuda["nvlink_link_label"],
            "nvlink_pairs": cuda["nvlink_pairs"],
        },
        "torch": {
            "cuda_version": cuda["torch_cuda_version"],
            "version": cuda["torch_version"],
        },
    }
    cuda_probe = {
        "argv": [
            RUNTIME_PYTHON,
            "-I",
            "-c",
            FIXED_CUDA_SOURCE,
            canonical(cuda_expected).decode("ascii"),
        ],
        "expected_result": cuda_expected,
        "schema": CUDA_PROBE_SCHEMA,
    }
    return import_probe, cuda_probe


def validate_seed_transformation(
    staging_root: Path,
    *,
    lock_payload: bytes,
    lock_document: Mapping[str, Any],
) -> tuple[bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    physical_root = _physical(staging_root, RUNTIME_ROOT)
    receipt_path = physical_root / SEED_TRANSFORMATION_NAME
    payload, receipt = _read_canonical(receipt_path, maximum=64 * 1024 * 1024)
    _exact_keys(receipt, contract.SEED_TRANSFORMATION_KEYS, "seed transformation receipt")
    seed = lock_document.get("seed")
    base = lock_document.get("base_image")
    if not isinstance(seed, dict) or not isinstance(base, dict) or not isinstance(
        base.get("python"), dict
    ):
        raise Refusal("binary-seed lock identity is unavailable")
    if (
        receipt["schema"] != SEED_TRANSFORMATION_SCHEMA
        or receipt["dependency_lock_sha256"] != digest_bytes(lock_payload)
        or receipt["seed_sha256"] != seed.get("sha256")
        or receipt["uncompressed_seed_sha256"]
        != seed.get("uncompressed_sha256")
        or receipt["base_python_sha256"] != base["python"].get("sha256")
    ):
        raise Refusal("seed transformation receipt identity differs")
    records = receipt["files"]
    if not isinstance(records, list) or not records or len(records) > MAX_FILES:
        raise Refusal("seed transformation file inventory is malformed")
    expected: dict[str, tuple[int, int, int, str]] = {}
    prior = ""
    for raw in records:
        if not isinstance(raw, dict):
            raise Refusal("seed transformation file record must be an object")
        _exact_keys(
            raw,
            contract.BINARY_SEED_RUNTIME_FILE_KEYS,
            "seed transformation file",
        )
        relative = _relative_path(raw["path"], "seed transformation file path")
        mode = raw["mode"]
        mtime_ns = raw["mtime_ns"]
        size = raw["size"]
        sha256 = raw["sha256"]
        if (
            relative <= prior
            or type(mode) is not int
            or mode not in (
                contract.RUNTIME_FILE_MODE,
                contract.RUNTIME_EXECUTABLE_MODE,
            )
            or type(mtime_ns) is not int
            or mtime_ns != contract.RUNTIME_MTIME_NS
            or type(size) is not int
            or size < 0
            or size > MAX_FILE_SIZE
            or not isinstance(sha256, str)
            or DIGEST_RE.fullmatch(sha256) is None
        ):
            raise Refusal("seed transformation file inventory is not canonical")
        expected[relative] = (mode, mtime_ns, size, sha256)
        prior = relative

    directory_records = receipt["directories"]
    if (
        not isinstance(directory_records, list)
        or not directory_records
        or len(directory_records) > MAX_FILES
    ):
        raise Refusal("seed transformation directory inventory is malformed")
    expected_directories: dict[str, tuple[int, int]] = {}
    prior = ""
    for raw in directory_records:
        if not isinstance(raw, dict):
            raise Refusal("seed transformation directory record must be an object")
        _exact_keys(
            raw,
            contract.BINARY_SEED_RUNTIME_DIRECTORY_KEYS,
            "seed transformation directory",
        )
        relative = _relative_path(raw["path"], "seed transformation directory path")
        mode = raw["mode"]
        mtime_ns = raw["mtime_ns"]
        if (
            relative <= prior
            or type(mode) is not int
            or mode != contract.RUNTIME_DIRECTORY_MODE
            or type(mtime_ns) is not int
            or mtime_ns != contract.RUNTIME_MTIME_NS
        ):
            raise Refusal("seed transformation directory inventory is not canonical")
        expected_directories[relative] = (mode, mtime_ns)
        prior = relative

    actual: dict[str, tuple[int, int, int, str]] = {}
    actual_directories: dict[str, tuple[int, int]] = {}
    for current, directories, files in os.walk(physical_root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            directory = current_path / name
            info = os.lstat(directory)
            relative = directory.relative_to(physical_root).as_posix()
            mode = stat.S_IMODE(info.st_mode)
            if (
                not stat.S_ISDIR(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or mode != contract.RUNTIME_DIRECTORY_MODE
                or info.st_mtime_ns != contract.RUNTIME_MTIME_NS
            ):
                raise Refusal(f"seed runtime contains an unsafe directory: {directory}")
            actual_directories[relative] = (mode, info.st_mtime_ns)
        for name in files:
            entry = current_path / name
            relative = entry.relative_to(physical_root).as_posix()
            info = os.lstat(entry)
            if relative == SEED_TRANSFORMATION_NAME:
                if (
                    not stat.S_ISREG(info.st_mode)
                    or stat.S_ISLNK(info.st_mode)
                    or info.st_nlink != 1
                    or stat.S_IMODE(info.st_mode) != contract.RUNTIME_FILE_MODE
                    or info.st_mtime_ns != contract.RUNTIME_MTIME_NS
                ):
                    raise Refusal("seed transformation receipt is unsafe")
                continue
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) not in (
                    contract.RUNTIME_FILE_MODE,
                    contract.RUNTIME_EXECUTABLE_MODE,
                )
                or info.st_mtime_ns != contract.RUNTIME_MTIME_NS
                or info.st_size > MAX_FILE_SIZE
            ):
                raise Refusal(f"seed runtime contains an unsafe file: {entry}")
            actual[relative] = (
                stat.S_IMODE(info.st_mode),
                info.st_mtime_ns,
                info.st_size,
                digest_file(entry),
            )
            if len(actual) > MAX_FILES:
                raise Refusal("seed runtime contains too many files")
    if actual != expected or actual_directories != expected_directories:
        raise Refusal("seed runtime tree differs from its transformation receipt")
    return payload, records, directory_records


def _publish(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short metadata write")
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, mode)
        os.utime(fd, ns=(contract.RUNTIME_MTIME_NS, contract.RUNTIME_MTIME_NS))
    finally:
        os.close(fd)


def generate(
    staging_root: Path,
    lock_path: Path,
    spec_path: Path,
    *,
    seed_path: Path | None = None,
    uncompressed_seed_path: Path | None = None,
) -> dict[str, Any]:
    lock_payload, locked, locked_python_version = validate_lock(
        lock_path,
        seed_path=seed_path,
        uncompressed_seed_path=uncompressed_seed_path,
    )
    lock_document = json.loads(lock_payload.decode("ascii"))
    seed_mode = lock_document.get("schema") == BINARY_SEED_LOCK_SCHEMA
    spec = validate_spec(spec_path, locked, locked_python_version)
    physical_root = _physical(staging_root, RUNTIME_ROOT)
    python_path = _physical(staging_root, RUNTIME_PYTHON)
    python_info = os.lstat(python_path)
    if (
        not stat.S_ISREG(python_info.st_mode)
        or stat.S_ISLNK(python_info.st_mode)
        or python_info.st_nlink != 1
        or stat.S_IMODE(python_info.st_mode) != 0o555
        or python_path.resolve(strict=True) != python_path.absolute()
    ):
        raise Refusal("runtime Python must be a canonical mode-0555 single-link file")
    seed_receipt_payload: bytes | None = None
    runtime_files: list[dict[str, Any]] | None = None
    runtime_directories: list[dict[str, Any]] | None = None
    if seed_mode:
        seed_receipt_payload, runtime_files, runtime_directories = validate_seed_transformation(
            staging_root, lock_payload=lock_payload, lock_document=lock_document
        )
        python_sha256 = digest_file(python_path)
        base_python_sha256 = lock_document["base_image"]["python"]["sha256"]
        runtime_python_relative = python_path.relative_to(physical_root).as_posix()
        python_records = [
            item for item in runtime_files if item["path"] == runtime_python_relative
        ]
        if (
            python_sha256 != base_python_sha256
            or len(python_records) != 1
            or python_records[0]["sha256"] != base_python_sha256
            or python_records[0]["mode"] != contract.RUNTIME_EXECUTABLE_MODE
        ):
            raise Refusal("runtime Python bytes differ from the pinned base image")
    installed = inventory(
        staging_root,
        spec,
        locked,
        runtime_files=runtime_files,
        runtime_directories=runtime_directories,
    )
    import_probe, cuda_probe = probe_documents(spec)
    members = {
        "dependency.lock": lock_payload,
        "installed-distributions.json": canonical(installed),
        "required-import-probe.json": canonical(import_probe),
        "cuda-compatibility-probe.json": canonical(cuda_probe),
    }
    for name in members:
        if os.path.lexists(physical_root / name):
            raise Refusal(f"runtime evidence destination already exists: {name}")
    for name, payload in members.items():
        _publish(physical_root / name, payload)
    unsigned_manifest = {
        "cuda_compatibility_probe_sha256": digest_bytes(
            members["cuda-compatibility-probe.json"]
        ),
        "dependency_lock_sha256": digest_bytes(members["dependency.lock"]),
        "installed_distributions_sha256": digest_bytes(
            members["installed-distributions.json"]
        ),
        "python": {
            "path": RUNTIME_PYTHON,
            "realpath": RUNTIME_PYTHON,
            "sha256": digest_file(python_path),
            "version": spec["python"]["version"],
        },
        "required_import_probe_sha256": digest_bytes(
            members["required-import-probe.json"]
        ),
        "runtime_root": RUNTIME_ROOT,
        "schema": BINARY_SEED_MANIFEST_SCHEMA if seed_mode else MANIFEST_SCHEMA,
    }
    if seed_receipt_payload is not None:
        unsigned_manifest["seed_transformation_sha256"] = digest_bytes(
            seed_receipt_payload
        )
    manifest = dict(unsigned_manifest)
    manifest["runtime_manifest_sha256"] = digest_bytes(canonical(unsigned_manifest))
    _publish(physical_root / "runtime-manifest.json", canonical(manifest))
    if seed_mode:
        os.chmod(physical_root, contract.RUNTIME_DIRECTORY_MODE)
        os.utime(
            physical_root,
            ns=(contract.RUNTIME_MTIME_NS, contract.RUNTIME_MTIME_NS),
            follow_symlinks=False,
        )
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument("--build-spec", required=True, type=Path)
    parser.add_argument("--binary-seed", type=Path)
    parser.add_argument("--uncompressed-seed", type=Path)
    parser.add_argument("--staging-root", type=Path, default=Path("/"))
    parser.add_argument("--check-inputs-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        _, locked, locked_python_version = validate_lock(
            args.dependency_lock,
            seed_path=args.binary_seed,
            uncompressed_seed_path=args.uncompressed_seed,
        )
        validate_spec(args.build_spec, locked, locked_python_version)
        if not args.check_inputs_only:
            generate(
                args.staging_root.resolve(strict=True),
                args.dependency_lock,
                args.build_spec,
                seed_path=args.binary_seed,
                uncompressed_seed_path=args.uncompressed_seed,
            )
    except (OSError, Refusal) as exc:
        print(f"generate_runtime_metadata: REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
