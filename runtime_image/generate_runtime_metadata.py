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
    if path.as_posix() != value or any(ord(char) < 33 or ord(char) > 126 for char in value):
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


def validate_lock(
    path: Path,
) -> tuple[bytes, dict[str, tuple[str, dict[str, object]]], str]:
    payload, document = _read_canonical(path, maximum=16 * 1024 * 1024)
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
    _exact_keys(
        document,
        contract.SPEC_KEYS,
        "build spec",
    )
    if document["schema"] != SPEC_SCHEMA or document["runtime_root"] != RUNTIME_ROOT:
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
        contract.CUDA_SPEC_KEYS,
        "cuda build spec",
    )
    torch_distribution = _name(cuda["torch_distribution"], "torch distribution")
    torch_version = _version(cuda["torch_version"], "torch version")
    torch_cuda_version = _version(cuda["torch_cuda_version"], "torch CUDA version")
    if torch_distribution not in locked or locked[torch_distribution][0] != torch_version:
        raise Refusal("CUDA torch identity does not match the dependency lock")
    device_count = cuda["device_count"]
    if isinstance(device_count, bool) or device_count != contract.EXACT_DEVICE_COUNT:
        raise Refusal("D043 H100 runtime requires exact device_count 1")
    capability = cuda["compute_capability"]
    if capability != contract.EXACT_COMPUTE_CAPABILITY:
        raise Refusal("D043 H100 runtime requires exact compute capability [9,0]")
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
    if document["python"]["version"] != python_version or not torch_cuda_version:
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
) -> dict[str, Any]:
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
            if relative in all_paths:
                raise Refusal(f"installed file is claimed more than once: {relative}")
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
    if actual_site_files != claimed_site_files:
        raise Refusal("installed inventory is not complete for site-packages")
    return {
        "distributions": distributions,
        "python_path": RUNTIME_PYTHON,
        "runtime_root": RUNTIME_ROOT,
        "schema": INVENTORY_SCHEMA,
    }


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
        "gpu": {
            "available": True,
            "compute_capabilities": [
                list(cuda["compute_capability"]) for _ in range(cuda["device_count"])
            ],
            "count": cuda["device_count"],
        },
        "python": python,
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


def _publish(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short metadata write")
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, mode)
    finally:
        os.close(fd)


def generate(staging_root: Path, lock_path: Path, spec_path: Path) -> dict[str, Any]:
    lock_payload, locked, locked_python_version = validate_lock(lock_path)
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
    installed = inventory(staging_root, spec, locked)
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
        "schema": MANIFEST_SCHEMA,
    }
    manifest = dict(unsigned_manifest)
    manifest["runtime_manifest_sha256"] = digest_bytes(canonical(unsigned_manifest))
    _publish(physical_root / "runtime-manifest.json", canonical(manifest))
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument("--build-spec", required=True, type=Path)
    parser.add_argument("--staging-root", type=Path, default=Path("/"))
    parser.add_argument("--check-inputs-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        _, locked, locked_python_version = validate_lock(args.dependency_lock)
        validate_spec(args.build_spec, locked, locked_python_version)
        if not args.check_inputs_only:
            generate(args.staging_root.resolve(strict=True), args.dependency_lock, args.build_spec)
    except (OSError, Refusal) as exc:
        print(f"generate_runtime_metadata: REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
