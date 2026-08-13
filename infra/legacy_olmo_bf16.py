"""Audit and validate the pre-marker OLMo-3.1-32B BF16 artifact.

This module deliberately does not infer how the legacy artifact was converted.
It establishes a narrower claim: every observed file is hashed, every tensor
header matches the immutable upstream model layout after an F32-to-BF16 width
change, and the resulting snapshot is committed under a distinct path.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from infra.olmo31_32b_legacy_bf16_reference import (
    REFERENCE_JSON_SHA256,
    load_reference,
)

LEGACY_SOURCE_PATH = "/workspace/models/olmo32-bf16"
AUDITED_MODEL_PATH = "/workspace/models/olmo32-bf16-legacy-audited-fc84a4f"
AUDIT_MARKER_NAME = ".legacy-bf16-audit.json"
AUDIT_SCHEMA_VERSION = 1
AUDITOR_ID = "olmo-legacy-bf16-byte-and-layout-audit-v1"
UNKNOWN_CONVERSION_PROVENANCE = "unknown; this manifest does not claim a converter"
LEGACY_CONFIG_SEMANTIC_SHA256 = (
    "4d405ff6526c7ace6a8c9d7c73b319944aa2647f14d97e06b5ac4927d7b89c37"
)
LEGACY_CONFIG_FILE_SHA256 = (
    "11741aebc7a2197d1d2d8af09f7f3b5e2e16413ba6afb98ba2304110ea22e8d6"
)
CONFIG_NORMALIZATION = "set dtype and torch_dtype to bfloat16; sort JSON keys"
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
LEGACY_REQUIRED_SUPPORT_FILES = (
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class LegacyArtifactError(RuntimeError):
    """The legacy artifact cannot be safely audited or validated."""


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LegacyArtifactError(f"artifact member is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after or size != after.st_size:
            raise LegacyArtifactError(f"artifact member changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def _unique_object(data: bytes, *, label: str) -> dict:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        value: dict = {}
        for key, item in pairs:
            if key in value:
                raise LegacyArtifactError(f"{label} has duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(data, object_pairs_hook=reject_duplicates)
    except LegacyArtifactError:
        raise
    except Exception as exc:
        raise LegacyArtifactError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise LegacyArtifactError(f"{label} must contain a JSON object")
    return value


def _read_json(path: Path, *, label: str) -> dict:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LegacyArtifactError(f"cannot read {label} {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LegacyArtifactError(f"{label} is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read()
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or len(data) != after.st_size:
            raise LegacyArtifactError(f"{label} changed while reading: {path}")
    finally:
        os.close(descriptor)
    return _unique_object(data, label=label)


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_source_config(path: Path) -> dict:
    config = _read_json(path, label="legacy model config")
    if config.get("architectures") != ["Olmo3ForCausalLM"]:
        raise LegacyArtifactError("legacy config does not select Olmo3ForCausalLM")
    if config.get("dtype") != "float32" or config.get("torch_dtype") != "bfloat16":
        raise LegacyArtifactError(
            "legacy config must preserve dtype=float32 and declare torch_dtype=bfloat16"
        )
    upstream_semantics = dict(config)
    upstream_semantics.pop("torch_dtype")
    if _canonical_json_sha256(upstream_semantics) != LEGACY_CONFIG_SEMANTIC_SHA256:
        raise LegacyArtifactError(
            "legacy config differs semantically from the pinned upstream revision"
        )
    digest, _ = _sha256_file(path)
    if digest != LEGACY_CONFIG_FILE_SHA256:
        raise LegacyArtifactError("legacy config bytes differ from the audited source")
    return config


def _normalized_config_bytes(source_path: Path, reference: dict) -> bytes:
    config = _validate_source_config(source_path)
    config["dtype"] = "bfloat16"
    config["torch_dtype"] = "bfloat16"
    payload = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode()
    if hashlib.sha256(payload).hexdigest() != reference["output_config_sha256"]:
        raise LegacyArtifactError(
            "normalized runtime config differs from the pinned output"
        )
    return payload


def _validate_runtime_config(path: Path, reference: dict) -> None:
    config = _read_json(path, label="audited runtime model config")
    if (
        config.get("architectures") != ["Olmo3ForCausalLM"]
        or config.get("dtype") != "bfloat16"
        or config.get("torch_dtype") != "bfloat16"
    ):
        raise LegacyArtifactError("audited runtime config does not select OLMo3 BF16")
    digest, _ = _sha256_file(path)
    if digest != reference["output_config_sha256"]:
        raise LegacyArtifactError(
            "audited runtime config failed its pinned byte identity"
        )


def _write_owned_file(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise LegacyArtifactError(
            f"cannot write derived audit member {path}: {exc}"
        ) from exc


def _expected_weight_map(reference: dict) -> dict[str, str]:
    return {
        key: shard
        for shard, record in reference["shards"].items()
        for key in record["tensors"]
    }


def _validate_index(path: Path, reference: dict) -> None:
    index = _read_json(path, label="legacy safetensors index")
    expected_metadata = {"total_size": reference["expected_bf16_tensor_bytes"]}
    if index.get("metadata") != expected_metadata:
        raise LegacyArtifactError(
            "legacy index metadata differs from the pinned BF16 aggregate"
        )
    if index.get("weight_map") != _expected_weight_map(reference):
        raise LegacyArtifactError(
            "legacy index weight_map differs from the pinned upstream revision"
        )
    if set(index) != {"metadata", "weight_map"}:
        raise LegacyArtifactError("legacy index contains unexpected top-level fields")


def _read_exact(handle: BinaryIO, size: int, *, label: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise LegacyArtifactError(f"truncated {label}")
    return data


def _inspect_and_hash_shard(path: Path, expected: dict) -> dict:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LegacyArtifactError(
                f"safetensors shard is not a regular file: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            prefix = _read_exact(handle, 8, label=f"safetensors prefix in {path.name}")
            digest.update(prefix)
            header_bytes = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_bytes <= MAX_SAFETENSORS_HEADER_BYTES:
                raise LegacyArtifactError(
                    f"unsafe safetensors header length in {path.name}: {header_bytes}"
                )
            header_payload = _read_exact(
                handle, header_bytes, label=f"safetensors header in {path.name}"
            )
            digest.update(header_payload)
            header = _unique_object(header_payload, label=f"header for {path.name}")
            metadata = header.pop("__metadata__", {})
            if not isinstance(metadata, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in metadata.items()
            ):
                raise LegacyArtifactError(
                    f"invalid safetensors metadata in {path.name}"
                )
            if set(header) != set(expected["tensors"]):
                missing = sorted(set(expected["tensors"]) - set(header))
                extra = sorted(set(header) - set(expected["tensors"]))
                raise LegacyArtifactError(
                    f"tensor-name mismatch in {path.name}; "
                    f"missing={missing[:3]}, extra={extra[:3]}"
                )
            tensor_bytes = 0
            for key, expected_tensor in expected["tensors"].items():
                observed = header[key]
                expected_record = {
                    "dtype": "BF16",
                    "shape": expected_tensor["shape"],
                    "data_offsets": expected_tensor["data_offsets"],
                }
                if observed != expected_record:
                    raise LegacyArtifactError(
                        f"dtype/shape/offset mismatch for {key!r} in {path.name}"
                    )
                tensor_bytes = max(tensor_bytes, expected_tensor["data_offsets"][1])
            expected_file_bytes = 8 + header_bytes + tensor_bytes
            if before.st_size != expected_file_bytes:
                raise LegacyArtifactError(
                    f"unexpected file/data boundary in {path.name}: "
                    f"file={before.st_size}, expected={expected_file_bytes}"
                )
            scanned = 8 + header_bytes
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
                scanned += len(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after or scanned != after.st_size:
            raise LegacyArtifactError(
                f"safetensors shard changed while auditing: {path}"
            )
    finally:
        os.close(descriptor)
    return {
        "sha256": digest.hexdigest(),
        "file_bytes": scanned,
        "header_bytes": header_bytes,
        "header_sha256": hashlib.sha256(header_payload).hexdigest(),
        "tensor_bytes": tensor_bytes,
        "tensor_count": len(expected["tensors"]),
        "metadata": metadata,
    }


def _safe_root(path: Path, *, label: str, must_exist: bool) -> Path:
    absolute = Path(os.path.abspath(path))
    if must_exist:
        if (
            absolute.is_symlink()
            or not absolute.is_dir()
            or absolute.resolve() != absolute
        ):
            raise LegacyArtifactError(
                f"{label} must be an exact non-symlink directory: {absolute}"
            )
    elif absolute.exists() or absolute.is_symlink():
        raise LegacyArtifactError(
            f"{label} already exists; refusing to overwrite: {absolute}"
        )
    return absolute


def _materialize(source: Path, destination: Path, mode: str) -> str:
    if source.is_symlink() or not source.is_file():
        raise LegacyArtifactError(f"source member is missing or unsafe: {source}")
    if destination.exists() or destination.is_symlink():
        raise LegacyArtifactError(f"destination member already exists: {destination}")
    if mode in {"hardlink", "auto"}:
        try:
            os.link(source, destination, follow_symlinks=False)
            source_stat = source.stat(follow_symlinks=False)
            destination_stat = destination.stat(follow_symlinks=False)
            if (source_stat.st_dev, source_stat.st_ino) != (
                destination_stat.st_dev,
                destination_stat.st_ino,
            ):
                raise LegacyArtifactError(
                    f"hardlink identity check failed for {source.name}"
                )
            return "hardlink"
        except OSError as exc:
            fallback_errors = {
                errno.EXDEV,
                errno.EPERM,
                errno.EACCES,
                getattr(errno, "ENOTSUP", errno.EPERM),
                getattr(errno, "EOPNOTSUPP", errno.EPERM),
            }
            if mode == "hardlink" or exc.errno not in fallback_errors:
                raise LegacyArtifactError(f"cannot hardlink {source}: {exc}") from exc
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
    except Exception as exc:
        raise LegacyArtifactError(f"cannot copy {source}: {exc}") from exc
    return "copy"


def _expected_files(reference: dict) -> list[str]:
    return [
        "config.json",
        "model.safetensors.index.json",
        *LEGACY_REQUIRED_SUPPORT_FILES,
        *sorted(reference["shards"]),
    ]


def audit_legacy_artifact(
    source_dir: Path,
    output_dir: Path,
    *,
    link_mode: str = "hardlink",
    progress=print,
) -> dict:
    """Create and audit a new snapshot without writing the legacy source path."""

    if link_mode not in {"hardlink", "copy", "auto"}:
        raise LegacyArtifactError("link_mode must be hardlink, copy, or auto")
    source = _safe_root(source_dir, label="legacy source", must_exist=True)
    output = _safe_root(output_dir, label="audited output", must_exist=False)
    if source == output or source in output.parents or output in source.parents:
        raise LegacyArtifactError(
            "source and audited output must be distinct sibling trees"
        )
    reference = load_reference()
    if _canonical_json_sha256(reference) != REFERENCE_JSON_SHA256:
        raise LegacyArtifactError(
            "embedded upstream layout reference failed integrity check"
        )

    staging = output.with_name(f".{output.name}.audit-{os.getpid()}.partial")
    if staging.exists() or staging.is_symlink():
        raise LegacyArtifactError(f"audit staging path already exists: {staging}")
    staging.mkdir(mode=0o755)
    try:
        files: dict[str, dict] = {}
        materializations: set[str] = set()
        names = _expected_files(reference)
        progress(f"auditing {len(reference['shards'])} BF16 shards into {output}")
        for position, name in enumerate(names, 1):
            source_path = source / name
            destination = staging / name
            if name == "config.json":
                _write_owned_file(
                    destination, _normalized_config_bytes(source_path, reference)
                )
            else:
                used = _materialize(source_path, destination, link_mode)
                materializations.add(used)
            if name == "model.safetensors.index.json":
                _validate_index(destination, reference)
            if name in reference["shards"]:
                record = _inspect_and_hash_shard(destination, reference["shards"][name])
                record["kind"] = "safetensors"
                files[name] = record
                progress(
                    f"[{position}/{len(names)}] {name}: {record['file_bytes']} bytes, "
                    f"sha256={record['sha256']}"
                )
            else:
                digest, size = _sha256_file(destination)
                if name == "config.json":
                    files[name] = {
                        "kind": "normalized-config",
                        "sha256": digest,
                        "file_bytes": size,
                        "source_sha256": LEGACY_CONFIG_FILE_SHA256,
                        "normalization": CONFIG_NORMALIZATION,
                    }
                    continue
                if (
                    name not in {"config.json", "model.safetensors.index.json"}
                    and digest != reference["support_sha256"][name]
                ):
                    raise LegacyArtifactError(
                        f"support file differs from pinned upstream revision: {name}"
                    )
                files[name] = {
                    "kind": "support",
                    "sha256": digest,
                    "file_bytes": size,
                }

        shard_records = [files[name] for name in sorted(reference["shards"])]
        aggregate = {
            "weight_file_bytes": sum(record["file_bytes"] for record in shard_records),
            "tensor_bytes": sum(record["tensor_bytes"] for record in shard_records),
            "tensor_count": sum(record["tensor_count"] for record in shard_records),
        }
        expected_aggregate = {
            "weight_file_bytes": reference["expected_weight_file_bytes"],
            "tensor_bytes": reference["expected_bf16_tensor_bytes"],
            "tensor_count": sum(
                len(record["tensors"]) for record in reference["shards"].values()
            ),
        }
        if aggregate != expected_aggregate:
            raise LegacyArtifactError(
                f"legacy BF16 aggregate differs from pinned layout: {aggregate!r}"
            )
        manifest = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "auditor": AUDITOR_ID,
            "complete": True,
            "artifact_class": "observed-legacy-bf16-snapshot",
            "conversion_provenance": UNKNOWN_CONVERSION_PROVENANCE,
            "audited_at": datetime.now(UTC).isoformat(),
            "source": {
                "path": str(source),
                "repo": reference["repo"],
                "revision": reference["revision"],
                "reference_json_sha256": REFERENCE_JSON_SHA256,
            },
            "materialization": sorted(materializations),
            "files": files,
            **aggregate,
        }
        marker = staging / AUDIT_MARKER_NAME
        marker_payload = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        with marker.open("xb") as handle:
            handle.write(marker_payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        # This process created this exact PID-scoped tree and has not committed
        # it. Removing it drops only our partial links/copies, never source data.
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    progress(f"committed audited legacy BF16 snapshot: {output}")
    return manifest


def validate_audited_artifact(root: Path, *, progress=None) -> dict:
    """Re-hash and structurally validate a committed audited snapshot."""

    artifact = _safe_root(root, label="audited legacy artifact", must_exist=True)
    reference = load_reference()
    if _canonical_json_sha256(reference) != REFERENCE_JSON_SHA256:
        raise LegacyArtifactError(
            "embedded upstream layout reference failed integrity check"
        )
    manifest = _read_json(
        artifact / AUDIT_MARKER_NAME, label="legacy BF16 audit marker"
    )
    source = manifest.get("source")
    expected_top_level = {
        "schema_version",
        "auditor",
        "complete",
        "artifact_class",
        "conversion_provenance",
        "audited_at",
        "source",
        "materialization",
        "files",
        "weight_file_bytes",
        "tensor_bytes",
        "tensor_count",
    }
    audited_at = manifest.get("audited_at")
    try:
        parsed_audited_at = (
            datetime.fromisoformat(audited_at) if isinstance(audited_at, str) else None
        )
    except ValueError:
        parsed_audited_at = None
    if (
        set(manifest) != expected_top_level
        or manifest.get("schema_version") != AUDIT_SCHEMA_VERSION
        or manifest.get("auditor") != AUDITOR_ID
        or manifest.get("complete") is not True
        or manifest.get("artifact_class") != "observed-legacy-bf16-snapshot"
        or manifest.get("conversion_provenance") != UNKNOWN_CONVERSION_PROVENANCE
        or not isinstance(source, dict)
        or set(source) != {"path", "repo", "revision", "reference_json_sha256"}
        or source.get("path") != LEGACY_SOURCE_PATH
        or source.get("repo") != reference["repo"]
        or source.get("revision") != reference["revision"]
        or source.get("reference_json_sha256") != REFERENCE_JSON_SHA256
        or parsed_audited_at is None
        or parsed_audited_at.tzinfo is None
        or manifest.get("materialization")
        not in (["hardlink"], ["copy"], ["copy", "hardlink"])
    ):
        raise LegacyArtifactError("legacy BF16 audit marker identity is invalid")
    expected_names = set(_expected_files(reference))
    children = list(artifact.iterdir())
    observed_names = {
        path.name for path in children if path.is_file() and not path.is_symlink()
    }
    if observed_names != expected_names | {AUDIT_MARKER_NAME}:
        raise LegacyArtifactError(
            "audited legacy artifact file set is incomplete or unexpected"
        )
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise LegacyArtifactError("audited legacy artifact contains an unsafe member")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != expected_names:
        raise LegacyArtifactError("legacy BF16 audit marker file set is inconsistent")
    _validate_runtime_config(artifact / "config.json", reference)
    _validate_index(artifact / "model.safetensors.index.json", reference)
    weight_file_bytes = tensor_bytes = tensor_count = 0
    for position, name in enumerate(sorted(expected_names), 1):
        recorded = files[name]
        if not isinstance(recorded, dict):
            raise LegacyArtifactError(f"invalid audit record for {name}")
        path = artifact / name
        if name == "config.json":
            digest, size = _sha256_file(path)
            actual = {
                "kind": "normalized-config",
                "sha256": digest,
                "file_bytes": size,
                "source_sha256": LEGACY_CONFIG_FILE_SHA256,
                "normalization": CONFIG_NORMALIZATION,
            }
        elif name in reference["shards"]:
            actual = _inspect_and_hash_shard(path, reference["shards"][name])
            actual["kind"] = "safetensors"
            weight_file_bytes += actual["file_bytes"]
            tensor_bytes += actual["tensor_bytes"]
            tensor_count += actual["tensor_count"]
        else:
            digest, size = _sha256_file(path)
            actual = {"kind": "support", "sha256": digest, "file_bytes": size}
        if actual != recorded:
            raise LegacyArtifactError(f"audited bytes/record differ for {name}")
        if progress is not None:
            progress(f"[{position}/{len(expected_names)}] verified {name}")
    if (
        manifest.get("weight_file_bytes") != weight_file_bytes
        or manifest.get("tensor_bytes") != tensor_bytes
        or manifest.get("tensor_count") != tensor_count
        or weight_file_bytes != reference["expected_weight_file_bytes"]
        or tensor_bytes != reference["expected_bf16_tensor_bytes"]
        or tensor_count
        != sum(len(record["tensors"]) for record in reference["shards"].values())
    ):
        raise LegacyArtifactError("legacy BF16 audited aggregate is inconsistent")
    return manifest
