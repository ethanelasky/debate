#!/usr/bin/env python3
"""Convert a pinned Hugging Face safetensors model to a BF16 artifact.

The converter deliberately stages at most one weight shard at a time.  It is
safe to rerun after an interruption: completed output shards carry conversion
provenance in their safetensors metadata and are validated before being
skipped.  The output and staging directories are owned by marker files; a
non-empty unmarked directory, an identity mismatch, or an unexpected file is
rejected rather than overwritten.

Remote usage (``--repo`` and ``--revision`` are always explicit)::

    HF_TOKEN=... python scripts/convert_hf_safetensors_bf16.py \
      --repo allenai/Olmo-3.1-32B-Instruct-DPO \
      --revision <40-character-commit> \
      --output-dir /workspace/models/olmo32-bf16 \
      --staging-dir /workspace/model-conversion-staging \
      --expected-size-gb-min 55 --expected-size-gb-max 75

``--source-dir`` provides an offline/local source with the same layout as a
Hugging Face model repository.  It exists both for tests and for converting a
previously mirrored, pinned artifact.  Authentication is read only from the
``HF_TOKEN`` environment variable; there is intentionally no token CLI flag.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, Protocol


INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
OUTPUT_MARKER_NAME = ".bf16-conversion.json"
STAGING_MARKER_NAME = ".bf16-conversion-staging.json"
LOCK_SUFFIX = ".bf16-conversion.lock"
MARKER_SCHEMA_VERSION = 1
CONVERTER_ID = "hf-safetensors-floating-to-bfloat16-v1"
PROVENANCE_PREFIX = "bf16_conversion."
HF_CANONICAL_ENDPOINT = "https://huggingface.co"

# These are the files needed to reconstruct the tokenizer/model interface.
# A repository may contain only the subset appropriate to its tokenizer.
SUPPORT_FILE_NAMES = (
    CONFIG_NAME,
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)
TOKENIZER_PAYLOADS = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
)
_EXACT_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}")
_REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class ConversionError(RuntimeError):
    """The conversion could not be performed without risking a mixed artifact."""


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class ConversionResult:
    output_dir: Path
    shards: int
    tensor_bytes: int
    weight_file_bytes: int


class ModelSource(Protocol):
    files: Mapping[str, RemoteFile]

    def read_bytes(self, name: str) -> bytes: ...

    def weight_identity(self, name: str) -> RemoteFile: ...

    def stage_file(self, name: str, destination: Path) -> None: ...


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ConversionError(f"invalid empty repository filename: {name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ConversionError(f"unsafe repository filename: {name!r}")
    normalized = pure.as_posix()
    if normalized != name or "\\" in name:
        raise ConversionError(f"non-canonical repository filename: {name!r}")
    return name


def _child(root: Path, name: str) -> Path:
    _safe_relative_name(name)
    return root.joinpath(*PurePosixPath(name).parts)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise ConversionError(f"refusing pre-existing atomic-write temporary: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # This temporary is uniquely named by this process and contains only
        # reproducible derived output.  Never touch any other pre-existing file.
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def _atomic_write_json(path: Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write_bytes(path, encoded)


def _load_json_bytes(data: bytes, *, label: str) -> dict:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"{label} must contain a JSON object")
    return value


def _load_json_file(path: Path, *, label: str) -> dict:
    try:
        return _load_json_bytes(path.read_bytes(), label=label)
    except OSError as exc:
        raise ConversionError(f"cannot read {label} at {path}: {exc}") from exc


def _validate_identity(repo: str, revision: str) -> None:
    if not _REPO_RE.fullmatch(repo):
        raise ConversionError(
            "--repo must be an exact Hugging Face model id in owner/name form"
        )
    if not _EXACT_REVISION_RE.fullmatch(revision):
        raise ConversionError(
            "--revision must be an exact 40-character hexadecimal commit id"
        )


def _assert_distinct_roots(paths: Mapping[str, Path]) -> None:
    resolved = {label: path.resolve(strict=False) for label, path in paths.items()}
    labels = list(resolved)
    for index, left_label in enumerate(labels):
        left = resolved[left_label]
        for right_label in labels[index + 1 :]:
            right = resolved[right_label]
            if left == right or left in right.parents or right in left.parents:
                raise ConversionError(
                    f"{left_label} and {right_label} must be distinct, non-nested directories: "
                    f"{left} vs {right}"
                )


def _iter_files(root: Path) -> Iterable[Path]:
    if root.is_symlink():
        raise ConversionError(f"directory may not be a symlink: {root}")
    if not root.exists():
        return ()
    if not root.is_dir():
        raise ConversionError(f"expected a directory: {root}")
    found: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ConversionError(f"symlinks are not accepted in conversion directories: {path}")
        if path.is_file():
            found.append(path)
    return found


class LocalModelSource:
    """A network-free source used for mirrors and tiny offline tests."""

    def __init__(self, root: Path):
        if root.is_symlink():
            raise ConversionError(f"local source directory may not be a symlink: {root}")
        self.root = root.resolve(strict=False)
        if not self.root.is_dir():
            raise ConversionError(f"local source directory is unavailable or unsafe: {root}")
        files: dict[str, RemoteFile] = {}
        for path in _iter_files(self.root):
            name = path.relative_to(self.root).as_posix()
            _safe_relative_name(name)
            files[name] = RemoteFile(name=name, size=path.stat().st_size)
        self.files = files

    def _path(self, name: str) -> Path:
        path = _child(self.root, name)
        if not path.is_file() or path.is_symlink():
            raise ConversionError(f"local source file is unavailable or unsafe: {name}")
        return path

    def read_bytes(self, name: str) -> bytes:
        return self._path(name).read_bytes()

    def weight_identity(self, name: str) -> RemoteFile:
        path = self._path(name)
        descriptor = RemoteFile(
            name=name,
            size=path.stat().st_size,
            sha256=_sha256_file(path),
        )
        self.files[name] = descriptor
        return descriptor

    def stage_file(self, name: str, destination: Path) -> None:
        source = self._path(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())


class _SameOriginAuthRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow signed CDN redirects without forwarding the Hugging Face token."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        old_origin = (req.type, req.host)
        new_origin = (redirected.type, redirected.host)
        if old_origin != new_origin:
            redirected.remove_header("Authorization")
        return redirected


class HubModelSource:
    """A pinned Hugging Face source with streaming, cache-free downloads."""

    def __init__(self, repo: str, revision: str):
        configured_endpoint = os.environ.get("HF_ENDPOINT")
        if (
            configured_endpoint
            and configured_endpoint.rstrip("/") != HF_CANONICAL_ENDPOINT
        ):
            raise ConversionError(
                "refusing non-canonical HF_ENDPOINT; remote conversion is pinned to "
                f"{HF_CANONICAL_ENDPOINT}"
            )
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - exercised on cold pods
            raise ConversionError("huggingface_hub is required for remote conversion") from exc

        self.repo = repo
        self.revision = revision
        # Authentication has exactly one ingress.  It is never accepted as an
        # argument and never included in progress or error messages.
        self._token = os.environ.get("HF_TOKEN") or None
        self._opener = urllib.request.build_opener(_SameOriginAuthRedirectHandler())
        try:
            info = HfApi(endpoint=HF_CANONICAL_ENDPOINT, token=False).model_info(
                repo_id=repo,
                revision=revision,
                token=self._token if self._token is not None else False,
                files_metadata=True,
            )
        except Exception as exc:
            raise ConversionError(
                f"could not inspect pinned Hugging Face repo {repo}@{revision}: "
                f"{type(exc).__name__}"
            ) from exc
        resolved_revision = str(getattr(info, "sha", "") or "")
        if resolved_revision.lower() != revision.lower():
            raise ConversionError(
                f"Hugging Face resolved {repo}@{revision} to unexpected commit "
                f"{resolved_revision or '<missing>'}"
            )

        files: dict[str, RemoteFile] = {}
        for sibling in info.siblings:
            name = _safe_relative_name(sibling.rfilename)
            lfs = getattr(sibling, "lfs", None)
            if isinstance(lfs, dict):
                digest = lfs.get("sha256")
                lfs_size = lfs.get("size")
            else:
                digest = getattr(lfs, "sha256", None)
                lfs_size = getattr(lfs, "size", None)
            size = getattr(sibling, "size", None) or lfs_size
            files[name] = RemoteFile(
                name=name,
                size=int(size) if size is not None else None,
                sha256=str(digest) if digest else None,
            )
        self.files = files

    def _open(self, name: str):
        try:
            from huggingface_hub import hf_hub_url
        except ImportError as exc:  # pragma: no cover - exercised on cold pods
            raise ConversionError("huggingface_hub is required for remote conversion") from exc
        url = hf_hub_url(
            self.repo,
            name,
            revision=self.revision,
            repo_type="model",
            endpoint=HF_CANONICAL_ENDPOINT,
        )
        if not url.startswith(f"{HF_CANONICAL_ENDPOINT}/"):
            raise ConversionError(f"huggingface_hub produced a non-canonical URL for {name}")
        headers = {"User-Agent": "debate-bf16-converter/1"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            return self._opener.open(request, timeout=120)
        except Exception as exc:
            raise ConversionError(
                f"could not download {name!r} from {self.repo}@{self.revision}: "
                f"{type(exc).__name__}"
            ) from exc

    def read_bytes(self, name: str) -> bytes:
        with self._open(name) as response:
            data = response.read()
        descriptor = self.files[name]
        if descriptor.size is not None and len(data) != descriptor.size:
            raise ConversionError(
                f"downloaded size mismatch for {name}: got {len(data)}, "
                f"expected {descriptor.size}"
            )
        if descriptor.sha256 is not None and _sha256_bytes(data) != descriptor.sha256:
            raise ConversionError(f"downloaded SHA-256 mismatch for {name}")
        return data

    def weight_identity(self, name: str) -> RemoteFile:
        descriptor = self.files[name]
        digest = descriptor.sha256
        if (
            descriptor.size is None
            or digest is None
            or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
        ):
            raise ConversionError(
                f"Hugging Face metadata has no verifiable size/SHA-256 identity for "
                f"weight shard {name!r}"
            )
        verified = RemoteFile(
            name=name,
            size=descriptor.size,
            sha256=digest.lower(),
        )
        self.files[name] = verified
        return verified

    def stage_file(self, name: str, destination: Path) -> None:
        descriptor = self.files[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with self._open(name) as response, destination.open("xb") as outgoing:
            while chunk := response.read(8 * 1024 * 1024):
                outgoing.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if descriptor.size is not None and size != descriptor.size:
            raise ConversionError(
                f"downloaded size mismatch for {name}: got {size}, expected {descriptor.size}"
            )
        if descriptor.sha256 is not None and digest.hexdigest() != descriptor.sha256:
            raise ConversionError(f"downloaded SHA-256 mismatch for {name}")


def _discover_plan(source: ModelSource, repo: str, revision: str) -> tuple[dict, dict[str, bytes]]:
    required = {INDEX_NAME, CONFIG_NAME, "tokenizer_config.json"}
    missing = sorted(required - source.files.keys())
    if missing:
        raise ConversionError(f"source is missing required model files: {', '.join(missing)}")
    if not any(name in source.files for name in TOKENIZER_PAYLOADS):
        raise ConversionError(
            "source has no tokenizer payload (expected one of "
            + ", ".join(TOKENIZER_PAYLOADS)
            + ")"
        )

    index_bytes = source.read_bytes(INDEX_NAME)
    index = _load_json_bytes(index_bytes, label=INDEX_NAME)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ConversionError(f"{INDEX_NAME} has no non-empty weight_map")
    if not all(isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()):
        raise ConversionError(f"{INDEX_NAME} weight_map must map string tensor names to files")
    shards = sorted({_safe_relative_name(name) for name in weight_map.values()})
    missing_shards = [name for name in shards if name not in source.files]
    if missing_shards:
        raise ConversionError(
            f"source index references unavailable shards: {', '.join(missing_shards[:5])}"
        )
    if any(not name.endswith(".safetensors") for name in shards):
        raise ConversionError("source index contains a non-safetensors weight shard")

    # Hash local shards sequentially without retaining them, while remote mode
    # uses the pinned repository's LFS SHA-256 metadata.  No second weight
    # shard is downloaded or staged to discover this identity.
    shard_identity: dict[str, dict[str, int | str]] = {}
    for name in shards:
        descriptor = source.weight_identity(name)
        if descriptor.size is None or descriptor.sha256 is None:
            raise ConversionError(f"source has no complete integrity identity for {name}")
        if descriptor.size <= 0 or re.fullmatch(r"[0-9a-f]{64}", descriptor.sha256) is None:
            raise ConversionError(f"source has an invalid integrity identity for {name}")
        shard_identity[name] = {
            "size": descriptor.size,
            "sha256": descriptor.sha256,
        }

    support_names = [name for name in SUPPORT_FILE_NAMES if name in source.files]
    support_bytes = {name: source.read_bytes(name) for name in support_names}
    # Parse now so an invalid config cannot leave a partially converted output.
    _load_json_bytes(support_bytes[CONFIG_NAME], label=CONFIG_NAME)
    immutable = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "converter": CONVERTER_ID,
        "source": {
            "repo": repo,
            "revision": revision.lower(),
            "index_sha256": _sha256_bytes(index_bytes),
            "support_sha256": {
                name: _sha256_bytes(data) for name, data in sorted(support_bytes.items())
            },
        },
        "weight_shards": shards,
        "weight_shard_identity": shard_identity,
        "support_files": support_names,
    }
    plan = dict(immutable)
    plan.update({"complete": False, "shards": {}})
    return plan, {INDEX_NAME: index_bytes, **support_bytes}


def _marker_identity(marker: Mapping[str, object]) -> dict:
    return {
        key: marker.get(key)
        for key in (
            "schema_version",
            "converter",
            "source",
            "weight_shards",
            "weight_shard_identity",
            "support_files",
        )
    }


def _recover_initial_marker_partial(
    root: Path,
    marker_name: str,
    files: list[Path],
    expected_identity: dict,
) -> dict | None:
    """Recover the sole exact temp left while establishing directory ownership."""

    if len(files) != 1 or files[0].parent != root:
        raise ConversionError(
            f"refusing non-empty unowned directory {root}; missing {marker_name}"
        )
    partial = files[0]
    match = re.fullmatch(
        rf"\.{re.escape(marker_name)}\.([0-9]+)\.partial",
        partial.name,
    )
    if match is None:
        raise ConversionError(
            f"refusing non-empty unowned directory {root}; missing {marker_name}"
        )
    try:
        data = partial.read_bytes()
    except OSError as exc:
        raise ConversionError(f"cannot inspect interrupted marker {partial}: {exc}") from exc
    try:
        marker = _load_json_bytes(data, label=str(partial))
    except ConversionError:
        # The stable adjacent lock proves there is no active conforming writer,
        # and the otherwise-empty directory plus exact reserved filename prove
        # this is a converter-owned incomplete marker, not arbitrary user data.
        partial.unlink()
        return None
    if _marker_identity(marker) != expected_identity:
        raise ConversionError(f"conversion identity mismatch in interrupted {partial}")
    marker_path = root / marker_name
    os.replace(partial, marker_path)
    directory_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return marker


def _inspect_directory(root: Path, marker_name: str, expected_identity: dict) -> dict | None:
    files = list(_iter_files(root))
    marker_path = root / marker_name
    if not files:
        return None
    if not marker_path.is_file() or marker_path.is_symlink():
        return _recover_initial_marker_partial(
            root,
            marker_name,
            files,
            expected_identity,
        )
    marker = _load_json_file(marker_path, label=str(marker_path))
    if _marker_identity(marker) != expected_identity:
        raise ConversionError(f"conversion identity mismatch in {marker_path}")
    return marker


def _relative_file_set(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in _iter_files(root)}


def _validate_directory_contents(
    output_dir: Path,
    staging_dir: Path,
    plan: Mapping[str, object],
) -> None:
    shards = set(plan["weight_shards"])
    support = set(plan["support_files"])
    output_allowed = shards | support | {INDEX_NAME, OUTPUT_MARKER_NAME}
    unexpected_output = sorted(_relative_file_set(output_dir) - output_allowed)
    if unexpected_output:
        raise ConversionError(
            f"unexpected files in owned output directory: {', '.join(unexpected_output[:5])}"
        )

    staging_files = _relative_file_set(staging_dir) - {STAGING_MARKER_NAME}
    unexpected_staging = sorted(staging_files - shards)
    if unexpected_staging:
        raise ConversionError(
            f"unexpected files in owned staging directory: {', '.join(unexpected_staging[:5])}"
        )
    # A normal remote conversion stages only one shard at a time.  Also admit
    # multiple expected shards in an already marker-owned staging directory so
    # operators can prefetch a pinned model efficiently.  Each file is still
    # checked against the immutable source size/SHA-256 immediately before it
    # is converted, and unexpected names continue to fail closed above.


@contextlib.contextmanager
def _exclusive_path_lock(root: Path):
    """Lock one exact output/staging path before inspecting or creating it."""

    resolved = root.resolve(strict=False)
    if not resolved.name:
        raise ConversionError(f"cannot lock filesystem root as a conversion directory: {root}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lock_path = resolved.parent / f".{resolved.name}{LOCK_SUFFIX}"
    if lock_path.is_symlink():
        raise ConversionError(f"conversion lock may not be a symlink: {lock_path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ConversionError(f"cannot open conversion lock {lock_path}: {exc}") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConversionError(
                f"another converter invocation owns conversion path {resolved}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextlib.contextmanager
def _exclusive_conversion_locks(output_dir: Path, staging_dir: Path):
    """Acquire stable adjacent locks in path order to cover both directories."""

    roots = sorted(
        {output_dir.resolve(strict=False), staging_dir.resolve(strict=False)},
        key=str,
    )
    with contextlib.ExitStack() as stack:
        for root in roots:
            stack.enter_context(_exclusive_path_lock(root))
        yield


def _clear_owned_partials(
    root: Path,
    *,
    logical_names: Iterable[str],
    download_partials: bool,
) -> list[str]:
    """Remove only exact converter partial names from a marker-owned directory.

    The caller holds stable locks for both conversion paths, so a matching
    partial cannot belong to a live conforming converter.  This is the
    hard-kill recovery path; arbitrary files and lookalike names remain
    fail-closed.
    """

    patterns: dict[tuple[str, ...], list[re.Pattern[str]]] = {}
    for logical_name in logical_names:
        pure = PurePosixPath(_safe_relative_name(logical_name))
        base = re.escape(pure.name)
        if download_partials:
            # The no-PID spelling was emitted by converter v1 before locking
            # was added; retain a narrow recovery path for those artifacts.
            suffix = rf"\.{base}\.(?:[0-9]+\.)?download\.partial"
        else:
            suffix = rf"\.{base}\.[0-9]+\.partial"
        patterns.setdefault(tuple(pure.parent.parts), []).append(
            re.compile(rf"^{suffix}$")
        )

    removed: list[str] = []
    for path in list(_iter_files(root)):
        relative = path.relative_to(root)
        parent_patterns = patterns.get(tuple(relative.parent.parts), [])
        if not any(pattern.fullmatch(relative.name) for pattern in parent_patterns):
            continue
        path.unlink()
        removed.append(relative.as_posix())
    return sorted(removed)


def _expected_keys_by_shard(index: Mapping[str, object]) -> dict[str, set[str]]:
    by_shard: dict[str, set[str]] = {}
    weight_map = index["weight_map"]
    assert isinstance(weight_map, dict)
    for tensor_name, shard_name in weight_map.items():
        assert isinstance(tensor_name, str) and isinstance(shard_name, str)
        by_shard.setdefault(shard_name, set()).add(tensor_name)
    return by_shard


def _provenance(
    repo: str,
    revision: str,
    index_sha256: str,
    shard: str,
    shard_identity: Mapping[str, int | str],
) -> dict[str, str]:
    return {
        f"{PROVENANCE_PREFIX}converter": CONVERTER_ID,
        f"{PROVENANCE_PREFIX}repo": repo,
        f"{PROVENANCE_PREFIX}revision": revision.lower(),
        f"{PROVENANCE_PREFIX}index_sha256": index_sha256,
        f"{PROVENANCE_PREFIX}source_shard": shard,
        f"{PROVENANCE_PREFIX}source_shard_size": str(shard_identity["size"]),
        f"{PROVENANCE_PREFIX}source_shard_sha256": str(shard_identity["sha256"]),
    }


def _validate_file_identity(
    path: Path,
    expected: Mapping[str, int | str],
    *,
    label: str,
) -> None:
    expected_size = expected.get("size")
    expected_sha256 = expected.get("sha256")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size <= 0
        or not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ConversionError(f"invalid immutable source identity for {label}")
    if path.stat().st_size != expected_size:
        raise ConversionError(f"{label} size differs from the immutable conversion plan")
    if _sha256_file(path) != expected_sha256:
        raise ConversionError(f"{label} SHA-256 differs from the immutable conversion plan")


def _shard_record(path: Path, tensor_bytes: int) -> dict[str, int | str]:
    return {
        "sha256": _sha256_file(path),
        "file_bytes": path.stat().st_size,
        "tensor_bytes": tensor_bytes,
    }


def _validate_recorded_shard(
    shard: str,
    recorded: object,
    actual: Mapping[str, int | str],
) -> None:
    if not isinstance(recorded, dict) or recorded != actual:
        raise ConversionError(
            f"recorded output integrity mismatch for {shard}; refusing to bless changed bytes"
        )


def _inspect_safetensors(
    path: Path,
    *,
    expected_keys: set[str],
    required_metadata: Mapping[str, str] | None = None,
    require_bfloat16: bool = False,
) -> tuple[int, dict[str, str]]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - exercised on cold pods
        raise ConversionError("torch and safetensors are required for conversion") from exc

    if not path.is_file() or path.is_symlink():
        raise ConversionError(f"safetensors shard is unavailable or unsafe: {path}")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != expected_keys:
                missing = sorted(expected_keys - keys)
                extra = sorted(keys - expected_keys)
                raise ConversionError(
                    f"tensor-name mismatch in {path.name}; missing={missing[:3]}, extra={extra[:3]}"
                )
            metadata = dict(handle.metadata() or {})
            if required_metadata is not None:
                for key, value in required_metadata.items():
                    if metadata.get(key) != value:
                        raise ConversionError(
                            f"conversion provenance mismatch for {path.name}: {key}"
                        )
            total = 0
            for key in sorted(keys):
                tensor = handle.get_tensor(key)
                if require_bfloat16 and tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
                    raise ConversionError(
                        f"floating tensor {key!r} in {path.name} has {tensor.dtype}, expected bfloat16"
                    )
                total += tensor.numel() * tensor.element_size()
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(f"invalid safetensors shard {path}: {type(exc).__name__}: {exc}") from exc
    return total, metadata


def _stage_source_shard(
    source: ModelSource,
    shard: str,
    staging_dir: Path,
    expected_keys: set[str],
    expected_identity: Mapping[str, int | str],
) -> Path:
    destination = _child(staging_dir, shard)
    if destination.exists() or destination.is_symlink():
        # A prior interrupted invocation may leave exactly its one owned shard.
        _inspect_safetensors(destination, expected_keys=expected_keys)
        _validate_file_identity(
            destination,
            expected_identity,
            label=f"staged shard {shard}",
        )
        return destination

    partial = destination.with_name(
        f".{destination.name}.{os.getpid()}.download.partial"
    )
    if partial.exists() or partial.is_symlink():
        raise ConversionError(f"refusing pre-existing staged partial download: {partial}")
    try:
        source.stage_file(shard, partial)
        _inspect_safetensors(partial, expected_keys=expected_keys)
        _validate_file_identity(
            partial,
            expected_identity,
            label=f"freshly staged shard {shard}",
        )
        os.replace(partial, destination)
    except BaseException:
        # Only this invocation's uniquely named derived download is removed.
        if partial.exists() and not partial.is_symlink():
            partial.unlink()
        raise
    return destination


def _convert_shard(
    source_path: Path,
    output_path: Path,
    *,
    expected_keys: set[str],
    provenance: Mapping[str, str],
) -> None:
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - exercised on cold pods
        raise ConversionError("torch and safetensors are required for conversion") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.partial")
    if output_path.exists() or output_path.is_symlink():
        raise ConversionError(f"refusing to overwrite output shard: {output_path}")
    if temporary.exists() or temporary.is_symlink():
        raise ConversionError(f"refusing pre-existing shard temporary: {temporary}")

    try:
        with safe_open(source_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != expected_keys:
                raise ConversionError(f"source tensor-name mismatch in {source_path.name}")
            source_metadata = dict(handle.metadata() or {})
            collisions = sorted(key for key in source_metadata if key.startswith(PROVENANCE_PREFIX))
            if collisions:
                raise ConversionError(
                    f"source metadata uses reserved conversion keys: {', '.join(collisions)}"
                )
            metadata = {**source_metadata, **provenance}
            converted = {}
            for key in sorted(keys):
                tensor = handle.get_tensor(key)
                if tensor.is_floating_point():
                    converted[key] = tensor.to(dtype=torch.bfloat16, copy=True).contiguous()
                else:
                    converted[key] = tensor.clone().contiguous()
            save_file(converted, temporary, metadata=metadata)
        _inspect_safetensors(
            temporary,
            expected_keys=expected_keys,
            required_metadata=provenance,
            require_bfloat16=True,
        )
        os.replace(temporary, output_path)
        _inspect_safetensors(
            output_path,
            expected_keys=expected_keys,
            required_metadata=provenance,
            require_bfloat16=True,
        )
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def _write_or_validate(path: Path, expected: bytes) -> None:
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink():
            raise ConversionError(f"output support path is unsafe: {path}")
        if _sha256_file(path) != _sha256_bytes(expected):
            raise ConversionError(f"refusing to overwrite mismatched output file: {path}")
        return
    _atomic_write_bytes(path, expected)


def _converted_config(source_bytes: bytes) -> bytes:
    config = _load_json_bytes(source_bytes, label=CONFIG_NAME)
    # ``torch_dtype`` is the serialized Transformers model dtype field.  Newer
    # configs may also carry its replacement ``dtype``; update it when present
    # without introducing an unrelated key into older model configs.
    config["torch_dtype"] = "bfloat16"
    if "dtype" in config:
        config["dtype"] = "bfloat16"
    return (json.dumps(config, indent=2, sort_keys=True) + "\n").encode()


def _converted_index(source_bytes: bytes, tensor_bytes: int) -> bytes:
    index = _load_json_bytes(source_bytes, label=INDEX_NAME)
    metadata = index.get("metadata")
    if metadata is None:
        metadata = {}
        index["metadata"] = metadata
    if not isinstance(metadata, dict):
        raise ConversionError(f"{INDEX_NAME} metadata must be a JSON object")
    metadata["total_size"] = tensor_bytes
    return (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()


def _validate_final(
    output_dir: Path,
    *,
    plan: Mapping[str, object],
    source_files: Mapping[str, bytes],
    expected_size_bytes: tuple[int, int] | None,
) -> tuple[int, int, dict[str, dict[str, int | str]]]:
    index_path = output_dir / INDEX_NAME
    index = _load_json_file(index_path, label=str(index_path))
    by_shard = _expected_keys_by_shard(index)
    expected_shards = set(plan["weight_shards"])
    if set(by_shard) != expected_shards:
        raise ConversionError("final index shard set differs from the pinned source index")

    source_identity = plan["source"]
    assert isinstance(source_identity, dict)
    weight_identity = plan["weight_shard_identity"]
    assert isinstance(weight_identity, dict)
    records: dict[str, dict[str, int | str]] = {}
    tensor_bytes = 0
    weight_file_bytes = 0
    for shard in sorted(expected_shards):
        path = _child(output_dir, shard)
        provenance = _provenance(
            str(source_identity["repo"]),
            str(source_identity["revision"]),
            str(source_identity["index_sha256"]),
            shard,
            weight_identity[shard],
        )
        shard_tensor_bytes, _ = _inspect_safetensors(
            path,
            expected_keys=by_shard[shard],
            required_metadata=provenance,
            require_bfloat16=True,
        )
        file_bytes = path.stat().st_size
        tensor_bytes += shard_tensor_bytes
        weight_file_bytes += file_bytes
        records[shard] = _shard_record(path, shard_tensor_bytes)

    metadata = index.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("total_size") != tensor_bytes:
        raise ConversionError(
            f"final index total_size does not match tensors: "
            f"{metadata.get('total_size') if isinstance(metadata, dict) else None} != {tensor_bytes}"
        )
    expected_index = _converted_index(source_files[INDEX_NAME], tensor_bytes)
    if index_path.read_bytes() != expected_index:
        raise ConversionError("final index differs from the deterministic converted index")

    config_path = output_dir / CONFIG_NAME
    if config_path.read_bytes() != _converted_config(source_files[CONFIG_NAME]):
        raise ConversionError("final config does not declare bfloat16 deterministically")
    for name in plan["support_files"]:
        if name == CONFIG_NAME:
            continue
        path = _child(output_dir, name)
        if path.read_bytes() != source_files[name]:
            raise ConversionError(f"final support file differs from pinned source: {name}")

    if expected_size_bytes is not None:
        minimum, maximum = expected_size_bytes
        if minimum < 0 or maximum < minimum:
            raise ConversionError(f"invalid expected byte range: {expected_size_bytes}")
        if not minimum <= weight_file_bytes <= maximum:
            raise ConversionError(
                f"converted weight files total {weight_file_bytes} bytes; expected "
                f"{minimum}..{maximum} bytes"
            )
    return tensor_bytes, weight_file_bytes, records


def convert_repository(
    *,
    repo: str,
    revision: str,
    output_dir: Path,
    staging_dir: Path,
    source_dir: Path | None = None,
    expected_size_bytes: tuple[int, int] | None = None,
    progress: Callable[[str], None] = print,
) -> ConversionResult:
    """Convert one pinned model repository, resuming verified output shards."""

    _validate_identity(repo, revision)
    output_dir = Path(output_dir)
    staging_dir = Path(staging_dir)
    roots = {"output_dir": output_dir, "staging_dir": staging_dir}
    if source_dir is not None:
        roots["source_dir"] = Path(source_dir)
    _assert_distinct_roots(roots)

    # These persistent adjacent locks are independent of the staging choice
    # and are held before source discovery, directory inspection, or marker
    # creation.  Thus two invocations sharing either output or staging cannot
    # race even when the other path differs.
    with _exclusive_conversion_locks(output_dir, staging_dir):
        return _convert_repository_locked(
            repo=repo,
            revision=revision,
            output_dir=output_dir,
            staging_dir=staging_dir,
            source_dir=Path(source_dir) if source_dir is not None else None,
            expected_size_bytes=expected_size_bytes,
            progress=progress,
        )


def _convert_repository_locked(
    *,
    repo: str,
    revision: str,
    output_dir: Path,
    staging_dir: Path,
    source_dir: Path | None,
    expected_size_bytes: tuple[int, int] | None,
    progress: Callable[[str], None],
) -> ConversionResult:
    """Conversion body; caller owns stable output and staging path locks."""

    source: ModelSource
    if source_dir is None:
        source = HubModelSource(repo, revision)
    else:
        source = LocalModelSource(Path(source_dir))

    plan, source_files = _discover_plan(source, repo, revision)
    source_index = _load_json_bytes(source_files[INDEX_NAME], label=INDEX_NAME)
    by_shard = _expected_keys_by_shard(source_index)
    expected_identity = _marker_identity(plan)

    output_marker = _inspect_directory(output_dir, OUTPUT_MARKER_NAME, expected_identity)
    staging_marker = _inspect_directory(staging_dir, STAGING_MARKER_NAME, expected_identity)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    if output_marker is None:
        _atomic_write_json(output_dir / OUTPUT_MARKER_NAME, plan)
        output_marker = dict(plan)
    if staging_marker is None:
        _atomic_write_json(staging_dir / STAGING_MARKER_NAME, plan)
    with contextlib.ExitStack():  # keep state updates in one explicit scope
        output_partial_names = [
            OUTPUT_MARKER_NAME,
            INDEX_NAME,
            *plan["weight_shards"],
            *plan["support_files"],
        ]
        cleared = _clear_owned_partials(
            output_dir,
            logical_names=output_partial_names,
            download_partials=False,
        )
        cleared += _clear_owned_partials(
            staging_dir,
            logical_names=plan["weight_shards"],
            download_partials=True,
        )
        cleared += _clear_owned_partials(
            staging_dir,
            logical_names=[STAGING_MARKER_NAME],
            download_partials=False,
        )
        if cleared:
            progress(f"cleared {len(cleared)} owned partial file(s) from an interrupted run")
        _validate_directory_contents(output_dir, staging_dir, plan)

        shards_record = output_marker.get("shards")
        if not isinstance(shards_record, dict):
            raise ConversionError("output marker has invalid shards record")
        unexpected_records = set(shards_record) - set(plan["weight_shards"])
        if unexpected_records:
            raise ConversionError(
                "output marker records unexpected shards: "
                + ", ".join(sorted(unexpected_records)[:5])
            )

        source_identity = plan["source"]
        assert isinstance(source_identity, dict)
        weight_identity = plan["weight_shard_identity"]
        assert isinstance(weight_identity, dict)
        for position, shard in enumerate(plan["weight_shards"], 1):
            assert isinstance(shard, str)
            output_path = _child(output_dir, shard)
            staged_path = _child(staging_dir, shard)
            provenance = _provenance(
                repo,
                revision,
                str(source_identity["index_sha256"]),
                shard,
                weight_identity[shard],
            )
            if shard not in shards_record and (
                output_path.exists() or output_path.is_symlink()
            ):
                if not output_path.is_file() or output_path.is_symlink():
                    raise ConversionError(f"uncommitted output shard is unsafe: {output_path}")
                # A SIGKILL can land after the atomic shard rename but before
                # its marker record commits.  Provenance metadata alone is not
                # an integrity commitment, so discard only this exact expected,
                # marker-owned derived file and reproduce it from the immutable
                # source identity instead of blessing its current bytes.
                output_path.unlink()
                progress(f"discarded uncommitted output shard {shard}; reconverting")
            if (
                shard in shards_record
                and not output_path.exists()
                and not output_path.is_symlink()
            ):
                raise ConversionError(
                    f"output marker records {shard}, but the output shard is missing"
                )
            if output_path.exists() or output_path.is_symlink():
                existing_tensor_bytes, _ = _inspect_safetensors(
                    output_path,
                    expected_keys=by_shard[shard],
                    required_metadata=provenance,
                    require_bfloat16=True,
                )
                actual_record = _shard_record(output_path, existing_tensor_bytes)
                if shard in shards_record:
                    _validate_recorded_shard(
                        shard,
                        shards_record[shard],
                        actual_record,
                    )
                progress(f"[{position}/{len(plan['weight_shards'])}] verified existing {shard}")
            else:
                progress(f"[{position}/{len(plan['weight_shards'])}] staging {shard}")
                staged_path = _stage_source_shard(
                    source,
                    shard,
                    staging_dir,
                    by_shard[shard],
                    weight_identity[shard],
                )
                progress(f"[{position}/{len(plan['weight_shards'])}] converting {shard}")
                _convert_shard(
                    staged_path,
                    output_path,
                    expected_keys=by_shard[shard],
                    provenance=provenance,
                )

            # The shard is an owned download/copy in the dedicated staging tree.
            # Delete it only after the corresponding atomic output has validated.
            if staged_path.is_file() and not staged_path.is_symlink():
                staged_path.unlink()

            shard_tensor_bytes, _ = _inspect_safetensors(
                output_path,
                expected_keys=by_shard[shard],
                required_metadata=provenance,
                require_bfloat16=True,
            )
            shards_record[shard] = _shard_record(output_path, shard_tensor_bytes)
            output_marker["complete"] = False
            _atomic_write_json(output_dir / OUTPUT_MARKER_NAME, output_marker)

        total_tensor_bytes = sum(
            int(shards_record[shard]["tensor_bytes"])
            for shard in plan["weight_shards"]
        )
        for name in plan["support_files"]:
            assert isinstance(name, str)
            data = (
                _converted_config(source_files[name])
                if name == CONFIG_NAME
                else source_files[name]
            )
            _write_or_validate(_child(output_dir, name), data)
        _write_or_validate(
            output_dir / INDEX_NAME,
            _converted_index(source_files[INDEX_NAME], total_tensor_bytes),
        )

        tensor_bytes, weight_file_bytes, shard_records = _validate_final(
            output_dir,
            plan=plan,
            source_files=source_files,
            expected_size_bytes=expected_size_bytes,
        )
        output_marker["complete"] = True
        output_marker["shards"] = shard_records
        output_marker["tensor_bytes"] = tensor_bytes
        output_marker["weight_file_bytes"] = weight_file_bytes
        # The source index/config hashes above attest the pinned FP32 inputs,
        # but conversion deterministically rewrites their dtype/total_size.
        # Record the committed output control files separately so launch
        # preflight never mistakes a source hash for an output hash.
        output_marker["output_index_sha256"] = _sha256_file(
            output_dir / INDEX_NAME
        )
        output_marker["output_support_sha256"] = {
            name: _sha256_file(_child(output_dir, name))
            for name in plan["support_files"]
        }
        _atomic_write_json(output_dir / OUTPUT_MARKER_NAME, output_marker)
        _validate_directory_contents(output_dir, staging_dir, plan)
    progress(
        f"validated {len(plan['weight_shards'])} BF16 shards: "
        f"{weight_file_bytes} file bytes, {tensor_bytes} tensor bytes"
    )
    return ConversionResult(
        output_dir=output_dir,
        shards=len(plan["weight_shards"]),
        tensor_bytes=tensor_bytes,
        weight_file_bytes=weight_file_bytes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="exact Hugging Face model id (owner/name)")
    parser.add_argument(
        "--revision",
        required=True,
        help="exact 40-character Hugging Face commit id (branches/tags are refused)",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--staging-dir", required=True, type=Path)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="read a local repository mirror instead of accessing Hugging Face",
    )
    parser.add_argument(
        "--expected-size-gb-min",
        type=float,
        help="optional minimum total on-disk weight size in decimal GB (use 55 for OLMo 32B)",
    )
    parser.add_argument(
        "--expected-size-gb-max",
        type=float,
        help="optional maximum total on-disk weight size in decimal GB (use 75 for OLMo 32B)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.expected_size_gb_min is None) != (args.expected_size_gb_max is None):
        parser.error("--expected-size-gb-min and --expected-size-gb-max must be supplied together")
    expected = None
    if args.expected_size_gb_min is not None:
        expected = (
            int(args.expected_size_gb_min * 1_000_000_000),
            int(args.expected_size_gb_max * 1_000_000_000),
        )
    try:
        convert_repository(
            repo=args.repo,
            revision=args.revision,
            output_dir=args.output_dir,
            staging_dir=args.staging_dir,
            source_dir=args.source_dir,
            expected_size_bytes=expected,
        )
    except ConversionError as exc:
        print(f"conversion refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
