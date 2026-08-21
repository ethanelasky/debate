#!/usr/bin/env python3
"""Retained-volume supervisor for the one approved scheduler debate command.

This is an inner, Debate-owned supervisor.  The job scheduler remains the sole
provider/deadline owner and the sole writer of the outer ``terminal.json``.
Here, cgroup v2 contains the trainer, its detached judge, Ray/vLLM workers, and
checkpoint synchronizers; the wrapper drains/kills that tree, reaps adopted
orphans, and writes a hash-qualified retained evidence tree for collection.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping, Optional


SCHEMA = "debate-scheduler-workload/v1"
RUN_NAME = "mathl5_qwen35_pc_debate_verl"
LAUNCH_ARGV = (
    "bash",
    "scripts/pod_run.sh",
    "debate",
    RUN_NAME,
)
NAMESPACE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ENV_SHA256_RE = re.compile(r"sha256:([0-9a-f]{64})")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
ANSI_RE = re.compile(rb"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
SECRET_RE = re.compile(
    rb"(?i)(authorization\s*:\s*(?:bearer\s+)?|"
    rb"(?:api[_-]?key|token|secret|password)\s*[=:]\s*)[^\s,;]+"
)
URL_CREDENTIAL_RE = re.compile(rb"(https?://)[^/@\s]+@", re.IGNORECASE)
PR_SET_CHILD_SUBREAPER = 36
CONTAINMENT_PROOF_FD = 9
CONTAINMENT_SCHEMA = "runpod-remote.containment-proof/v1"
CHECKPOINT_DESTINATION_FD = 10
CHECKPOINT_DESTINATION_PATH = "/proc/self/fd/10"
RUNTIME_PYTHON = Path(
    "/opt/job-scheduler/debate-runtime/v1/python/bin/python3.12"
)
MAX_PENDING_LOG_RECORD = 8 * 1024 * 1024


class Refusal(RuntimeError):
    """A precondition or evidence boundary failed closed."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_existing_directory(raw: str, *, field: str) -> Path:
    if not raw or not os.path.isabs(raw):
        raise Refusal(f"{field} must be an absolute path")
    path = Path(raw)
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise Refusal(f"{field} must exist: {path}") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise Refusal(f"{field} must be a real directory: {path}")
    resolved = path.resolve(strict=True)
    if str(path) != str(resolved):
        raise Refusal(f"{field} must be canonical: {resolved}")
    return path


def _below_workspace(path: Path, *, field: str) -> None:
    try:
        path.relative_to("/workspace")
    except ValueError as exc:
        raise Refusal(f"{field} must be on the retained /workspace volume") from exc


def _validate_runtime_python() -> None:
    """Refuse a mutable, redirected, or legacy shared-volume runtime."""
    try:
        info = os.lstat(RUNTIME_PYTHON)
        resolved = RUNTIME_PYTHON.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise Refusal(f"fixed runtime Python is unavailable: {RUNTIME_PYTHON}") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o555
        or resolved != RUNTIME_PYTHON
    ):
        raise Refusal(
            "fixed runtime Python must be a canonical root-owned mode-0555 "
            "single-link regular file"
        )


def _required_env(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if not value:
        raise Refusal(f"missing required scheduler field {name}")
    if "\x00" in value:
        raise Refusal(f"{name} contains NUL")
    return value


def _environment_sha256(environ: Mapping[str, str], name: str) -> str:
    """Validate the scheduler's canonical digest and return proof-form hex."""
    value = _required_env(environ, name)
    matched = ENV_SHA256_RE.fullmatch(value)
    if matched is None:
        raise Refusal(f"{name} must be canonical sha256:<64 lowercase hex>")
    return matched.group(1)


def _deadline(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise Refusal("DEBATE_DEADLINE_EPOCH must be a finite epoch") from exc
    if not (parsed > 0 and parsed < float("inf")):
        raise Refusal("DEBATE_DEADLINE_EPOCH must be a finite positive epoch")
    return parsed


def _read_destination(path_raw: str) -> tuple[bytes, dict[str, object]]:
    if path_raw != CHECKPOINT_DESTINATION_PATH:
        raise Refusal(
            "DEBATE_CHECKPOINT_DESTINATION_FILE must be exact sealed FD 10"
        )
    try:
        info = os.fstat(CHECKPOINT_DESTINATION_FD)
    except OSError as exc:
        raise Refusal("sealed checkpoint destination FD 10 is unavailable") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
        or info.st_nlink != 0
        or stat.S_IMODE(info.st_mode) != 0o400
    ):
        raise Refusal(
            "checkpoint destination must be a root-owned mode-0400 memfd"
        )
    seal_names = ("F_SEAL_SEAL", "F_SEAL_SHRINK", "F_SEAL_GROW", "F_SEAL_WRITE")
    if not hasattr(fcntl, "F_GET_SEALS") or any(
        not hasattr(fcntl, name) for name in seal_names
    ):
        raise Refusal("kernel/Python lacks sealed-memfd destination support")
    required_seals = sum(int(getattr(fcntl, name)) for name in seal_names)
    try:
        observed_seals = int(
            fcntl.fcntl(CHECKPOINT_DESTINATION_FD, fcntl.F_GET_SEALS)
        )
    except OSError as exc:
        raise Refusal("cannot verify checkpoint destination memfd seals") from exc
    if observed_seals & required_seals != required_seals:
        raise Refusal("checkpoint destination memfd is not fully sealed")
    try:
        os.lseek(CHECKPOINT_DESTINATION_FD, 0, os.SEEK_SET)
        payload = os.read(CHECKPOINT_DESTINATION_FD, 64 * 1024 + 1)
        if len(payload) > 64 * 1024 or not payload:
            raise Refusal("checkpoint destination must contain 1..65536 bytes")
        if os.read(CHECKPOINT_DESTINATION_FD, 1):
            raise Refusal("checkpoint destination exceeds 65536 bytes")
        final_info = os.fstat(CHECKPOINT_DESTINATION_FD)
        final_seals = int(
            fcntl.fcntl(CHECKPOINT_DESTINATION_FD, fcntl.F_GET_SEALS)
        )
        if (
            (final_info.st_dev, final_info.st_ino, final_info.st_size)
            != (info.st_dev, info.st_ino, info.st_size)
            or final_seals != observed_seals
        ):
            raise Refusal("checkpoint destination changed while being read")
    finally:
        # Sync descendants inherit the same immutable description from offset
        # zero; the supervisor retains FD 10 until its entire subtree exits.
        os.lseek(CHECKPOINT_DESTINATION_FD, 0, os.SEEK_SET)
    if not payload or len(payload) > 64 * 1024:
        raise Refusal("checkpoint destination file must contain 1..65536 bytes")
    try:
        document = json.loads(payload, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Refusal(f"invalid checkpoint destination JSON: {exc}") from exc
    if not isinstance(document, dict) or document.get("kind") not in {"local", "bucket"}:
        raise Refusal("checkpoint destination kind must be local or bucket")
    canonical = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    if payload != canonical:
        raise Refusal("checkpoint destination JSON must be canonical ASCII with one newline")
    # ckpt_sync.sh owns the complete schema and provider-specific validation.
    # This early check only prevents the wrapper from starting with a shape the
    # synchronizer can never accept.
    if document["kind"] == "local":
        if set(document) != {"kind", "directory"}:
            raise Refusal("local checkpoint destination has unknown or missing keys")
        if not isinstance(document["directory"], str):
            raise Refusal("destination.directory must be a string")
        directory = _canonical_existing_directory(
            document["directory"], field="destination.directory"
        )
        _below_workspace(directory, field="destination.directory")
    else:
        if set(document) != {"kind", "endpoint", "region", "bucket", "prefix"}:
            raise Refusal("bucket checkpoint destination has unknown or missing keys")
        for field in ("endpoint", "region", "bucket", "prefix"):
            value = document[field]
            if not isinstance(value, str) or not value or any(
                ord(character) < 32 or ord(character) == 127
                for character in value
            ):
                raise Refusal(
                    f"bucket checkpoint destination {field} must be a plain string"
                )
        if "@" in document["endpoint"]:
            raise Refusal("bucket checkpoint endpoint must not contain user information")
    return payload, document


@dataclass(frozen=True)
class ContainmentProof:
    protocol_version: int
    attempt_id: str
    attempt_identity_sha256: str
    namespace: str
    cgroup_relative_path: str
    workload_uid: int
    workload_gid: int
    wrapper_release: str
    wrapper_sha256: str
    worker_ref: str
    deadline_epoch: float
    artifact_root: str
    attempt_root: str
    evidence_root: str
    workload_output_root: str
    checkpoint_working_root: str
    snapshot_sha256: str


def _plain_string(document: dict[str, object], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise Refusal(f"containment proof field {key} must be a nonempty plain string")
    return value


def _canonical_absolute(value: str, *, field: str) -> str:
    path = Path(value)
    if (
        not path.is_absolute()
        or value != os.path.normpath(value)
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise Refusal(f"containment proof field {field} is not a canonical absolute path")
    return value


def _parse_containment_document(document: object) -> ContainmentProof:
    fields = {
        "schema",
        "protocol_version",
        "attempt_id",
        "attempt_identity_sha256",
        "namespace",
        "cgroup_relative_path",
        "workload_uid",
        "workload_gid",
        "wrapper_release",
        "wrapper_sha256",
        "worker_ref",
        "deadline_epoch",
        "artifact_root",
        "attempt_root",
        "evidence_root",
        "workload_output_root",
        "checkpoint_working_root",
        "snapshot_sha256",
    }
    if not isinstance(document, dict) or set(document) != fields:
        raise Refusal("containment proof has unknown or missing fields")
    if document.get("schema") != CONTAINMENT_SCHEMA:
        raise Refusal("containment proof schema is not supported")
    protocol_version = document.get("protocol_version")
    if protocol_version != 1 or isinstance(protocol_version, bool):
        raise Refusal("containment proof protocol_version must be integer 1")
    attempt_id = _plain_string(document, "attempt_id")
    if SAFE_ID_RE.fullmatch(attempt_id) is None:
        raise Refusal("containment proof attempt_id has an invalid format")
    attempt_identity = _plain_string(document, "attempt_identity_sha256")
    namespace = _plain_string(document, "namespace")
    wrapper_sha256 = _plain_string(document, "wrapper_sha256")
    snapshot_sha256 = _plain_string(document, "snapshot_sha256")
    if any(
        SHA256_RE.fullmatch(value) is None
        for value in (attempt_identity, wrapper_sha256, snapshot_sha256)
    ):
        raise Refusal("containment proof digests must be lowercase SHA-256 values")
    if NAMESPACE_RE.fullmatch(namespace) is None:
        raise Refusal("containment proof namespace has an invalid format")
    uid, gid = document.get("workload_uid"), document.get("workload_gid")
    if uid != 10001 or gid != 10001 or isinstance(uid, bool) or isinstance(gid, bool):
        raise Refusal("containment proof must bind dedicated uid/gid 10001")
    cgroup_path = _canonical_absolute(
        _plain_string(document, "cgroup_relative_path"),
        field="cgroup_relative_path",
    )
    if cgroup_path == "/":
        raise Refusal("containment proof cannot bind the cgroup root")
    wrapper_release = _plain_string(document, "wrapper_release")
    if SAFE_ID_RE.fullmatch(wrapper_release) is None:
        raise Refusal("containment proof wrapper_release has an invalid format")
    worker_ref = _plain_string(document, "worker_ref")
    if SAFE_ID_RE.fullmatch(worker_ref) is None:
        raise Refusal("containment proof worker_ref has an invalid format")
    deadline_value = document.get("deadline_epoch")
    if isinstance(deadline_value, bool) or not isinstance(deadline_value, (int, float)):
        raise Refusal("containment proof deadline_epoch must be numeric")
    deadline_epoch = _deadline(str(deadline_value))
    return ContainmentProof(
        protocol_version=protocol_version,
        attempt_id=attempt_id,
        attempt_identity_sha256=attempt_identity,
        namespace=namespace,
        cgroup_relative_path=cgroup_path,
        workload_uid=uid,
        workload_gid=gid,
        wrapper_release=wrapper_release,
        wrapper_sha256=wrapper_sha256,
        worker_ref=worker_ref,
        deadline_epoch=deadline_epoch,
        artifact_root=_canonical_absolute(
            _plain_string(document, "artifact_root"), field="artifact_root"
        ),
        attempt_root=_canonical_absolute(
            _plain_string(document, "attempt_root"), field="attempt_root"
        ),
        evidence_root=_canonical_absolute(
            _plain_string(document, "evidence_root"), field="evidence_root"
        ),
        workload_output_root=_canonical_absolute(
            _plain_string(document, "workload_output_root"),
            field="workload_output_root",
        ),
        checkpoint_working_root=_canonical_absolute(
            _plain_string(document, "checkpoint_working_root"),
            field="checkpoint_working_root",
        ),
        snapshot_sha256=snapshot_sha256,
    )


def _read_containment_proof() -> ContainmentProof:
    # The installed root wrapper owns this ABI. Only its sealed memfd on the
    # fixed descriptor is accepted; an environment-selected descriptor would
    # be mutable by the unprivileged workload and is deliberately unsupported.
    try:
        info = os.fstat(CONTAINMENT_PROOF_FD)
    except OSError as exc:
        raise Refusal("sealed containment proof FD 9 is unavailable") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
        or info.st_nlink != 0
        or stat.S_IMODE(info.st_mode) != 0o400
    ):
        raise Refusal("containment proof must be a root-owned mode-0400 memfd")
    seal_names = ("F_SEAL_SEAL", "F_SEAL_SHRINK", "F_SEAL_GROW", "F_SEAL_WRITE")
    if not hasattr(fcntl, "F_GET_SEALS") or any(
        not hasattr(fcntl, name) for name in seal_names
    ):
        raise Refusal("kernel/Python lacks sealed-memfd containment proof support")
    required_seals = sum(int(getattr(fcntl, name)) for name in seal_names)
    observed_seals = int(fcntl.fcntl(CONTAINMENT_PROOF_FD, fcntl.F_GET_SEALS))
    if observed_seals & required_seals != required_seals:
        raise Refusal("containment proof memfd is not write/grow/shrink/seal sealed")
    try:
        os.lseek(CONTAINMENT_PROOF_FD, 0, os.SEEK_SET)
        payload = os.read(CONTAINMENT_PROOF_FD, 64 * 1024 + 1)
        if len(payload) > 64 * 1024 or not payload:
            raise Refusal("containment proof must contain 1..65536 bytes")
        if os.read(CONTAINMENT_PROOF_FD, 1):
            raise Refusal("containment proof exceeds 65536 bytes")
    finally:
        os.close(CONTAINMENT_PROOF_FD)
    try:
        document = json.loads(payload, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Refusal(f"invalid containment proof JSON: {exc}") from exc
    proof = _parse_containment_document(document)
    canonical = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    if payload != canonical:
        raise Refusal("containment proof JSON must be canonical ASCII with one newline")
    return proof


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    artifact_root: Path
    attempt_root: Path
    output_root: Path
    pod_state_root: Path
    checkpoint_dir: Path
    destination_bytes: bytes
    destination: dict[str, object]
    cgroup_path: Path
    namespace: str
    attempt_id: str
    attempt_identity: str
    snapshot_sha256: str
    deadline_epoch: float
    pod_id: str
    containment: ContainmentProof


def _proc_status() -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = Path("/proc/self/status").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise Refusal("cannot read Linux process security status") from exc
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            result[key] = value.strip()
    return result


def _current_cgroup_relative_path() -> str:
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise Refusal("cannot read current cgroup-v2 membership") from exc
    if len(lines) != 1 or not lines[0].startswith("0::"):
        raise Refusal("workload must have one exact unified cgroup-v2 membership")
    value = lines[0][3:]
    return _canonical_absolute(value, field="current cgroup-v2 path")


def _validate_unprivileged_containment(proof: ContainmentProof) -> Path:
    if os.geteuid() != proof.workload_uid or os.getegid() != proof.workload_gid:
        raise Refusal("effective uid/gid do not match the root-sealed containment proof")
    if os.getgroups():
        raise Refusal("dedicated workload must have no supplementary groups")
    status = _proc_status()
    expected_ids = ["10001"] * 4
    if status.get("Uid", "").split() != expected_ids or status.get(
        "Gid", ""
    ).split() != expected_ids:
        raise Refusal("real/effective/saved/fs uid/gid must all be dedicated identity 10001")
    if status.get("CapEff") != "0000000000000000":
        raise Refusal("dedicated workload must have an empty effective capability set")
    if status.get("NoNewPrivs") != "1":
        raise Refusal("dedicated workload must have no_new_privs enabled")
    current = _current_cgroup_relative_path()
    if current != proof.cgroup_relative_path:
        raise Refusal("current cgroup does not match the root-sealed containment proof")
    cgroup_path = Path("/sys/fs/cgroup") / current.lstrip("/")
    resolved = _canonical_existing_directory(str(cgroup_path), field="proved cgroup")
    cgroup_info = os.lstat(resolved)
    if cgroup_info.st_uid != 0 or cgroup_info.st_mode & (
        stat.S_IWGRP | stat.S_IWOTH
    ):
        raise Refusal("proved cgroup must remain root-owned and non-writable by workload")
    for required in ("cgroup.procs", "cgroup.events"):
        control = resolved / required
        if not control.is_file():
            raise Refusal(f"proved cgroup lacks readable {required}")
        if os.access(control, os.W_OK):
            raise Refusal(f"unprivileged workload must not write proved {required}")
    try:
        members = {
            int(line)
            for line in (resolved / "cgroup.procs")
            .read_text(encoding="ascii")
            .splitlines()
            if line
        }
    except (OSError, ValueError) as exc:
        raise Refusal("cannot read proved cgroup membership") from exc
    if os.getpid() not in members:
        raise Refusal("proved cgroup does not contain this workload supervisor")
    return resolved


def load_settings(repo_root_raw: str, environ: Mapping[str, str]) -> Settings:
    containment = _read_containment_proof()
    proved_cgroup = _validate_unprivileged_containment(containment)
    repo_root = _canonical_existing_directory(repo_root_raw, field="repo root")
    expected_repo = Path(__file__).resolve(strict=True).parent.parent
    if repo_root != expected_repo:
        raise Refusal(f"--repo-root does not name this frozen script tree: {expected_repo}")
    _below_workspace(repo_root, field="repo root")
    _validate_runtime_python()

    namespace = _required_env(environ, "DEBATE_LAUNCH_NAMESPACE")
    if NAMESPACE_RE.fullmatch(namespace) is None:
        raise Refusal("DEBATE_LAUNCH_NAMESPACE has an invalid format")
    attempt_identity = _environment_sha256(
        environ, "DEBATE_ATTEMPT_IDENTITY_SHA256"
    )
    snapshot_sha256 = _environment_sha256(environ, "DEBATE_SNAPSHOT_SHA256")
    pod_id = _required_env(environ, "RUNPOD_POD_ID")
    if SAFE_ID_RE.fullmatch(pod_id) is None:
        raise Refusal("RUNPOD_POD_ID has an invalid format")

    deadline_epoch = _deadline(_required_env(environ, "DEBATE_DEADLINE_EPOCH"))
    expected_checkpoint_dir = str(
        Path("/workspace/checkpoints") / RUN_NAME / namespace
    )
    proof_bindings = {
        "attempt_identity_sha256": (containment.attempt_identity_sha256, attempt_identity),
        "namespace": (containment.namespace, namespace),
        "worker_ref": (containment.worker_ref, pod_id),
        "deadline_epoch": (containment.deadline_epoch, deadline_epoch),
        "snapshot_sha256": (containment.snapshot_sha256, snapshot_sha256),
        "artifact_root": (
            containment.artifact_root,
            _required_env(environ, "DEBATE_ARTIFACT_ROOT"),
        ),
        "checkpoint_working_root": (
            containment.checkpoint_working_root,
            expected_checkpoint_dir,
        ),
    }
    mismatches = [
        name for name, (proved, supplied) in proof_bindings.items() if proved != supplied
    ]
    if mismatches:
        raise Refusal(
            "scheduler fields do not match root-sealed containment proof: "
            + ", ".join(mismatches)
        )

    artifact_root = _canonical_existing_directory(
        _required_env(environ, "DEBATE_ARTIFACT_ROOT"), field="DEBATE_ARTIFACT_ROOT"
    )
    _below_workspace(artifact_root, field="DEBATE_ARTIFACT_ROOT")
    if artifact_root == Path("/workspace"):
        raise Refusal("DEBATE_ARTIFACT_ROOT must be a dedicated /workspace child")
    if os.lstat(artifact_root).st_uid != os.geteuid():
        raise Refusal("DEBATE_ARTIFACT_ROOT must be owned by the workload user")
    if stat.S_IMODE(os.lstat(artifact_root).st_mode) != 0o700:
        raise Refusal("DEBATE_ARTIFACT_ROOT must have exact mode 0700")

    destination_bytes, destination = _read_destination(
        _required_env(environ, "DEBATE_CHECKPOINT_DESTINATION_FILE")
    )
    checkpoint_dir = Path("/workspace/checkpoints") / RUN_NAME / namespace
    checkpoint_parent = _canonical_existing_directory(
        str(checkpoint_dir.parent), field="checkpoint working parent"
    )
    checkpoint_parent_info = os.lstat(checkpoint_parent)
    if checkpoint_parent_info.st_uid != os.geteuid() or checkpoint_parent_info.st_mode & (
        stat.S_IWGRP | stat.S_IWOTH
    ):
        raise Refusal(
            "checkpoint working parent must be workload-owned and not group/world writable"
        )
    if os.path.lexists(checkpoint_dir):
        raise Refusal(
            f"checkpoint attempt destination already exists; refusing adoption: {checkpoint_dir}"
        )

    attempt_root = artifact_root / namespace
    if str(attempt_root) != containment.attempt_root:
        raise Refusal("derived attempt root does not match root-sealed containment proof")
    attempt_root = _canonical_existing_directory(
        str(attempt_root), field="proved attempt root"
    )
    evidence_root = Path(containment.evidence_root)
    _below_workspace(evidence_root, field="containment evidence_root")
    try:
        evidence_root.relative_to(attempt_root)
    except ValueError:
        pass
    else:
        raise Refusal(
            "installed-wrapper evidence_root must remain outside Debate's attempt root"
        )
    if (
        os.lstat(attempt_root).st_uid != os.geteuid()
        or stat.S_IMODE(os.lstat(attempt_root).st_mode) != 0o700
    ):
        raise Refusal("proved attempt root must be workload-owned mode 0700")
    output_root = _canonical_existing_directory(
        containment.workload_output_root,
        field="proved workload_output_root",
    )
    if output_root != attempt_root / "scheduler-output":
        raise Refusal(
            "workload output root must be the exact scheduler-output leaf of the "
            "proved Debate attempt"
        )
    if (
        os.lstat(output_root).st_uid != os.geteuid()
        or stat.S_IMODE(os.lstat(output_root).st_mode) != 0o700
    ):
        raise Refusal("workload output root must be workload-owned mode 0700")
    if any(output_root.iterdir()):
        raise Refusal("workload output root must be empty before its exclusive claim")
    claim = output_root / ".debate-wrapper-claim-v1"
    claim_fd = os.open(
        claim,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        os.write(claim_fd, f"{attempt_identity}\n".encode("ascii"))
        os.fsync(claim_fd)
    finally:
        os.close(claim_fd)
    pod_state_root = output_root / "pod-run"
    pod_state_root.mkdir(mode=0o700)
    output_fd = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(output_fd)
    finally:
        os.close(output_fd)

    return Settings(
        repo_root=repo_root,
        artifact_root=artifact_root,
        attempt_root=attempt_root,
        output_root=output_root,
        pod_state_root=pod_state_root,
        checkpoint_dir=checkpoint_dir,
        destination_bytes=destination_bytes,
        destination=destination,
        cgroup_path=proved_cgroup,
        namespace=namespace,
        attempt_id=containment.attempt_id,
        attempt_identity=attempt_identity,
        snapshot_sha256=snapshot_sha256,
        deadline_epoch=deadline_epoch,
        pod_id=pod_id,
        containment=containment,
    )


def _publish_json(path: Path, document: object) -> None:
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short evidence write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _manifest_tree(
    root: Path, *, excluded: Optional[set[Path]] = None
) -> list[dict[str, object]]:
    excluded = excluded or set()
    records: list[dict[str, object]] = []
    root_info = os.lstat(root)
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise Refusal(f"manifest root is unsafe: {root}")
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            entry = current_path / name
            info = os.lstat(entry)
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise Refusal(f"manifest refuses non-directory or symlink: {entry}")
        for name in files:
            entry = current_path / name
            if entry in excluded:
                continue
            info = os.lstat(entry)
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
                raise Refusal(f"manifest refuses non-regular, linked evidence: {entry}")
            records.append(
                {
                    "path": entry.relative_to(root).as_posix(),
                    "size": info.st_size,
                    "sha256": _sha256_file(entry),
                }
            )
    records.sort(key=lambda item: str(item["path"]))
    return records


def _set_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise Refusal(f"cannot become child subreaper: errno {errno}")


class UpstreamCgroup:
    """Read-only view of the root wrapper's already-owned attempt cgroup."""

    def __init__(self, path: Path):
        self.path = path

    def pids(self) -> tuple[int, ...]:
        values = []
        for raw in (self.path / "cgroup.procs").read_text(encoding="ascii").splitlines():
            if raw:
                values.append(int(raw))
        return tuple(sorted(set(values)))

    def attach(self, pid: int) -> None:
        # A child inherits this supervisor's cgroup. Check that kernel fact
        # while it is still blocked behind the exec barrier; never write a
        # cgroup control from the unprivileged workload.
        if pid not in self.pids():
            raise Refusal(f"child pid {pid} did not inherit the proved attempt cgroup")

    def workload_pids(self) -> tuple[int, ...]:
        return tuple(pid for pid in self.pids() if pid != os.getpid())

    def populated(self) -> bool:
        return bool(self.workload_pids())

    @staticmethod
    def _require_same_uid(pid: int) -> None:
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
        except FileNotFoundError:
            return
        uid_line = next((line for line in status.splitlines() if line.startswith("Uid:")), "")
        fields = uid_line.split()[1:]
        if fields != [str(os.geteuid())] * 4:
            raise Refusal(f"refusing to signal unexpected-uid cgroup member pid {pid}")

    def signal_all(self, sig: int) -> None:
        for pid in self.workload_pids():
            self._require_same_uid(pid)
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def kill_all(self) -> None:
        self.signal_all(signal.SIGKILL)

    def close(self) -> None:
        if self.workload_pids():
            raise Refusal("proved attempt cgroup still has workload descendants")


def _process_identity(pid: int) -> Optional[dict[str, object]]:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    prefix = text[: close + 1]
    rest = text[close + 2 :].split()
    if len(rest) < 20:
        return None
    comm = prefix[prefix.find("(") + 1 : -1]
    try:
        return {
            "pid": pid,
            "ppid": int(rest[1]),
            "pgrp": int(rest[2]),
            "session": int(rest[3]),
            "start_ticks": int(rest[19]),
            "comm": comm[:128],
        }
    except ValueError:
        return None


def _sanitize(data: bytes) -> bytes:
    data = ANSI_RE.sub(b"", data).replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    data = SECRET_RE.sub(lambda match: match.group(1) + b"[REDACTED]", data)
    data = URL_CREDENTIAL_RE.sub(rb"\1[REDACTED]@", data)
    return bytes(byte for byte in data if byte in (9, 10) or byte >= 32)


def _pump(source: BinaryIO, destination: Path, inherited_fd: int) -> None:
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        pending = bytearray()
        while True:
            chunk = source.read(64 * 1024)
            if not chunk:
                break
            pending.extend(chunk)
            while True:
                newline = pending.find(b"\n")
                carriage = pending.find(b"\r")
                separators = [index for index in (newline, carriage) if index >= 0]
                if not separators:
                    break
                separator = min(separators)
                if pending[separator] == 13 and separator + 1 == len(pending):
                    break
                end = separator + 1
                if pending[separator] == 13 and pending[end : end + 1] == b"\n":
                    end += 1
                record = bytes(pending[:end])
                del pending[:end]
                clean = _sanitize(record)
                if clean:
                    _write_all(fd, clean)
                    try:
                        _write_all(inherited_fd, clean)
                    except BrokenPipeError:
                        pass
            if len(pending) > MAX_PENDING_LOG_RECORD:
                raise Refusal("log record exceeded the bounded sanitizer buffer")
        clean = _sanitize(bytes(pending))
        if clean:
            _write_all(fd, clean)
            try:
                _write_all(inherited_fd, clean)
            except BrokenPipeError:
                pass
        os.fsync(fd)
    finally:
        os.close(fd)
        source.close()


def _pump_guarded(
    source: BinaryIO,
    destination: Path,
    inherited_fd: int,
    errors: list[BaseException],
) -> None:
    try:
        _pump(source, destination, inherited_fd)
    except BaseException as exc:
        errors.append(exc)
        try:
            source.close()
        except Exception:
            pass


@dataclass
class OwnedProcess:
    process: subprocess.Popen[bytes]
    pumps: tuple[threading.Thread, threading.Thread]
    pump_errors: list[BaseException]
    initial_identity: dict[str, object]

    def wait(self, timeout: Optional[float] = None) -> int:
        result = self.process.wait(timeout=timeout)
        for thread in self.pumps:
            thread.join(timeout=30)
            if thread.is_alive():
                raise Refusal("log pump did not finish after child exit")
        if self.pump_errors:
            error = self.pump_errors[0]
            raise Refusal(
                f"log capture failed: {type(error).__name__}: {error}"
            ) from error
        return result


def _spawn(
    *,
    argv: tuple[str, ...],
    cwd: Path,
    environ: Mapping[str, str],
    cgroup: UpstreamCgroup,
    stdout_path: Path,
    stderr_path: Path,
) -> OwnedProcess:
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    barrier = 'IFS= read -r -n 1 <&"$1"; shift; exec "$@"'
    command = ("bash", "-c", barrier, "scheduler-cgroup-barrier", str(read_fd), *argv)
    process: Optional[subprocess.Popen[bytes]] = None
    initial_identity: Optional[dict[str, object]] = None
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(environ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(read_fd, CHECKPOINT_DESTINATION_FD),
            start_new_session=True,
        )
        os.close(read_fd)
        read_fd = -1
        cgroup.attach(process.pid)
        initial_identity = _process_identity(process.pid)
        if initial_identity is None:
            raise Refusal("cannot capture child identity behind the cgroup barrier")
        os.write(write_fd, b"G")
    except BaseException:
        if process is not None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            process.wait()
        raise
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        os.close(write_fd)
    assert process.stdout is not None and process.stderr is not None
    pump_errors: list[BaseException] = []
    out_thread = threading.Thread(
        target=_pump_guarded,
        args=(process.stdout, stdout_path, 1, pump_errors),
        daemon=False,
    )
    err_thread = threading.Thread(
        target=_pump_guarded,
        args=(process.stderr, stderr_path, 2, pump_errors),
        daemon=False,
    )
    out_thread.start()
    err_thread.start()
    assert initial_identity is not None
    return OwnedProcess(
        process,
        (out_thread, err_thread),
        pump_errors,
        initial_identity,
    )


def _sync_environment(settings: Settings, *, once: bool) -> dict[str, str]:
    state = settings.output_root / "ckpt-sync-state"
    state.mkdir(mode=0o700, exist_ok=True)
    result = dict(os.environ)
    # The worker never receives provider lifecycle authority. Checkpoint sync
    # may inherit only its separately scoped ambient S3 credentials.
    for forbidden in (
        "RUNPOD_API_KEY",
        "RUNPOD_CONFIG",
        "RUNPOD_CONFIG_FILE",
        "DEBATE_CHECKPOINT_DESTINATION_FILE",
        "DEBATE_WORKLOAD_OUTPUT_ROOT",
        "DEBATE_ATTEMPT_ID",
        "DEBATE_ATTEMPT_IDENTITY_SHA256",
        "DEBATE_SNAPSHOT_SHA256",
        "DEBATE_DEADLINE_EPOCH",
    ):
        result.pop(forbidden, None)
    result.update(
        {
            "CKPT_DIR": str(settings.checkpoint_dir),
            "RUN_NAME": RUN_NAME,
            "DEBATE_LAUNCH_NAMESPACE": settings.namespace,
            "DEBATE_CHECKPOINT_DESTINATION_FILE": CHECKPOINT_DESTINATION_PATH,
            "CKPT_DESTINATION_JSON": settings.destination_bytes.decode("utf-8"),
            "CKPT_SYNC_STATE": str(state / "completed.state"),
            "CKPT_SYNC_PID_FILE": str(state / "sync.pid"),
            "CKPT_SYNC_LOCK_FILE": str(state / "sync.lock"),
            "CKPT_SYNC_ONCE": "1" if once else "0",
            "QUIESCENT_SECS": "0" if once else "90",
            "INTERVAL": "120",
            "PYBIN": str(RUNTIME_PYTHON),
            # Scheduler mode accepts provider-injected ambient credential names
            # only. /proc/self cannot acquire this nonexistent leaf, so the
            # manual /root/.runpod/s3.env compatibility input is unreachable.
            "S3_ENV_FILE": "/proc/self/debate-scheduler-no-credential-file",
        }
    )
    return result


def _observe(
    cgroup: UpstreamCgroup,
    observed: dict[tuple[int, int], dict[str, object]],
) -> None:
    for pid in cgroup.workload_pids():
        identity = _process_identity(pid)
        if identity is not None:
            observed.setdefault((pid, int(identity["start_ticks"])), identity)


def _reap_adopted() -> int:
    reaped = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return reaped
        if pid == 0:
            return reaped
        reaped += 1


def _drain_cgroup(
    cgroup: UpstreamCgroup,
    observed: dict[tuple[int, int], dict[str, object]],
) -> None:
    cgroup.signal_all(signal.SIGTERM)
    deadline = time.monotonic() + 30
    while cgroup.populated() and time.monotonic() < deadline:
        _observe(cgroup, observed)
        _reap_adopted()
        time.sleep(0.1)
    if cgroup.populated():
        cgroup.kill_all()
    deadline = time.monotonic() + 30
    while cgroup.populated() and time.monotonic() < deadline:
        _observe(cgroup, observed)
        _reap_adopted()
        time.sleep(0.1)
    _reap_adopted()
    if cgroup.populated():
        raise Refusal("attempt cgroup retained workload descendants after SIGKILL")


def _stop_owned(process: Optional[OwnedProcess]) -> Optional[int]:
    if process is None:
        return None
    if process.process.poll() is None:
        try:
            os.killpg(process.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait(timeout=30)


def _run_final_sync(
    settings: Settings,
    cgroup: UpstreamCgroup,
    *,
    observed: dict[tuple[int, int], dict[str, object]],
) -> int:
    process = _spawn(
        argv=("bash", "scripts/ckpt_sync.sh"),
        cwd=settings.repo_root,
        environ=_sync_environment(settings, once=True),
        cgroup=cgroup,
        stdout_path=settings.output_root / "checkpoint-sync-final.stdout",
        stderr_path=settings.output_root / "checkpoint-sync-final.stderr",
    )
    observed[
        (
            int(process.initial_identity["pid"]),
            int(process.initial_identity["start_ticks"]),
        )
    ] = process.initial_identity
    deadline = time.monotonic() + 300
    while process.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.25)
    if process.process.poll() is not None:
        return process.wait()
    try:
        os.killpg(process.process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            cgroup.kill_all()
        except ProcessLookupError:
            pass
        process.wait(timeout=30)
    return 124


def _known_escapes(observed: dict[tuple[int, int], dict[str, object]]) -> list[dict[str, object]]:
    escapes = []
    for (pid, start_ticks), old in observed.items():
        current = _process_identity(pid)
        if current is not None and int(current["start_ticks"]) == start_ticks:
            escapes.append(old)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return sorted(escapes, key=lambda item: (int(item["pid"]), int(item["start_ticks"])))


def _has_nonempty_files(path: Path, *, suffix: str) -> bool:
    if not path.is_dir() or path.is_symlink():
        return False
    for entry in path.rglob(f"*{suffix}"):
        try:
            info = os.lstat(entry)
        except FileNotFoundError:
            return False
        if (
            stat.S_ISREG(info.st_mode)
            and not stat.S_ISLNK(info.st_mode)
            and info.st_nlink == 1
            and info.st_size > 0
        ):
            return True
    return False


def run(settings: Settings) -> int:
    _set_subreaper()
    cgroup = UpstreamCgroup(settings.cgroup_path)
    observed: dict[tuple[int, int], dict[str, object]] = {}
    received_signal: Optional[int] = None

    def on_signal(signum: int, _frame: object) -> None:
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, on_signal)

    destination_digest = hashlib.sha256(settings.destination_bytes).hexdigest()
    _publish_json(
        settings.output_root / "provider-handshake.json",
        {
            "schema": SCHEMA,
            "event": "workload-ready",
            "provider": "runpod",
            "worker_ref": settings.pod_id,
            "attempt_id": settings.attempt_id,
            "launch_namespace": settings.namespace,
            "attempt_identity_sha256": settings.attempt_identity,
            "scheduler_asserted_snapshot_sha256": settings.snapshot_sha256,
            "deadline_epoch": settings.deadline_epoch,
            "repo_root": str(settings.repo_root),
            "artifact_attempt_root": str(settings.attempt_root),
            "workload_output_root": str(settings.output_root),
            "upstream_evidence_root": settings.containment.evidence_root,
            "proved_cgroup_relative_path": (
                settings.containment.cgroup_relative_path
            ),
            "upstream_wrapper_release": settings.containment.wrapper_release,
            "upstream_wrapper_sha256": settings.containment.wrapper_sha256,
            "checkpoint_dir": str(settings.checkpoint_dir),
            "checkpoint_destination_sha256": destination_digest,
            "checkpoint_destination": settings.destination,
            "launch_argv": list(LAUNCH_ARGV),
            "scheduler_mode": True,
            "pod_idle_stop": False,
            "warm_start": False,
            "created_at_epoch": time.time(),
        },
    )

    child_env = dict(os.environ)
    child_env.update(
        {
            "CONFIG": "configs/math_pc_debate.yaml",
            "POD_IDLE_STOP": "0",
            "DEBATE_SCHEDULER_MODE": "1",
            "POD_RUN_STATE_DIR": str(settings.pod_state_root),
            "PY": str(RUNTIME_PYTHON),
            "PYBIN": str(RUNTIME_PYTHON),
            "DEBATE_ARTIFACT_ROOT": str(settings.artifact_root),
            "DEBATE_LAUNCH_NAMESPACE": settings.namespace,
        }
    )
    for forbidden in (
        "CKPT_DESTINATION_JSON",
        "CKPT_DIR",
        "CKPT_SYNC_ONCE",
        "WANDB_RESUME",
        "WANDB_RUN_ID",
        "RUNPOD_API_KEY",
        "RUNPOD_CONFIG",
        "RUNPOD_CONFIG_FILE",
        "DEBATE_CHECKPOINT_DESTINATION_FILE",
        "DEBATE_WORKLOAD_OUTPUT_ROOT",
        "DEBATE_ATTEMPT_ID",
        "DEBATE_ATTEMPT_IDENTITY_SHA256",
        "DEBATE_SNAPSHOT_SHA256",
        "DEBATE_DEADLINE_EPOCH",
    ):
        child_env.pop(forbidden, None)

    workload = _spawn(
        argv=LAUNCH_ARGV,
        cwd=settings.repo_root,
        environ=child_env,
        cgroup=cgroup,
        stdout_path=settings.output_root / "stdout",
        stderr_path=settings.output_root / "stderr",
    )
    observed[
        (
            int(workload.initial_identity["pid"]),
            int(workload.initial_identity["start_ticks"]),
        )
    ] = workload.initial_identity
    continuous: Optional[OwnedProcess] = None
    continuous_started = False
    continuous_rc: Optional[int] = None
    start_monotonic = time.monotonic()
    checkpoint_ready_at: Optional[float] = None
    forced_reason: Optional[str] = None

    while workload.process.poll() is None:
        _observe(cgroup, observed)
        if received_signal is not None:
            forced_reason = f"wrapper received signal {received_signal}"
            break
        if not continuous_started and settings.checkpoint_dir.is_dir():
            checkpoint_ready_at = time.time()
            continuous_started = True
            continuous = _spawn(
                argv=("bash", "scripts/ckpt_sync.sh"),
                cwd=settings.repo_root,
                environ=_sync_environment(settings, once=False),
                cgroup=cgroup,
                stdout_path=settings.output_root / "checkpoint-sync-continuous.stdout",
                stderr_path=settings.output_root / "checkpoint-sync-continuous.stderr",
            )
            observed[
                (
                    int(continuous.initial_identity["pid"]),
                    int(continuous.initial_identity["start_ticks"]),
                )
            ] = continuous.initial_identity
        elif not continuous_started and time.monotonic() - start_monotonic > 900:
            forced_reason = "checkpoint directory was not claimed within 900 seconds"
            break
        if continuous is not None and continuous.process.poll() is not None:
            continuous_rc = continuous.wait()
            continuous = None
            if settings.destination.get("kind") != "local" or continuous_rc != 0:
                forced_reason = f"continuous checkpoint sync exited early with rc {continuous_rc}"
                break
        time.sleep(0.25)

    if forced_reason is not None:
        cgroup.signal_all(signal.SIGTERM)
    workload_rc = workload.wait(timeout=60) if workload.process.poll() is not None else None
    if workload_rc is None:
        try:
            workload_rc = workload.wait(timeout=30)
        except subprocess.TimeoutExpired:
            cgroup.kill_all()
            workload_rc = workload.wait(timeout=30)

    if continuous is not None:
        continuous_rc = _stop_owned(continuous)
    _drain_cgroup(cgroup, observed)

    final_sync_rc: Optional[int] = None
    if settings.checkpoint_dir.is_dir():
        final_sync_rc = _run_final_sync(
            settings,
            cgroup,
            observed=observed,
        )
        _observe(cgroup, observed)
        _drain_cgroup(cgroup, observed)
    escapes = _known_escapes(observed)
    census_clean = not escapes and not cgroup.workload_pids()
    cgroup.close()

    checkpoint_manifest_records: list[dict[str, object]] = []
    checkpoint_manifest_error: Optional[str] = None
    if settings.checkpoint_dir.is_dir():
        try:
            checkpoint_manifest_records = _manifest_tree(settings.checkpoint_dir)
        except Exception as exc:  # retained failure evidence, never hidden
            checkpoint_manifest_error = f"{type(exc).__name__}: {exc}"
    else:
        checkpoint_manifest_error = "checkpoint directory is absent"
    _publish_json(
        settings.output_root / "checkpoint-manifest.json",
        {
            "schema": SCHEMA,
            "checkpoint_dir": str(settings.checkpoint_dir),
            "destination_sha256": destination_digest,
            "destination": settings.destination,
            "files": checkpoint_manifest_records,
            "error": checkpoint_manifest_error,
        },
    )
    _publish_json(
        settings.output_root / "checkpoint-sync-receipt.json",
        {
            "schema": SCHEMA,
            "checkpoint_dir": str(settings.checkpoint_dir),
            "checkpoint_ready_at_epoch": checkpoint_ready_at,
            "destination_sha256": destination_digest,
            "destination": settings.destination,
            "continuous_exit_code": continuous_rc,
            "final_drain_exit_code": final_sync_rc,
            "final_manifest_file_count": len(checkpoint_manifest_records),
            "qualified": final_sync_rc == 0 and checkpoint_manifest_error is None,
        },
    )
    _publish_json(
        settings.output_root / "process-census.json",
        {
            "schema": SCHEMA,
            "mechanism": "cgroup-v2-plus-subreaper",
            "cgroup_path": str(settings.cgroup_path),
            "upstream_wrapper_release": settings.containment.wrapper_release,
            "upstream_wrapper_sha256": settings.containment.wrapper_sha256,
            "authoritative": False,
            "observed": sorted(
                observed.values(),
                key=lambda item: (int(item["pid"]), int(item["start_ticks"])),
            ),
            "escaped_survivors": escapes,
            "clean": census_clean,
            "reason": "proved cgroup had no workload descendants and no observed identity survived"
            if census_clean
            else "observed descendant survived or remained in the proved cgroup",
        },
    )

    required_missing: list[str] = []
    metrics_path = settings.attempt_root / "training-metrics/events.jsonl"
    try:
        metrics_info = os.lstat(metrics_path)
    except FileNotFoundError:
        metrics_info = None
    if (
        metrics_info is None
        or not stat.S_ISREG(metrics_info.st_mode)
        or stat.S_ISLNK(metrics_info.st_mode)
        or metrics_info.st_nlink != 1
        or metrics_info.st_size == 0
    ):
        required_missing.append("training-metrics/events.jsonl:nonempty-regular")
    for relative in (f"docent/{RUN_NAME}", f"transcripts/{RUN_NAME}"):
        if not _has_nonempty_files(settings.attempt_root / relative, suffix=".jsonl"):
            required_missing.append(f"{relative}:nonempty-jsonl")
    if not checkpoint_manifest_records:
        required_missing.append("checkpoint-manifest:nonempty")
    manifest_path = settings.output_root / "evidence-manifest.json"
    workload_terminal_path = settings.output_root / "workload-terminal.json"
    evidence_manifest_error: Optional[str] = None
    try:
        _manifest_tree(
            settings.attempt_root,
            excluded={manifest_path, workload_terminal_path},
        )
    except Exception as exc:
        evidence_manifest_error = f"{type(exc).__name__}: {exc}"
    sync_qualified = final_sync_rc == 0 and checkpoint_manifest_error is None
    overall_success = (
        workload_rc == 0
        and received_signal is None
        and forced_reason is None
        and census_clean
        and sync_qualified
        and not required_missing
        and evidence_manifest_error is None
    )
    _publish_json(
        settings.output_root / "workload-terminal.json",
        {
            "schema": SCHEMA,
            "exit_code": workload_rc if workload_rc is not None and workload_rc >= 0 else None,
            "signal": (
                -workload_rc
                if workload_rc is not None and workload_rc < 0
                else received_signal
            ),
            "wrapper_signal": received_signal,
            "forced_reason": forced_reason,
            "census_clean": census_clean,
            "checkpoint_sync_qualified": sync_qualified,
            "required_evidence_missing": required_missing,
            "evidence_manifest_error": evidence_manifest_error,
            "succeeded": overall_success,
            "finished_at_epoch": time.time(),
        },
    )

    evidence_records = (
        _manifest_tree(settings.attempt_root, excluded={manifest_path})
        if evidence_manifest_error is None
        else []
    )
    _publish_json(
        manifest_path,
        {
            "schema": SCHEMA,
            "root": str(settings.attempt_root),
            "files": evidence_records,
            "checkpoint_manifest": (
                settings.output_root / "checkpoint-manifest.json"
            ).relative_to(settings.attempt_root).as_posix(),
            "error": evidence_manifest_error,
        },
    )
    return 0 if overall_success else 1


def _publish_emergency_terminal(settings: Settings, error: BaseException) -> None:
    terminal = settings.output_root / "workload-terminal.json"
    if os.path.lexists(terminal):
        return
    try:
        _publish_json(
            terminal,
            {
                "schema": SCHEMA,
                "exit_code": None,
                "signal": None,
                "wrapper_signal": None,
                "forced_reason": "inner-supervisor-failure",
                "error_type": type(error).__name__,
                "census_clean": False,
                "checkpoint_sync_qualified": False,
                "required_evidence_missing": ["inner-supervisor-completion"],
                "evidence_manifest_error": "inner-supervisor-did-not-finish",
                "succeeded": False,
                "finished_at_epoch": time.time(),
            },
        )
    except Exception:
        # The root-installed supervisor owns the authoritative terminal/census
        # and will report this inner evidence-write failure without replacement.
        pass


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args(argv)
    settings: Optional[Settings] = None
    try:
        settings = load_settings(args.repo_root, os.environ)
        return run(settings)
    except Refusal as exc:
        if settings is not None:
            _publish_emergency_terminal(settings, exc)
        print(f"scheduler_debate_supervisor: REFUSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        if settings is not None:
            _publish_emergency_terminal(settings, exc)
        print(
            f"scheduler_debate_supervisor: FAILED: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
