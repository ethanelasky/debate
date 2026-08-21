"""Durable scheduler-local training metrics.

The scheduler is the authority for ``DEBATE_ARTIFACT_ROOT``.  Debate never
accepts a YAML/config override for it: when the variable is present, one launch
claims exactly::

    <artifact-root>/<DEBATE_LAUNCH_NAMESPACE>/training-metrics/events.jsonl

The file is an append-only event stream.  Every complete record is flushed and
fsynced before control returns to the training loop, so a successful workload
cannot outrun its local analysis evidence.  The namespace-specific directory
and file are exclusively created; an existing destination is evidence of a
collision and is never resumed or overwritten.

Only scalar numeric metrics already emitted by the training logger cross this
boundary.  In particular, exception messages, environment values, config, and
credentials are never serialized.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from infra.launch_namespace import (
    ENV_VAR as LAUNCH_NAMESPACE_ENV,
    claim_directory,
    open_claimed_text_file,
    validate_launch_namespace,
)


ARTIFACT_ROOT_ENV = "DEBATE_ARTIFACT_ROOT"
_OUTPUT_DIRECTORY = "training-metrics"
_OUTPUT_FILENAME = "events.jsonl"
_SCHEMA = "debate-training-metrics/v1"
_METRIC_KEY_RE = re.compile(r"[A-Za-z0-9_./:-]{1,256}")


def _artifact_root(environ: Mapping[str, str]) -> str | None:
    if ARTIFACT_ROOT_ENV not in environ:
        return None
    root = environ[ARTIFACT_ROOT_ENV]
    if not root:
        raise ValueError(f"{ARTIFACT_ROOT_ENV} must not be empty")
    if "\x00" in root or not os.path.isabs(root):
        raise ValueError(
            f"{ARTIFACT_ROOT_ENV} must be an absolute retained-volume path"
        )
    try:
        root_stat = os.lstat(root)
    except FileNotFoundError:
        raise ValueError(
            f"{ARTIFACT_ROOT_ENV} must name an existing scheduler-owned directory"
        ) from None
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise ValueError(
            f"{ARTIFACT_ROOT_ENV} must name a real directory, not a link or file"
        )
    filesystem_root = os.stat(os.path.sep)
    if (root_stat.st_dev, root_stat.st_ino) == (
        filesystem_root.st_dev,
        filesystem_root.st_ino,
    ):
        raise ValueError(f"{ARTIFACT_ROOT_ENV} must not be the filesystem root")
    if root_stat.st_uid != os.geteuid() or root_stat.st_mode & (
        stat.S_IWGRP | stat.S_IWOTH
    ):
        raise ValueError(
            f"{ARTIFACT_ROOT_ENV} must be owner-controlled and not group/world writable"
        )
    return root


def scheduler_artifact_attempt_root(
    launch_namespace: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Derive the scheduler-owned attempt root without creating anything."""
    source = os.environ if environ is None else environ
    root = _artifact_root(source)
    if root is None:
        return None
    resolved = validate_launch_namespace(launch_namespace)
    if LAUNCH_NAMESPACE_ENV not in source:
        raise ValueError(
            f"{ARTIFACT_ROOT_ENV} requires scheduler-owned {LAUNCH_NAMESPACE_ENV}"
        )
    scheduler_namespace = validate_launch_namespace(source[LAUNCH_NAMESPACE_ENV])
    if resolved != scheduler_namespace:
        raise ValueError(
            "resolved training namespace does not match scheduler-owned "
            f"{LAUNCH_NAMESPACE_ENV}: resolved={resolved!r}, "
            f"scheduler={scheduler_namespace!r}"
        )
    return os.path.join(root, resolved)


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, int | float]:
    """Return the exact scalar metric values, rejecting secret-capable shapes."""
    out: dict[str, int | float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TypeError("local metric evidence keys must be strings")
        if _METRIC_KEY_RE.fullmatch(key) is None:
            raise ValueError(
                "local metric evidence keys must be 1-256 safe metric-name "
                f"characters, got {key!r}"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                "local metric evidence accepts only scalar numeric values; "
                f"metric {key!r} has type {type(value).__name__}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"local metric evidence requires finite JSON numbers; metric {key!r} is nonfinite"
            )
        out[key] = value
    return out


class LocalMetricsEvidence:
    """Exclusive, append-only JSONL evidence for one launch namespace."""

    __slots__ = (
        "launch_namespace",
        "output_dir",
        "path",
        "_file",
        "_sequence",
        "_finalized",
        "_poisoned",
    )

    def __init__(self, artifact_root: str, launch_namespace: str):
        namespace = validate_launch_namespace(launch_namespace)
        output_dir = os.path.join(
            artifact_root, namespace, _OUTPUT_DIRECTORY
        )
        claimed = claim_directory(output_dir)
        handle = None
        try:
            handle = open_claimed_text_file(claimed, _OUTPUT_FILENAME)
            fd = handle.fileno()
            os.fchmod(fd, 0o600)
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_APPEND)
        except BaseException:
            if handle is not None:
                handle.close()
            raise

        self.launch_namespace = namespace
        self.output_dir = str(claimed)
        self.path = str(Path(claimed) / _OUTPUT_FILENAME)
        self._file = handle
        self._sequence = 0
        self._finalized = False
        self._poisoned = False
        try:
            self._append({"event": "started"})
            # Durably link the file, training-metrics directory, and namespace
            # into their respective parents.  claim_directory itself provides
            # exclusive/no-follow authority but does not fsync parent entries.
            self._fsync_directory(claimed, mode=0o700)
            self._fsync_directory(Path(claimed).parent)
            self._fsync_directory(artifact_root)
        except BaseException:
            self._file.close()
            raise

    @staticmethod
    def _fsync_directory(
        path: str | os.PathLike[str], *, mode: int | None = None
    ) -> None:
        directory_fd = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            opened = os.fstat(directory_fd)
            if not stat.S_ISDIR(opened.st_mode):
                raise RuntimeError(f"evidence path is not a directory: {path}")
            if mode is not None:
                os.fchmod(directory_fd, mode)
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @classmethod
    def from_environment(
        cls,
        launch_namespace: str,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> LocalMetricsEvidence | None:
        source = os.environ if environ is None else environ
        attempt_root = scheduler_artifact_attempt_root(
            launch_namespace, environ=source
        )
        if attempt_root is None:
            return None
        return cls(str(Path(attempt_root).parent), launch_namespace)

    def _append(self, fields: dict[str, Any]) -> None:
        if self._poisoned:
            raise RuntimeError(
                "local metric evidence is poisoned after an uncertain append"
            )
        if self._file.closed:
            raise RuntimeError("local metric evidence is already closed")
        record = {
            "schema": _SCHEMA,
            "sequence": self._sequence,
            "launch_namespace": self.launch_namespace,
            **fields,
        }
        encoded = (
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
        view = memoryview(encoded)
        fd = self._file.fileno()
        try:
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError(
                        "short write while appending local metric evidence"
                    )
                view = view[written:]
            os.fsync(fd)
        except BaseException:
            # Never append another record after an uncertain/partial append;
            # that could concatenate valid terminal JSON onto a torn fragment.
            self._poisoned = True
            raise
        self._sequence += 1

    def metrics(self, step: int, metrics: dict[str, Any]) -> None:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(
                f"metric step must be a nonnegative integer, got {step!r}"
            )
        self._append(
            {
                "event": "metrics",
                "step": step,
                "metrics": _numeric_metrics(metrics),
            }
        )

    def finalize(self, *, succeeded: bool) -> None:
        """Append exactly one terminal record and close the owned descriptor."""
        if self._finalized:
            return
        if self._poisoned:
            self._file.close()
            return
        fields: dict[str, Any] = {
            "event": "finalized",
            "status": "succeeded" if succeeded else "failed",
        }
        try:
            self._append(fields)
            self._finalized = True
        finally:
            self._file.close()
