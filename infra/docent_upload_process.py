#!/usr/bin/env python3
"""Dedicated, disposable Docent uploader entered only through fixed FDs.

Keep this module's top level stdlib-only: the hard deadline is installed from
the inherited protocol before Docent or debate code can import.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import signal
import stat
import sys
import time
from typing import Any

_DOCENT_JSONL_FD = 198
_DOCENT_RECEIPT_FD = 199
_DOCENT_TOTAL_BUDGET_SECONDS = 5.0 * 60.0


class _ChildHardDeadline(BaseException):
    pass


def _raise_deadline(_signum: int, _frame: Any) -> None:
    raise _ChildHardDeadline()


def _validate_fixed_fds() -> None:
    jsonl_stat = os.fstat(_DOCENT_JSONL_FD)
    receipt_stat = os.fstat(_DOCENT_RECEIPT_FD)
    if not stat.S_ISREG(jsonl_stat.st_mode):
        raise ValueError("Docent JSONL FD is not a regular file")
    if not stat.S_ISFIFO(receipt_stat.st_mode):
        raise ValueError("Docent receipt FD is not a pipe")
    if (fcntl.fcntl(_DOCENT_JSONL_FD, fcntl.F_GETFL) & os.O_ACCMODE) != os.O_RDONLY:
        raise ValueError("Docent JSONL FD is not read-only")
    if (fcntl.fcntl(_DOCENT_RECEIPT_FD, fcntl.F_GETFL) & os.O_ACCMODE) != os.O_WRONLY:
        raise ValueError("Docent receipt FD is not write-only")
    if os.lseek(_DOCENT_JSONL_FD, 0, os.SEEK_CUR) != 0:
        raise ValueError("Docent JSONL FD did not start at offset zero")


def _load_canonical_runs(agent_run_type: type[Any]) -> list[Any]:
    runs: list[Any] = []
    with os.fdopen(os.dup(_DOCENT_JSONL_FD), "rb", closefd=True) as source:
        for raw_line in source:
            if not raw_line.endswith(b"\n") or raw_line.strip() == b"":
                raise ValueError("Docent JSONL contains a blank or trailing fragment")
            source_bytes = raw_line[:-1]
            run = agent_run_type.model_validate_json(source_bytes)
            if run.model_dump_json().encode("utf-8") != source_bytes:
                raise ValueError("Docent JSONL is not canonical AgentRun encoding")
            # The pinned SDK distinguishes generated IDs from caller-supplied
            # IDs using Pydantic's fields-set metadata. JSON round-tripping
            # marks the already-generated IDs explicit even though their bytes
            # are authoritative local evidence. Restore only that provenance
            # bit; values and serialized scientific bytes remain identical.
            run.__pydantic_fields_set__.discard("id")
            for transcript in run.transcripts:
                transcript.__pydantic_fields_set__.discard("id")
            for group in run.transcript_groups:
                group.__pydantic_fields_set__.discard("id")
            if run.model_dump_json().encode("utf-8") != source_bytes:
                raise ValueError("Docent AgentRun bytes changed during ID handoff")
            runs.append(run)
    if not runs:
        raise ValueError("Docent JSONL contains no AgentRun")
    return runs


def _write_receipt(frame: dict[str, Any]) -> None:
    payload = (
        json.dumps(frame, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
        + b"\n"
    )
    pipe_buf = os.fpathconf(_DOCENT_RECEIPT_FD, "PC_PIPE_BUF")
    if len(payload) > min(4096, pipe_buf):
        raise ValueError("Docent receipt frame exceeds the atomic pipe boundary")
    view = memoryview(payload)
    while view:
        try:
            written = os.write(_DOCENT_RECEIPT_FD, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise BrokenPipeError("Docent receipt pipe closed")
        view = view[written:]


def main() -> int:
    collection_name = "invalid"
    launch_namespace = "invalid"
    collection_id: str | None = None
    alarm_installed = False
    try:
        _validate_fixed_fds()
        deadline = float(os.environ["DEBATE_DOCENT_DEADLINE_MONOTONIC"])
        remaining = deadline - time.monotonic()
        if (
            not math.isfinite(deadline)
            or remaining <= 0
            or remaining > _DOCENT_TOTAL_BUDGET_SECONDS
        ):
            raise ValueError("Docent child deadline is invalid")
        signal.signal(signal.SIGALRM, _raise_deadline)
        signal.setitimer(signal.ITIMER_REAL, remaining, 0.0)
        alarm_installed = True

        # Isolated mode deliberately omits both cwd and PYTHONPATH. The worker
        # derives this reviewed absolute project root only after its alarm is live.
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, os.fspath(project_root))

        from docent.data_models import AgentRun
        from infra.envs.debate.docent_export import (
            _DocentTimeBudget,
            _upload_in_child_process,
            DocentUploadFailure,
            collection_name_for_launch,
            validate_launch_namespace,
        )

        launch_namespace = validate_launch_namespace(
            os.environ["DEBATE_DOCENT_LAUNCH_NAMESPACE"]
        )
        collection_name = os.environ["DEBATE_DOCENT_COLLECTION_NAME"]
        # Reconstructing from its base proves the name contains the exact
        # namespace without accepting a second protocol identity.
        suffix = f"--launch-{launch_namespace}"
        if not collection_name.endswith(suffix):
            raise ValueError("Docent collection name does not contain its namespace")
        base_name = collection_name[: -len(suffix)]
        if collection_name_for_launch(base_name, launch_namespace) != collection_name:
            raise ValueError("Docent collection name is not canonical")
        runs = _load_canonical_runs(AgentRun)
        if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
            raise RuntimeError("Docent upload child imported torch")

        def confirmed(value: str) -> None:
            nonlocal collection_id
            collection_id = value
            _write_receipt({"event": "collection", "id": value})

        result = _upload_in_child_process(
            runs,
            collection_name,
            launch_namespace,
            _budget=_DocentTimeBudget(
                deadline=deadline,
                clock=time.monotonic,
                sleeper=time.sleep,
            ),
            on_collection_confirmed=confirmed,
        )
        if result.collection_id is not None:
            collection_id = result.collection_id
        if isinstance(result, DocentUploadFailure):
            _write_receipt(
                {"event": "terminal", "ok": False, "error": result.error_type}
            )
            return 2
        _write_receipt({"event": "terminal", "ok": True})
        return 0
    except _ChildHardDeadline:
        error_type = "DocentUploadDeadlineError"
    except BaseException as exc:
        name = type(exc).__name__
        error_type = (
            name
            if isinstance(name, str)
            and name.isidentifier()
            and len(name) <= 128
            else "UnknownError"
        )
    finally:
        if alarm_installed:
            signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
    try:
        _write_receipt({"event": "terminal", "ok": False, "error": error_type})
    except BaseException:
        pass
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
