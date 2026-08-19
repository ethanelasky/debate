"""Export debate rollouts to Docent (transluce) for transcript analysis.

Comprehensive by design: one AgentRun per debate carrying
  - an "omniscient" transcript: every slot in flat order, including private /
    ephemeral slots and thinking (as reasoning content) — the analyst's view;
  - one transcript per speaker: the EXACT rendered context of that speaker's
    final generation (regenerated via render_context, which is a pure function
    of the record store) plus its final response — "what the model saw";
  - metadata: task, verdict (winner, per-seat rulings, BOTH confidence
    channels + logit status), rewards, failure state, bindings.

Usage:
    runs = agent_runs(env)                      # after env.rollout(...)
    export_jsonl(runs, "debates.jsonl")         # local file
    export_jsonl_claimed(runs, directory, name) # reserved launch sink
    fd = open_claimed_read_fd(directory, "docent.jsonl")
    upload(fd, base_collection_name="math-pc-rl", launch_namespace=namespace)
                                                # needs DOCENT_API_KEY; close fd
"""

from __future__ import annotations

import json
import fcntl
import os
import re
import select
import signal
import stat
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import (
    AssistantMessage,
    ContentReasoning,
    ContentText,
    SystemMessage,
    UserMessage,
)

from infra.envs.debate.round import DebateState, render_context
from infra.envs.debate.protocol import Kind
from infra.launch_namespace import open_claimed_text_file, validate_launch_namespace


_REVIEWED_DOCENT_VERSION = "0.1.77"
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_COLLECTION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._~-]{0,127}")
_JOB_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._~-]{0,255}")
_DOCENT_CONNECT_TIMEOUT_SECONDS = 10.0
_DOCENT_READ_TIMEOUT_SECONDS = 120.0
_DOCENT_TOTAL_BUDGET_SECONDS = 5.0 * 60.0
_DOCENT_STATUS_CHUNK_SIZE = 100
_DOCENT_JSONL_FD = 198
_DOCENT_RECEIPT_FD = 199
_DOCENT_RECEIPT_MAX_BYTES = 16 * 1024
_DOCENT_RECEIPT_MAX_LINE_BYTES = 4096
_DOCENT_LOCAL_TERMINATION_GRACE_SECONDS = 0.25
_DOCENT_FORCE_KILL_REAP_RESERVE_SECONDS = 0.05
_DOCENT_WORKER_PATH = (
    Path(__file__).resolve().parents[2] / "docent_upload_process.py"
)
_DOCENT_SPAWN_LOCK = threading.Lock()


class DocentMutationRedirectError(RuntimeError):
    """A Docent mutation tried to redirect after its one transport send."""


class DocentMutationHTTPStatusError(RuntimeError):
    """A Docent POST returned a non-success status after its one send."""


class DocentIngestionAcknowledgementError(RuntimeError):
    """Docent did not confirm every submitted ingestion batch."""


class DocentIngestionStatusError(RuntimeError):
    """Docent returned an incomplete, malformed, or failed status census."""


class DocentCollectionAcknowledgementError(RuntimeError):
    """Docent returned no bounded URL-safe collection identity."""


class DocentUploadDeadlineError(TimeoutError):
    """The fixed total Docent upload budget was exhausted."""


class DocentUploadCleanupError(RuntimeError):
    """The dedicated upload process group could not be safely cleaned up."""


class DocentUploadChildOwnershipError(RuntimeError):
    """The supervisor lost authoritative wait/reap ownership of its child."""


@dataclass(frozen=True, slots=True)
class DocentUploadResult:
    """Sanitized confirmed identity for one completed external upload."""

    collection_name: str
    launch_namespace: str
    collection_id: str


@dataclass(frozen=True, slots=True)
class DocentUploadFailure:
    """Sanitized failure result; it deliberately has no traceback or cause."""

    collection_name: str
    launch_namespace: str
    collection_id: str | None
    error_type: str


@dataclass(frozen=True, slots=True)
class DocentUploadControlFlow:
    """Sanitized request to re-raise operator control flow after a receipt."""

    failure: DocentUploadFailure
    kind: str
    exit_code: int | None = None


@dataclass(slots=True)
class _DocentPostTracker:
    agent_batch_posts: int = 0
    job_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _DocentTimeBudget:
    """One monotonic deadline shared by auth, mutations, and polling.

    Manual uploads use the approved fixed five-minute budget.  Clock and
    sleeper injection are private test seams, not operator configuration.
    """

    deadline: float
    clock: Callable[[], float]
    sleeper: Callable[[float], None]

    @classmethod
    def fixed(
        cls,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> "_DocentTimeBudget":
        clock = time.monotonic if clock is None else clock
        sleeper = time.sleep if sleeper is None else sleeper
        return cls(
            deadline=clock() + _DOCENT_TOTAL_BUDGET_SECONDS,
            clock=clock,
            sleeper=sleeper,
        )

    def remaining(self) -> float:
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise DocentUploadDeadlineError("Docent upload deadline exhausted")
        return remaining

    def request_timeout(self) -> tuple[float, float]:
        remaining = self.remaining()
        return (
            min(_DOCENT_CONNECT_TIMEOUT_SECONDS, remaining),
            min(_DOCENT_READ_TIMEOUT_SECONDS, remaining),
        )

    def sleep(self, seconds: float) -> None:
        remaining = self.remaining()
        self.sleeper(min(seconds, remaining))


def _sanitized_error_type(exc: BaseException) -> str:
    name = type(exc).__name__
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", name) is None:
        return "UnknownError"
    return name


def _assistant(text: str, thinking: Optional[str] = None, **meta) -> AssistantMessage:
    content = []
    if thinking:
        content.append(ContentReasoning(reasoning=thinking))
    content.append(ContentText(text=text))
    return AssistantMessage(content=content, metadata=meta or {})


def _answer_format_valid(env, state: DebateState, rec) -> bool:
    """The parsing-only format result recorded by the rollout census.

    Explicit ``states=`` callers may export states that did not pass through
    DebateEnv.rollout, so fall back to the same family parser when no retained
    census value exists.
    """
    retained = state.meta.get("solution_answer_format_valid", {})
    key = str(rec.slot.index)
    if key in retained:
        return bool(retained[key])
    return bool(env.family.parse_answers(rec.text).answer_format_valid)


def _omniscient_transcript(state: DebateState, env) -> Transcript:
    messages: list[Any] = [
        SystemMessage(
            content="Omniscient debate view: every slot in generation order, "
            "including private/ephemeral slots and reasoning no other "
            "participant saw."
        )
    ]
    for rec in state.records:
        cs = rec.slot
        metadata = {
            "speaker": cs.speaker,
            "display_name": state.bindings.get(cs.speaker, {}).get("NAME", cs.speaker),
            "turn": cs.turn,
            "slot": cs.slot.name,
            "kind": cs.slot.kind.value,
            "visibility": cs.slot.visibility.value,
            "extracted": None if rec.extracted is None else str(rec.extracted),
            "retries": rec.retries,
        }
        if cs.slot.kind == Kind.SOLUTION:
            metadata["answer_format_valid"] = _answer_format_valid(env, state, rec)
        messages.append(
            _assistant(
                rec.text,
                thinking=rec.thinking,
                **metadata,
            )
        )
    return Transcript(name="omniscient", messages=messages)


def _speaker_view(state: DebateState, speaker: str, env) -> Optional[Transcript]:
    last = None
    for rec in state.records:
        if rec.slot.speaker == speaker:
            last = rec
    if last is None:
        return None
    rendered = render_context(state_before(state, last), last.slot, env.prompts)
    messages: list[Any] = []
    for m in rendered:
        role = m["role"]
        if role == "system":
            messages.append(SystemMessage(content=m["content"]))
        elif role == "user":
            messages.append(UserMessage(content=m["content"]))
        else:
            messages.append(_assistant(m["content"]))
    metadata = {"slot": last.slot.slot.name}
    if last.slot.slot.kind == Kind.SOLUTION:
        metadata["answer_format_valid"] = _answer_format_valid(env, state, last)
    messages.append(_assistant(last.text, thinking=last.thinking, **metadata))
    return Transcript(name=f"view:{speaker}", messages=messages)


def _solo_turn1_view(state: DebateState, env) -> Optional[Transcript]:
    """The slot-0 GENERATION context, verbatim: what the solo author actually
    saw when producing its first speech (first_speech_non_debate_aware — no
    debate framing), plus that speech. The author's regular view presents the
    same answer under the debate framing instead, so without this transcript
    the context that really elicited turn 1 never reaches the export."""
    solo = state.first_slot_messages
    if solo is None:
        return None
    rec = next((r for r in state.records if r.slot.index == 0), None)
    if rec is None:
        return None
    messages: list[Any] = []
    for m in solo:
        if m["role"] == "system":
            messages.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            messages.append(UserMessage(content=m["content"]))
        else:
            messages.append(_assistant(m["content"]))
    metadata = {
        "slot": rec.slot.slot.name,
        "extracted": None if rec.extracted is None else str(rec.extracted),
        "retries": rec.retries,
    }
    if rec.slot.slot.kind == Kind.SOLUTION:
        metadata["answer_format_valid"] = _answer_format_valid(env, state, rec)
    messages.append(_assistant(rec.text, thinking=rec.thinking, **metadata))
    return Transcript(name=f"view:{rec.slot.speaker}@turn1", messages=messages)


def state_before(state: DebateState, rec) -> DebateState:
    """A shallow view of the state as it was before `rec` was generated.
    dataclasses.replace so every field (present and future) carries over —
    a hand-copied clone here silently dropped first_slot_messages once."""
    return replace(state, records=[r for r in state.records if r.slot.index < rec.slot.index])


def _verdict_metadata(env, state: DebateState) -> dict[str, Any]:
    v = env._verdict(state)
    if v is None:
        return {"verdict": None}
    return {
        "verdict": {
            "ok": v.ok,
            "winner": v.winner,
            "seats": {k: s.value for k, s in v.seats.items()},
            "confidence": {
                k: {"json": c.json, "logit": c.logit, "logit_status": c.logit_status.name}
                for k, c in v.confidence.items()
            },
            "attempts": v.attempts,
        }
    }


def agent_runs(env, states: Optional[list[DebateState]] = None) -> list[AgentRun]:
    """Build one AgentRun per debate from the env's last rollout (or given
    states). Includes failed debates — those are often the interesting ones."""
    states = states if states is not None else getattr(env, "last_states", [])
    runs = []
    for i, st in enumerate(states):
        transcripts = [_omniscient_transcript(st, env)]
        solo_view = _solo_turn1_view(st, env)
        if solo_view is not None:
            transcripts.append(solo_view)
        for speaker in env.protocol.speakers:
            view = _speaker_view(st, speaker, env)
            if view is not None:
                transcripts.append(view)
        meta: dict[str, Any] = {
            "debate_index": i,
            "failed": st.failed,
            "flipped": bool(st.meta.get("flipped", False)),
            "empty_slots": st.meta.get("empty_slots", []),
            # This is the source-owned, already-redacted transcript boundary.
            # Never fall back to raw st.meta["task"]: it carries private
            # verifier material (including scalar secrets in some families).
            "task": dict(st.meta.get("task_export", {})),
            "bindings": {s: b.get("NAME") for s, b in st.bindings.items()},
            # Ground-truth label per solution slot. Private verifier inputs do
            # not cross task_export, so this is the record of correctness that
            # survives export — see DebateEnv._attach_labels.
            "grades": st.meta.get("grades", {}),
        }
        if st.failed is None:
            meta.update(_verdict_metadata(env, st))
        runs.append(
            AgentRun(
                name=f"debate-{i}",
                description=str(st.meta.get("task_export", {}).get("question", ""))[:300],
                transcripts=transcripts,
                metadata=meta,
            )
        )
    return runs


def export_jsonl(runs: list[AgentRun], path: str) -> str:
    # Standalone callers still get exclusive create rather than truncation;
    # production launch sinks use export_jsonl_claimed below for retained
    # directory authority as well.
    with open(path, "x") as f:
        for run in runs:
            f.write(run.model_dump_json() + "\n")
    return path


def export_jsonl_claimed(
    runs: list[AgentRun],
    directory: str | os.PathLike[str],
    filename: str,
) -> str:
    """Write the unchanged AgentRun JSONL bytes through claimed authority.

    Production launch sinks use this descriptor-relative boundary so an
    unclaimed directory cannot be adopted and a replaced ancestor cannot
    redirect the local transcript write. ``export_jsonl`` remains the
    standalone one-file convenience API for callers without a launch sink.
    """
    with open_claimed_text_file(directory, filename) as f:
        for run in runs:
            f.write(run.model_dump_json() + "\n")
    return os.path.join(os.fspath(directory), filename)


def validate_base_collection_name(base_collection_name: str) -> str:
    """Validate the operator-supplied collection base before any rollout."""
    if not isinstance(base_collection_name, str) or not base_collection_name.strip():
        raise ValueError("Docent base collection name must be a non-empty string")
    return base_collection_name


def collection_name_for_launch(base_collection_name: str, launch_namespace: str) -> str:
    """Name one external Docent collection for exactly one launch attempt.

    The base is the operator-facing experiment name.  The validated namespace
    is preserved exactly: it selects the collection, but never enters an
    ``AgentRun`` or the local transcript bytes.
    """
    base_collection_name = validate_base_collection_name(base_collection_name)
    namespace = validate_launch_namespace(launch_namespace)
    return f"{base_collection_name}--launch-{namespace}"


def _validate_docent_sdk_class(client_type: type[Any]) -> None:
    """Validate pinned class-level seams without constructing a client."""
    try:
        installed_version = version("docent")
        shim_version = version("docent-python")
    except PackageNotFoundError as exc:
        raise RuntimeError("cannot verify the installed Docent SDK") from exc
    add_impl = getattr(client_type, "add_agent_runs", None)
    create_impl = getattr(client_type, "create_collection", None)
    init_impl = getattr(client_type, "__init__", None)
    login_impl = getattr(client_type, "_login", None)
    retry_impl = getattr(client_type, "_post_with_retry", None)
    wait_impl = getattr(client_type, "_wait_for_jobs", None)
    statuses_impl = getattr(client_type, "get_agent_run_job_statuses", None)
    add_code = getattr(add_impl, "__code__", None)
    create_code = getattr(create_impl, "__code__", None)
    init_code = getattr(init_impl, "__code__", None)
    login_code = getattr(login_impl, "__code__", None)
    wait_code = getattr(wait_impl, "__code__", None)
    statuses_code = getattr(statuses_impl, "__code__", None)
    add_parameters = (
        signature(add_impl).parameters if callable(add_impl) else {}
    )
    if (
        installed_version != _REVIEWED_DOCENT_VERSION
        or shim_version != _REVIEWED_DOCENT_VERSION
    ):
        raise RuntimeError(
            "refusing unreviewed Docent SDK version for no-repeat upload: "
            f"expected {_REVIEWED_DOCENT_VERSION}, got "
            f"docent={installed_version}, docent-python={shim_version}"
        )
    if (
        init_code is None
        or "_login" not in init_code.co_names
        or "Session" not in init_code.co_names
        or getattr(init_impl, "__module__", None) != "docent.sdk._base"
        or login_code is None
        or "_session" not in login_code.co_names
        or "get" not in login_code.co_names
        or "_handle_response_errors" not in login_code.co_names
        or getattr(login_impl, "__module__", None) != "docent.sdk._base"
        or tuple(signature(login_impl).parameters) != ("self", "api_key")
    ):
        raise RuntimeError(
            "Docent authentication seam no longer supports the reviewed bounded login"
        )
    if (
        add_code is None
        or "_post_with_retry" not in add_code.co_names
        or "_session" in add_code.co_names
        or "_wait_for_jobs" not in add_code.co_names
        or getattr(add_impl, "__module__", None) != "docent.sdk._collections"
        or "wait" not in add_parameters
        or add_parameters["wait"].default is not True
        or not callable(retry_impl)
    ):
        raise RuntimeError(
            "Docent add_agent_runs no longer uses only the reviewed retry hook"
        )
    if (
        create_code is None
        or "_session" not in create_code.co_names
        or "post" not in create_code.co_names
        or "_handle_response_errors" not in create_code.co_names
        or "_post_with_retry" in create_code.co_names
        or getattr(create_impl, "__module__", None) != "docent.sdk._collections"
    ):
        raise RuntimeError(
            "Docent create_collection no longer uses the reviewed one-POST seam"
        )
    retry_parameter = signature(retry_impl).parameters.get("max_retries")
    if (
        getattr(retry_impl, "__module__", None) != "docent.sdk._base"
        or retry_parameter is None
        or retry_parameter.kind
        not in {
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        }
    ):
        raise RuntimeError(
            "Docent retry hook cannot enforce max_retries=0; refusing upload"
        )
    wait_parameters = (
        signature(wait_impl).parameters if callable(wait_impl) else {}
    )
    statuses_parameters = (
        signature(statuses_impl).parameters if callable(statuses_impl) else {}
    )
    if (
        wait_code is None
        or "get_agent_run_job_statuses" not in wait_code.co_names
        or getattr(wait_impl, "__module__", None) != "docent.sdk._collections"
        or tuple(wait_parameters) != (
            "self",
            "collection_id",
            "job_ids",
            "poll_interval",
        )
        or wait_parameters["poll_interval"].default != 1.0
    ):
        raise RuntimeError("Docent ingestion wait seam no longer matches the reviewed SDK")
    if (
        statuses_code is None
        or "_session" not in statuses_code.co_names
        or "post" not in statuses_code.co_names
        or "_handle_response_errors" not in statuses_code.co_names
        or getattr(statuses_impl, "__module__", None) != "docent.sdk._collections"
        or tuple(statuses_parameters) != ("self", "collection_id", "job_ids")
    ):
        raise RuntimeError("Docent ingestion status seam no longer matches the reviewed SDK")


def _validate_single_send_session(session: Any) -> requests.Session:
    """Require a real Session whose adapters cannot retry any HTTP method."""
    if not isinstance(session, requests.Session):
        raise RuntimeError("Docent SDK no longer uses the reviewed requests.Session")
    if not session.adapters:
        raise RuntimeError("Docent requests.Session has no transport adapters")
    for prefix, adapter in session.adapters.items():
        adapter_retries = getattr(adapter, "max_retries", None)
        if (
            not isinstance(adapter, HTTPAdapter)
            or not isinstance(adapter_retries, Retry)
            or adapter_retries.total != 0
        ):
            raise RuntimeError(
                "Docent transport adapter can retry HTTP calls; refusing upload "
                f"for adapter {prefix!r}"
            )
    return session


def _bounded_docent_type(client_type: type[Any], budget: _DocentTimeBudget) -> type[Any]:
    """Override login before the pinned constructor makes its auth request."""

    class BoundedDocent(client_type):
        def _login(self, api_key: str) -> None:
            session = _validate_single_send_session(self._session)
            # The dedicated child has an allowlisted environment, and the
            # pinned Session additionally refuses ambient proxy/netrc/CA
            # discovery before its first authentication request.
            session.trust_env = False
            session.headers.update({"Authorization": f"Bearer {api_key}"})
            response = session.get(
                f"{self._api_url}/api-keys/test",
                allow_redirects=False,
                timeout=budget.request_timeout(),
            )
            # A per-read socket timeout is not itself a total runtime bound.
            # Recheck the monotonic budget after the complete response arrives.
            budget.remaining()
            if response.status_code in _REDIRECT_STATUS_CODES:
                raise DocentMutationRedirectError(
                    "Docent authentication redirect refused after one transport send"
                )
            if not 200 <= response.status_code < 300:
                raise DocentMutationHTTPStatusError(
                    "Docent authentication returned a non-success status after one send"
                )

    BoundedDocent.__name__ = "BoundedDocent"
    BoundedDocent.__qualname__ = "BoundedDocent"
    return BoundedDocent


def _disable_pinned_docent_tqdm_monitor_for_child(client_type: type[Any]) -> None:
    """Disable the pinned SDK's monitor in the disposable upload child."""
    add_impl = getattr(client_type, "add_agent_runs", None)
    add_globals = getattr(add_impl, "__globals__", None)
    tqdm_type = add_globals.get("tqdm") if isinstance(add_globals, dict) else None
    if (
        not isinstance(tqdm_type, type)
        or getattr(tqdm_type, "__module__", None) != "tqdm.std"
        or getattr(tqdm_type, "__name__", None) != "tqdm"
        or not hasattr(tqdm_type, "monitor_interval")
        or not hasattr(tqdm_type, "monitor")
    ):
        raise RuntimeError("Docent tqdm seam no longer matches the reviewed SDK")

    previous_interval = tqdm_type.monitor_interval
    if (
        isinstance(previous_interval, bool)
        or not isinstance(previous_interval, (int, float))
        or previous_interval < 0
    ):
        raise RuntimeError("Docent tqdm monitor interval is not safely controllable")
    # This process is dedicated to one upload and exits immediately after it,
    # so no restoration is needed and no monitor thread can outlive the child.
    tqdm_type.monitor_interval = 0


def _bind_single_attempt_mutation_posts(
    client: Any,
    *,
    class_validated: bool = False,
    _budget: _DocentTimeBudget | None = None,
) -> _DocentPostTracker:
    """Make every collection/create and agent-run POST one transport send.

    Docent 0.1.77 routes every ingestion batch through ``_post_with_retry``.
    Its default repeats ambiguous 5xx responses three times, which can
    duplicate a persisted batch. Requests also follows POST redirects by
    default. Bind the reviewed hook to ``max_retries=0``, force
    ``allow_redirects=False`` on the shared Session, and reject an unreviewed
    SDK/session seam before creating the collection.
    """
    if not class_validated:
        _validate_docent_sdk_class(type(client))

    budget = _DocentTimeBudget.fixed() if _budget is None else _budget
    session = _validate_single_send_session(getattr(client, "_session", None))

    original_retry = client._post_with_retry
    original_session_post = session.post
    original_statuses = client.get_agent_run_job_statuses
    tracker = _DocentPostTracker()

    def post_without_redirect(url: str, **kwargs: Any):
        if kwargs.get("allow_redirects") not in {None, False}:
            raise RuntimeError("Docent SDK requested mutation redirects; refusing upload")
        if "timeout" in kwargs:
            raise RuntimeError("Docent SDK supplied an unexpected HTTP timeout")
        kwargs["allow_redirects"] = False
        kwargs["timeout"] = budget.request_timeout()
        response = original_session_post(url, **kwargs)
        if response.status_code in _REDIRECT_STATUS_CODES:
            raise DocentMutationRedirectError(
                "Docent mutation redirect refused after one transport send"
            )
        if not 200 <= response.status_code < 300:
            raise DocentMutationHTTPStatusError(
                "Docent mutation returned a non-success status after one transport send"
            )
        return response

    def post_once(url: str, **kwargs: Any):
        if "max_retries" in kwargs:
            raise RuntimeError("Docent SDK supplied an unexpected retry override")
        tracker.agent_batch_posts += 1
        response = original_retry(url, max_retries=0, **kwargs)
        budget.remaining()
        try:
            body = response.json()
        except Exception:
            raise DocentIngestionAcknowledgementError(
                "Docent ingestion batch response was not valid JSON"
            ) from None
        job_id = body.get("job_id") if isinstance(body, dict) else None
        if (
            not isinstance(job_id, str)
            or _JOB_ID_RE.fullmatch(job_id) is None
            or job_id in tracker.job_ids
        ):
            raise DocentIngestionAcknowledgementError(
                "Docent ingestion batch returned no unique bounded job identifier"
            )
        tracker.job_ids.append(job_id)
        return response

    def wait_for_jobs(
        collection_id: str,
        job_ids: list[str],
        poll_interval: float = 1.0,
    ) -> None:
        """Pinned fail-closed replacement for Docent 0.1.77's wait loop."""
        if (
            not isinstance(job_ids, list)
            or not job_ids
            or any(
                not isinstance(job_id, str)
                or _JOB_ID_RE.fullmatch(job_id) is None
                for job_id in job_ids
            )
            or len(set(job_ids)) != len(job_ids)
        ):
            raise DocentIngestionStatusError(
                "Docent wait received malformed ingestion job identifiers"
            )

        pending = set(job_ids)
        while pending:
            requested = [job_id for job_id in job_ids if job_id in pending]
            next_pending: set[str] = set()
            for offset in range(0, len(requested), _DOCENT_STATUS_CHUNK_SIZE):
                chunk = requested[offset : offset + _DOCENT_STATUS_CHUNK_SIZE]
                try:
                    rows = original_statuses(collection_id, chunk)
                    budget.remaining()
                except (requests.Timeout, DocentUploadDeadlineError):
                    raise
                except Exception:
                    raise DocentIngestionStatusError(
                        "Docent status request did not return a validated census"
                    ) from None
                if not isinstance(rows, list) or len(rows) != len(chunk):
                    raise DocentIngestionStatusError(
                        "Docent status response did not exactly cover pending jobs"
                    )

                by_id: dict[str, str] = {}
                for row in rows:
                    if not isinstance(row, dict):
                        raise DocentIngestionStatusError(
                            "Docent status response contained a malformed row"
                        )
                    job_id = row.get("job_id")
                    status = row.get("status")
                    if (
                        not isinstance(job_id, str)
                        or _JOB_ID_RE.fullmatch(job_id) is None
                        or not isinstance(status, str)
                        or job_id in by_id
                    ):
                        raise DocentIngestionStatusError(
                            "Docent status response contained a malformed or duplicate row"
                        )
                    by_id[job_id] = status

                if set(by_id) != set(chunk):
                    raise DocentIngestionStatusError(
                        "Docent status response had missing or extra job identifiers"
                    )
                for job_id in chunk:
                    status = by_id[job_id]
                    if status == "completed":
                        continue
                    if status in {"pending", "running"}:
                        next_pending.add(job_id)
                        continue
                    # canceled, failed, and every unknown state are terminal
                    # partial failures. Never leave them spinning in the loop.
                    raise DocentIngestionStatusError(
                        "Docent ingestion job did not complete successfully"
                    )
            pending = next_pending
            if pending:
                # One sleep per complete sweep, never one per 100-job chunk.
                budget.sleep(poll_interval)

    session.post = post_without_redirect
    client._post_with_retry = post_once
    client._wait_for_jobs = wait_for_jobs
    return tracker


# Backward-compatible private alias retained for focused callers/tests written
# while the no-repeat boundary covered only agent-run POSTs.
_bind_single_attempt_agent_run_posts = _bind_single_attempt_mutation_posts


def _validated_collection_id(value: Any) -> str:
    if not isinstance(value, str) or _COLLECTION_ID_RE.fullmatch(value) is None:
        raise DocentCollectionAcknowledgementError(
            "Docent collection identifier is not one bounded URL-safe component"
        )
    return value


def _create_collection_once(
    client: Any,
    collection_name: str,
    *,
    on_confirmed: Callable[[str], None] | None = None,
) -> str:
    """Create without invoking the SDK's pre-validation raw-ID logging."""
    response = client._session.post(
        f"{client._api_url}/create",
        json={
            "collection_id": None,
            "name": collection_name,
            "description": None,
        },
    )
    try:
        body = response.json()
    except Exception:
        raise DocentCollectionAcknowledgementError(
            "Docent collection response was not valid JSON"
        ) from None
    if not isinstance(body, dict):
        raise DocentCollectionAcknowledgementError(
            "Docent collection response was not a mapping"
        )
    collection_id = _validated_collection_id(body.get("collection_id"))
    # The child publishes this identity before the first batch mutation. A
    # SIGKILL between the 2xx and this callback remains honestly unknowable.
    if on_confirmed is not None:
        on_confirmed(collection_id)
    return collection_id


def _validate_ingestion_result(
    result: Any,
    *,
    expected_runs: int,
    expected_batch_posts: int,
    expected_job_ids: list[str] | None = None,
) -> None:
    if not isinstance(result, dict):
        raise DocentIngestionAcknowledgementError(
            "Docent ingestion result is not a confirmed result mapping"
        )
    job_ids = result.get("job_ids")
    valid_job_ids = (
        isinstance(job_ids, list)
        and len(job_ids) == expected_batch_posts
        and len(job_ids) > 0
        and all(
            isinstance(job_id, str) and _JOB_ID_RE.fullmatch(job_id) is not None
            for job_id in job_ids
        )
        and len(set(job_ids)) == len(job_ids)
    )
    total_runs_added = result.get("total_runs_added")
    if (
        result.get("status") != "success"
        or isinstance(total_runs_added, bool)
        or not isinstance(total_runs_added, int)
        or total_runs_added != expected_runs
        or expected_batch_posts <= 0
        or not valid_job_ids
        or (expected_job_ids is not None and job_ids != expected_job_ids)
    ):
        raise DocentIngestionAcknowledgementError(
            "Docent did not confirm and complete every ingestion batch"
        )


def _upload_in_child_process(
    runs: list[AgentRun],
    collection_name: str,
    launch_namespace: str,
    *,
    _budget: _DocentTimeBudget,
    on_collection_confirmed: Callable[[str], None],
) -> DocentUploadResult | DocentUploadFailure:
    """Run the reviewed SDK seam inside the disposable dedicated child."""
    if not runs:
        raise ValueError("Docent upload requires at least one AgentRun")
    namespace = validate_launch_namespace(launch_namespace)
    if not isinstance(collection_name, str) or not collection_name:
        raise ValueError("Docent collection name is invalid")
    budget = _budget
    collection_id: str | None = None
    try:
        from docent import Docent

        _validate_docent_sdk_class(Docent)
        _disable_pinned_docent_tqdm_monitor_for_child(Docent)
        bounded_client_type = _bounded_docent_type(Docent, budget)
        client = bounded_client_type(
            api_key=os.environ["DOCENT_API_KEY"],
            config_file=os.devnull,
        )
        tracker = _bind_single_attempt_mutation_posts(
            client,
            class_validated=True,
            _budget=budget,
        )
        collection_id = _create_collection_once(
            client,
            collection_name,
            on_confirmed=on_collection_confirmed,
        )
        budget.remaining()
        ingestion_result = client.add_agent_runs(collection_id, runs)
        _validate_ingestion_result(
            ingestion_result,
            expected_runs=len(runs),
            expected_batch_posts=tracker.agent_batch_posts,
            expected_job_ids=tracker.job_ids,
        )
        budget.remaining()
    except Exception as exc:
        return DocentUploadFailure(
            collection_name=collection_name,
            launch_namespace=namespace,
            collection_id=collection_id,
            error_type=_sanitized_error_type(exc),
        )
    assert collection_id is not None
    return DocentUploadResult(
        collection_name=collection_name,
        launch_namespace=namespace,
        collection_id=collection_id,
    )


@dataclass(slots=True)
class _DocentReceiptState:
    collection_name: str
    launch_namespace: str
    collection_id: str | None = None
    terminal: DocentUploadResult | DocentUploadFailure | None = None
    protocol_error: str | None = None
    receipt_bytes_seen: int = 0


def _json_without_duplicate_keys(raw: bytes) -> Any:
    def build(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=build)


def _accept_docent_receipt_frame(state: _DocentReceiptState, raw: bytes) -> None:
    """Apply one strict child event without retaining untrusted detail."""
    if state.protocol_error is not None:
        return
    try:
        if not raw or len(raw) > _DOCENT_RECEIPT_MAX_LINE_BYTES:
            raise ValueError("receipt frame length invalid")
        frame = _json_without_duplicate_keys(raw)
        if not isinstance(frame, dict):
            raise ValueError("receipt frame is not an object")
        event = frame.get("event")
        if event == "collection_confirmed":
            if set(frame) != {
                "event",
                "collection_name",
                "launch_namespace",
                "collection_id",
            }:
                raise ValueError("collection frame schema invalid")
            if state.collection_id is not None or state.terminal is not None:
                raise ValueError("collection frame duplicated or late")
            if (
                frame["collection_name"] != state.collection_name
                or frame["launch_namespace"] != state.launch_namespace
            ):
                raise ValueError("collection frame identity changed")
            state.collection_id = _validated_collection_id(frame["collection_id"])
            return
        if event != "terminal" or state.terminal is not None:
            raise ValueError("terminal frame missing, duplicate, or unknown")
        status = frame.get("status")
        if status == "confirmed":
            if set(frame) != {
                "event",
                "status",
                "collection_name",
                "launch_namespace",
                "collection_id",
            }:
                raise ValueError("confirmed terminal schema invalid")
            if state.collection_id is None:
                raise ValueError("confirmed terminal preceded collection identity")
            if frame["collection_id"] != state.collection_id:
                raise ValueError("terminal collection identity changed")
            result: DocentUploadResult | DocentUploadFailure = DocentUploadResult(
                collection_name=state.collection_name,
                launch_namespace=state.launch_namespace,
                collection_id=state.collection_id,
            )
        elif status == "ambiguous_or_unconfirmed":
            if set(frame) != {
                "event",
                "status",
                "collection_name",
                "launch_namespace",
                "collection_id",
                "error_type",
            }:
                raise ValueError("failure terminal schema invalid")
            error_type = frame["error_type"]
            if (
                not isinstance(error_type, str)
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_type)
                is None
            ):
                raise ValueError("failure type invalid")
            if frame["collection_id"] != state.collection_id:
                raise ValueError("failure terminal collection identity changed")
            result = DocentUploadFailure(
                collection_name=state.collection_name,
                launch_namespace=state.launch_namespace,
                collection_id=state.collection_id,
                error_type=error_type,
            )
        else:
            raise ValueError("terminal status invalid")
        if (
            frame["collection_name"] != state.collection_name
            or frame["launch_namespace"] != state.launch_namespace
        ):
            raise ValueError("terminal launch identity changed")
        state.terminal = result
    except Exception:
        state.protocol_error = "DocentUploadProtocolError"


def _consume_docent_receipt_bytes(
    state: _DocentReceiptState,
    buffer: bytearray,
    chunk: bytes,
    *,
    eof: bool = False,
) -> None:
    if state.protocol_error is not None:
        return
    state.receipt_bytes_seen += len(chunk)
    if state.receipt_bytes_seen > _DOCENT_RECEIPT_MAX_BYTES:
        state.protocol_error = "DocentUploadProtocolError"
        return
    buffer.extend(chunk)
    while True:
        newline = buffer.find(b"\n")
        if newline < 0:
            break
        raw = bytes(buffer[:newline])
        del buffer[: newline + 1]
        _accept_docent_receipt_frame(state, raw)
    if len(buffer) > _DOCENT_RECEIPT_MAX_LINE_BYTES or (eof and buffer):
        state.protocol_error = "DocentUploadProtocolError"


def _waitpid_nointr(pid: int, options: int) -> tuple[int, int]:
    while True:
        try:
            return os.waitpid(pid, options)
        except InterruptedError:
            continue
        except ChildProcessError:
            raise DocentUploadChildOwnershipError(
                "lost authoritative wait status for the Docent upload child"
            ) from None


def _kill_docent_process_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        if sys.platform == "darwin":
            try:
                # Darwin reports EPERM for a group whose only remaining
                # member is the known unreaped zombie leader. WNOWAIT proves
                # that narrow state while still pinning PGID until waitpid.
                if _child_exited_without_reaping(pid):
                    return
            except (OSError, DocentUploadChildOwnershipError):
                pass
        raise DocentUploadCleanupError(
            "permission denied while cleaning the dedicated Docent process group"
        ) from None


def _drain_receipt_fd(
    receipt_fd: int,
    state: _DocentReceiptState,
    buffer: bytearray,
) -> bool:
    """Drain a bounded amount of currently readable data; return true on EOF."""
    eof = False
    for _ in range(4):
        if state.protocol_error is not None:
            break
        try:
            chunk = os.read(receipt_fd, 4096)
        except BlockingIOError:
            break
        except InterruptedError:
            continue
        if not chunk:
            eof = True
            _consume_docent_receipt_bytes(state, buffer, b"", eof=True)
            break
        _consume_docent_receipt_bytes(state, buffer, chunk)
    return eof


def _child_exited_without_reaping(pid: int) -> bool:
    """Observe exit while retaining the leader zombie to pin its process group."""
    if not hasattr(os, "waitid"):
        # CPython on macOS does not expose waitid even though Darwin provides
        # the required XSI syscall. Keep the same WNOWAIT invariant through a
        # narrow libc binding.
        if sys.platform != "darwin":
            raise RuntimeError("waitid WNOWAIT is unavailable")
        import ctypes

        class Siginfo(ctypes.Structure):
            _fields_ = [
                ("si_signo", ctypes.c_int),
                ("si_errno", ctypes.c_int),
                ("si_code", ctypes.c_int),
                ("si_pid", ctypes.c_int),
                ("si_uid", ctypes.c_uint),
                ("si_status", ctypes.c_int),
                ("si_addr", ctypes.c_void_p),
                ("si_value", ctypes.c_void_p),
                ("si_band", ctypes.c_long),
                ("pad", ctypes.c_ulong * 7),
            ]
        libc = ctypes.CDLL(None, use_errno=True)
        waitid = libc.waitid
        waitid.argtypes = [
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.POINTER(Siginfo),
            ctypes.c_int,
        ]
        waitid.restype = ctypes.c_int
        while True:
            info = Siginfo()
            if waitid(1, pid, ctypes.byref(info), 0x04 | 0x20 | 0x01) == 0:
                return info.si_pid == pid
            error = ctypes.get_errno()
            if error == 4:  # EINTR
                continue
            raise OSError(error, os.strerror(error))
    while True:
        try:
            result = os.waitid(
                os.P_PID,
                pid,
                os.WEXITED | os.WNOWAIT | os.WNOHANG,
            )
            return result is not None
        except InterruptedError:
            continue


def _sweep_and_reap_docent_group(
    pid: int,
    receipt_fd: int,
    state: _DocentReceiptState,
    buffer: bytearray,
) -> int:
    """Kill descendants while the unreaped leader pins the fresh PGID."""
    _kill_docent_process_group(pid, signal.SIGKILL)
    _drain_receipt_fd(receipt_fd, state, buffer)
    _waited_pid, status = _waitpid_nointr(pid, 0)
    _drain_receipt_fd(receipt_fd, state, buffer)
    return status


def _terminate_and_reap_docent_child(
    pid: int,
    receipt_fd: int,
    state: _DocentReceiptState,
    buffer: bytearray,
    *,
    force_kill_at: float,
) -> int:
    """Send TERM, briefly drain receipt frames, then force-KILL and reap."""
    term_sent = False
    while True:
        try:
            if not term_sent:
                _kill_docent_process_group(pid, signal.SIGTERM)
                term_sent = True
            if time.monotonic() < force_kill_at:
                _drain_receipt_fd(receipt_fd, state, buffer)
                try:
                    select.select(
                        [receipt_fd],
                        [],
                        [],
                        min(0.01, max(0.0, force_kill_at - time.monotonic())),
                    )
                except InterruptedError:
                    pass
                continue
            return _sweep_and_reap_docent_group(
                pid, receipt_fd, state, buffer
            )
        except (KeyboardInterrupt, SystemExit):
            # The first operator control flow is already represented by the
            # caller's sanitized carrier. Cleanup is idempotent and must finish
            # even if another asynchronous control-flow exception lands here.
            continue


def _safe_spawn_source_fd(fd: int) -> int:
    duplicate = fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 200)
    while duplicate in {_DOCENT_JSONL_FD, _DOCENT_RECEIPT_FD}:
        replacement = fcntl.fcntl(duplicate, fcntl.F_DUPFD_CLOEXEC, 200)
        os.close(duplicate)
        duplicate = replacement
    return duplicate


def _child_default_signals() -> tuple[int, ...]:
    excluded = {signal.SIGKILL, signal.SIGSTOP}
    return tuple(
        sorted(int(sig) for sig in signal.valid_signals() if sig not in excluded)
    )


def _docent_child_environment(
    collection_name: str,
    launch_namespace: str,
    deadline: float,
) -> dict[str, str]:
    api_key = os.environ.get("DOCENT_API_KEY")
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("DOCENT_API_KEY is required for external Docent upload")
    api_url = os.environ.get("DOCENT_API_URL")
    frontend_url = os.environ.get("DOCENT_FRONTEND_URL")
    domain = os.environ.get("DOCENT_DOMAIN")
    if (api_url is None) != (frontend_url is None):
        raise ValueError("DOCENT_API_URL and DOCENT_FRONTEND_URL must be supplied together")
    if domain is not None and api_url is not None:
        raise ValueError("DOCENT_DOMAIN cannot be combined with explicit Docent URLs")
    environment = {
        "DOCENT_API_KEY": api_key,
        "DEBATE_DOCENT_COLLECTION_NAME": collection_name,
        "DEBATE_DOCENT_LAUNCH_NAMESPACE": launch_namespace,
        "DEBATE_DOCENT_DEADLINE_MONOTONIC": format(deadline, ".17g"),
    }
    if api_url is not None:
        environment["DOCENT_API_URL"] = api_url
        environment["DOCENT_FRONTEND_URL"] = frontend_url  # type: ignore[assignment]
    elif domain is not None:
        environment["DOCENT_DOMAIN"] = domain
    return environment


def _explicit_inheritable_fds() -> list[int]:
    """Snapshot open inheritable descriptors for explicit spawn closure."""
    directory = "/dev/fd" if os.path.isdir("/dev/fd") else "/proc/self/fd"
    result: list[int] = []
    for entry in os.listdir(directory):
        try:
            fd = int(entry)
        except ValueError:
            continue
        if fd <= 2:
            continue
        try:
            if os.get_inheritable(fd):
                result.append(fd)
        except OSError:
            continue
    return sorted(set(result))


def _spawn_docent_upload_child(
    jsonl_fd: int,
    receipt_write_fd: int,
    collection_name: str,
    launch_namespace: str,
    deadline: float,
    *,
    worker_path: Path = _DOCENT_WORKER_PATH,
) -> int:
    python = os.path.abspath(sys.executable)
    worker = worker_path.resolve()
    if not worker.is_absolute() or not worker.is_file():
        raise RuntimeError("Docent upload worker is unavailable")
    environment = _docent_child_environment(
        collection_name, launch_namespace, deadline
    )
    with _DOCENT_SPAWN_LOCK:
        jsonl_source = _safe_spawn_source_fd(jsonl_fd)
        receipt_source = _safe_spawn_source_fd(receipt_write_fd)
        try:
            inherited = _explicit_inheritable_fds()
            preserved = {
                jsonl_source,
                receipt_source,
                _DOCENT_JSONL_FD,
                _DOCENT_RECEIPT_FD,
            }
            actions = [
                (os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0),
                (os.POSIX_SPAWN_OPEN, 1, os.devnull, os.O_WRONLY, 0),
                (os.POSIX_SPAWN_OPEN, 2, os.devnull, os.O_WRONLY, 0),
            ]
            actions.extend(
                (os.POSIX_SPAWN_CLOSE, fd)
                for fd in inherited
                if fd not in preserved
            )
            actions.extend(
                [
                    (os.POSIX_SPAWN_DUP2, jsonl_source, _DOCENT_JSONL_FD),
                    (os.POSIX_SPAWN_DUP2, receipt_source, _DOCENT_RECEIPT_FD),
                    (os.POSIX_SPAWN_CLOSE, jsonl_source),
                    (os.POSIX_SPAWN_CLOSE, receipt_source),
                ]
            )
            return os.posix_spawn(
                python,
                [python, "-I", "-B", os.fspath(worker)],
                environment,
                file_actions=actions,
                setpgroup=0,
                setsigmask=(),
                setsigdef=_child_default_signals(),
            )
        finally:
            os.close(jsonl_source)
            os.close(receipt_source)


def upload(
    jsonl_fd: int,
    base_collection_name: str,
    launch_namespace: str,
    *,
    _deadline_seconds: float = _DOCENT_TOTAL_BUDGET_SECONDS,
    _worker_path: Path = _DOCENT_WORKER_PATH,
) -> DocentUploadResult | DocentUploadFailure | DocentUploadControlFlow:
    """Supervise one dedicated external-upload child from an open JSONL FD."""
    namespace = validate_launch_namespace(launch_namespace)
    collection_name = collection_name_for_launch(base_collection_name, namespace)
    if (
        isinstance(_deadline_seconds, bool)
        or not isinstance(_deadline_seconds, (int, float))
        or not 0 < float(_deadline_seconds) <= _DOCENT_TOTAL_BUDGET_SECONDS
    ):
        raise ValueError("Docent child deadline must be positive and at most five minutes")
    stat_result = os.fstat(jsonl_fd)
    if not stat.S_ISREG(stat_result.st_mode):
        raise ValueError("Docent JSONL descriptor must reference a regular file")
    if (fcntl.fcntl(jsonl_fd, fcntl.F_GETFL) & os.O_ACCMODE) != os.O_RDONLY:
        raise ValueError("Docent JSONL descriptor must be read-only")
    if stat_result.st_size <= 0:
        raise ValueError("Docent upload requires nonempty canonical JSONL")
    if len(collection_name.encode("utf-8")) > _DOCENT_RECEIPT_MAX_LINE_BYTES // 2:
        raise ValueError("Docent collection name is too large for the receipt protocol")
    if not hasattr(os, "waitid") and sys.platform != "darwin":
        raise RuntimeError("Docent process-group supervision requires waitid WNOWAIT")

    receipt_read_fd, receipt_write_fd = os.pipe()
    os.set_blocking(receipt_read_fd, False)
    pid: int | None = None
    state = _DocentReceiptState(collection_name, namespace)
    buffer = bytearray()
    child_status: int | None = None
    try:
        # This is the authoritative start of both the parent and child budget.
        deadline = time.monotonic() + float(_deadline_seconds)
        pid = _spawn_docent_upload_child(
            jsonl_fd,
            receipt_write_fd,
            collection_name,
            namespace,
            deadline,
            worker_path=_worker_path,
        )
        os.close(receipt_write_fd)
        receipt_write_fd = -1
        eof = False
        termination_error: str | None = None
        revocation_deadline = max(
            time.monotonic(),
            deadline - _DOCENT_LOCAL_TERMINATION_GRACE_SECONDS,
        )
        while True:
            now = time.monotonic()
            if state.protocol_error is not None:
                termination_error = state.protocol_error
                break
            if now >= revocation_deadline:
                termination_error = "DocentUploadDeadlineError"
                break
            eof = _drain_receipt_fd(receipt_read_fd, state, buffer) or eof
            if _child_exited_without_reaping(pid):
                # The unreaped leader pins PGID==PID while we terminate any
                # descendants, including on a nominally successful exit.
                child_status = _sweep_and_reap_docent_group(
                    pid, receipt_read_fd, state, buffer
                )
                break
            timeout = min(
                0.05,
                max(0.0, revocation_deadline - time.monotonic()),
            )
            try:
                select.select([receipt_read_fd], [], [], timeout)
            except InterruptedError:
                continue
        if termination_error is not None:
            cleanup_end = min(
                deadline,
                time.monotonic() + _DOCENT_LOCAL_TERMINATION_GRACE_SECONDS,
            )
            try:
                child_status = _terminate_and_reap_docent_child(
                    pid,
                    receipt_read_fd,
                    state,
                    buffer,
                    force_kill_at=max(
                        time.monotonic(),
                        cleanup_end - _DOCENT_FORCE_KILL_REAP_RESERVE_SECONDS,
                    ),
                )
            except Exception as cleanup_exc:
                termination_error = _sanitized_error_type(cleanup_exc)
            return DocentUploadFailure(
                collection_name=collection_name,
                launch_namespace=namespace,
                collection_id=state.collection_id,
                error_type=termination_error,
            )
        assert child_status is not None
        exit_code = os.waitstatus_to_exitcode(child_status)
        if state.protocol_error is not None:
            return DocentUploadFailure(
                collection_name=collection_name,
                launch_namespace=namespace,
                collection_id=state.collection_id,
                error_type=state.protocol_error,
            )
        if isinstance(state.terminal, DocentUploadResult) and exit_code == 0:
            return state.terminal
        if isinstance(state.terminal, DocentUploadFailure):
            return state.terminal
        return DocentUploadFailure(
            collection_name=collection_name,
            launch_namespace=namespace,
            collection_id=state.collection_id,
            error_type="DocentUploadChildExitError",
        )
    except (KeyboardInterrupt, SystemExit) as exc:
        cleanup_error_type: str | None = None
        if pid is not None:
            cleanup_end = min(
                deadline,
                time.monotonic() + _DOCENT_LOCAL_TERMINATION_GRACE_SECONDS,
            )
            try:
                _terminate_and_reap_docent_child(
                    pid,
                    receipt_read_fd,
                    state,
                    buffer,
                    force_kill_at=max(
                        time.monotonic(),
                        cleanup_end - _DOCENT_FORCE_KILL_REAP_RESERVE_SECONDS,
                    ),
                )
            except Exception as cleanup_exc:
                cleanup_error_type = _sanitized_error_type(cleanup_exc)
        kind = "KeyboardInterrupt" if isinstance(exc, KeyboardInterrupt) else "SystemExit"
        exit_code = None
        if isinstance(exc, SystemExit):
            exit_code = exc.code if type(exc.code) is int else 1
        return DocentUploadControlFlow(
            failure=DocentUploadFailure(
                collection_name=collection_name,
                launch_namespace=namespace,
                collection_id=state.collection_id,
                error_type=cleanup_error_type or kind,
            ),
            kind=kind,
            exit_code=exit_code,
        )
    except Exception as exc:
        if pid is not None:
            cleanup_end = min(
                deadline,
                time.monotonic() + _DOCENT_LOCAL_TERMINATION_GRACE_SECONDS,
            )
            try:
                _terminate_and_reap_docent_child(
                    pid,
                    receipt_read_fd,
                    state,
                    buffer,
                    force_kill_at=max(
                        time.monotonic(),
                        cleanup_end - _DOCENT_FORCE_KILL_REAP_RESERVE_SECONDS,
                    ),
                )
            except Exception as cleanup_exc:
                exc = cleanup_exc
        return DocentUploadFailure(
            collection_name=collection_name,
            launch_namespace=namespace,
            collection_id=state.collection_id,
            error_type=_sanitized_error_type(exc),
        )
    finally:
        if receipt_write_fd >= 0:
            os.close(receipt_write_fd)
        os.close(receipt_read_fd)
