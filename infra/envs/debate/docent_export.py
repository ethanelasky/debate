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
    upload(runs, base_collection_name="math-pc-rl", launch_namespace=namespace)
                                                # needs DOCENT_API_KEY
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from inspect import Parameter, signature
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


class DocentHardDeadlineUnavailableError(RuntimeError):
    """A process-wide hard deadline could not be installed safely."""


class _DocentHardDeadlineSignal(BaseException):
    """Private SIGALRM escape that SDK ``except Exception`` cannot swallow."""


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


@dataclass(slots=True)
class _DocentUploadAttemptState:
    collection_id: str | None = None


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


def _raise_docent_hard_deadline(_signum: int, _frame: Any) -> None:
    raise _DocentHardDeadlineSignal()


@contextmanager
def _docent_hard_wall_clock_guard(budget: _DocentTimeBudget):
    """Preempt the whole upload even when Requests keeps receiving bytes.

    ``requests`` interprets its read timeout as idle time between socket reads,
    not total wall time.  POSIX ``ITIMER_REAL`` is therefore the hard boundary
    around authentication through final confirmation.  It is process-global,
    so only the main thread may install it and an existing timer is a hard
    conflict rather than something this uploader may replace.
    """
    required_names = (
        "SIGALRM",
        "ITIMER_REAL",
        "SIG_BLOCK",
        "SIG_SETMASK",
        "getitimer",
        "setitimer",
        "getsignal",
        "signal",
        "pthread_sigmask",
        "sigpending",
        "sigwait",
    )
    required = {name: getattr(signal, name, None) for name in required_names}
    unavailable = any(value is None for value in required.values()) or any(
        not callable(required[name])
        for name in (
            "getitimer",
            "setitimer",
            "getsignal",
            "signal",
            "pthread_sigmask",
            "sigpending",
            "sigwait",
        )
    )
    if unavailable:
        raise DocentHardDeadlineUnavailableError(
            "POSIX wall-clock deadline support is unavailable"
        )
    if threading.current_thread() is not threading.main_thread():
        raise DocentHardDeadlineUnavailableError(
            "Docent hard deadline requires the main thread"
        )
    current_thread = threading.current_thread()
    other_threads = [
        thread
        for thread in threading.enumerate()
        if thread is not current_thread and thread.is_alive()
    ]
    if other_threads:
        raise DocentHardDeadlineUnavailableError(
            "another live Python thread conflicts with the process wall-clock timer"
        )

    sigalrm = signal.SIGALRM
    timer_kind = signal.ITIMER_REAL
    alarm_set = {sigalrm}
    previous_mask: set[signal.Signals] | None = None
    previous_handler: Any = None
    previous_timer: tuple[float, float] | None = None
    mask_blocked = False
    handler_installed = False
    timer_armed = False
    teardown_control: tuple[str, int | None] | None = None

    def remember_teardown_control(exc: BaseException) -> None:
        nonlocal teardown_control
        if teardown_control is not None:
            return
        if isinstance(exc, KeyboardInterrupt):
            teardown_control = ("KeyboardInterrupt", None)
        elif isinstance(exc, SystemExit):
            code = exc.code if exc.code is None or type(exc.code) is int else 1
            teardown_control = ("SystemExit", code)
        else:
            teardown_control = ("DocentHardDeadline", None)

    def interrupt_resilient_cleanup(action: Callable[[], Any]) -> None:
        """Retry an idempotent teardown transition after async control flow."""
        while True:
            try:
                action()
                # Keep the return in the protected suite: a line-trace
                # exception after the syscall's effect is caught and the
                # idempotent action is safely repeated.
                return
            except (KeyboardInterrupt, SystemExit, _DocentHardDeadlineSignal) as exc:
                remember_teardown_control(exc)

    def restore_process_signal_state() -> None:
        """Idempotent full teardown, safe to restart after an interrupt."""
        nonlocal mask_blocked
        if previous_mask is None:
            return
        if not mask_blocked:
            # Own the resulting state before the effect so an interrupt at the
            # call/assignment boundary cannot strand it untracked.
            mask_blocked = True
            interrupt_resilient_cleanup(
                lambda: signal.pthread_sigmask(signal.SIG_BLOCK, alarm_set)
            )
        if timer_armed or handler_installed:
            interrupt_resilient_cleanup(
                lambda: signal.setitimer(timer_kind, 0.0, 0.0)
            )
            # The timer can expire after delivery is blocked but before it is
            # canceled. Consume that guard-owned signal while our handler
            # remains installed, so restoring SIG_DFL cannot turn it into a
            # delayed process kill.
            while sigalrm in signal.sigpending():
                interrupt_resilient_cleanup(lambda: signal.sigwait(alarm_set))
        if handler_installed:
            interrupt_resilient_cleanup(
                lambda: signal.signal(sigalrm, previous_handler)
            )
            assert previous_timer is not None
            interrupt_resilient_cleanup(
                lambda: signal.setitimer(
                    timer_kind,
                    previous_timer[0],
                    previous_timer[1],
                )
            )
        # As on admission, claim the post-effect state before restoring the
        # prior mask. A restarted full teardown will re-block before touching
        # timer/handler state if an interrupt lands after this call's effect.
        mask_blocked = False
        interrupt_resilient_cleanup(
            lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        )

    try:
        try:
            # Atomically block delivery before inspecting or changing any
            # process-wide SIGALRM state.
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, alarm_set)
            mask_blocked = True
            if sigalrm in signal.sigpending():
                raise DocentHardDeadlineUnavailableError(
                    "SIGALRM is already pending"
                )
            if sigalrm in previous_mask:
                raise DocentHardDeadlineUnavailableError(
                    "SIGALRM is already blocked"
                )
            previous_handler = signal.getsignal(sigalrm)
            previous_timer = signal.getitimer(timer_kind)
            # A previously armed one-shot can expire after the first pending
            # check but before getitimer observes it.  Delivery is still
            # blocked here, so reject that newly pending signal rather than
            # later consuming it as though it belonged to our guard.
            if sigalrm in signal.sigpending():
                raise DocentHardDeadlineUnavailableError(
                    "SIGALRM became pending during deadline admission"
                )
            if previous_timer[0] > 0 or previous_timer[1] > 0:
                raise DocentHardDeadlineUnavailableError(
                    "an existing process wall-clock timer conflicts with Docent upload"
                )

            duration = min(_DOCENT_TOTAL_BUDGET_SECONDS, budget.remaining())
            # Claim restoration responsibility before the process-global
            # mutation.  KeyboardInterrupt/SystemExit can arrive after
            # signal.signal has taken effect but before it returns.
            handler_installed = True
            signal.signal(sigalrm, _raise_docent_hard_deadline)
            replaced_timer = signal.setitimer(timer_kind, duration, 0.0)
            # With no other live Python thread this cannot ordinarily race,
            # but setitimer's return value is the final atomic check.
            if replaced_timer[0] > 0 or replaced_timer[1] > 0:
                signal.setitimer(timer_kind, 0.0, 0.0)
                signal.signal(sigalrm, previous_handler)
                signal.setitimer(
                    timer_kind,
                    replaced_timer[0],
                    replaced_timer[1],
                )
                handler_installed = False
                raise DocentHardDeadlineUnavailableError(
                    "a concurrent process wall-clock timer conflicts with Docent upload"
                )
            timer_armed = True
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            mask_blocked = False
        except DocentHardDeadlineUnavailableError:
            raise
        except Exception:
            raise DocentHardDeadlineUnavailableError(
                "could not install the process wall-clock deadline"
            ) from None
        yield
    finally:
        active_exception = sys.exception()
        if previous_mask is not None:
            # The outer restart boundary also covers Python lines between the
            # individual syscall wrappers. Thus a first KI/SE anywhere during
            # teardown cannot skip the remaining restoration steps.
            interrupt_resilient_cleanup(restore_process_signal_state)
        # A first operator interrupt (or our own hard deadline) during
        # teardown outranks an ordinary upload exception: otherwise run_eval
        # would misclassify it as a best-effort external failure. An operator
        # control-flow exception already unwinding from the body keeps
        # priority over any later teardown interruption.
        active_has_control_priority = isinstance(
            active_exception,
            (KeyboardInterrupt, SystemExit, _DocentHardDeadlineSignal),
        )
        if not active_has_control_priority and teardown_control is not None:
            kind, exit_code = teardown_control
            if kind == "KeyboardInterrupt":
                raise KeyboardInterrupt()
            if kind == "SystemExit":
                raise SystemExit(exit_code)
            raise _DocentHardDeadlineSignal()


@contextmanager
def _block_docent_deadline_during_identity_handoff():
    """Let a confirmed ID reach outer state before a pending alarm fires."""
    alarm_set = {signal.SIGALRM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, alarm_set)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


@contextmanager
def _block_docent_identity_recovery_signals():
    """Make recovery of an already returned identity non-interruptible."""
    blocked = {signal.SIGALRM, signal.SIGINT}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


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


@contextmanager
def _disable_pinned_docent_tqdm_monitor(client_type: type[Any]):
    """Prevent the pinned SDK from spawning a timer-racing monitor thread."""
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
    # tqdm checks this class attribute before constructing TMonitor.  Keep the
    # visible progress context but prevent it from creating a Python thread
    # after the hard-deadline single-thread admission check.
    interval_changed = False
    teardown_control: tuple[str, int | None] | None = None
    try:
        with _block_docent_deadline_during_identity_handoff():
            # As with the signal handler, own restoration before mutating a
            # process-global: asynchronous BaseExceptions may arrive after
            # the assignment takes effect but before the next Python line.
            interval_changed = True
            tqdm_type.monitor_interval = 0
        yield
    finally:
        active_exception = sys.exception()
        while interval_changed:
            try:
                with _block_docent_deadline_during_identity_handoff():
                    tqdm_type.monitor_interval = previous_interval
                    interval_changed = False
            except (KeyboardInterrupt, SystemExit, _DocentHardDeadlineSignal) as exc:
                if teardown_control is None:
                    if isinstance(exc, KeyboardInterrupt):
                        teardown_control = ("KeyboardInterrupt", None)
                    elif isinstance(exc, SystemExit):
                        code = (
                            exc.code
                            if exc.code is None or type(exc.code) is int
                            else 1
                        )
                        teardown_control = ("SystemExit", code)
                    else:
                        teardown_control = ("DocentHardDeadline", None)
                # Assignment and flag update are inside the protected suite.
                # Whether the interrupt landed before or after the assignment,
                # repeating it is idempotent and restores the reviewed global.
                continue
        active_has_control_priority = isinstance(
            active_exception,
            (KeyboardInterrupt, SystemExit, _DocentHardDeadlineSignal),
        )
        if not active_has_control_priority and teardown_control is not None:
            kind, exit_code = teardown_control
            if kind == "KeyboardInterrupt":
                raise KeyboardInterrupt()
            if kind == "SystemExit":
                raise SystemExit(exit_code)
            raise _DocentHardDeadlineSignal()


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
    _attempt_state: _DocentUploadAttemptState | None = None,
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
        with _block_docent_deadline_during_identity_handoff():
            collection_id = _validated_collection_id(body.get("collection_id"))
            if _attempt_state is not None:
                # Store outside the interruptible upload stack before returning.
                # If the timer expires during validation, SIGALRM remains pending
                # until this assignment completes, then the hard guard records a
                # failure carrying the confirmed identity.
                _attempt_state.collection_id = collection_id
        return collection_id
    except BaseException:
        # The POST has returned a 2xx response, so an operator interrupt may
        # land after a valid identity is known but before outer attempt state
        # receives it.  Re-read only this already-buffered response while both
        # relevant signals are masked.  Never send, log, or retain its body;
        # failure to recover simply leaves the identity unconfirmed.
        with _block_docent_identity_recovery_signals():
            try:
                recovery_body = response.json()
                if isinstance(recovery_body, dict):
                    recovered_id = _validated_collection_id(
                        recovery_body.get("collection_id")
                    )
                    if _attempt_state is not None:
                        _attempt_state.collection_id = recovered_id
                recovery_body = None
                recovered_id = None
            except BaseException:
                pass
        response = None
        raise


def _validate_ingestion_result(
    result: Any,
    *,
    expected_runs: int,
    expected_batch_posts: int,
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
    ):
        raise DocentIngestionAcknowledgementError(
            "Docent did not confirm and complete every ingestion batch"
        )


def upload(
    runs: list[AgentRun],
    base_collection_name: str,
    launch_namespace: str,
) -> DocentUploadResult | DocentUploadFailure | DocentUploadControlFlow:
    """Push runs to this launch's new Docent collection.

    External upload remains best-effort at the runner boundary.  This function
    always creates the deterministic per-launch collection and passes the
    scientific ``AgentRun`` objects through unchanged.
    """
    return _guarded_upload_with_budget(
        runs,
        base_collection_name,
        launch_namespace,
        _budget=_DocentTimeBudget.fixed(),
    )


def _guarded_upload_with_budget(
    runs: list[AgentRun],
    base_collection_name: str,
    launch_namespace: str,
    *,
    _budget: _DocentTimeBudget,
) -> DocentUploadResult | DocentUploadFailure | DocentUploadControlFlow:
    """Private tiny-budget seam around the production hard-deadline path."""
    if not runs:
        raise ValueError("Docent upload requires at least one AgentRun")
    namespace = validate_launch_namespace(launch_namespace)
    collection_name = collection_name_for_launch(base_collection_name, namespace)
    attempt_state = _DocentUploadAttemptState()
    try:
        with _docent_hard_wall_clock_guard(_budget):
            return _upload_with_budget(
                runs,
                base_collection_name,
                namespace,
                _budget=_budget,
                _attempt_state=attempt_state,
            )
    except _DocentHardDeadlineSignal:
        error_type = "DocentUploadDeadlineError"
    except (DocentHardDeadlineUnavailableError, DocentUploadDeadlineError) as exc:
        error_type = _sanitized_error_type(exc)
    except KeyboardInterrupt:
        failure = DocentUploadFailure(
            collection_name=collection_name,
            launch_namespace=namespace,
            collection_id=attempt_state.collection_id,
            error_type="KeyboardInterrupt",
        )
        return DocentUploadControlFlow(
            failure=failure,
            kind="KeyboardInterrupt",
        )
    except SystemExit as exc:
        # Preserve an integer exit status, but never retain an arbitrary
        # message/object that could carry response bodies or credentials.
        if exc.code is None or type(exc.code) is int:
            exit_code = exc.code
        else:
            exit_code = 1
        failure = DocentUploadFailure(
            collection_name=collection_name,
            launch_namespace=namespace,
            collection_id=attempt_state.collection_id,
            error_type="SystemExit",
        )
        return DocentUploadControlFlow(
            failure=failure,
            kind="SystemExit",
            exit_code=exit_code,
        )
    failure = DocentUploadFailure(
        collection_name=collection_name,
        launch_namespace=namespace,
        collection_id=attempt_state.collection_id,
        error_type=error_type,
    )
    return failure


def _upload_with_budget(
    runs: list[AgentRun],
    base_collection_name: str,
    launch_namespace: str,
    *,
    _budget: _DocentTimeBudget,
    _attempt_state: _DocentUploadAttemptState | None = None,
) -> DocentUploadResult | DocentUploadFailure:
    """Private deterministic-clock seam for no-network deadline tests."""
    if not runs:
        raise ValueError("Docent upload requires at least one AgentRun")
    namespace = validate_launch_namespace(launch_namespace)
    collection_name = collection_name_for_launch(base_collection_name, namespace)
    attempt_state = (
        _DocentUploadAttemptState() if _attempt_state is None else _attempt_state
    )
    # Starts before SDK import/class validation and, critically, before the
    # constructor's authentication request.  This is fixed for manual/run_eval
    # callers; it is deliberately not operator-configurable.
    budget = _budget
    try:
        from docent import Docent

        # Exact versions and bytecode-visible class seams are checked before
        # construction because Docent.__init__ performs authentication I/O.
        _validate_docent_sdk_class(Docent)
        with _disable_pinned_docent_tqdm_monitor(Docent):
            bounded_client_type = _bounded_docent_type(Docent, budget)
            client = bounded_client_type(api_key=os.environ["DOCENT_API_KEY"])
            tracker = _bind_single_attempt_mutation_posts(
                client,
                class_validated=True,
                _budget=budget,
            )
            collection_id = _create_collection_once(
                client,
                collection_name,
                _attempt_state=attempt_state,
            )
            # Assign the confirmed identity before enforcing the total deadline
            # so a late complete 2xx create response remains identifiable.
            budget.remaining()
            ingestion_result = client.add_agent_runs(collection_id, runs)
            _validate_ingestion_result(
                ingestion_result,
                expected_runs=len(runs),
                expected_batch_posts=tracker.agent_batch_posts,
            )
            budget.remaining()
    except Exception as exc:
        return DocentUploadFailure(
            collection_name=collection_name,
            launch_namespace=namespace,
            collection_id=attempt_state.collection_id,
            error_type=_sanitized_error_type(exc),
        )
    return DocentUploadResult(
        collection_name=collection_name,
        launch_namespace=namespace,
        collection_id=attempt_state.collection_id,
    )
