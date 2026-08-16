"""Single-turn docent export: RLVR rollout records -> AgentRuns, same output
path as the debate export so both run types ingest into docent identically."""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import textwrap
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from infra.envs.debate.docent_export import (
    _bind_single_attempt_agent_run_posts,
    _DocentTimeBudget,
    _guarded_upload_with_budget,
    _upload_with_budget,
    DocentIngestionStatusError,
    DocentMutationHTTPStatusError,
    DocentMutationRedirectError,
    DocentUploadDeadlineError,
    DocentUploadFailure,
    DocentUploadResult,
    collection_name_for_launch,
    export_jsonl,
    export_jsonl_claimed,
    upload,
)
from infra.envs.singleturn_docent import agent_runs
from infra.launch_namespace import claim_directory

RECORDS = [
    {
        "task_index": 0,
        "meta": {"gt": 42.0, "level": 5, "split": "train", "question": "q?"},
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "solve q"},
        ],
        "completion": "the answer is \\boxed{42}",
        "stop_reason": "stop",
        "reward": 1.1,
        "info": {
            "correct_strict": 1.0,
            "correct_relaxed": 1.0,
            "answer_format_valid": 1.0,
        },
    },
    {
        "task_index": 1,
        "meta": {"gt": 7.0, "level": 5, "split": "train", "question": "r?"},
        "messages": [{"role": "user", "content": "solve r"}],
        "completion": "the answer is 7",
        "stop_reason": "stop",
        "reward": 0.1,
        "info": {
            "correct_strict": 0.0,
            "correct_relaxed": 1.0,
            "answer_format_valid": 0.0,
        },
    },
    {
        "task_index": 2,
        "meta": {"gt": 9.0, "level": 5, "split": "train", "question": "s?"},
        "messages": [{"role": "user", "content": "solve s"}],
        "completion": "\\boxed{8}",
        "stop_reason": "length",
        "reward": 0.1,
        "info": {
            "correct_strict": 0.0,
            "correct_relaxed": 0.0,
            "answer_format_valid": 1.0,
        },
    },
]


@pytest.fixture(autouse=True)
def _stop_docent_tqdm_monitor_between_tests():
    """The pinned SDK's progress bar leaves a daemon monitor after success."""
    from tqdm.std import tqdm

    monitor_before = tqdm.monitor
    yield
    monitor = tqdm.monitor
    if monitor_before is None and monitor is not None:
        monitor.exit()
        tqdm.monitor = None


def test_one_run_per_record_with_completion_appended():
    runs = agent_runs(RECORDS)
    assert len(runs) == 3
    msgs = runs[0].transcripts[0].messages
    assert [m.role for m in msgs] == ["system", "user", "assistant"]
    assert "boxed{42}" in msgs[-1].text


def test_metadata_carries_grading_and_mutually_exclusive_status_in_name():
    runs = agent_runs(RECORDS)
    assert runs[0].metadata["reward"] == 1.1
    assert runs[0].metadata["task"]["gt"] == 42.0
    assert [run.name.rsplit(" -> ", 1)[-1] for run in runs] == [
        "strict-correct",
        "relaxed-only-correct",
        "incorrect",
    ]
    assert runs[2].metadata["stop_reason"] == "length"
    legacy_keys = {
        "correct",
        "has_boxed",
        "has_code",
        "answer_tag",
        "strict_boxed",
        "code_fence",
    }
    assert all(not legacy_keys.intersection(run.metadata["info"]) for run in runs)


def test_export_jsonl_roundtrips(tmp_path):
    path = str(tmp_path / "st.jsonl")
    export_jsonl(agent_runs(RECORDS), path)
    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == 3
    assert rows[0]["transcripts"][0]["messages"][-1]["role"] == "assistant"
    assert rows[0]["metadata"]["info"]["correct_relaxed"] == 1.0


def test_claimed_export_preserves_exact_agent_run_bytes(tmp_path):
    runs = agent_runs(RECORDS)
    standalone = tmp_path / "standalone.jsonl"
    export_jsonl(runs, str(standalone))
    claimed = claim_directory(tmp_path / "docent" / "run" / "attempt")

    path = export_jsonl_claimed(runs, claimed, "step-00001.jsonl")

    assert Path(path).read_bytes() == standalone.read_bytes()


def test_docent_collection_name_is_stable_per_namespace_and_distinct_per_attempt():
    assert collection_name_for_launch("math-pc-rl", "run-A") == (
        "math-pc-rl--launch-run-A"
    )
    assert collection_name_for_launch("math-pc-rl", "run-A") == (
        "math-pc-rl--launch-run-A"
    )
    assert collection_name_for_launch("math-pc-rl", "run-B") == (
        "math-pc-rl--launch-run-B"
    )


@pytest.mark.parametrize("namespace", ["", "../attempt", "attempt/child", "two words"])
def test_docent_collection_name_refuses_invalid_namespace(namespace):
    with pytest.raises(ValueError, match="DEBATE_LAUNCH_NAMESPACE"):
        collection_name_for_launch("math-pc-rl", namespace)


def test_docent_upload_names_collection_without_changing_agent_run_bytes(
    monkeypatch,
):
    import docent

    runs = agent_runs(RECORDS)
    before = [run.model_dump_json().encode("utf-8") for run in runs]
    observed = {}

    session, adapter = _sequenced_session(
        [(200, {"collection_id": "collection-123"})]
    )
    client = object.__new__(docent.Docent)
    client._api_url = "https://docent.invalid/rest"
    client._session = session

    def one_batch_post(url, max_retries=3, **kwargs):
        observed["max_retries"] = max_retries
        return object()

    def add_agent_runs(collection_id, added_runs):
        observed["collection_id"] = collection_id
        observed["same_list"] = added_runs is runs
        observed["runs"] = added_runs
        client._post_with_retry("https://docent.invalid/agent-runs")
        return {
            "status": "success",
            "total_runs_added": len(added_runs),
            "job_ids": ["job-123"],
        }

    client._post_with_retry = one_batch_post
    client.add_agent_runs = add_agent_runs

    class ReturningDocent(docent.Docent):
        def __new__(cls, **kwargs):
            return client

    monkeypatch.setattr(docent, "Docent", ReturningDocent)
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")

    result = upload(runs, "math-pc-rl", "launch_2026.08-14")
    assert result == DocentUploadResult(
        collection_name="math-pc-rl--launch-launch_2026.08-14",
        launch_namespace="launch_2026.08-14",
        collection_id="collection-123",
    )
    create_payload = json.loads(adapter.calls[0][0].body)
    assert create_payload["name"] == "math-pc-rl--launch-launch_2026.08-14"
    assert observed["collection_id"] == "collection-123"
    assert observed["max_retries"] == 0
    assert observed["same_list"] is True
    assert [run.model_dump_json().encode("utf-8") for run in observed["runs"]] == before


def test_invalid_namespace_refuses_before_docent_client_or_upload(monkeypatch):
    runs = agent_runs(RECORDS)
    constructed = False

    class MustNotConstruct:
        def __init__(self, **kwargs):
            nonlocal constructed
            constructed = True
            raise AssertionError("invalid namespace must refuse before Docent client creation")

    import docent

    monkeypatch.setattr(docent, "Docent", MustNotConstruct, raising=False)
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")

    with pytest.raises(ValueError, match="DEBATE_LAUNCH_NAMESPACE"):
        upload(runs, "math-pc-rl", "../wrong-collection")
    assert constructed is False


class _SequencedAdapter(HTTPAdapter):
    """No-network transport underneath a real requests.Session."""

    def __init__(self, responses, *, max_retries=0):
        super().__init__(max_retries=max_retries)
        self.responses = list(responses)
        self.calls = []

    def send(self, request, **kwargs):
        self.calls.append((request, kwargs))
        status_code, body = self.responses.pop(0)
        response = requests.Response()
        response.status_code = status_code
        response._content = json.dumps(body).encode("utf-8")
        response.headers["content-type"] = "application/json"
        if status_code in {301, 302, 303, 307, 308}:
            response.headers["location"] = "https://docent.invalid/replayed-mutation"
        response.url = request.url
        response.request = request
        return response


def _sequenced_session(responses, *, max_retries=0):
    session = requests.Session()
    adapter = _SequencedAdapter(responses, max_retries=max_retries)
    session.mount("https://", adapter)
    return session, adapter


def _bare_installed_docent(session):
    from docent import Docent

    client = object.__new__(Docent)
    client._api_url = "https://docent.invalid/rest"
    client._frontend_url = "https://docent.invalid"
    client._logger = logging.getLogger("test-docent-no-repeat")
    client._session = session
    return client


class _FakeMonotonicClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.advance(seconds)


class _TimedSequencedAdapter(HTTPAdapter):
    """Real Session transport whose steps can consume fake monotonic time."""

    def __init__(self, clock, responses):
        super().__init__(max_retries=0)
        self.clock = clock
        self.responses = list(responses)
        self.calls = []

    def send(self, request, **kwargs):
        self.calls.append((request, kwargs))
        status_code, body, elapsed = self.responses.pop(0)
        self.clock.advance(elapsed)
        response = requests.Response()
        response.status_code = status_code
        response._content = json.dumps(body).encode("utf-8")
        response.headers["content-type"] = "application/json"
        response.url = request.url
        response.request = request
        return response


class _TricklingAdapter(HTTPAdapter):
    """Ignores Requests' idle timeout by simulating regular incoming bytes."""

    def __init__(self, immediate_responses=()):
        super().__init__(max_retries=0)
        self.immediate_responses = list(immediate_responses)
        self.calls = []
        self.trickles = 0

    def send(self, request, **kwargs):
        self.calls.append((request, kwargs))
        if self.immediate_responses:
            status_code, body = self.immediate_responses.pop(0)
            response = requests.Response()
            response.status_code = status_code
            response._content = json.dumps(body).encode("utf-8")
            response.headers["content-type"] = "application/json"
            response.url = request.url
            response.request = request
            return response
        while True:
            self.trickles += 1
            # A socket yielding a byte more frequently than its read timeout
            # can otherwise keep Requests alive forever.
            time.sleep(0.01)


def _install_adapter_on_new_sessions(monkeypatch, adapter):
    """Intercept the pinned SDK constructor without replacing Session itself."""
    original_init = requests.Session.__init__

    def initialize(session, *args, **kwargs):
        original_init(session, *args, **kwargs)
        session.mount("https://", adapter)

    monkeypatch.setattr(requests.Session, "__init__", initialize)


def _docent_test_environment(monkeypatch):
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")
    monkeypatch.setenv("DOCENT_API_URL", "https://docent.invalid/rest")
    monkeypatch.setenv("DOCENT_FRONTEND_URL", "https://docent.invalid")


def _tiny_real_time_budget(seconds):
    return _DocentTimeBudget(
        deadline=time.monotonic() + seconds,
        clock=time.monotonic,
        sleeper=time.sleep,
    )


def test_hard_deadline_preempts_trickling_constructor_auth_and_restores_timer(
    monkeypatch,
):
    if not hasattr(signal, "setitimer"):
        pytest.skip("POSIX interval timers unavailable")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0 or previous_timer[1] > 0:
        pytest.skip("test process already owns ITIMER_REAL")

    adapter = _TricklingAdapter()
    _install_adapter_on_new_sessions(monkeypatch, adapter)
    _docent_test_environment(monkeypatch)
    started = time.monotonic()

    result = _guarded_upload_with_budget(
        agent_runs(RECORDS),
        "math-pc-rl",
        "run-A",
        _budget=_tiny_real_time_budget(0.25),
    )

    elapsed = time.monotonic() - started
    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id=None,
        error_type="DocentUploadDeadlineError",
    )
    assert len(adapter.calls) == 1
    assert adapter.calls[0][0].method == "GET"
    assert adapter.trickles > 0
    assert elapsed < 1.0
    assert signal.getsignal(signal.SIGALRM) is previous_handler
    assert signal.getitimer(signal.ITIMER_REAL) == previous_timer


def test_hard_deadline_after_confirmed_create_retains_id_and_never_retries(
    monkeypatch,
):
    if not hasattr(signal, "setitimer"):
        pytest.skip("POSIX interval timers unavailable")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0 or previous_timer[1] > 0:
        pytest.skip("test process already owns ITIMER_REAL")

    adapter = _TricklingAdapter(
        [
            (200, {}),
            (200, {"collection_id": "confirmed-hard-timeout-id"}),
        ]
    )
    _install_adapter_on_new_sessions(monkeypatch, adapter)
    _docent_test_environment(monkeypatch)
    started = time.monotonic()

    result = _guarded_upload_with_budget(
        agent_runs(RECORDS),
        "math-pc-rl",
        "run-A",
        _budget=_tiny_real_time_budget(0.5),
    )

    elapsed = time.monotonic() - started
    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="confirmed-hard-timeout-id",
        error_type="DocentUploadDeadlineError",
    )
    assert [request.method for request, _kwargs in adapter.calls] == [
        "GET",
        "POST",
        "POST",
    ]
    assert adapter.calls[1][0].url.endswith("/rest/create")
    assert adapter.calls[2][0].url.endswith(
        "/rest/confirmed-hard-timeout-id/agent_runs"
    )
    assert adapter.trickles > 0
    assert elapsed < 1.5
    assert signal.getsignal(signal.SIGALRM) is previous_handler
    assert signal.getitimer(signal.ITIMER_REAL) == previous_timer


def test_pending_alarm_during_collection_id_validation_retains_confirmed_id(
    monkeypatch,
):
    import infra.envs.debate.docent_export as docent_export

    if not hasattr(signal, "pthread_sigmask"):
        pytest.skip("POSIX signal masks unavailable")
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0 or previous_timer[1] > 0:
        pytest.skip("test process already owns ITIMER_REAL")

    adapter = _SequencedAdapter(
        [
            (200, {}),
            (200, {"collection_id": "critical-section-id"}),
        ]
    )
    _install_adapter_on_new_sessions(monkeypatch, adapter)
    _docent_test_environment(monkeypatch)
    original_validate = docent_export._validated_collection_id

    def validate_then_expire(value):
        collection_id = original_validate(value)
        os.kill(os.getpid(), signal.SIGALRM)
        # SIGALRM is blocked by the identity handoff, so execution must reach
        # the outer attempt-state assignment before delivery resumes.
        assert signal.SIGALRM in signal.sigpending()
        return collection_id

    monkeypatch.setattr(
        docent_export,
        "_validated_collection_id",
        validate_then_expire,
    )

    result = _guarded_upload_with_budget(
        agent_runs(RECORDS),
        "math-pc-rl",
        "run-A",
        _budget=_tiny_real_time_budget(2.0),
    )

    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="critical-section-id",
        error_type="DocentUploadDeadlineError",
    )
    assert len(adapter.calls) == 2
    assert signal.SIGALRM not in signal.sigpending()
    assert signal.getitimer(signal.ITIMER_REAL) == previous_timer


def test_deadline_admission_rechecks_alarm_pending_after_getitimer(monkeypatch):
    """An old timer expiring in the admission gap is never adopted as ours."""
    import docent
    import infra.envs.debate.docent_export as docent_export

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0 or previous_timer[1] > 0:
        pytest.skip("test process already owns ITIMER_REAL")
    delivered = []
    constructed = False
    original_getitimer = signal.getitimer

    class MustNotConstruct:
        def __init__(self, **_kwargs):
            nonlocal constructed
            constructed = True

    def prior_handler(signum, _frame):
        delivered.append(signum)

    fired = False

    def expire_between_checks(timer_kind):
        nonlocal fired
        value = original_getitimer(timer_kind)
        if not fired:
            fired = True
            os.kill(os.getpid(), signal.SIGALRM)
        return value

    signal.signal(signal.SIGALRM, prior_handler)
    monkeypatch.setattr(docent, "Docent", MustNotConstruct)
    monkeypatch.setattr(signal, "getitimer", expire_between_checks)
    try:
        result = _guarded_upload_with_budget(
            agent_runs(RECORDS),
            "math-pc-rl",
            "run-A",
            _budget=_tiny_real_time_budget(1.0),
        )
        monkeypatch.setattr(signal, "getitimer", original_getitimer)
    finally:
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(
            signal.ITIMER_REAL,
            previous_timer[0],
            previous_timer[1],
        )

    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id=None,
        error_type="DocentHardDeadlineUnavailableError",
    )
    assert constructed is False
    assert delivered == [signal.SIGALRM]


@pytest.mark.parametrize("control_type", [KeyboardInterrupt, SystemExit])
def test_deadline_handler_mutation_interrupt_restores_previous_handler(
    monkeypatch, control_type
):
    import infra.envs.debate.docent_export as docent_export

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0 or previous_timer[1] > 0:
        pytest.skip("test process already owns ITIMER_REAL")
    original_signal = signal.signal
    interrupted = False

    def interrupt_after_effect(signum, handler):
        nonlocal interrupted
        result = original_signal(signum, handler)
        if handler is docent_export._raise_docent_hard_deadline and not interrupted:
            interrupted = True
            raise control_type("RAW-HANDLER-BOUNDARY")
        return result

    monkeypatch.setattr(signal, "signal", interrupt_after_effect)
    try:
        with pytest.raises(control_type):
            with docent_export._docent_hard_wall_clock_guard(
                _tiny_real_time_budget(1.0)
            ):
                raise AssertionError("guard body must not run")
        retained_handler = signal.getsignal(signal.SIGALRM)
        retained_timer = signal.getitimer(signal.ITIMER_REAL)
    finally:
        monkeypatch.setattr(signal, "signal", original_signal)
        original_signal(signal.SIGALRM, previous_handler)
        signal.setitimer(
            signal.ITIMER_REAL,
            previous_timer[0],
            previous_timer[1],
        )

    assert interrupted is True
    assert retained_handler is previous_handler
    assert retained_timer == previous_timer


@pytest.mark.parametrize("control_type", [KeyboardInterrupt, SystemExit])
def test_tqdm_interval_interrupt_after_mutation_restores_global(
    monkeypatch, control_type
):
    import inspect
    import infra.envs.debate.docent_export as docent_export
    from docent import Docent
    from tqdm.std import tqdm

    previous_interval = tqdm.monitor_interval
    generator_impl = docent_export._disable_pinned_docent_tqdm_monitor.__wrapped__
    source, first_line = inspect.getsourcelines(generator_impl)
    yield_line = next(
        first_line + index
        for index, line in enumerate(source)
        if line.strip() == "yield"
    )

    def trace(frame, event, _arg):
        if (
            event == "line"
            and frame.f_code
            is generator_impl.__code__
            and frame.f_lineno == yield_line
        ):
            sys.settrace(None)
            raise control_type("RAW-TQDM-BOUNDARY")
        return trace

    sys.settrace(trace)
    try:
        with pytest.raises(control_type):
            with docent_export._disable_pinned_docent_tqdm_monitor(Docent):
                raise AssertionError("context body must not run")
    finally:
        sys.settrace(None)

    assert tqdm.monitor_interval == previous_interval


@pytest.mark.parametrize("control_type", [KeyboardInterrupt, SystemExit])
def test_tqdm_interval_interrupt_at_restore_assignment_retries_and_restores(
    control_type,
):
    import inspect
    import infra.envs.debate.docent_export as docent_export
    from docent import Docent
    from tqdm.std import tqdm

    previous_interval = tqdm.monitor_interval
    generator_impl = docent_export._disable_pinned_docent_tqdm_monitor.__wrapped__
    source, first_line = inspect.getsourcelines(generator_impl)
    restore_line = next(
        first_line + index
        for index, line in enumerate(source)
        if line.strip() == "tqdm_type.monitor_interval = previous_interval"
    )
    fired = False

    def trace(frame, event, _arg):
        nonlocal fired
        if (
            not fired
            and event == "line"
            and frame.f_code is generator_impl.__code__
            and frame.f_lineno == restore_line
        ):
            fired = True
            raise control_type("RAW-TQDM-RESTORE-BOUNDARY")
        return trace

    sys.settrace(trace)
    try:
        with pytest.raises(control_type) as caught:
            with docent_export._disable_pinned_docent_tqdm_monitor(Docent):
                assert tqdm.monitor_interval == 0
    finally:
        sys.settrace(None)

    assert fired is True
    assert caught.value.args in {(), (1,)}
    assert tqdm.monitor_interval == previous_interval
    assert "RAW-TQDM-RESTORE-BOUNDARY" not in repr(caught.value)


@pytest.mark.parametrize(
    ("transition", "control_kind", "body_error"),
    [
        (transition, control_kind, False)
        for transition in (
            "block",
            "timer_cancel",
            "handler_restore",
            "timer_restore",
            "mask_restore",
        )
        for control_kind in ("KeyboardInterrupt", "SystemExit")
    ]
    + [
        ("timer_cancel", "KeyboardInterrupt", True),
        ("timer_cancel", "SystemExit", True),
    ],
)
def test_subprocess_teardown_interrupt_restores_every_signal_transition(
    transition, control_kind, body_error
):
    """An interrupt after each teardown syscall effect cannot corrupt state."""
    probe = textwrap.dedent(
        f"""
        import signal
        import time
        from infra.envs.debate.docent_export import (
            _DocentTimeBudget,
            _docent_hard_wall_clock_guard,
        )

        transition = {transition!r}
        control_kind = {control_kind!r}
        body_error = {body_error!r}
        old_handler = signal.getsignal(signal.SIGALRM)
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        old_timer = signal.getitimer(signal.ITIMER_REAL)
        original_mask = signal.pthread_sigmask
        original_timer = signal.setitimer
        original_signal = signal.signal
        fired = False
        zero_timer_calls = 0

        def interrupt():
            if control_kind == "KeyboardInterrupt":
                raise KeyboardInterrupt("RAW-TEARDOWN-BOUNDARY")
            raise SystemExit("RAW-TEARDOWN-BOUNDARY")

        def mask_wrapper(how, signals):
            global fired
            result = original_mask(how, signals)
            target = (
                (transition == "block" and how == signal.SIG_BLOCK)
                or (transition == "mask_restore" and how == signal.SIG_SETMASK)
            )
            if target and not fired:
                fired = True
                interrupt()
            return result

        def timer_wrapper(kind, seconds, interval=0.0):
            global fired, zero_timer_calls
            result = original_timer(kind, seconds, interval)
            if seconds == 0.0:
                zero_timer_calls += 1
                target_number = 1 if transition == "timer_cancel" else 2
                if (
                    transition in {{"timer_cancel", "timer_restore"}}
                    and zero_timer_calls == target_number
                    and not fired
                ):
                    fired = True
                    interrupt()
            return result

        def signal_wrapper(signum, handler):
            global fired
            result = original_signal(signum, handler)
            if transition == "handler_restore" and handler is old_handler and not fired:
                fired = True
                interrupt()
            return result

        budget = _DocentTimeBudget(
            deadline=time.monotonic() + 5.0,
            clock=time.monotonic,
            sleeper=time.sleep,
        )
        caught = None
        try:
            with _docent_hard_wall_clock_guard(budget):
                signal.pthread_sigmask = mask_wrapper
                signal.setitimer = timer_wrapper
                signal.signal = signal_wrapper
                if body_error:
                    raise RuntimeError("ordinary-upload-error")
        except KeyboardInterrupt:
            caught = "KeyboardInterrupt"
        except SystemExit as exc:
            caught = "SystemExit"
            assert exc.code == 1
        except RuntimeError:
            caught = "RuntimeError"
        finally:
            signal.pthread_sigmask = original_mask
            signal.setitimer = original_timer
            signal.signal = original_signal

        assert fired
        assert caught == control_kind
        assert signal.getsignal(signal.SIGALRM) is old_handler
        assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == old_mask
        assert signal.getitimer(signal.ITIMER_REAL) == old_timer
        assert signal.SIGALRM not in signal.sigpending()
        print("teardown-restored", flush=True)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "teardown-restored"
    assert "RAW-TEARDOWN-BOUNDARY" not in completed.stderr


@pytest.mark.parametrize("control_kind", ["KeyboardInterrupt", "SystemExit"])
def test_teardown_operator_control_supersedes_ordinary_upload_failure(
    monkeypatch, control_kind
):
    import infra.envs.debate.docent_export as docent_export

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0 or previous_timer[1] > 0:
        pytest.skip("test process already owns ITIMER_REAL")
    adapter = _SequencedAdapter(
        [
            (200, {}),
            (200, {"collection_id": "confirmed-teardown-control"}),
            (500, {"raw": "ordinary-upload-secret"}),
        ]
    )
    _install_adapter_on_new_sessions(monkeypatch, adapter)
    _docent_test_environment(monkeypatch)
    original_setitimer = signal.setitimer
    fired = False

    def interrupt_after_cancel(kind, seconds, interval=0.0):
        nonlocal fired
        result = original_setitimer(kind, seconds, interval)
        if seconds == 0.0 and not fired:
            fired = True
            if control_kind == "KeyboardInterrupt":
                raise KeyboardInterrupt("RAW-TEARDOWN-CONTROL")
            raise SystemExit("RAW-TEARDOWN-CONTROL")
        return result

    monkeypatch.setattr(signal, "setitimer", interrupt_after_cancel)
    result = _guarded_upload_with_budget(
        agent_runs(RECORDS),
        "math-pc-rl",
        "run-A",
        _budget=_tiny_real_time_budget(2.0),
    )
    monkeypatch.setattr(signal, "setitimer", original_setitimer)

    assert fired is True
    assert result == docent_export.DocentUploadControlFlow(
        failure=DocentUploadFailure(
            collection_name="math-pc-rl--launch-run-A",
            launch_namespace="run-A",
            collection_id="confirmed-teardown-control",
            error_type=control_kind,
        ),
        kind=control_kind,
        exit_code=1 if control_kind == "SystemExit" else None,
    )
    assert len(adapter.calls) == 3
    assert signal.getsignal(signal.SIGALRM) is previous_handler
    assert signal.getitimer(signal.ITIMER_REAL) == previous_timer
    assert "RAW-TEARDOWN-CONTROL" not in repr(result)
    assert "ordinary-upload-secret" not in repr(result)


def test_hard_deadline_conflict_refuses_before_client_and_leaves_timer_untouched(
    monkeypatch,
):
    if not hasattr(signal, "setitimer"):
        pytest.skip("POSIX interval timers unavailable")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0 or previous_timer[1] > 0:
        pytest.skip("test process already owns ITIMER_REAL")

    import docent

    constructed = False

    class MustNotConstruct:
        def __init__(self, **kwargs):
            nonlocal constructed
            constructed = True

    def conflicting_handler(_signum, _frame):
        raise AssertionError("conflicting timer should not fire during this test")

    monkeypatch.setattr(docent, "Docent", MustNotConstruct)
    signal.signal(signal.SIGALRM, conflicting_handler)
    signal.setitimer(signal.ITIMER_REAL, 5.0, 0.0)
    try:
        result = _guarded_upload_with_budget(
            agent_runs(RECORDS),
            "math-pc-rl",
            "run-A",
            _budget=_tiny_real_time_budget(1.0),
        )
        retained_handler = signal.getsignal(signal.SIGALRM)
        retained_timer = signal.getitimer(signal.ITIMER_REAL)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(
            signal.ITIMER_REAL,
            previous_timer[0],
            previous_timer[1],
        )

    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id=None,
        error_type="DocentHardDeadlineUnavailableError",
    )
    assert constructed is False
    assert retained_handler is conflicting_handler
    assert retained_timer[0] > 4.0
    assert retained_timer[1] == 0.0


def test_hard_deadline_non_main_thread_refuses_before_client(monkeypatch):
    import docent

    constructed = False

    class MustNotConstruct:
        def __init__(self, **kwargs):
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(docent, "Docent", MustNotConstruct)
    results = []
    errors = []

    def invoke():
        try:
            results.append(
                _guarded_upload_with_budget(
                    agent_runs(RECORDS),
                    "math-pc-rl",
                    "run-A",
                    _budget=_tiny_real_time_budget(1.0),
                )
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert errors == []
    assert results == [
        DocentUploadFailure(
            collection_name="math-pc-rl--launch-run-A",
            launch_namespace="run-A",
            collection_id=None,
            error_type="DocentHardDeadlineUnavailableError",
        )
    ]
    assert constructed is False


def test_hard_deadline_refuses_main_thread_when_second_python_thread_is_live(
    monkeypatch,
):
    import docent

    constructed = False

    class MustNotConstruct:
        def __init__(self, **kwargs):
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(docent, "Docent", MustNotConstruct)
    started = threading.Event()
    release = threading.Event()

    def remain_live():
        started.set()
        release.wait(timeout=2.0)

    other = threading.Thread(target=remain_live)
    other.start()
    assert started.wait(timeout=1.0)
    try:
        result = _guarded_upload_with_budget(
            agent_runs(RECORDS),
            "math-pc-rl",
            "run-A",
            _budget=_tiny_real_time_budget(1.0),
        )
    finally:
        release.set()
        other.join(timeout=1.0)

    assert not other.is_alive()
    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id=None,
        error_type="DocentHardDeadlineUnavailableError",
    )
    assert constructed is False


def test_hard_deadline_unavailable_refuses_before_client(monkeypatch):
    import docent

    constructed = False

    class MustNotConstruct:
        def __init__(self, **kwargs):
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(docent, "Docent", MustNotConstruct)
    monkeypatch.setattr(signal, "setitimer", None)

    result = _guarded_upload_with_budget(
        agent_runs(RECORDS),
        "math-pc-rl",
        "run-A",
        _budget=_tiny_real_time_budget(1.0),
    )

    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id=None,
        error_type="DocentHardDeadlineUnavailableError",
    )
    assert constructed is False


def test_subprocess_initially_blocked_pending_alarm_refuses_without_construction():
    probe = textwrap.dedent(
        """
        import os
        import signal
        import time
        import docent
        from infra.envs.debate.docent_export import (
            _DocentTimeBudget,
            _guarded_upload_with_budget,
            DocentUploadFailure,
        )

        constructed = False
        class MustNotConstruct:
            def __init__(self, **kwargs):
                global constructed
                constructed = True
        docent.Docent = MustNotConstruct

        old_handler = signal.getsignal(signal.SIGALRM)
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        signal.signal(signal.SIGALRM, lambda *_args: None)
        os.kill(os.getpid(), signal.SIGALRM)
        assert signal.SIGALRM in signal.sigpending()
        try:
            budget = _DocentTimeBudget(
                deadline=time.monotonic() + 1.0,
                clock=time.monotonic,
                sleeper=time.sleep,
            )
            result = _guarded_upload_with_budget(
                [object()], "base", "run-A", _budget=budget
            )
            assert isinstance(result, DocentUploadFailure)
            assert result.error_type == "DocentHardDeadlineUnavailableError"
            assert result.collection_id is None
            assert not constructed
            current_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
            assert signal.SIGALRM in current_mask
            assert signal.SIGALRM in signal.sigpending()
            print("blocked-pending-refusal-ok", flush=True)
        finally:
            if signal.SIGALRM in signal.sigpending():
                signal.sigwait({signal.SIGALRM})
            signal.signal(signal.SIGALRM, old_handler)
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "blocked-pending-refusal-ok"


def test_subprocess_initially_blocked_alarm_refuses_without_construction():
    probe = textwrap.dedent(
        """
        import signal
        import time
        import docent
        from infra.envs.debate.docent_export import (
            _DocentTimeBudget,
            _guarded_upload_with_budget,
        )

        constructed = False
        class MustNotConstruct:
            def __init__(self, **kwargs):
                global constructed
                constructed = True
        docent.Docent = MustNotConstruct

        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        try:
            assert signal.SIGALRM not in signal.sigpending()
            budget = _DocentTimeBudget(
                deadline=time.monotonic() + 1.0,
                clock=time.monotonic,
                sleeper=time.sleep,
            )
            result = _guarded_upload_with_budget(
                [object()], "base", "run-A", _budget=budget
            )
            assert result.error_type == "DocentHardDeadlineUnavailableError"
            assert signal.SIGALRM in signal.pthread_sigmask(signal.SIG_BLOCK, set())
            assert not constructed
            print("blocked-refusal-ok", flush=True)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "blocked-refusal-ok"


def test_subprocess_second_python_thread_refuses_before_construction():
    probe = textwrap.dedent(
        """
        import threading
        import time
        import docent
        from infra.envs.debate.docent_export import (
            _DocentTimeBudget,
            _guarded_upload_with_budget,
        )

        constructed = False
        class MustNotConstruct:
            def __init__(self, **kwargs):
                global constructed
                constructed = True
        docent.Docent = MustNotConstruct

        started = threading.Event()
        release = threading.Event()
        def remain_live():
            started.set()
            release.wait(2.0)
        other = threading.Thread(target=remain_live)
        other.start()
        assert started.wait(1.0)
        try:
            budget = _DocentTimeBudget(
                deadline=time.monotonic() + 1.0,
                clock=time.monotonic,
                sleeper=time.sleep,
            )
            result = _guarded_upload_with_budget(
                [object()], "base", "run-A", _budget=budget
            )
            assert result.error_type == "DocentHardDeadlineUnavailableError"
            assert not constructed
            assert other.is_alive()
            print("second-thread-refusal-ok", flush=True)
        finally:
            release.set()
            other.join(1.0)
        assert not other.is_alive()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "second-thread-refusal-ok"


def test_subprocess_guard_consumes_owned_pending_alarm_before_delayed_unmask():
    probe = textwrap.dedent(
        """
        import signal
        import time
        from infra.envs.debate.docent_export import (
            _DocentTimeBudget,
            _docent_hard_wall_clock_guard,
        )

        old_handler = signal.getsignal(signal.SIGALRM)
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        old_timer = signal.getitimer(signal.ITIMER_REAL)
        assert signal.SIGALRM not in old_mask
        assert old_timer == (0.0, 0.0)
        budget = _DocentTimeBudget(
            deadline=time.monotonic() + 5.0,
            clock=time.monotonic,
            sleeper=time.sleep,
        )
        with _docent_hard_wall_clock_guard(budget):
            # Deterministically queue an alarm while the guard owns the
            # handler. Teardown must consume it before restoring the prior
            # handler; the separate preemption probes exercise timer expiry.
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
            signal.raise_signal(signal.SIGALRM)
            assert signal.SIGALRM in signal.sigpending()

        assert signal.getsignal(signal.SIGALRM) is old_handler
        assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == old_mask
        assert signal.getitimer(signal.ITIMER_REAL) == old_timer
        assert signal.SIGALRM not in signal.sigpending()
        # If teardown restored SIG_DFL before consuming the alarm, this delayed
        # unmask/window would terminate the subprocess instead of printing.
        time.sleep(0.10)
        print("pending-consumed-restored-survived", flush=True)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "pending-consumed-restored-survived"


def test_production_docent_http_calls_all_use_approved_connect_read_timeouts(
    monkeypatch,
):
    adapter = _SequencedAdapter(
        [
            (200, {}),
            (200, {"collection_id": "collection-123"}),
            (202, {"job_id": "job-123"}),
            (200, {"jobs": [{"job_id": "job-123", "status": "completed"}]}),
        ]
    )
    _install_adapter_on_new_sessions(monkeypatch, adapter)
    _docent_test_environment(monkeypatch)

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")

    assert result == DocentUploadResult(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="collection-123",
    )
    assert [request.method for request, _ in adapter.calls] == [
        "GET",
        "POST",
        "POST",
        "POST",
    ]
    assert [kwargs["timeout"] for _, kwargs in adapter.calls] == [
        (10.0, 120.0),
        (10.0, 120.0),
        (10.0, 120.0),
        (10.0, 120.0),
    ]
    assert adapter.responses == []


def test_two_sequential_production_uploads_spawn_no_tqdm_monitor_thread(
    monkeypatch,
):
    from docent import Docent

    sdk_tqdm = Docent.add_agent_runs.__globals__["tqdm"]
    previous_interval = sdk_tqdm.monitor_interval
    threads_before = set(threading.enumerate())
    responses = []
    for attempt in (1, 2):
        responses.extend(
            [
                (200, {}),
                (200, {"collection_id": f"collection-{attempt}"}),
                (202, {"job_id": f"job-{attempt}"}),
                (
                    200,
                    {
                        "jobs": [
                            {
                                "job_id": f"job-{attempt}",
                                "status": "completed",
                            }
                        ]
                    },
                ),
            ]
        )
    adapter = _SequencedAdapter(responses)
    _install_adapter_on_new_sessions(monkeypatch, adapter)
    _docent_test_environment(monkeypatch)

    first = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")
    second = upload(agent_runs(RECORDS), "math-pc-rl", "run-B")

    assert first == DocentUploadResult(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="collection-1",
    )
    assert second == DocentUploadResult(
        collection_name="math-pc-rl--launch-run-B",
        launch_namespace="run-B",
        collection_id="collection-2",
    )
    assert set(threading.enumerate()) == threads_before
    assert sdk_tqdm.monitor is None
    assert sdk_tqdm.monitor_interval == previous_interval
    assert len(adapter.calls) == 8


@pytest.mark.parametrize(
    ("phase", "responses", "expected_collection_id", "expected_sends"),
    [
        ("auth", [(200, {}, 301.0)], None, 1),
        (
            "create",
            [(200, {}, 0.0), (200, {"collection_id": "collection-123"}, 301.0)],
            "collection-123",
            2,
        ),
        (
            "batch",
            [
                (200, {}, 0.0),
                (200, {"collection_id": "collection-123"}, 0.0),
                (202, {"job_id": "job-123"}, 301.0),
            ],
            "collection-123",
            3,
        ),
        (
            "status",
            [
                (200, {}, 0.0),
                (200, {"collection_id": "collection-123"}, 0.0),
                (202, {"job_id": "job-123"}, 0.0),
                (
                    200,
                    {"jobs": [{"job_id": "job-123", "status": "completed"}]},
                    301.0,
                ),
            ],
            "collection-123",
            4,
        ),
    ],
)
def test_total_docent_deadline_expires_safely_in_every_http_phase(
    monkeypatch,
    phase,
    responses,
    expected_collection_id,
    expected_sends,
):
    clock = _FakeMonotonicClock()
    adapter = _TimedSequencedAdapter(clock, responses)
    _install_adapter_on_new_sessions(monkeypatch, adapter)
    _docent_test_environment(monkeypatch)
    budget = _DocentTimeBudget.fixed(clock=clock, sleeper=clock.sleep)

    result = _upload_with_budget(
        agent_runs(RECORDS),
        "math-pc-rl",
        "run-A",
        _budget=budget,
    )

    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id=expected_collection_id,
        error_type="DocentUploadDeadlineError",
    ), phase
    assert len(adapter.calls) == expected_sends
    assert adapter.responses == []


def test_docent_timeout_tuple_clamps_both_values_to_total_time_remaining():
    clock = _FakeMonotonicClock()
    budget = _DocentTimeBudget.fixed(clock=clock, sleeper=clock.sleep)
    clock.advance(295.0)
    session, adapter = _sequenced_session([(200, {"collection_id": "collection-123"})])
    client = _bare_installed_docent(session)
    _bind_single_attempt_agent_run_posts(client, _budget=budget)

    client.create_collection(name="math-pc-rl--launch-run-A")

    assert len(adapter.calls) == 1
    assert adapter.calls[0][1]["timeout"] == (5.0, 5.0)


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (301, "DocentMutationRedirectError"),
        (500, "DocentMutationHTTPStatusError"),
    ],
)
def test_constructor_auth_failure_never_redirects_or_retries(
    monkeypatch, status_code, error_type, capsys
):
    adapter = _SequencedAdapter(
        [
            (status_code, {"detail": "RAW-AUTH-BODY-MARKER"}),
            (200, {"detail": "must-not-be-sent"}),
        ]
    )
    _install_adapter_on_new_sessions(monkeypatch, adapter)
    _docent_test_environment(monkeypatch)

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")

    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id=None,
        error_type=error_type,
    )
    assert len(adapter.calls) == 1
    assert len(adapter.responses) == 1
    captured = capsys.readouterr()
    assert "RAW-AUTH-BODY-MARKER" not in captured.out
    assert "RAW-AUTH-BODY-MARKER" not in captured.err


def test_constructor_refuses_retrying_adapter_before_auth_send(monkeypatch):
    adapter = _SequencedAdapter(
        [(200, {"detail": "must-not-be-sent"})],
        max_retries=Retry(total=1, allowed_methods=None),
    )
    _install_adapter_on_new_sessions(monkeypatch, adapter)
    _docent_test_environment(monkeypatch)

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")

    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id=None,
        error_type="RuntimeError",
    )
    assert adapter.calls == []


def test_installed_docent_ambiguous_500_agent_batch_is_not_repeated():
    """A persisted-like 500 followed by 200 must leave the 200 untouched."""
    session, adapter = _sequenced_session(
        [
            (500, {"detail": "persisted, acknowledgement lost"}),
            (200, {"job_id": "duplicate-if-retried"}),
        ]
    )
    client = _bare_installed_docent(session)
    _bind_single_attempt_agent_run_posts(client)

    with pytest.raises(DocentMutationHTTPStatusError):
        client.add_agent_runs(
            "collection-123",
            agent_runs(RECORDS),
            compression="none",
            wait=False,
        )

    assert len(adapter.calls) == 1
    assert adapter.responses == [(200, {"job_id": "duplicate-if-retried"})]


def test_installed_docent_single_attempt_agent_batch_succeeds_normally():
    session, adapter = _sequenced_session(
        [(202, {"job_id": "job-123"})]
    )
    client = _bare_installed_docent(session)
    _bind_single_attempt_agent_run_posts(client)

    result = client.add_agent_runs(
        "collection-123",
        agent_runs(RECORDS),
        compression="none",
        wait=False,
    )

    assert len(adapter.calls) == 1
    assert result == {
        "status": "enqueued",
        "total_runs_added": len(RECORDS),
        "job_ids": ["job-123"],
    }


def test_installed_docent_preserves_distinct_batch_posts(monkeypatch):
    from docent import Docent

    # Each fixture AgentRun is below 1 KiB, while pairs are above it. Exercise
    # the installed size batcher rather than replacing it with a synthetic one.
    monkeypatch.setitem(
        Docent.add_agent_runs.__globals__, "MAX_AGENT_RUN_PAYLOAD_BYTES", 1_000
    )
    session, adapter = _sequenced_session(
        [
            (202, {"job_id": "job-1"}),
            (202, {"job_id": "job-2"}),
            (202, {"job_id": "job-3"}),
        ]
    )
    client = _bare_installed_docent(session)
    _bind_single_attempt_agent_run_posts(client)

    result = client.add_agent_runs(
        "collection-123",
        agent_runs(RECORDS),
        compression="none",
        wait=False,
    )

    assert len(adapter.calls) == 3
    assert result["job_ids"] == ["job-1", "job-2", "job-3"]


def test_docent_sdk_version_drift_refuses_before_collection_mutation(monkeypatch):
    import infra.envs.debate.docent_export as docent_export

    session, adapter = _sequenced_session([])
    client = _bare_installed_docent(session)
    monkeypatch.setattr(
        docent_export,
        "version",
        lambda distribution: "0.1.78" if distribution == "docent" else "0.1.77",
    )

    with pytest.raises(RuntimeError, match="unreviewed Docent SDK version"):
        _bind_single_attempt_agent_run_posts(client)
    assert adapter.calls == []


def test_upload_sdk_drift_refuses_before_constructor_or_post(monkeypatch):
    import docent
    import infra.envs.debate.docent_export as docent_export

    constructed = False

    class MustNotConstruct(docent.Docent):
        def __new__(cls, **kwargs):
            nonlocal constructed
            constructed = True
            raise AssertionError("version drift must refuse before construction")

    monkeypatch.setattr(docent, "Docent", MustNotConstruct)
    monkeypatch.setattr(
        docent_export,
        "version",
        lambda distribution: "0.1.78" if distribution == "docent" else "0.1.77",
    )
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")
    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id=None,
        error_type="RuntimeError",
    )
    assert constructed is False


def test_upload_sdk_class_seam_drift_refuses_before_constructor(monkeypatch):
    import docent

    constructed = False

    class DriftedDocent(docent.Docent):
        def __new__(cls, **kwargs):
            nonlocal constructed
            constructed = True
            raise AssertionError("class drift must refuse before construction")

        def add_agent_runs(self, collection_id, agent_runs):
            raise AssertionError("drifted seam")

    monkeypatch.setattr(docent, "Docent", DriftedDocent)
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")
    assert isinstance(result, DocentUploadFailure)
    assert result.collection_id is None
    assert result.error_type == "RuntimeError"
    assert constructed is False


def test_upload_auth_seam_drift_refuses_before_constructor(monkeypatch):
    import docent

    constructed = False

    class DriftedLoginDocent(docent.Docent):
        def __new__(cls, **kwargs):
            nonlocal constructed
            constructed = True
            raise AssertionError("auth seam drift must refuse before construction")

        def _login(self, api_key):
            raise AssertionError("unreviewed auth seam")

    monkeypatch.setattr(docent, "Docent", DriftedLoginDocent)
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")

    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id=None,
        error_type="RuntimeError",
    )
    assert constructed is False


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("operation", ["create", "batch"])
def test_installed_docent_redirect_never_replays_mutation(operation, status_code):
    """A real Session receives a redirect plus a tempting success response."""
    session, adapter = _sequenced_session(
        [
            (status_code, {"detail": "mutation may already have persisted"}),
            (200, {"collection_id": "duplicate-if-followed"}),
        ]
    )
    client = _bare_installed_docent(session)
    _bind_single_attempt_agent_run_posts(client)

    with pytest.raises(DocentMutationRedirectError):
        if operation == "create":
            client.create_collection(name="math-pc-rl--launch-run-A")
        else:
            client.add_agent_runs(
                "collection-123",
                agent_runs(RECORDS),
                compression="none",
                wait=False,
            )

    assert len(adapter.calls) == 1
    assert adapter.responses == [(200, {"collection_id": "duplicate-if-followed"})]


@pytest.mark.parametrize("status_code", [300, 400, 500])
@pytest.mark.parametrize("operation", ["create", "batch"])
def test_installed_docent_non_2xx_never_retries_mutation(operation, status_code):
    session, adapter = _sequenced_session(
        [
            (status_code, {"detail": "persisted, acknowledgement lost"}),
            (200, {"collection_id": "duplicate-if-retried"}),
        ]
    )
    client = _bare_installed_docent(session)
    _bind_single_attempt_agent_run_posts(client)

    with pytest.raises(DocentMutationHTTPStatusError):
        if operation == "create":
            client.create_collection(name="math-pc-rl--launch-run-A")
        else:
            client.add_agent_runs(
                "collection-123",
                agent_runs(RECORDS),
                compression="none",
                wait=False,
            )

    assert len(adapter.calls) == 1
    assert len(adapter.responses) == 1


def test_installed_docent_create_then_batch_each_send_once_normally():
    session, adapter = _sequenced_session(
        [
            (200, {"collection_id": "collection-123"}),
            (202, {"job_id": "job-123"}),
        ]
    )
    client = _bare_installed_docent(session)
    _bind_single_attempt_agent_run_posts(client)

    collection_id = client.create_collection(name="math-pc-rl--launch-run-A")
    result = client.add_agent_runs(
        collection_id,
        agent_runs(RECORDS),
        compression="none",
        wait=False,
    )

    assert collection_id == "collection-123"
    assert result["job_ids"] == ["job-123"]
    assert len(adapter.calls) == 2


def test_docent_retrying_session_adapter_refuses_before_mutation():
    session, adapter = _sequenced_session(
        [(200, {"collection_id": "must-not-be-created"})],
        max_retries=Retry(total=1, allowed_methods=None),
    )
    client = _bare_installed_docent(session)

    with pytest.raises(RuntimeError, match="transport adapter can retry"):
        _bind_single_attempt_agent_run_posts(client)
    assert adapter.calls == []


def test_upload_precreate_failure_has_no_collection_id_or_raw_message(monkeypatch):
    import docent

    session, adapter = _sequenced_session(
        [(500, {"detail": "sensitive pre-create upstream message"})]
    )
    session.headers["Authorization"] = "Bearer test-only-secret-token"
    client = _bare_installed_docent(session)
    class ReturningDocent(docent.Docent):
        def __new__(cls, **kwargs):
            return client

    monkeypatch.setattr(docent, "Docent", ReturningDocent)
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")

    failure = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")
    assert isinstance(failure, DocentUploadFailure)
    assert failure.collection_name == "math-pc-rl--launch-run-A"
    assert failure.launch_namespace == "run-A"
    assert failure.collection_id is None
    assert failure.error_type == "DocentMutationHTTPStatusError"
    assert not isinstance(failure, BaseException)
    traceback_cursor = getattr(failure, "__traceback__", None)
    traceback_frames = []
    while traceback_cursor is not None:
        traceback_frames.append(traceback_cursor.tb_frame.f_locals)
        traceback_cursor = traceback_cursor.tb_next
    assert traceback_frames == []
    # There is no exception traceback to retain upload() frames. The complete
    # result representation contains only the four sanitized receipt fields.
    rendered = repr(failure)
    assert "sensitive" not in rendered
    assert "test-only-secret-token" not in rendered
    assert RECORDS[0]["completion"] not in rendered
    assert len(adapter.calls) == 1


def test_empty_upload_refuses_before_client_construction(monkeypatch):
    import docent

    constructed = False

    class MustNotConstruct:
        def __init__(self, **kwargs):
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(docent, "Docent", MustNotConstruct)
    with pytest.raises(ValueError, match="at least one AgentRun"):
        upload([], "math-pc-rl", "run-A")
    assert constructed is False


@pytest.mark.parametrize(
    "collection_id",
    ["", "   ", ".", "..", "../other", "folder/id", "a%2Fb", "a" * 129],
)
def test_upload_rejects_malformed_collection_id_before_any_batch(
    monkeypatch, collection_id
):
    import docent

    session, adapter = _sequenced_session(
        [(200, {"collection_id": collection_id})]
    )
    client = _bare_installed_docent(session)

    class ReturningDocent(docent.Docent):
        def __new__(cls, **kwargs):
            return client

    monkeypatch.setattr(docent, "Docent", ReturningDocent)
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")
    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id=None,
        error_type="DocentCollectionAcknowledgementError",
    )
    assert len(adapter.calls) == 1
    assert adapter.calls[0][0].url.endswith("/rest/create")


def test_upload_rejects_202_without_job_id_after_one_batch(monkeypatch):
    import docent

    session, adapter = _sequenced_session(
        [
            (200, {"collection_id": "collection-123"}),
            (202, {}),
        ]
    )
    client = _bare_installed_docent(session)

    class ReturningDocent(docent.Docent):
        def __new__(cls, **kwargs):
            return client

    monkeypatch.setattr(docent, "Docent", ReturningDocent)
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")
    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="collection-123",
        error_type="DocentIngestionAcknowledgementError",
    )
    assert len(adapter.calls) == 2


def test_upload_accepts_real_sdk_result_only_after_job_completion(monkeypatch):
    import docent

    session, adapter = _sequenced_session(
        [
            (200, {"collection_id": "collection-123"}),
            (202, {"job_id": "job-123"}),
            (
                200,
                {"jobs": [{"job_id": "job-123", "status": "completed"}]},
            ),
        ]
    )
    client = _bare_installed_docent(session)

    class ReturningDocent(docent.Docent):
        def __new__(cls, **kwargs):
            return client

    monkeypatch.setattr(docent, "Docent", ReturningDocent)
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")
    assert result == DocentUploadResult(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="collection-123",
    )
    assert len(adapter.calls) == 3
    assert adapter.calls[-1][0].url.endswith("/agent_runs/jobs/batch_status")


@pytest.mark.parametrize("terminal_status", ["failed", "canceled", "unknown-state"])
def test_upload_terminal_or_unknown_job_status_fails_without_polling_forever(
    monkeypatch, terminal_status
):
    import docent

    session, adapter = _sequenced_session(
        [
            (200, {"collection_id": "collection-123"}),
            (202, {"job_id": "job-123"}),
            (
                200,
                {"jobs": [{"job_id": "job-123", "status": terminal_status}]},
            ),
        ]
    )
    client = _bare_installed_docent(session)

    class ReturningDocent(docent.Docent):
        def __new__(cls, **kwargs):
            return client

    monkeypatch.setattr(docent, "Docent", ReturningDocent)
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")
    monkeypatch.setattr(
        "infra.envs.debate.docent_export.time.sleep",
        lambda _seconds: pytest.fail("terminal status must not sleep or poll again"),
    )

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")
    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="collection-123",
        error_type="DocentIngestionStatusError",
    )
    assert len(adapter.calls) == 3


@pytest.mark.parametrize(
    ("requested", "rows"),
    [
        (["job-1"], []),
        (
            ["job-1"],
            [
                {"job_id": "job-1", "status": "completed"},
                {"job_id": "job-extra", "status": "completed"},
            ],
        ),
        (
            ["job-1", "job-2"],
            [
                {"job_id": "job-1", "status": "completed"},
                {"job_id": "job-1", "status": "completed"},
            ],
        ),
        (["job-1"], [{"job_id": "job-1"}]),
        (["job-1"], {"job_id": "job-1", "status": "completed"}),
    ],
    ids=["missing", "extra", "duplicate", "malformed-row", "not-a-list"],
)
def test_bound_status_census_rejects_non_exact_rows_after_one_send(
    requested, rows
):
    session, adapter = _sequenced_session([(200, {"jobs": rows})])
    client = _bare_installed_docent(session)
    _bind_single_attempt_agent_run_posts(client)

    with pytest.raises(DocentIngestionStatusError):
        client._wait_for_jobs("collection-123", requested, poll_interval=0)
    assert len(adapter.calls) == 1


def test_bound_status_polling_chunks_101_jobs_and_sleeps_once_per_sweep():
    job_ids = [f"job-{i:03d}" for i in range(101)]

    def rows(ids, status):
        return {"jobs": [{"job_id": job_id, "status": status} for job_id in ids]}

    session, adapter = _sequenced_session(
        [
            (200, rows(job_ids[:100], "pending")),
            (200, rows(job_ids[100:], "running")),
            (200, rows(job_ids[:100], "completed")),
            (200, rows(job_ids[100:], "completed")),
        ]
    )
    clock = _FakeMonotonicClock()
    budget = _DocentTimeBudget.fixed(clock=clock, sleeper=clock.sleep)
    client = _bare_installed_docent(session)
    _bind_single_attempt_agent_run_posts(client, _budget=budget)

    client._wait_for_jobs("collection-123", job_ids)

    assert len(adapter.calls) == 4
    requested_chunks = [
        json.loads(request.body)["job_ids"] for request, _kwargs in adapter.calls
    ]
    assert requested_chunks == [
        job_ids[:100],
        job_ids[100:],
        job_ids[:100],
        job_ids[100:],
    ]
    assert all(len(chunk) <= 100 for chunk in requested_chunks)
    assert clock.sleeps == [1.0]


def test_status_poll_clamps_final_sleep_and_refuses_next_send_at_deadline():
    clock = _FakeMonotonicClock()
    adapter = _TimedSequencedAdapter(
        clock,
        [
            (
                200,
                {"jobs": [{"job_id": "job-1", "status": "pending"}]},
                299.5,
            )
        ],
    )
    session = requests.Session()
    session.mount("https://", adapter)
    budget = _DocentTimeBudget.fixed(clock=clock, sleeper=clock.sleep)
    client = _bare_installed_docent(session)
    _bind_single_attempt_agent_run_posts(client, _budget=budget)

    with pytest.raises(DocentUploadDeadlineError):
        client._wait_for_jobs("collection-123", ["job-1"])

    assert len(adapter.calls) == 1
    assert clock.sleeps == [0.5]


def test_exhausted_budget_refuses_sleep_without_calling_sleeper():
    clock = _FakeMonotonicClock()
    budget = _DocentTimeBudget.fixed(clock=clock, sleeper=clock.sleep)
    clock.advance(300.0)

    with pytest.raises(DocentUploadDeadlineError):
        budget.sleep(1.0)

    assert clock.sleeps == []


@pytest.mark.parametrize(
    "job_ids",
    [
        ["job-1", "job-1"],
        ["job-1", "../malformed"],
    ],
    ids=["duplicate-original-ids", "malformed-original-id"],
)
def test_bound_status_polling_rejects_invalid_original_job_ids_before_send(job_ids):
    session, adapter = _sequenced_session([])
    client = _bare_installed_docent(session)
    _bind_single_attempt_agent_run_posts(client)

    with pytest.raises(DocentIngestionStatusError):
        client._wait_for_jobs("collection-123", job_ids)

    assert adapter.calls == []


def test_upload_pending_and_running_then_completed_polls_boundedly(monkeypatch):
    import docent

    session, adapter = _sequenced_session(
        [
            (200, {"collection_id": "collection-123"}),
            (202, {"job_id": "job-123"}),
            (200, {"jobs": [{"job_id": "job-123", "status": "pending"}]}),
            (200, {"jobs": [{"job_id": "job-123", "status": "running"}]}),
            (200, {"jobs": [{"job_id": "job-123", "status": "completed"}]}),
        ]
    )
    client = _bare_installed_docent(session)

    class ReturningDocent(docent.Docent):
        def __new__(cls, **kwargs):
            return client

    monkeypatch.setattr(docent, "Docent", ReturningDocent)
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")
    sleeps = []
    monkeypatch.setattr(
        "infra.envs.debate.docent_export.time.sleep", sleeps.append
    )

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")
    assert result == DocentUploadResult(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="collection-123",
    )
    assert len(adapter.calls) == 5
    assert sleeps == [1.0, 1.0]


def test_upload_rejects_partial_job_ids_across_real_sdk_batches(
    monkeypatch,
):
    import docent
    from docent import Docent

    monkeypatch.setitem(
        Docent.add_agent_runs.__globals__, "MAX_AGENT_RUN_PAYLOAD_BYTES", 1_000
    )
    session, adapter = _sequenced_session(
        [
            (200, {"collection_id": "collection-123"}),
            (202, {"job_id": "job-1"}),
            (202, {}),
            (202, {"job_id": "job-3"}),
            (
                200,
                {
                    "jobs": [
                        {"job_id": "job-1", "status": "completed"},
                        {"job_id": "job-3", "status": "completed"},
                    ]
                },
            ),
        ]
    )
    client = _bare_installed_docent(session)

    class ReturningDocent(docent.Docent):
        def __new__(cls, **kwargs):
            return client

    monkeypatch.setattr(docent, "Docent", ReturningDocent)
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")
    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="collection-123",
        error_type="DocentIngestionAcknowledgementError",
    )
    ingestion_calls = [
        request
        for request, _kwargs in adapter.calls
        if request.url.endswith("/agent_runs")
    ]
    status_calls = [
        request
        for request, _kwargs in adapter.calls
        if request.url.endswith("/agent_runs/jobs/batch_status")
    ]
    assert len(adapter.calls) == 5
    assert len(ingestion_calls) == 3
    assert len(status_calls) == 1


def test_upload_rejects_malformed_job_id_before_status_send(monkeypatch):
    import docent

    session, adapter = _sequenced_session(
        [
            (200, {"collection_id": "collection-123"}),
            (202, {"job_id": "../malformed"}),
        ]
    )
    client = _bare_installed_docent(session)

    class ReturningDocent(docent.Docent):
        def __new__(cls, **kwargs):
            return client

    monkeypatch.setattr(docent, "Docent", ReturningDocent)
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")

    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="collection-123",
        error_type="DocentIngestionStatusError",
    )
    assert len(adapter.calls) == 2


def test_upload_rejects_duplicate_original_job_ids_before_status_send(
    monkeypatch,
):
    import docent
    from docent import Docent

    monkeypatch.setitem(
        Docent.add_agent_runs.__globals__, "MAX_AGENT_RUN_PAYLOAD_BYTES", 1_000
    )
    session, adapter = _sequenced_session(
        [
            (200, {"collection_id": "collection-123"}),
            (202, {"job_id": "job-duplicate"}),
            (202, {"job_id": "job-duplicate"}),
            (202, {"job_id": "job-duplicate"}),
        ]
    )
    client = _bare_installed_docent(session)

    class ReturningDocent(docent.Docent):
        def __new__(cls, **kwargs):
            return client

    monkeypatch.setattr(docent, "Docent", ReturningDocent)
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")

    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="collection-123",
        error_type="DocentIngestionStatusError",
    )
    assert len(adapter.calls) == 4


@pytest.mark.parametrize(
    ("batch_posts", "reported_total", "reported_job_ids"),
    [
        (1, len(RECORDS) - 1, ["job-1"]),
        (2, len(RECORDS), ["job-1"]),
    ],
    ids=["total-runs-mismatch", "batch-count-mismatch"],
)
def test_upload_rejects_total_runs_or_batch_count_mismatch(
    monkeypatch,
    batch_posts,
    reported_total,
    reported_job_ids,
):
    import docent

    session, adapter = _sequenced_session(
        [(200, {"collection_id": "collection-123"})]
        + [(202, {"job_id": f"server-job-{i}"}) for i in range(batch_posts)]
    )
    client = _bare_installed_docent(session)

    def mismatched_add(collection_id, added_runs):
        for _ in range(batch_posts):
            client._post_with_retry(
                f"{client._api_url}/{collection_id}/agent_runs"
            )
        return {
            "status": "success",
            "total_runs_added": reported_total,
            "job_ids": reported_job_ids,
        }

    client.add_agent_runs = mismatched_add

    class ReturningDocent(docent.Docent):
        def __new__(cls, **kwargs):
            return client

    monkeypatch.setattr(docent, "Docent", ReturningDocent)
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")

    result = upload(agent_runs(RECORDS), "math-pc-rl", "run-A")

    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="collection-123",
        error_type="DocentIngestionAcknowledgementError",
    )
    assert len(adapter.calls) == 1 + batch_posts


def test_docent_version_is_exactly_pinned_in_all_install_surfaces():
    repo_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((repo_root / "pyproject.toml").read_text())
    assert "docent-python==0.1.77" in project["project"]["dependencies"]
    assert "wandb==0.28.1" in project["project"]["dependencies"]

    for relative in ("scripts/provision_pod.sh", "scripts/provision_blackwell.sh"):
        provision = (repo_root / relative).read_text()
        assert '"docent-python==0.1.77"' in provision
        assert "docent-python>=" not in provision
        assert '"wandb==0.28.1"' in provision

    lock = (repo_root / "uv.lock").read_text()
    assert '{ name = "docent-python", specifier = "==0.1.77" }' in lock
    assert 'name = "docent-python"\nversion = "0.1.77"' in lock
    assert 'name = "docent"\nversion = "0.1.77"' in lock
    assert '{ name = "wandb", specifier = "==0.28.1" }' in lock


def test_runtime_docent_version_matches_reviewed_pin():
    assert version("docent-python") == "0.1.77"
    assert version("docent") == "0.1.77"
