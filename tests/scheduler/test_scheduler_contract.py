"""Executable contract for the deterministic RunPod scheduler.

These tests are intentionally provider-free.  They use a real SQLite file and
small recording ports around the approved ``reconcile_once`` seam; paid
RunPod behavior belongs in a separately marked integration suite.

Expected public surface in ``tools.scheduler``::

    store = SQLiteStore(path)
    job_id = store.submit_job(
        command=(...), profile="h200x1", inputs=(...), outputs=(...),
        max_attempts=3,
    )
    store.get_job(job_id).state
    store.list_attempts(job_id)
    store.close()
    reconcile_once(store, provider, transfer, remote_attempts)

The three ports are duck typed by the recording fakes below.  This specifies
behavior, not an inheritance hierarchy or a particular RunPod client.

The transfer port speaks the same two verbs as
``tests/scheduler/test_transfer_contract.py``: ``stage_inputs`` and
``collect_attempt``.  The scheduler-facing adapter is addressed by worker and
attempt with the artifacts declared for the job, and one ``collect_attempt``
call covers declared outputs plus stdout/stderr for both successful and failed
attempts.  Its ``command_exit_code`` comes from the terminal remote-attempt
observation, rather than being inferred from a scheduler state label.

Workers are provider qualified.  A provider names itself as ``provider.name``
(``None`` for the single unnamed provider the older tests below use), a profile
may carry that name as ``"<provider>:<flavor>"``, and a provider serves a job
only when the two agree.  Scheduler side worker identity is therefore the pair
(provider, external id), so two providers may return the same external id.

Deleting a worker is a different verb from stopping it: ``stop_worker`` returns
a worker to inventory, ``destroy_worker`` removes it for good.  Two optional
keyword arguments drive that policy::

    reconcile_once(store, provider, transfer, remote_attempts,
                   now=<seconds>, stopped_delete_after=<seconds>)

``now`` is the clock reading to use, so retention deadlines are deterministic
here and a worker stop time is recorded on the same clock.
``stopped_delete_after`` is the already-resolved profile retention duration.
The persistent application supplies 24 hours for ordinary paid profiles,
6 hours for explicitly ephemeral one-off test profiles, and ``None`` only for
profiles explicitly configured as free/keep-forever.  The reconcile core does
not infer cost from marketplace pricing.
Deletion applies only to workers whose creation by this scheduler is proven by
durable provenance in this SQLite store, only once they are stopped, evacuated
and idle, and never to registered, manual, or foreign workers.  Provider
inventory metadata is descriptive, not authority to destroy: even a
pre-existing worker that claims ``ownership="scheduler-created"`` is protected
when this store has no corresponding creation record.

Jobs are independent queue entries.  There is deliberately no dependency,
fan-in, or DAG behavior in this contract and no API for one.
"""

from __future__ import annotations

import importlib
import sqlite3
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

import pytest


# A fixed clock reading: retention arithmetic stays deterministic, and a
# wall-clock leak in the implementation shows up as a wrong deadline.
_T0 = 1_700_000_000.0
_HOUR = 60.0 * 60.0
_PAID_RETENTION = 24.0 * _HOUR
_EPHEMERAL_RETENTION = 6.0 * _HOUR


@pytest.fixture
def scheduler() -> ModuleType:
    """Import lazily so the contract suite collects before implementation."""
    try:
        return importlib.import_module("tools.scheduler")
    except ModuleNotFoundError as exc:
        if exc.name != "tools.scheduler":
            raise
        pytest.fail(
            "tools.scheduler does not exist yet; this is the expected red "
            "baseline for the scheduler contract",
            pytrace=False,
        )


@dataclass
class _Worker:
    worker_id: str
    profile: str
    state: str
    # Provider inventory may describe registered/manual/foreign ownership, but
    # this field is not destructive authority.  GC eligibility must come from
    # the SQLite store's durable record that create_worker returned this worker.
    ownership: str = "registered"


@dataclass(frozen=True)
class _AttemptObservation:
    state: str
    command_exit_code: int | None = None


class _Provider:
    def __init__(
        self,
        workers=(),
        events=None,
        *,
        name: str | None = None,
        resume_immediately: bool = True,
        stop_immediately: bool = True,
        resume_errors=(),
        destroy_errors=(),
        before_destroy: Callable[[str], None] | None = None,
    ):
        self.workers = {worker.worker_id: worker for worker in workers}
        self.events = events if events is not None else []
        self.name = name  # the provider identity a profile may qualify with
        self.resume_immediately = resume_immediately
        self.stop_immediately = stop_immediately
        self.list_errors = deque()
        self.resume_errors = deque(resume_errors)
        self.destroy_errors = deque(destroy_errors)
        self.before_destroy = before_destroy
        self.created: list[str] = []
        self.resumed: list[str] = []
        self.stopped: list[str] = []
        self.destroy_calls: list[str] = []  # every attempt, acknowledged or not

    def list_workers(self):
        if self.list_errors:
            raise self.list_errors.popleft()
        return list(self.workers.values())

    def resume_worker(self, worker_id: str):
        self.events.append(("resume", worker_id))
        self.resumed.append(worker_id)
        if self.resume_errors:
            raise self.resume_errors.popleft()
        self.workers[worker_id].state = (
            "running" if self.resume_immediately else "starting"
        )
        return self.workers[worker_id]

    def create_worker(self, profile: str):
        worker_id = f"created-{len(self.created) + 1}"
        self.events.append(("create", worker_id, profile))
        self.created.append(worker_id)
        worker = _Worker(worker_id, profile, "running", "scheduler-created")
        self.workers[worker_id] = worker
        return worker

    def stop_worker(self, worker_id: str):
        self.events.append(("stop", worker_id))
        self.stopped.append(worker_id)
        self.workers[worker_id].state = (
            "stopped" if self.stop_immediately else "stopping"
        )

    def destroy_worker(self, worker_id: str):
        # Let a test inspect committed scheduler state at provider-call entry,
        # before this fake records any provider-side effect or acknowledgement.
        if self.before_destroy is not None:
            self.before_destroy(worker_id)
        # Recorded before the acknowledgement, so a lost one stays visible.
        self.events.append(("destroy", worker_id))
        self.destroy_calls.append(worker_id)
        if self.destroy_errors:
            raise self.destroy_errors.popleft()
        self.workers.pop(worker_id, None)


class _Transfer:
    """Scheduler-facing adapter over the ``tools.scheduler.transfer`` verbs."""

    def __init__(
        self,
        *,
        stage_results=(),
        collect_results=(),
        logs=("stdout.log", "stderr.log"),
        events=None,
    ):
        self.stage_results = deque(stage_results)
        self.collect_results = deque(collect_results)
        self.logs = tuple(logs)
        self.events = events if events is not None else []
        self.stage_calls = []
        self.collect_calls = []

    def stage_inputs(self, worker_id: str, attempt_id: str, inputs: tuple[str, ...]):
        result = self.stage_results.popleft() if self.stage_results else True
        self.stage_calls.append((worker_id, attempt_id, tuple(inputs), result))
        self.events.append(("stage", attempt_id, result))
        return result

    def collect_attempt(
        self,
        worker_id: str,
        attempt_id: str,
        outputs: tuple[str, ...] = (),
        *,
        command_exit_code: int,
    ):
        # One verb for every terminal attempt: declared outputs plus both
        # standard logs, whether the command succeeded or failed.  The exact
        # command exit code drives which declared outputs are mandatory.
        result = self.collect_results.popleft() if self.collect_results else True
        self.collect_calls.append(
            (
                worker_id,
                attempt_id,
                tuple(outputs),
                self.logs,
                command_exit_code,
                result,
            )
        )
        self.events.append(("collect", attempt_id, result))
        return result


class _RemoteAttempts:
    def __init__(self, outcomes=(), events=None, *, start_results=()):
        self.outcomes = deque(outcomes)
        self.start_results = deque(start_results)
        self.events = events if events is not None else []
        self.starts = []
        self._states: dict[str, _AttemptObservation] = {}

    @staticmethod
    def _observation(outcome) -> _AttemptObservation:
        if isinstance(outcome, _AttemptObservation):
            return outcome
        if isinstance(outcome, tuple):
            state, command_exit_code = outcome
            return _AttemptObservation(state, command_exit_code)
        command_exit_code = 0 if outcome == "succeeded" else None
        if outcome == "failed":
            command_exit_code = 1
        return _AttemptObservation(outcome, command_exit_code)

    def start_attempt(
        self, worker_id: str, attempt_id: str, command: tuple[str, ...]
    ):
        self.events.append(("start", attempt_id))
        self.starts.append((worker_id, attempt_id, tuple(command)))
        self._states[attempt_id] = self._observation(
            self.outcomes.popleft() if self.outcomes else "running"
        )
        result = self.start_results.popleft() if self.start_results else True
        if isinstance(result, BaseException):
            raise result
        return result

    def inspect_attempt(self, worker_id: str, attempt_id: str):
        return self._states.get(attempt_id, _AttemptObservation("unknown"))

    def finish(
        self, attempt_id: str, state: str, command_exit_code: int | None = None
    ) -> None:
        """Let a test end a long-running attempt at a chosen point."""
        if command_exit_code is None and state == "succeeded":
            command_exit_code = 0
        self._states[attempt_id] = _AttemptObservation(state, command_exit_code)


def _store(scheduler: ModuleType, path: Path):
    return scheduler.SQLiteStore(path)


def _sqlite_has_durable_worker_state(
    path: Path, provider: str, worker_id: str, state: str
) -> bool:
    """Inspect committed state through a second connection, schema-neutrally."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as reader:
        tables = reader.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        for (table,) in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            for row in reader.execute(f"SELECT * FROM {quoted}"):
                values = {str(value) for value in row if value is not None}
                if {provider, worker_id, state} <= values:
                    return True
    return False


def _submit(
    store,
    name: str,
    *,
    profile: str = "h200x1",
    inputs: tuple[str, ...] = ("source.tar.zst",),
    outputs: tuple[str, ...] = ("result.json",),
    max_attempts: int | None = None,
):
    kwargs = dict(
        command=("run-job", name),
        profile=profile,
        inputs=inputs,
        outputs=outputs,
    )
    if max_attempts is not None:
        kwargs["max_attempts"] = max_attempts
    return store.submit_job(**kwargs)


def _reconcile_until(
    scheduler: ModuleType,
    store,
    provider,
    transfer,
    remote,
    predicate: Callable[[], bool],
    *,
    limit: int = 30,
    **options,
):
    for _ in range(limit):
        scheduler.reconcile_once(store, provider, transfer, remote, **options)
        if predicate():
            return
    pytest.fail("scheduler did not reach the expected state within 30 reconciliations")


def test_default_is_three_total_compute_attempts(scheduler, tmp_path):
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    job_id = _submit(store, "fails-three-times")  # exercise the default
    provider = _Provider([_Worker("warm", "h200x1", "running")])
    transfer = _Transfer()
    remote = _RemoteAttempts(["failed", "failed", "failed", "succeeded"])

    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: store.get_job(job_id).state == "failed",
    )

    assert len(remote.starts) == 3
    assert len(store.list_attempts(job_id)) == 3
    for _ in range(5):
        scheduler.reconcile_once(store, provider, transfer, remote)
    assert len(remote.starts) == 3, "retry exhaustion must remain terminal"


@pytest.mark.parametrize(
    ("workers", "chosen", "expected_resume", "expected_creates"),
    [
        (
            [
                _Worker("warm", "h200x1", "running"),
                _Worker("cold", "h200x1", "stopped"),
            ],
            "warm",
            [],
            0,
        ),
        (
            [
                _Worker("wrong-warm", "a100x1", "running"),
                _Worker("cold", "h200x1", "stopped"),
            ],
            "cold",
            ["cold"],
            0,
        ),
        (
            [
                _Worker("wrong-warm", "a100x1", "running"),
                _Worker("wrong-cold", "a100x1", "stopped"),
            ],
            "created-1",
            [],
            1,
        ),
    ],
    ids=["running-first", "resume-exact-profile", "create-when-no-exact-match"],
)
def test_worker_selection_is_running_then_stopped_then_create_with_exact_profiles(
    scheduler,
    tmp_path,
    workers,
    chosen,
    expected_resume,
    expected_creates,
):
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    _submit(store, "selection")
    provider = _Provider(workers)
    transfer = _Transfer()
    remote = _RemoteAttempts(["running"])

    _reconcile_until(
        scheduler, store, provider, transfer, remote, lambda: bool(remote.starts)
    )

    assert remote.starts[0][0] == chosen
    assert provider.resumed == expected_resume
    assert len(provider.created) == expected_creates


def test_resumed_worker_is_not_used_until_provider_reports_running(
    scheduler, tmp_path
):
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    _submit(store, "wait-for-resume")
    worker = _Worker("cold", "h200x1", "stopped")
    provider = _Provider([worker], resume_immediately=False)
    transfer = _Transfer()
    remote = _RemoteAttempts(["running"])

    for _ in range(5):
        scheduler.reconcile_once(store, provider, transfer, remote)

    assert provider.resumed == ["cold"]
    assert worker.state == "starting"
    assert not transfer.stage_calls
    assert not remote.starts

    worker.state = "running"
    _reconcile_until(
        scheduler, store, provider, transfer, remote, lambda: bool(remote.starts)
    )
    assert remote.starts[0][0] == "cold"


@pytest.mark.parametrize(
    "unavailable", ["resume-raises", "resume-never-reaches-running"]
)
def test_unavailable_resume_falls_back_to_a_fresh_worker_for_free(
    scheduler, tmp_path, unavailable
):
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    # A single attempt: a resume that never yields a usable worker must not
    # spend it, because no compute ran -- unlike a worker lost mid attempt.
    job_id = _submit(store, "resume-unavailable", max_attempts=1)
    worker = _Worker("cold", "h200x1", "stopped")
    provider = _Provider(
        [worker],
        resume_immediately=unavailable != "resume-never-reaches-running",
        resume_errors=(
            [RuntimeError("no capacity for this profile")]
            if unavailable == "resume-raises"
            else ()
        ),
    )
    transfer = _Transfer()
    remote = _RemoteAttempts(["succeeded"])

    _reconcile_until(
        scheduler, store, provider, transfer, remote, lambda: bool(provider.resumed)
    )
    if unavailable == "resume-never-reaches-running":
        worker.state = "lost"  # the provider withdrew the stopped worker

    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: store.get_job(job_id).state == "succeeded",
    )

    assert provider.resumed == ["cold"]
    assert provider.created == ["created-1"]
    assert [start[0] for start in remote.starts] == ["created-1"]
    assert len(store.list_attempts(job_id)) == 1
    assert {call[0] for call in transfer.stage_calls} == {"created-1"}


def test_fifo_keeps_one_active_job_per_worker_then_hands_it_over_hot(
    scheduler, tmp_path
):
    events = []
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    first = _submit(store, "first")
    second = _submit(store, "second")
    provider = _Provider([_Worker("worker", "h200x1", "running")], events)
    transfer = _Transfer(events=events)
    remote = _RemoteAttempts(["running", "running"], events)

    for _ in range(8):
        scheduler.reconcile_once(store, provider, transfer, remote)

    assert [start[2] for start in remote.starts] == [("run-job", "first")]
    assert store.get_job(first).state == "running"
    assert store.get_job(second).state == "queued"

    remote.finish(remote.starts[0][1], "succeeded")
    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: len(remote.starts) == 2,
    )

    assert store.get_job(first).state == "succeeded"
    assert store.get_job(second).state == "running"
    assert [start[0] for start in remote.starts] == ["worker", "worker"]
    first_start = next(i for i, event in enumerate(events) if event[0] == "start")
    second_start = next(
        i
        for i, event in enumerate(events[first_start + 1 :], first_start + 1)
        if event[0] == "start"
    )
    assert not any(
        event[0] in {"stop", "resume"}
        for event in events[first_start + 1 : second_start]
    )


def test_two_running_workers_fan_out_two_fifo_jobs(scheduler, tmp_path):
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    _submit(store, "first")
    _submit(store, "second")
    provider = _Provider(
        [
            _Worker("worker-a", "h200x1", "running"),
            _Worker("worker-b", "h200x1", "running"),
        ]
    )
    transfer = _Transfer()
    remote = _RemoteAttempts(["running", "running"])

    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: len(remote.starts) == 2,
    )

    assert [start[2] for start in remote.starts] == [
        ("run-job", "first"),
        ("run-job", "second"),
    ]
    assert {start[0] for start in remote.starts} == {"worker-a", "worker-b"}
    assert provider.resumed == []
    assert provider.created == []


@pytest.mark.parametrize(
    "ambiguity",
    ["lost-launch-ack", "worker-missing", "provider-list-error", "scheduler-restart"],
)
def test_ambiguity_is_acknowledged_without_duplicating_a_started_attempt(
    scheduler, tmp_path, ambiguity
):
    db_path = tmp_path / "scheduler.sqlite"
    store = _store(scheduler, db_path)
    job_id = _submit(store, f"ambiguous-{ambiguity}")
    worker = _Worker("worker", "h200x1", "running")
    provider = _Provider([worker])
    transfer = _Transfer()
    remote = _RemoteAttempts(
        ["running"],
        start_results=(
            [TimeoutError("launch acknowledgement lost")]
            if ambiguity == "lost-launch-ack"
            else ()
        ),
    )

    _reconcile_until(
        scheduler, store, provider, transfer, remote, lambda: bool(remote.starts)
    )

    if ambiguity == "worker-missing":
        provider.workers.pop(worker.worker_id)
    elif ambiguity == "provider-list-error":
        provider.list_errors.append(ConnectionError("transient provider outage"))
    elif ambiguity == "scheduler-restart":
        store.close()
        store = _store(scheduler, db_path)

    scheduler.reconcile_once(store, provider, transfer, remote)  # tick under doubt

    if ambiguity == "worker-missing":
        provider.workers[worker.worker_id] = worker
    for _ in range(6):
        scheduler.reconcile_once(store, provider, transfer, remote)

    assert store.get_job(job_id).state == "running"
    assert len(remote.starts) == 1
    assert len(store.list_attempts(job_id)) == 1
    assert provider.created == []


def test_provider_confirmed_worker_loss_consumes_an_attempt_and_retries(
    scheduler, tmp_path
):
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    job_id = _submit(store, "lost-then-retried", max_attempts=2)
    worker = _Worker("lost-worker", "h200x1", "running")
    provider = _Provider([worker])
    transfer = _Transfer()
    remote = _RemoteAttempts(["running", "succeeded"])

    _reconcile_until(
        scheduler, store, provider, transfer, remote, lambda: bool(remote.starts)
    )
    worker.state = "lost"  # authoritative provider state, not an SSH inference

    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: store.get_job(job_id).state == "succeeded",
    )

    assert len(remote.starts) == 2
    assert len(store.list_attempts(job_id)) == 2
    assert remote.starts[0][0] == "lost-worker"
    assert remote.starts[1][0] == "created-1"


def test_ambiguous_attempt_is_quarantined_without_blocking_other_capacity(
    scheduler, tmp_path
):
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    ambiguous = _submit(store, "ambiguous")
    waiting = _submit(store, "can-use-other-capacity")
    provider = _Provider(
        [
            _Worker("ambiguous-worker", "h200x1", "running"),
            _Worker("healthy-worker", "h200x1", "running"),
        ]
    )
    transfer = _Transfer()
    remote = _RemoteAttempts(["unknown", "succeeded"])

    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: store.get_job(waiting).state == "succeeded",
    )

    assert store.get_job(ambiguous).state == "unknown"
    assert [start[2] for start in remote.starts].count(
        ("run-job", "ambiguous")
    ) == 1
    assert len(remote.starts) == 2
    assert remote.starts[0][0] == "ambiguous-worker"
    assert remote.starts[1][0] == "healthy-worker"
    assert provider.created == []
    assert "ambiguous-worker" not in provider.stopped


def test_inputs_are_staged_before_start_and_stage_failures_are_free(
    scheduler, tmp_path
):
    events = []
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    job_id = _submit(
        store,
        "staged",
        inputs=("code.tar.zst", "dataset.jsonl"),
        max_attempts=1,
    )
    provider = _Provider([_Worker("worker", "h200x1", "running")], events)
    transfer = _Transfer(stage_results=[False, False, True], events=events)
    remote = _RemoteAttempts(["succeeded"], events)

    scheduler.reconcile_once(store, provider, transfer, remote)
    scheduler.reconcile_once(store, provider, transfer, remote)
    assert not remote.starts
    assert store.get_job(job_id).state != "failed"

    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: store.get_job(job_id).state == "succeeded",
    )

    assert transfer.stage_calls[0][2] == ("code.tar.zst", "dataset.jsonl")
    assert [event[0] for event in events].index("stage") < [
        event[0] for event in events
    ].index("start")
    assert len(remote.starts) == 1
    assert len(store.list_attempts(job_id)) == 1


@pytest.mark.parametrize(
    (
        "outcomes",
        "max_attempts",
        "final_state",
        "expected_starts",
        "expected_collected_exit_codes",
    ),
    [
        ([("succeeded", 0)], 1, "succeeded", 1, [0, 0]),
        (
            [("failed", 23), ("succeeded", 0)],
            2,
            "succeeded",
            2,
            [23, 23, 0],
        ),
        ([("failed", 17)], 1, "failed", 1, [17, 17]),
    ],
    ids=["success", "failure-before-retry", "terminal-failure"],
)
def test_terminal_attempts_verify_outputs_and_logs_before_success_retry_or_stop(
    scheduler,
    tmp_path,
    outcomes,
    max_attempts,
    final_state,
    expected_starts,
    expected_collected_exit_codes,
):
    events = []
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    job_id = _submit(
        store,
        "collect",
        outputs=("model.safetensors", "metrics.json"),
        max_attempts=max_attempts,
    )
    provider = _Provider([_Worker("worker", "h200x1", "running")], events)
    transfer = _Transfer(collect_results=[False, True], events=events)
    remote = _RemoteAttempts(outcomes, events)

    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: bool(transfer.collect_calls),
    )
    assert store.get_job(job_id).state not in {"succeeded", "failed"}
    assert not provider.stopped
    assert len(remote.starts) == 1, "a failed collection must not rerun compute"

    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: store.get_job(job_id).state == final_state and bool(provider.stopped),
    )

    assert len(remote.starts) == expected_starts
    assert len(store.list_attempts(job_id)) == expected_starts
    assert transfer.collect_calls[-1][2] == ("model.safetensors", "metrics.json")
    assert transfer.collect_calls[-1][3] == ("stdout.log", "stderr.log")
    assert [call[4] for call in transfer.collect_calls] == expected_collected_exit_codes

    verified = next(
        i for i, event in enumerate(events) if event[0] == "collect" and event[2]
    )
    starts = [i for i, event in enumerate(events) if event[0] == "start"]
    if expected_starts > 1:
        assert verified < starts[1]
    assert verified < next(i for i, event in enumerate(events) if event[0] == "stop")


def test_a_normally_stopped_worker_remains_inventory_for_the_next_job(
    scheduler, tmp_path
):
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    first = _submit(store, "first")
    provider = _Provider([_Worker("reusable", "h200x1", "running")])
    transfer = _Transfer()
    remote = _RemoteAttempts(["succeeded", "succeeded"])

    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: store.get_job(first).state == "succeeded",
    )
    assert provider.workers["reusable"].state == "stopped"

    second = _submit(store, "second")
    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: store.get_job(second).state == "succeeded",
    )

    assert [start[0] for start in remote.starts] == ["reusable", "reusable"]
    assert provider.resumed == ["reusable"]
    assert provider.created == []


@pytest.mark.parametrize(
    ("retention", "accepted"),
    [
        (None, True),
        (0.001, True),
        (0.0, False),
        (-1.0, False),
        (float("nan"), False),
        (float("inf"), False),
        (float("-inf"), False),
    ],
    ids=["never", "positive", "zero", "negative", "nan", "inf", "negative-inf"],
)
def test_stopped_retention_is_none_or_positive_finite_before_provider_mutation(
    scheduler, tmp_path, retention, accepted
):
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    _submit(store, "retention-validation")
    provider = _Provider([_Worker("worker", "h200x1", "running")])
    transfer = _Transfer()
    remote = _RemoteAttempts(["running"])

    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: bool(remote.starts),
        stopped_delete_after=None,
    )
    remote.finish(remote.starts[0][1], "succeeded")

    if accepted:
        scheduler.reconcile_once(
            store,
            provider,
            transfer,
            remote,
            stopped_delete_after=retention,
        )
    else:
        with pytest.raises(ValueError):
            scheduler.reconcile_once(
                store,
                provider,
                transfer,
                remote,
                stopped_delete_after=retention,
            )
        assert provider.stopped == []
        assert provider.destroy_calls == []


def test_queued_compatible_job_wins_over_destroy_at_retention_deadline(
    scheduler, tmp_path
):
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    profile = "runpod:h200x1"
    first = _submit(store, "first", profile=profile)
    provider = _Provider(name="runpod")
    transfer = _Transfer()
    remote = _RemoteAttempts(["succeeded", "succeeded"])

    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: store.get_job(first).state == "succeeded" and bool(provider.stopped),
        now=_T0,
        stopped_delete_after=60.0,
    )

    second = _submit(store, "second", profile=profile)
    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: store.get_job(second).state == "succeeded",
        now=_T0 + 60.0,
        stopped_delete_after=60.0,
    )

    assert provider.destroy_calls == []
    assert provider.resumed == ["created-1"]
    assert [start[0] for start in remote.starts] == ["created-1", "created-1"]


def test_worker_with_pending_stop_is_not_treated_as_stopped_inventory(
    scheduler, tmp_path
):
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    first = _submit(store, "first")
    worker = _Worker("draining", "h200x1", "running")
    provider = _Provider([worker], stop_immediately=False)
    transfer = _Transfer()
    remote = _RemoteAttempts(["succeeded", "running"])

    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: bool(provider.stopped),
    )
    assert store.get_job(first).state == "succeeded"
    assert worker.state == "stopping"

    _submit(store, "second")
    for _ in range(8):
        scheduler.reconcile_once(store, provider, transfer, remote)

    assert provider.stopped.count("draining") == 1
    assert all(start[0] != "draining" for start in remote.starts[1:])


def test_provider_qualification_keeps_equal_external_ids_distinct(
    scheduler, tmp_path
):
    db_path = tmp_path / "scheduler.sqlite"
    store = _store(scheduler, db_path)
    runpod_job = _submit(store, "runpod", profile="runpod:b300x1")
    vast_job = _submit(store, "vast", profile="vast:b300x1")
    transfer = _Transfer()
    runpod = _Provider(name="runpod")
    vast = _Provider(name="vast")
    runpod_remote = _RemoteAttempts(["succeeded"])
    vast_remote = _RemoteAttempts(["succeeded"])

    _reconcile_until(
        scheduler,
        store,
        runpod,
        transfer,
        runpod_remote,
        lambda: store.get_job(runpod_job).state == "succeeded",
    )
    assert store.get_job(vast_job).state == "queued"

    _reconcile_until(
        scheduler,
        store,
        vast,
        transfer,
        vast_remote,
        lambda: store.get_job(vast_job).state == "succeeded",
    )

    # Both providers deliberately returned the same external id.  Provider
    # qualification, rather than globally unique marketplace ids, keeps both
    # workers and their attempts independent.
    assert runpod.created == vast.created == ["created-1"]
    assert [call[0] for call in runpod_remote.starts] == ["created-1"]
    assert [call[0] for call in vast_remote.starts] == ["created-1"]

    store.close()
    store = _store(scheduler, db_path)
    runpod_attempt = store.list_attempts(runpod_job)[0]
    vast_attempt = store.list_attempts(vast_job)[0]
    assert (runpod_attempt.provider, runpod_attempt.worker_id) == (
        "runpod",
        "created-1",
    )
    assert (vast_attempt.provider, vast_attempt.worker_id) == ("vast", "created-1")


@pytest.mark.parametrize(
    ("case", "profile", "outcome", "collects", "retention", "should_destroy"),
    [
        ("paid-default-24h", "runpod:h200x1", "succeeded", True, _PAID_RETENTION, True),
        ("ephemeral-6h", "runpod:one-time-b300x1", "succeeded", True, _EPHEMERAL_RETENTION, True),
        ("retention-never", "cluster:free-h200x1", "succeeded", True, None, False),
        ("registered", "runpod:h200x1", "succeeded", True, _PAID_RETENTION, False),
        ("foreign", "runpod:h200x1", None, True, _PAID_RETENTION, False),
        ("spoofed-provider-owner", "runpod:h200x1", None, True, _PAID_RETENTION, False),
        ("active", "runpod:h200x1", "running", True, _PAID_RETENTION, False),
        ("unknown", "runpod:h200x1", "unknown", True, _PAID_RETENTION, False),
        ("collecting", "runpod:h200x1", "succeeded", False, _PAID_RETENTION, False),
    ],
)
def test_gc_requires_owned_stopped_evacuated_idle_worker_and_explicit_retention(
    scheduler, tmp_path, case, profile, outcome, collects, retention, should_destroy
):
    store = _store(scheduler, tmp_path / "scheduler.sqlite")
    if case == "registered":
        workers = [_Worker("manual", profile, "running", "registered")]
    elif case == "foreign":
        workers = [_Worker("foreign", profile, "stopped", "foreign")]
    elif case == "spoofed-provider-owner":
        # The provider can claim scheduler ownership, but this fresh store has
        # no durable evidence that its create_worker path produced this worker.
        workers = [_Worker("spoofed", profile, "stopped", "scheduler-created")]
    else:
        workers = []
    provider = _Provider(workers, name=profile.split(":", 1)[0])
    transfer = _Transfer(collect_results=[collects] if collects else [False] * 30)
    remote = _RemoteAttempts([outcome] if outcome is not None else [])

    if case not in {"foreign", "spoofed-provider-owner"}:
        job_id = _submit(store, case, profile=profile)
        if case == "collecting":
            _reconcile_until(
                scheduler,
                store,
                provider,
                transfer,
                remote,
                lambda: bool(transfer.collect_calls),
                now=_T0,
                stopped_delete_after=retention,
            )
            # Isolate the collection guard from the stopped-state guard: even
            # if inventory reports this owned worker stopped, its unverified
            # attempt artifacts still forbid permanent deletion.
            provider.workers["created-1"].state = "stopped"
            scheduler.reconcile_once(
                store,
                provider,
                transfer,
                remote,
                now=_T0,
                stopped_delete_after=retention,
            )
        else:
            for _ in range(12):
                scheduler.reconcile_once(
                    store,
                    provider,
                    transfer,
                    remote,
                    now=_T0,
                    stopped_delete_after=retention,
                )
    else:
        job_id = None
        # Establish that this pre-existing stopped worker has already aged for
        # the full retention window.  Provider inventory metadata is never
        # sufficient proof of local creation, even when it claims ownership;
        # the missing durable creation record must protect it from deletion.
        scheduler.reconcile_once(
            store,
            provider,
            transfer,
            remote,
            now=_T0,
            stopped_delete_after=retention,
        )

    # Reconcile once before and once at/after the resolved profile deadline.
    # For eligible cases this also proves stop and destroy are separate.
    deadline = retention if retention is not None else _PAID_RETENTION
    scheduler.reconcile_once(
        store,
        provider,
        transfer,
        remote,
        now=_T0 + deadline - 1.0,
        stopped_delete_after=retention,
    )
    assert not provider.destroy_calls
    scheduler.reconcile_once(
        store,
        provider,
        transfer,
        remote,
        now=_T0 + deadline,
        stopped_delete_after=retention,
    )

    if retention is None:
        assert job_id is not None and store.get_job(job_id).state == "succeeded"
        assert provider.stopped == ["created-1"]
        scheduler.reconcile_once(
            store,
            provider,
            transfer,
            remote,
            now=_T0 + 100.0 * 365.0 * 24.0 * _HOUR,
            stopped_delete_after=None,
        )

    if should_destroy:
        assert provider.stopped == ["created-1"]
        assert provider.destroy_calls == ["created-1"]
        assert events_in_order(provider.events, "stop", "destroy")
    else:
        assert provider.destroy_calls == []


def test_lost_destroy_ack_is_persisted_and_quarantined_across_restart(
    scheduler, tmp_path
):
    db_path = tmp_path / "scheduler.sqlite"
    store = _store(scheduler, db_path)
    first = _submit(store, "first", profile="runpod:h200x1")
    delete_entry_snapshots = []

    def inspect_delete_entry(worker_id: str) -> None:
        delete_entry_snapshots.append(
            _sqlite_has_durable_worker_state(
                db_path, "runpod", worker_id, "deleting"
            )
        )

    provider = _Provider(
        name="runpod",
        destroy_errors=[TimeoutError("destroy ack lost")],
        before_destroy=inspect_delete_entry,
    )
    transfer = _Transfer()
    remote = _RemoteAttempts(["succeeded", "running"])

    _reconcile_until(
        scheduler,
        store,
        provider,
        transfer,
        remote,
        lambda: store.get_job(first).state == "succeeded" and bool(provider.stopped),
        now=_T0,
        stopped_delete_after=10.0,
    )
    scheduler.reconcile_once(
        store,
        provider,
        transfer,
        remote,
        now=_T0 + 10.0,
        stopped_delete_after=10.0,
    )
    assert provider.destroy_calls == ["created-1"]
    assert delete_entry_snapshots == [True], (
        "delete intent must be committed before entering the provider verb"
    )

    store.close()
    store = _store(scheduler, db_path)
    second = _submit(store, "second", profile="runpod:h200x1")
    for _ in range(12):
        scheduler.reconcile_once(
            store,
            provider,
            transfer,
            remote,
            now=_T0 + 1_000.0,
            stopped_delete_after=10.0,
        )

    assert provider.destroy_calls == ["created-1"], "never repeat an ambiguous delete"
    assert provider.created == ["created-1", "created-2"]
    assert remote.starts[-1][0] == "created-2"
    assert store.get_job(second).state == "running"


def events_in_order(events, first: str, second: str) -> bool:
    """Keep ordering assertions readable without exposing scheduler internals."""
    return next(i for i, event in enumerate(events) if event[0] == first) < next(
        i for i, event in enumerate(events) if event[0] == second
    )
