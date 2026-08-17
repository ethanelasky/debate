"""Independent tests for the task-family parsing and grading contract."""

import concurrent.futures
import threading
from typing import Any, Optional

import pytest

from infra.envs import base as env_base
from infra.envs.tasks import TASK_FAMILIES
from infra.envs.tasks.base import (
    AnswerParse,
    GraderInfrastructureError,
    TaskFamily,
)


class ContractFamily(TaskFamily):
    def source(self, ds: dict):
        return None

    def parse_answers(self, text: str) -> AnswerParse:
        strict = text.removeprefix("strict:") if text.startswith("strict:") else None
        relaxed = strict if strict is not None else text.removeprefix("relaxed:")
        return AnswerParse(strict=strict, relaxed=relaxed)

    def grade(self, meta: dict[str, Any], solution: Any) -> Optional[bool]:
        failure = meta.get("failure")
        if failure == "ordinary":
            raise ValueError("bad candidate")
        if failure == "infrastructure":
            raise GraderInfrastructureError("worker died")
        return solution == meta.get("gold")


def test_answer_format_valid_is_derived_from_strict_candidate() -> None:
    missing = AnswerParse(strict=None, relaxed="fallback")
    present_but_falsey = AnswerParse(strict="", relaxed="fallback")

    assert missing.answer_format_valid is False
    assert present_but_falsey.answer_format_valid is True
    with pytest.raises((AttributeError, TypeError)):
        missing.strict = "mutated"
    with pytest.raises((AttributeError, TypeError)):
        missing.answer_format_valid = True


def test_parse_answers_exposes_strict_and_relaxed_candidates_directly() -> None:
    family = ContractFamily()

    fallback = family.parse_answers("relaxed:fallback")
    declared = family.parse_answers("strict:declared")

    assert fallback == AnswerParse(strict=None, relaxed="fallback")
    assert declared == AnswerParse(strict="declared", relaxed="declared")


@pytest.mark.parametrize("family_cls", TASK_FAMILIES.values(), ids=lambda cls: cls.__name__)
def test_concrete_families_have_no_legacy_extractor_api(family_cls) -> None:
    assert not hasattr(family_cls, "extractor")
    assert not hasattr(family_cls, "format_flags")


@pytest.mark.parametrize("grade_workers", [1, 2])
def test_ordinary_grade_exception_becomes_none_and_is_counted(
    grade_workers: int,
) -> None:
    family = ContractFamily()
    family.grade_workers = grade_workers

    grades = family.grade_batch(
        [
            ({"gold": "yes"}, "yes"),
            ({"failure": "ordinary"}, "candidate"),
        ]
    )

    assert grades == [True, None]
    assert family.last_grade_errors == 1


@pytest.mark.parametrize("grade_workers", [1, 2])
def test_fatal_grader_error_propagates_inline_and_threaded(
    grade_workers: int,
) -> None:
    family = ContractFamily()
    family.grade_workers = grade_workers
    items = [
        ({"gold": "yes"}, "yes"),
        ({"failure": "infrastructure"}, "candidate"),
    ]

    with pytest.raises(GraderInfrastructureError, match="worker died"):
        family.grade_batch(items)


def test_threaded_fatal_grade_aborts_queued_calls_and_reraises_original_error(
    monkeypatch,
) -> None:
    abort = threading.Event()
    running_started = threading.Event()
    all_submitted = threading.Event()
    failure = GraderInfrastructureError("first fatal worker")

    real_executor = concurrent.futures.ThreadPoolExecutor

    class DrainQueuedExecutor(real_executor):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.submissions = 0

        def submit(self, fn, /, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            self.submissions += 1
            if self.submissions == 4:
                all_submitted.set()
            return future

        def shutdown(self, wait=True, *, cancel_futures=False):
            return super().shutdown(wait=wait, cancel_futures=False)

    class CoordinatedFamily(ContractFamily):
        def __init__(self) -> None:
            self.entered: list[str] = []

        def grade(self, meta: dict[str, Any], solution: Any) -> Optional[bool]:
            kind = meta["kind"]
            self.entered.append(kind)
            if kind == "running":
                running_started.set()
                if not abort.wait(timeout=2):
                    raise TimeoutError("pool never published its abort")
                return True
            if kind == "fatal":
                if not running_started.wait(timeout=2):
                    raise TimeoutError("the first worker never started")
                if not all_submitted.wait(timeout=2):
                    raise TimeoutError("queued work was not submitted")
                raise failure
            raise AssertionError(f"queued grade entered after fatal error: {kind}")

    monkeypatch.setattr(env_base, "Event", lambda: abort)
    monkeypatch.setattr(
        env_base.concurrent.futures,
        "ThreadPoolExecutor",
        DrainQueuedExecutor,
    )
    monkeypatch.setattr(env_base.concurrent.futures.Future, "cancel", lambda self: False)
    family = CoordinatedFamily()
    family.grade_workers = 2
    items = [
        ({"kind": "running"}, None),
        ({"kind": "fatal"}, None),
        ({"kind": "queued-a"}, None),
        ({"kind": "queued-b"}, None),
    ]

    with pytest.raises(GraderInfrastructureError, match="first fatal worker") as caught:
        family.grade_batch(items)

    assert caught.value is failure
    assert set(family.entered) == {"running", "fatal"}


def test_threaded_grade_batch_preserves_order_after_out_of_order_completion() -> None:
    second_finished = threading.Event()

    class ReverseCompletionFamily(ContractFamily):
        def __init__(self) -> None:
            self.completion_order: list[int] = []

        def grade(self, meta: dict[str, Any], solution: Any) -> Optional[bool]:
            index = meta["index"]
            if index == 0:
                if not second_finished.wait(timeout=2):
                    raise TimeoutError("second grade never completed")
            else:
                second_finished.set()
            self.completion_order.append(index)
            return bool(solution)

    family = ReverseCompletionFamily()
    family.grade_workers = 2
    grades = family.grade_batch(
        [
            ({"index": 0}, False),
            ({"index": 1}, True),
        ]
    )

    assert family.completion_order == [1, 0]
    assert grades == [False, True]


def test_protocol_identity_and_close_defaults() -> None:
    family = ContractFamily()

    assert family.protocol_identity() == {}
    assert family.close() is None
