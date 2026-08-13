"""Independent tests for the task-family parsing and grading contract."""

from typing import Any, Optional

import pytest

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


def test_protocol_identity_and_close_defaults() -> None:
    family = ContractFamily()

    assert family.protocol_identity() == {}
    assert family.close() is None
