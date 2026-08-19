from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from infra.envs.base import Task
from infra.envs.tasks.base import AnswerParse, GraderInfrastructureError
from infra.envs.tasks.math_symbolic import (
    DATASET_ID,
    DATASET_REVISION,
    DEV_COUNT,
    EXACT_CANONICALIZATIONS,
    EXPECTED_COHORT_CATEGORY_COUNTS,
    SPLIT_SHA256,
    SymbolicMathEnv,
    SymbolicMathFamily,
    TEST_COUNT,
    TRAIN_COUNT,
    build_symbolic_cohorts,
    parse_symbolic_answers,
    visible_answer_text,
)
from infra.envs.tasks.math_verify_worker import (
    MathVerifyWorker,
    canonicalize_math_expression,
)
from _dataset_tqdm_monitor import dataset_tqdm_monitor_owner


@pytest.fixture(scope="module")
def dataset_monitor_call():
    with dataset_tqdm_monitor_owner() as monitor_call:
        yield monitor_call


class FakeWorker:
    def __init__(self, *, unparsable=(), fatal: Exception | None = None):
        self.unparsable = set(unparsable)
        self.fatal = fatal
        self.parse_calls: list[str] = []
        self.grade_calls: list[tuple[str, str]] = []
        self.grade_many_calls: list[list[tuple[str, str]]] = []
        self.close_calls = 0

    def is_parseable(self, candidate: str) -> bool:
        self.parse_calls.append(candidate)
        if self.fatal is not None:
            raise self.fatal
        return bool(candidate.strip()) and candidate not in self.unparsable

    def grade(self, gold: str, candidate: str):
        self.grade_calls.append((gold, candidate))
        if self.fatal is not None:
            raise self.fatal
        if candidate in self.unparsable:
            return None
        return gold.replace(" ", "") == candidate.replace(" ", "")

    def grade_many(self, items: list[tuple[str, str]]):
        self.grade_many_calls.append(list(items))
        return [self.grade(gold, candidate) for gold, candidate in items]

    def close(self):
        self.close_calls += 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("before <think>hidden</think> after", "before  after"),
        ("a <THINK>x <think>y</think> z</ThInK> b", "a  b"),
        ("a </think> b", "a  b"),
        ("shown <think>hidden forever \\boxed{9}", "shown "),
        ("<think>x</think>ok<think>unfinished", "ok"),
    ],
)
def test_visible_answer_text_depth_and_unclosed_semantics(text, expected):
    assert visible_answer_text(text) == expected


@pytest.mark.parametrize(
    ("text", "strict", "relaxed"),
    [
        (r"work \boxed{2+2} trailing prose", "2+2", "2+2"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}", r"\frac{1}{2}"),
        (r"\boxed{1} then \boxed{x+{2}}.", "x+{2}", "x+{2}"),
        (r"no box; final $x+1$", None, "x+1"),
        (r"first $x$ then \(y+1\)", None, "y+1"),
        (r"first \[x\] then $$z^2$$", None, "z^2"),
        ("therefore -1,234.50e-2", None, "-1234.50e-2"),
        ("therefore +.5", None, "+.5"),
        (r"<think>\boxed{99}</think> visible $4$", None, "4"),
    ],
)
def test_symbolic_extraction_happy_paths(text, strict, relaxed):
    worker = FakeWorker()
    assert parse_symbolic_answers(text, worker) == AnswerParse(strict, relaxed)


def test_rightmost_balanced_unparseable_box_poison_blocks_all_fallback():
    worker = FakeWorker(unparsable={"bad"})
    parsed = parse_symbolic_answers(r"$7$ \boxed{4} then \boxed{bad}; 9", worker)
    assert parsed == AnswerParse(None, None)
    assert worker.parse_calls == ["bad"]


def test_wrong_but_parseable_box_is_authoritative():
    worker = FakeWorker()
    assert parse_symbolic_answers(r"$7$ then \boxed{4}; correction: 7", worker) == AnswerParse(
        "4", "4"
    )


def test_unbalanced_rightmost_box_allows_marked_then_number_fallback():
    worker = FakeWorker()
    assert parse_symbolic_answers(r"\boxed{oops then $x+1$ trailing 8", worker) == AnswerParse(
        None, "x+1"
    )

    worker = FakeWorker(unparsable={"x+1"})
    assert parse_symbolic_answers(r"\boxed{oops then $x+1$ trailing 8", worker) == AnswerParse(
        None, "8"
    )
    assert worker.parse_calls == ["x+1", "8"]


def test_rightmost_box_command_without_opening_brace_blocks_earlier_box():
    parsed = parse_symbolic_answers(r"\boxed{3}; correction: $4$ then \boxed", FakeWorker())
    assert parsed == AnswerParse(None, "4")


def test_rightmost_marked_span_is_not_rescued_by_earlier_span():
    worker = FakeWorker(unparsable={"bad"})
    parsed = parse_symbolic_answers(r"earlier $x+1$ later \[bad\] and number 12", worker)
    assert parsed == AnswerParse(None, "12")
    assert worker.parse_calls == ["bad", "12"]


@pytest.mark.parametrize(
    "text",
    [
        "12,34",       # invalid comma grouping cannot produce a suffix match
        "abc1.5def",   # embedded in a word
        "1,234,56",    # invalid final group
        "version 1.2.3",  # dotted version is not a scalar token
    ],
)
def test_number_fallback_rejects_malformed_or_embedded_numbers(text):
    assert parse_symbolic_answers(text, FakeWorker()) == AnswerParse(None, None)


def test_candidate_size_bound_prevents_worker_input():
    worker = FakeWorker()
    parsed = parse_symbolic_answers("\\boxed{" + "x" * (64 * 1024 + 1) + "}", worker)
    assert parsed == AnswerParse(None, None)
    assert worker.parse_calls == []


def test_escaped_math_markers_are_not_candidates():
    assert parse_symbolic_answers(r"literal \$x\$ and no answer", FakeWorker()) == AnswerParse(
        None, None
    )


def test_parse_worker_infrastructure_failure_propagates():
    fatal = GraderInfrastructureError("dead worker")
    with pytest.raises(GraderInfrastructureError, match="dead worker"):
        parse_symbolic_answers(r"\boxed{x}", FakeWorker(fatal=fatal))


def _cohorts_for_env():
    def row(row_id: str, gt: str):
        return {
            "row_id": row_id,
            "problem": f"problem {row_id}",
            "gt": gt,
            "category": "Algebra",
            "level": 5,
        }

    return {
        "train": [row("train", "x+1")],
        "dev": [row("dev", "x+1")],
        "test": [row("test", "x+1")],
    }


def _env(worker=None, **rewards):
    family = SymbolicMathFamily(worker or FakeWorker())
    env = SymbolicMathEnv(
        family,
        _cohorts_for_env(),
        seed=0,
        prompt_file="infra/prompts/tasks/math.yaml",
        correct_reward=rewards.get("correct_reward", 1.0),
        relaxed_correct_bonus=rewards.get("relaxed_correct_bonus", 0.1),
        format_reward=rewards.get("format_reward", 0.0),
        think_overshoot_penalty=rewards.get("think_overshoot_penalty", 0.0),
    )
    return family, env


def test_direct_reward_strict_relaxed_and_incorrect_defaults():
    _, env = _env()
    task = Task(messages=[], meta={"gt": "x+1"})
    assert env.reward(task, r"\boxed{x+1}") == (
        1.0,
        {"correct_strict": 1.0, "correct_relaxed": 1.0, "answer_format_valid": 1.0},
    )
    assert env.reward(task, r"answer: $x+1$") == (
        0.1,
        {"correct_strict": 0.0, "correct_relaxed": 1.0, "answer_format_valid": 0.0},
    )
    assert env.reward(task, r"\boxed{x+2}") == (
        0.0,
        {"correct_strict": 0.0, "correct_relaxed": 0.0, "answer_format_valid": 1.0},
    )


def test_format_reward_is_zero_by_default_and_configurable():
    task = Task(messages=[], meta={"gt": "x+1"})
    _, default = _env()
    _, configured = _env(format_reward=0.25)
    assert default.reward(task, r"\boxed{x+2}")[0] == 0.0
    assert configured.reward(task, r"\boxed{x+2}")[0] == 0.25


def test_identical_strict_relaxed_candidate_is_graded_once():
    worker = FakeWorker()
    _, env = _env(worker)
    env.reward(Task(messages=[], meta={"gt": "x+1"}), r"\boxed{x+1}")
    assert worker.grade_many_calls == [[("x+1", "x+1")]]
    assert worker.grade_calls == [("x+1", "x+1")]


def test_grade_batch_alignment_and_unparseable_candidate():
    worker = FakeWorker(unparsable={"bad"})
    family = SymbolicMathFamily(worker)
    assert family.grade_batch(
        [({"gt": "x"}, "x"), ({"gt": "x"}, "bad"), ({}, "x"), ({"gt": "x"}, None)]
    ) == [True, None, None, None]
    assert worker.grade_many_calls == [[("x", "x"), ("x", "bad")]]


def test_family_close_is_idempotent():
    worker = FakeWorker()
    family = SymbolicMathFamily(worker)
    family.close()
    family.close()
    assert worker.close_calls == 1
    with pytest.raises(GraderInfrastructureError, match="closed"):
        family.parse_answers(r"\boxed{x}")


def _synthetic_sources():
    categories = list(EXPECTED_COHORT_CATEGORY_COUNTS["dev"])

    def rows(total: int, split: str):
        result = []
        for index in range(total):
            answer = str(index + (10000 if split == "test" else 0))
            result.append(
                {
                    "_source_index": index,
                    "problem": f"{split} problem {index}",
                    "level": "Level 5",
                    "solution": f"work \\boxed{{{answer}}}",
                    "type": categories[index % len(categories)],
                    "answer": answer,
                }
            )
        return result

    return rows(DEV_COUNT + 70, "train"), rows(TEST_COUNT + 70, "test")


def test_stratified_split_is_input_order_independent_with_stable_source_indexes():
    train, test = _synthetic_sources()
    first, first_digest = build_symbolic_cohorts(train, test, validate_production=False)
    second, second_digest = build_symbolic_cohorts(
        list(reversed(copy.deepcopy(train))),
        list(reversed(copy.deepcopy(test))),
        validate_production=False,
    )
    assert first_digest == second_digest
    assert {
        split: [row["row_id"] for row in rows] for split, rows in first.items()
    } == {split: [row["row_id"] for row in rows] for split, rows in second.items()}
    assert {split: len(rows) for split, rows in first.items()} == {
        "train": 70,
        "dev": DEV_COUNT,
        "test": TEST_COUNT,
    }
    assert all("solution" not in row for rows in first.values() for row in rows)


def test_cohort_construction_fails_on_gold_box_disagreement():
    train, test = _synthetic_sources()
    train[0]["answer"] = "different"
    with pytest.raises(RuntimeError, match="answer disagrees"):
        build_symbolic_cohorts(train, test, validate_production=False)


def test_cohort_construction_rejects_duplicate_source_position():
    train, test = _synthetic_sources()
    train[1]["_source_index"] = train[0]["_source_index"]
    with pytest.raises(RuntimeError, match="duplicate train source index"):
        build_symbolic_cohorts(train, test, validate_production=False)


def test_pinned_cached_dataset_digest_counts_and_quotas(dataset_monitor_call):
    # Explicitly offline: absent cache is an allowed skip; a present but
    # changed/corrupt pinned revision must fail rather than silently skip.
    from datasets import DownloadConfig, config, load_dataset

    processed_cache = Path(config.HF_DATASETS_CACHE) / "the-jb___hendrycks-math"
    if not any(processed_cache.glob(f"**/{DATASET_REVISION}")):
        pytest.skip("pinned Hendrycks-MATH revision is not cached")
    dataset = dataset_monitor_call(
        load_dataset,
        DATASET_ID,
        revision=DATASET_REVISION,
        download_config=DownloadConfig(local_files_only=True),
    )
    cohorts, digest = build_symbolic_cohorts(dataset["train"], dataset["test"])
    assert digest == SPLIT_SHA256
    assert {split: len(rows) for split, rows in cohorts.items()} == {
        "train": TRAIN_COUNT,
        "dev": DEV_COUNT,
        "test": TEST_COUNT,
    }
    assert {
        split: dict(Counter(row["category"] for row in rows))
        for split, rows in cohorts.items()
    } == EXPECTED_COHORT_CATEGORY_COUNTS
    assert all("solution" not in row for rows in cohorts.values() for row in rows)


def test_pinned_cached_selected_golds_all_parse_in_real_worker_without_cohort_mutation(
    dataset_monitor_call,
):
    """Independent production audit: every exposed raw gold must be gradeable.

    Only an absent pinned dataset cache can skip this. Missing dependencies,
    parse failures, worker failures, or cohort drift are protocol failures.
    """
    from datasets import DownloadConfig, config, load_dataset

    processed_cache = Path(config.HF_DATASETS_CACHE) / "the-jb___hendrycks-math"
    if not any(processed_cache.glob(f"**/{DATASET_REVISION}")):
        pytest.skip("pinned Hendrycks-MATH revision is not cached")
    dataset = dataset_monitor_call(
        load_dataset,
        DATASET_ID,
        revision=DATASET_REVISION,
        download_config=DownloadConfig(local_files_only=True),
    )
    cohorts, digest_before = build_symbolic_cohorts(dataset["train"], dataset["test"])
    raw_before = {
        split: [(row["row_id"], row["gt"]) for row in rows]
        for split, rows in cohorts.items()
    }
    assert digest_before == SPLIT_SHA256
    assert sum(len(rows) for rows in cohorts.values()) == TRAIN_COUNT + DEV_COUNT + TEST_COUNT
    assert all(
        any(row["gt"] == raw for rows in cohorts.values() for row in rows)
        for raw, _canonical in EXACT_CANONICALIZATIONS
    )

    worker = MathVerifyWorker()
    try:
        failures = [
            (split, row["source_split"], row["source_index"], row["gt"])
            for split, rows in cohorts.items()
            for row in rows
            if not worker.is_parseable(row["gt"])
        ]
    finally:
        worker.close()
    assert failures == []

    # Canonicalization is worker-local and symmetric; cohort raw gold,
    # membership, ordering, and its authoritative digest remain unchanged.
    assert {
        split: [(row["row_id"], row["gt"]) for row in rows]
        for split, rows in cohorts.items()
    } == raw_before
    assert build_symbolic_cohorts(dataset["train"], dataset["test"])[1] == digest_before


def test_source_protocol_identity_and_task_meta_are_exact(monkeypatch):
    import infra.envs.tasks.math_symbolic as module

    cohorts = _cohorts_for_env()
    monkeypatch.setattr(module, "_load_symbolic_dataset", lambda: {"train": [], "test": []})
    monkeypatch.setattr(
        module,
        "build_symbolic_cohorts",
        lambda *_args, **_kwargs: (copy.deepcopy(cohorts), "a" * 64),
    )
    family = SymbolicMathFamily(FakeWorker())
    env = family.source(
        {
            "seed": 7,
            "correct_reward": 2,
            "relaxed_correct_bonus": 0.25,
            "format_reward": 0.125,
            "think_overshoot_penalty": 0.5,
        }
    )
    identity = family.protocol_identity()
    assert identity["dataset_id"] == DATASET_ID
    assert identity["dataset_revision"] == DATASET_REVISION
    assert identity["cohort_seed"] == "0"
    assert identity["training_seed"] == "7"
    assert identity["correct_reward"] == "2.0"
    assert identity["relaxed_correct_bonus"] == "0.25"
    assert identity["format_reward"] == "0.125"
    assert identity["think_overshoot_penalty"] == "0.5"
    assert identity["split_sha256"] == "a" * 64
    assert identity["math_verify_version"] == "0.9.0"
    assert identity["extraction_protocol"] == "math-symbolic-visible-answer-v2"
    assert identity["grading_protocol"] == "math-verify-latex-v2"
    assert identity["worker_protocol"] == "math-verify-worker-v3"
    assert identity["canonicalization_protocol"] == "math-symbolic-exact-canonicalization-v1"
    assert json.loads(identity["canonicalizations"]) == [
        [raw, canonical] for raw, canonical in EXACT_CANONICALIZATIONS
    ]
    assert (
        identity["canonicalizations_sha256"]
        == "b6587c954aeb511a863c22b39e788d934d5d578b68cf99ba416cefa0a7fa848b"
    )
    assert json.loads(identity["normalization"])["malformed_operators"] is True
    assert len(identity["prompt_sha256"]) == 64
    identity["training_seed"] = "mutated"
    assert family.protocol_identity()["training_seed"] == "7"

    task = env.tasks(1, "dev")[0]
    assert set(task.meta) == {"question", "gt", "category", "level", "split", "row_id"}
    assert "solution" not in task.meta


def test_family_identity_canonicalizations_match_worker_exactly():
    for raw, canonical in EXACT_CANONICALIZATIONS:
        assert canonicalize_math_expression(raw) == canonical
        assert canonicalize_math_expression(f"  {raw}\n") == canonical
    assert canonicalize_math_expression(r"\approx 8.25 \text{ mph}") == (
        r"\approx 8.25 \text{ mph}"
    )
    assert canonicalize_math_expression("a + b + d") == "a + b + d"
    assert canonicalize_math_expression(r"\sin^2 x") == r"\sin^2 x"


def test_symbolic_family_rejects_unknown_config_before_dataset_load(monkeypatch):
    import infra.envs.tasks.math_symbolic as module

    monkeypatch.setattr(
        module,
        "_load_symbolic_dataset",
        lambda: pytest.fail("dataset load must not run for invalid config"),
    )
    with pytest.raises(ValueError, match="eval_subset_size"):
        SymbolicMathFamily(FakeWorker()).source({"eval_subset_size": 12})


def test_legacy_numeric_families_still_reject_symbolic_expression():
    from infra.envs.tasks.aime import AimeFamily
    from infra.envs.tasks.math import MathFamily

    assert MathFamily().parse_answers(r"\boxed{x+y}") == AnswerParse(None, None)
    assert AimeFamily().parse_answers(r"\boxed{x+y}") == AnswerParse(None, None)


@pytest.mark.skipif(
    importlib.util.find_spec("math_verify") is None,
    reason="real Math-Verify stack is not installed",
)
def test_real_worker_symbolic_equivalence_smoke():
    pairs = [
        ("2x", "x+x"),
        (r"\frac{1}{2}", r"\frac{2}{4}"),
        (r"\{1,2\}", r"\{2,1\}"),
        (r"(1,2)", r"(1,2)"),
        (r"\sqrt{8}", r"2\sqrt{2}"),
        (r"\$5", "5"),
        ("1+i", "i+1"),
    ]
    family = SymbolicMathFamily()
    try:
        for gold, equivalent in pairs:
            assert family.parse_answers(rf"\boxed{{{equivalent}}}").answer_format_valid
            assert family.grade({"gt": gold}, equivalent) is True
    finally:
        family.close()
