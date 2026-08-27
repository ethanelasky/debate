"""Symbolic Hendrycks-MATH family graded by a serialized Math-Verify worker.

This is deliberately a separate protocol from :mod:`infra.envs.tasks.math`:
the legacy numeric MATH/AIME cohorts and rewards remain unchanged.  Cohort
membership, answer extraction, dependency semantics, and reward coefficients
are all part of this family's immutable protocol identity.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from datasets import load_dataset

from infra.envs.base import Env, SingleTurnEnv, Task
from infra.envs.shaping import FlagTerm, shaped_sample_reward
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file
from infra.envs.tasks.base import (
    AnswerParse,
    GraderInfrastructureError,
    TaskFamily,
    reject_unknown_keys,
)
from infra.envs.tasks.math_verify_worker import (
    CANONICALIZATION_PROTOCOL,
    EXACT_CANONICALIZATIONS,
    MATH_VERIFY_WORKER_PROTOCOL,
    MathVerifyWorker,
)


DATASET_ID = "the-jb/hendrycks-math"
DATASET_REVISION = "af6b99a181a909b1aec1424451f10e875fd97377"
DATASET_LEVEL = 5
PROMPT_FILE = "math.yaml"

COHORT_SEED = 0
SPLIT_PROTOCOL = "math-symbolic-l5-stratified-sha256-v1"
SPLIT_SHA256 = "27208772ebb9743ecf1ffd6fa544118a1206c1d8d2b148220837766bb4b9df08"
TRAIN_COUNT = 2074
DEV_COUNT = 230
TEST_COUNT = 512

EXTRACTION_PROTOCOL = "math-symbolic-visible-answer-v2"
GRADING_PROTOCOL = "math-verify-latex-v2"
MAX_CANDIDATE_BYTES = 64 * 1024

# These are protocol pins, not best-effort environment observations.  The
# worker validates the installed stack before doing any parsing or grading.
DEPENDENCY_VERSIONS = {
    "math_verify_version": "0.9.0",
    "latex2sympy2_extended_version": "1.11.0",
    "sympy_version": "1.14.0",
    "antlr4_python3_runtime_version": "4.13.2",
}

EXPECTED_SOURCE_COUNTS = {"train": 2304, "test": 1324}
EXPECTED_CATEGORY_COUNTS = {
    "train": {
        "Algebra": 436,
        "Counting & Probability": 276,
        "Geometry": 421,
        "Intermediate Algebra": 429,
        "Number Theory": 313,
        "Prealgebra": 252,
        "Precalculus": 177,
    },
    "test": {
        "Algebra": 307,
        "Counting & Probability": 123,
        "Geometry": 132,
        "Intermediate Algebra": 280,
        "Number Theory": 154,
        "Prealgebra": 193,
        "Precalculus": 135,
    },
}
EXPECTED_COHORT_CATEGORY_COUNTS = {
    "train": {
        "Algebra": 393,
        "Counting & Probability": 248,
        "Geometry": 379,
        "Intermediate Algebra": 386,
        "Number Theory": 282,
        "Prealgebra": 227,
        "Precalculus": 159,
    },
    "dev": {
        "Algebra": 43,
        "Counting & Probability": 28,
        "Geometry": 42,
        "Intermediate Algebra": 43,
        "Number Theory": 31,
        "Prealgebra": 25,
        "Precalculus": 18,
    },
    "test": {
        "Algebra": 119,
        "Counting & Probability": 48,
        "Geometry": 51,
        "Intermediate Algebra": 108,
        "Number Theory": 59,
        "Prealgebra": 75,
        "Precalculus": 52,
    },
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_level_five(value: Any) -> bool:
    return str(value).strip().lower() in {"5", "level 5"}


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return bool(backslashes % 2)


_THINK_TAG_RE = re.compile(r"<\s*(/?)\s*think\s*>", re.IGNORECASE)


def visible_answer_text(text: str) -> str:
    """Remove nested think blocks and hide everything after an unclosed one.

    Stray closing tags are discarded.  This scanner intentionally does not
    change the shared legacy numeric extractor's historical behavior.
    """

    chunks: list[str] = []
    depth = 0
    outside_start = 0
    for match in _THINK_TAG_RE.finditer(text):
        closing = bool(match.group(1))
        if closing:
            if depth == 0:
                chunks.append(text[outside_start : match.start()])
                outside_start = match.end()
            else:
                depth -= 1
                if depth == 0:
                    outside_start = match.end()
        else:
            if depth == 0:
                chunks.append(text[outside_start : match.start()])
            depth += 1
    if depth == 0:
        chunks.append(text[outside_start:])
    return "".join(chunks)


def _matching_brace(text: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(text)):
        char = text[index]
        if _is_escaped(text, index):
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


_BOX_COMMAND_RE = re.compile(r"\\boxed(?![A-Za-z])")


def _rightmost_box(text: str) -> tuple[str, str | None]:
    """Return (absent|unbalanced|balanced, raw inner content)."""

    matches = [
        match
        for match in _BOX_COMMAND_RE.finditer(text)
        if not _is_escaped(text, match.start())
    ]
    if not matches:
        return "absent", None
    match = matches[-1]
    opening = match.end()
    while opening < len(text) and text[opening].isspace():
        opening += 1
    if opening >= len(text) or text[opening] != "{":
        return "unbalanced", None
    closing = _matching_brace(text, opening)
    if closing is None:
        return "unbalanced", None
    return "balanced", text[opening + 1 : closing]


def _candidate_is_bounded(candidate: str) -> bool:
    return bool(candidate.strip()) and len(candidate.encode("utf-8")) <= MAX_CANDIDATE_BYTES


def _unescaped_token_offsets(text: str, token: str) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        index = text.find(token, start)
        if index < 0:
            return offsets
        if not _is_escaped(text, index):
            offsets.append(index)
        start = index + len(token)


def _paired_spans(text: str, opening: str, closing: str) -> Iterable[tuple[int, str]]:
    starts = _unescaped_token_offsets(text, opening)
    ends = _unescaped_token_offsets(text, closing)
    end_cursor = 0
    for start in starts:
        while end_cursor < len(ends) and ends[end_cursor] < start + len(opening):
            end_cursor += 1
        if end_cursor >= len(ends):
            return
        end = ends[end_cursor]
        yield end + len(closing), text[start + len(opening) : end]
        end_cursor += 1


def _dollar_spans(text: str, *, display: bool) -> Iterable[tuple[int, str]]:
    token = "$$" if display else "$"
    offsets: list[int] = []
    index = 0
    while index < len(text):
        found = text.find(token, index)
        if found < 0:
            break
        if _is_escaped(text, found):
            index = found + len(token)
            continue
        if not display and (
            (found > 0 and text[found - 1] == "$")
            or (found + 1 < len(text) and text[found + 1] == "$")
        ):
            index = found + 1
            continue
        offsets.append(found)
        index = found + len(token)
    for pair in range(0, len(offsets) - 1, 2):
        start, end = offsets[pair], offsets[pair + 1]
        yield end + len(token), text[start + len(token) : end]


def _rightmost_math_span(text: str) -> str | None:
    # The numeric tie value pins deterministic behavior for the theoretical
    # case where different delimiter scans report the same closing offset.
    spans: list[tuple[int, int, str]] = []
    spans.extend((end, 0, inner) for end, inner in _dollar_spans(text, display=True))
    spans.extend((end, 1, inner) for end, inner in _dollar_spans(text, display=False))
    spans.extend((end, 2, inner) for end, inner in _paired_spans(text, r"\(", r"\)"))
    spans.extend((end, 3, inner) for end, inner in _paired_spans(text, r"\[", r"\]"))
    if not spans:
        return None
    return max(spans, key=lambda item: (item[0], item[1]))[2]


_NUMBER_RE = re.compile(
    r"(?<![\w.,])"
    r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
    r"(?![\w.,])"
)


def _last_sanitized_number(text: str) -> str | None:
    matches = list(_NUMBER_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(0).replace(",", "")


def parse_symbolic_answers(text: str, worker: MathVerifyWorker) -> AnswerParse:
    """Extract the strict and relaxed symbolic candidates from one response."""

    visible = visible_answer_text(text)
    box_status, boxed = _rightmost_box(visible)
    if box_status == "balanced":
        if boxed is not None and _candidate_is_bounded(boxed) and worker.is_parseable(boxed):
            return AnswerParse(strict=boxed, relaxed=boxed)
        # A declared, balanced final answer is authoritative even when it is
        # unparsable: prose or earlier expressions must not rescue it.
        return AnswerParse(strict=None, relaxed=None)

    marked = _rightmost_math_span(visible)
    if marked is not None and _candidate_is_bounded(marked) and worker.is_parseable(marked):
        return AnswerParse(strict=None, relaxed=marked)

    number = _last_sanitized_number(visible)
    if number is not None and _candidate_is_bounded(number) and worker.is_parseable(number):
        return AnswerParse(strict=None, relaxed=number)
    return AnswerParse(strict=None, relaxed=None)


def _row_id(source_split: str, source_index: int, fields: dict[str, Any]) -> str:
    return _sha256_json(
        [
            "math-symbolic-row-v1",
            DATASET_ID,
            DATASET_REVISION,
            source_split,
            source_index,
            fields,
        ]
    )


def _row_rank(source_split: str, category: str, row_id: str) -> str:
    return _sha256_json([SPLIT_PROTOCOL, COHORT_SEED, source_split, category, row_id])


def _hamilton_quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if target < 0 or target > total:
        raise ValueError(f"invalid stratified target {target} for population {total}")
    quotas = {category: target * count // total for category, count in counts.items()}
    remaining = target - sum(quotas.values())
    by_remainder = sorted(
        counts,
        key=lambda category: (-(target * counts[category] % total), category),
    )
    for category in by_remainder[:remaining]:
        quotas[category] += 1
    return quotas


def _annotate_rows(rows: Sequence[dict[str, Any]], source_split: str) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_source_indexes: set[int] = set()
    for fallback_index, example in enumerate(rows):
        source_index = int(example.get("_source_index", fallback_index))
        if source_index in seen_source_indexes:
            raise RuntimeError(
                f"symbolic MATH duplicate {source_split} source index {source_index}"
            )
        seen_source_indexes.add(source_index)
        if not _is_level_five(example.get("level")):
            continue
        fields = {
            key: example.get(key)
            for key in ("problem", "level", "solution", "type", "answer")
        }
        if not all(isinstance(fields[key], str) for key in fields):
            raise RuntimeError(
                f"symbolic MATH {source_split}[{source_index}] has malformed public fields"
            )
        row_id = _row_id(source_split, source_index, fields)
        if row_id in seen:
            raise RuntimeError(f"symbolic MATH duplicate row id {row_id}")
        seen.add(row_id)

        status, boxed = _rightmost_box(fields["solution"])
        if status != "balanced" or boxed is None:
            raise RuntimeError(
                f"symbolic MATH {source_split}[{source_index}] has no balanced final box"
            )
        if fields["answer"].strip() != boxed.strip():
            raise RuntimeError(
                f"symbolic MATH {source_split}[{source_index}] answer disagrees with final box"
            )

        category = fields["type"]
        annotated.append(
            {
                "source_split": source_split,
                "source_index": source_index,
                "row_id": row_id,
                "rank": _row_rank(source_split, category, row_id),
                "problem": fields["problem"],
                "gt": fields["answer"],
                "category": category,
                "level": DATASET_LEVEL,
            }
        )
    return annotated


def _select_stratified(
    rows: Sequence[dict[str, Any]], target: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    counts = {category: len(group) for category, group in grouped.items()}
    quotas = _hamilton_quotas(counts, target)
    selected_ids: set[str] = set()
    for category, group in grouped.items():
        ordered = sorted(group, key=lambda row: (row["rank"], row["row_id"]))
        selected_ids.update(row["row_id"] for row in ordered[: quotas[category]])
    selected = [row for row in rows if row["row_id"] in selected_ids]
    complement = [row for row in rows if row["row_id"] not in selected_ids]
    key = lambda row: (row["rank"], row["row_id"])
    return sorted(selected, key=key), sorted(complement, key=key), quotas


def _split_digest(cohorts: dict[str, Sequence[dict[str, Any]]]) -> str:
    return _sha256_json(
        {
            split: [
                [row["source_split"], row["source_index"], row["row_id"]]
                for row in cohorts[split]
            ]
            for split in ("train", "dev", "test")
        }
    )


def build_symbolic_cohorts(
    source_train: Sequence[dict[str, Any]],
    source_test: Sequence[dict[str, Any]],
    *,
    validate_production: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Build fixed stratified cohorts, discarding public worked solutions."""

    train_population = _annotate_rows(source_train, "train")
    test_population = _annotate_rows(source_test, "test")
    if validate_production:
        actual_sources = {"train": len(train_population), "test": len(test_population)}
        actual_categories = {
            "train": dict(Counter(row["category"] for row in train_population)),
            "test": dict(Counter(row["category"] for row in test_population)),
        }
        if actual_sources != EXPECTED_SOURCE_COUNTS:
            raise RuntimeError(
                f"symbolic MATH source counts changed: {actual_sources} != {EXPECTED_SOURCE_COUNTS}"
            )
        if actual_categories != EXPECTED_CATEGORY_COUNTS:
            raise RuntimeError(
                "symbolic MATH source category populations changed: "
                f"{actual_categories} != {EXPECTED_CATEGORY_COUNTS}"
            )

    dev, train, _ = _select_stratified(train_population, DEV_COUNT)
    test, _, _ = _select_stratified(test_population, TEST_COUNT)
    cohorts = {"train": train, "dev": dev, "test": test}
    digest = _split_digest(cohorts)

    if validate_production:
        counts = {split: len(rows) for split, rows in cohorts.items()}
        expected_counts = {"train": TRAIN_COUNT, "dev": DEV_COUNT, "test": TEST_COUNT}
        categories = {
            split: dict(Counter(row["category"] for row in rows))
            for split, rows in cohorts.items()
        }
        if counts != expected_counts:
            raise RuntimeError(
                f"symbolic MATH cohort counts changed: {counts} != {expected_counts}"
            )
        if categories != EXPECTED_COHORT_CATEGORY_COUNTS:
            raise RuntimeError(
                "symbolic MATH cohort category counts changed: "
                f"{categories} != {EXPECTED_COHORT_CATEGORY_COUNTS}"
            )
        if digest != SPLIT_SHA256:
            raise RuntimeError(f"symbolic MATH split digest changed: {digest} != {SPLIT_SHA256}")
    return cohorts, digest


def _load_symbolic_dataset():
    return load_dataset(DATASET_ID, revision=DATASET_REVISION)


class SymbolicMathEnv(SingleTurnEnv):
    grade_workers = 1

    def __init__(
        self,
        family: "SymbolicMathFamily",
        cohorts: dict[str, list[dict[str, Any]]],
        *,
        seed: int,
        prompt_file: str | Path,
        correct_reward: float = 1.0,
        relaxed_correct_bonus: float = 0.1,
        format_reward: float = 0.0,
        think_overshoot_penalty: float = 0.0,
        soft_token_budget: Optional[int] = None,
        overshoot_penalty: float = 0.0,
        overshoot_mode: str = "flat",
    ):
        self.family = family
        self.rng = random.Random(seed)
        self.prompts = load_generation_prompts(prompt_file)
        self.train_rows = cohorts["train"]
        self.dev_rows = cohorts["dev"]
        self.test_rows = cohorts["test"]
        self.correct_reward = float(correct_reward)
        self.relaxed_correct_bonus = float(relaxed_correct_bonus)
        self.format_reward = float(format_reward)
        self.think_overshoot_penalty = float(think_overshoot_penalty)
        if self.think_overshoot_penalty < 0:
            raise ValueError(
                "math_symbolic: think_overshoot_penalty must be >= 0 "
                f"(it is subtracted), got {think_overshoot_penalty}"
            )
        # Length budget on the ANSWER GENERATION itself, priced by SingleTurnEnv.
        # Distinct from think_overshoot_penalty above, which is a boolean on
        # "the think phase was force-closed at its cap" and carries no magnitude
        # -- budget_forced_sample injects </think> at the cap, so the sample
        # physically cannot exceed it and there is nothing to be proportional to.
        # This one has the magnitude, which is the whole reason it exists.
        self.soft_token_budget = int(soft_token_budget) if soft_token_budget else None
        self.overshoot_penalty = float(overshoot_penalty)
        if self.overshoot_penalty < 0:
            raise ValueError(
                "math_symbolic: overshoot_penalty must be >= 0 "
                f"(it is subtracted), got {overshoot_penalty}"
            )
        if overshoot_mode not in ("flat", "proportional"):
            raise ValueError(
                "math_symbolic: overshoot_mode must be 'flat' or 'proportional', "
                f"got {overshoot_mode!r}"
            )
        self.overshoot_mode = overshoot_mode
        if self.overshoot_penalty and self.soft_token_budget is None:
            raise ValueError(
                "math_symbolic: overshoot_penalty is set but soft_token_budget is "
                "not, so the penalty can never fire. State the budget or drop the "
                "penalty -- a priced term that silently never applies is the "
                "value-shaped void this check exists to close."
            )

    def _task(self, row: dict[str, Any], split: str) -> Task:
        return Task(
            messages=self.prompts.render({"PROBLEM": row["problem"]}),
            meta={
                "question": row["problem"],
                "gt": row["gt"],
                "category": row["category"],
                "level": row["level"],
                "split": split,
                "row_id": row["row_id"],
            },
        )

    def tasks(self, n: int, split: str = "train") -> list[Task]:
        if n < 0:
            raise ValueError("task count must be nonnegative")
        if split == "train":
            return [self._task(self.rng.choice(self.train_rows), split) for _ in range(n)]
        if split not in {"dev", "test"}:
            raise ValueError(f"unknown symbolic MATH split {split!r}")
        rows = self.dev_rows if split == "dev" else self.test_rows
        return [self._task(row, split) for row in rows[:n]]

    def reward(self, task: Task, text: str) -> tuple[float, dict[str, Any]]:
        parsed = self.family.parse_answers(text)
        candidates: list[str] = []
        for candidate in (parsed.strict, parsed.relaxed):
            if candidate is not None and candidate not in candidates:
                candidates.append(candidate)
        grades = self.family.grade_batch([(task.meta, candidate) for candidate in candidates])
        by_candidate = dict(zip(candidates, grades))
        strict_correct = parsed.strict is not None and by_candidate.get(parsed.strict) is True
        relaxed_correct = parsed.relaxed is not None and by_candidate.get(parsed.relaxed) is True

        reward = self.format_reward if parsed.answer_format_valid else 0.0
        if strict_correct:
            reward += self.correct_reward
        elif relaxed_correct:
            reward += self.relaxed_correct_bonus
        info = {
            "correct_strict": float(strict_correct),
            "correct_relaxed": float(relaxed_correct),
            "answer_format_valid": float(parsed.answer_format_valid),
        }
        return reward, info

    def sample_terms(self) -> list[FlagTerm]:
        """Sample-level shaping this env prices; see MathEnv.sample_terms. The
        format bonus stays in reward(), on THIS family's answer_format_valid
        predicate rather than MathEnv's raw-box one."""
        return [FlagTerm(self.think_overshoot_penalty, "think_overshoot", sign=-1)]

    def reward_sample(self, task: Task, sample) -> tuple[float, dict[str, Any]]:
        reward, info = self.reward(task, sample.text)
        return shaped_sample_reward(reward, info, sample, self.sample_terms())


class SymbolicMathFamily(TaskFamily):
    grade_workers = 1

    def __init__(self, worker: MathVerifyWorker | None = None):
        self._worker: MathVerifyWorker | None = worker
        self._closed = False
        self._protocol_identity: dict[str, str] = {}

    def _get_worker(self) -> MathVerifyWorker:
        if self._closed:
            raise GraderInfrastructureError("symbolic MATH family is closed")
        if self._worker is None:
            self._worker = MathVerifyWorker()
        return self._worker

    def source(self, ds: dict) -> Env:
        reject_unknown_keys(
            ds,
            {
                "seed",
                "prompt_file",
                "correct_reward",
                "relaxed_correct_bonus",
                "format_reward",
                "think_overshoot_penalty",
                "soft_token_budget",
                "overshoot_penalty",
                "overshoot_mode",
            },
            "math_symbolic",
        )
        seed = int(ds.get("seed", 0))
        prompt_path = resolve_prompt_file(
            str(ds["prompt_file"]) if ds.get("prompt_file") else None,
            PROMPT_FILE,
        )
        correct_reward = float(ds.get("correct_reward", 1.0))
        relaxed_correct_bonus = float(ds.get("relaxed_correct_bonus", 0.1))
        format_reward = float(ds.get("format_reward", 0.0))
        think_overshoot_penalty = float(ds.get("think_overshoot_penalty", 0.0))
        if think_overshoot_penalty < 0:
            raise ValueError("math_symbolic: think_overshoot_penalty must be >= 0")
        soft_token_budget = int(ds["soft_token_budget"]) if ds.get("soft_token_budget") else None
        overshoot_penalty = float(ds.get("overshoot_penalty", 0.0))
        overshoot_mode = str(ds.get("overshoot_mode", "flat"))

        dataset = _load_symbolic_dataset()
        if "train" not in dataset or "test" not in dataset:
            raise RuntimeError("pinned symbolic MATH dataset must contain train and test splits")
        cohorts, digest = build_symbolic_cohorts(dataset["train"], dataset["test"])
        env = SymbolicMathEnv(
            self,
            cohorts,
            seed=seed,
            prompt_file=prompt_path,
            correct_reward=correct_reward,
            relaxed_correct_bonus=relaxed_correct_bonus,
            format_reward=format_reward,
            think_overshoot_penalty=think_overshoot_penalty,
            soft_token_budget=soft_token_budget,
            overshoot_penalty=overshoot_penalty,
            overshoot_mode=overshoot_mode,
        )
        self._protocol_identity = {
            "family": "math_symbolic",
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_level": str(DATASET_LEVEL),
            "split_protocol": SPLIT_PROTOCOL,
            "cohort_seed": str(COHORT_SEED),
            "split_sha256": digest,
            "source_train_count": str(EXPECTED_SOURCE_COUNTS["train"]),
            "source_test_count": str(EXPECTED_SOURCE_COUNTS["test"]),
            "train_count": str(len(cohorts["train"])),
            "dev_count": str(len(cohorts["dev"])),
            "test_count": str(len(cohorts["test"])),
            "extraction_protocol": EXTRACTION_PROTOCOL,
            "grading_protocol": GRADING_PROTOCOL,
            "worker_protocol": MATH_VERIFY_WORKER_PROTOCOL,
            "canonicalization_protocol": CANONICALIZATION_PROTOCOL,
            "canonicalizations": _canonical_json(EXACT_CANONICALIZATIONS).decode("utf-8"),
            "canonicalizations_sha256": _sha256_json(EXACT_CANONICALIZATIONS),
            "parse_route": "latex_dollar_envelope;fallback=no_fallback;raise_on_error=true",
            "normalization": _canonical_json(
                {
                    "basic_latex": True,
                    "boxed": "all",
                    "equations": False,
                    "malformed_operators": True,
                    "nits": False,
                    "units": True,
                }
            ).decode("utf-8"),
            "equivalence": _canonical_json(
                {
                    "allow_set_relation_comp": False,
                    "float_rounding": 6,
                    "numeric_precision": 15,
                    "strict": True,
                }
            ).decode("utf-8"),
            "operation_timeout_seconds": "5",
            "candidate_max_bytes": str(MAX_CANDIDATE_BYTES),
            "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
            "correct_reward": repr(correct_reward),
            "relaxed_correct_bonus": repr(relaxed_correct_bonus),
            "format_reward": repr(format_reward),
            "think_overshoot_penalty": repr(think_overshoot_penalty),
            "soft_token_budget": "none" if soft_token_budget is None else str(soft_token_budget),
            "overshoot_penalty": repr(overshoot_penalty),
            "overshoot_mode": overshoot_mode,
            "training_seed": str(seed),
            **DEPENDENCY_VERSIONS,
        }
        return env

    def parse_answers(self, text: str) -> AnswerParse:
        return parse_symbolic_answers(text, self._get_worker())

    def grade(self, meta: dict[str, Any], solution: Any) -> Optional[bool]:
        gt = meta.get("gt")
        if not isinstance(gt, str) or not isinstance(solution, str):
            return None
        return self._get_worker().grade(gt, solution)

    def grade_batch(self, items: list[tuple[dict[str, Any], Any]]) -> list[Optional[bool]]:
        self.last_grade_errors = 0
        results: list[Optional[bool]] = [None] * len(items)
        valid: list[tuple[int, str, str]] = []
        for index, (meta, solution) in enumerate(items):
            gt = meta.get("gt")
            if isinstance(gt, str) and isinstance(solution, str):
                valid.append((index, gt, solution))
        if not valid:
            return results
        grades = self._get_worker().grade_many([(gt, solution) for _, gt, solution in valid])
        if len(grades) != len(valid):
            raise GraderInfrastructureError(
                f"symbolic grader returned {len(grades)} results for {len(valid)} requests"
            )
        for (index, _, _), grade in zip(valid, grades):
            results[index] = grade
        return results

    def protocol_identity(self) -> dict[str, str]:
        return dict(self._protocol_identity)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            self._worker.close()


__all__ = [
    "COHORT_SEED",
    "DATASET_ID",
    "DATASET_REVISION",
    "DEV_COUNT",
    "SPLIT_PROTOCOL",
    "SPLIT_SHA256",
    "SymbolicMathEnv",
    "SymbolicMathFamily",
    "TEST_COUNT",
    "TRAIN_COUNT",
    "build_symbolic_cohorts",
    "parse_symbolic_answers",
    "visible_answer_text",
]
