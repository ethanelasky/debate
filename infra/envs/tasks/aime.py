"""AIME task family: the 1983-2024 pool (933 problems, integer answers
000-999). MATH L5 saturated for the 7B under the unified prompt (~76% floor,
2026-08-06); AIME is the next difficulty tier that keeps the existing
boxed-number extraction and numeric-tolerance grading unchanged.

Reuses the math family's answer prompts (single source, so RLVR and debate
render byte-identical proposals here too), extractors, and grading. The
held-out split is a seeded-shuffle carve, the same convention as MathEnv's
no-test-split branch."""

from __future__ import annotations

import math
import random
from typing import Any, Optional

from datasets import load_dataset

from infra.envs.base import Env, SingleTurnEnv, Task
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file
from infra.envs.tasks.base import TaskFamily, reject_unknown_keys
from infra.envs.tasks.math import MathEnv, relaxed_extract, strict_extract
from infra.envs.answer_parsing import extract_number_from_boxed_answer, parse_number

DATASET_ID = "di-zhang-fdu/AIME_1983_2024"
PROMPT_FILE = "math.yaml"  # deliberately the math family's prompt: one answer prompt


def _rows(split) -> list[dict[str, Any]]:
    rows = []
    for ex in split:
        problem = str(ex.get("Question", "")).strip()
        gt = parse_number(str(ex.get("Answer", "")).strip())
        if not problem or gt is None or not math.isfinite(gt):
            continue
        rows.append({"problem": problem, "gt": float(gt), "id": str(ex.get("ID", ""))})
    return rows


class AimeEnv(SingleTurnEnv):
    def __init__(
        self,
        seed: int = 0,
        eval_subset_size: int = 512,
        correct_reward: float = 1.0,
        format_reward: float = 0.1,
        relaxed_correct_bonus: float = 0.1,
        prompt_file: str | None = None,
    ):
        self.rng = random.Random(seed)
        self.prompts = load_generation_prompts(resolve_prompt_file(prompt_file, PROMPT_FILE))
        self.correct_reward = correct_reward
        self.format_reward = format_reward
        self.relaxed_correct_bonus = relaxed_correct_bonus
        self.shaped_reward = 0.0

        rows = _rows(load_dataset(DATASET_ID)["train"])
        rng = random.Random(seed + 22222)
        rng.shuffle(rows)
        n_test = max(64, len(rows) // 10)
        self.test_rows, self.train_rows = rows[:n_test], rows[n_test:]
        self.test_rows = self.test_rows[:eval_subset_size]
        if len(self.train_rows) < 2 or not self.test_rows:
            raise RuntimeError(
                f"aime env: too few rows (train={len(self.train_rows)}, test={len(self.test_rows)})"
            )

    def _task(self, row: dict[str, Any], split: str) -> Task:
        return Task(
            messages=self.prompts.render({"PROBLEM": row["problem"]}),
            meta={"gt": row["gt"], "split": split, "question": row["problem"], "id": row["id"]},
        )

    tasks = MathEnv.tasks
    reward = MathEnv.reward


class AimeFamily(TaskFamily):
    def source(self, ds: dict) -> Env:
        reject_unknown_keys(ds, {"seed", "eval_subset_size", "prompt_file"}, "aime")
        return AimeEnv(
            seed=int(ds.get("seed", 0)),
            eval_subset_size=int(ds.get("eval_subset_size", 512)),
            prompt_file=(str(ds["prompt_file"]) if ds.get("prompt_file") else None),
        )

    def extractor(self, relaxed: bool):
        return relaxed_extract if relaxed else strict_extract

    def grade(self, meta: dict[str, Any], solution: Any) -> Optional[bool]:
        gt = meta.get("gt")
        if gt is None or solution is None:
            return None
        try:
            return abs(float(solution) - float(gt)) < 1e-6
        except (TypeError, ValueError):
            return None

    def format_flags(self, text: str) -> dict[str, float]:
        return {"strict_boxed": float(extract_number_from_boxed_answer(text) is not None)}
