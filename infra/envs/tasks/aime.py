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

from infra.envs.answer_parsing import parse_number
from infra.envs.base import Env, SingleTurnEnv, Task
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file
from infra.envs.tasks.base import AnswerParse, TaskFamily, reject_unknown_keys
from infra.envs.tasks.math import MathEnv, _numeric_protocol_identity, parse_numeric_answers

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
        think_overshoot_penalty: float = 0.0,
        prompt_file: str | None = None,
        pool_held_out: bool = False,
    ):
        self.rng = random.Random(seed)
        self.prompts = load_generation_prompts(resolve_prompt_file(prompt_file, PROMPT_FILE))
        self.correct_reward = correct_reward
        self.format_reward = format_reward
        self.relaxed_correct_bonus = relaxed_correct_bonus
        self.shaped_reward = 0.0
        # Same knob, validation placement (before the dataset download), and
        # reward_sample as MathEnv — the methods are shared by assignment
        # below, and the overshoot fact rides the Sample's regions.
        self.think_overshoot_penalty = float(think_overshoot_penalty)
        if self.think_overshoot_penalty < 0:
            raise ValueError(
                f"aime: think_overshoot_penalty must be >= 0 (it is SUBTRACTED "
                f"from the reward on overshoot), got {think_overshoot_penalty}"
            )

        rows = _rows(load_dataset(DATASET_ID)["train"])
        rng = random.Random(seed + 22222)
        rng.shuffle(rows)
        n_test = max(64, len(rows) // 10)
        self.test_rows, self.train_rows = rows[:n_test], rows[n_test:]
        self.test_rows = self.test_rows[:eval_subset_size]
        # Eager dev carve, same rationale and shape as MathEnv's: test rows
        # stay byte-identical to the campaign's historical eval split.
        k = min(len(self.test_rows), max(1, len(self.train_rows) // 10))
        self.dev_rows, self.train_rows = self.train_rows[:k], self.train_rows[k:]
        # A dev/test split buys model selection on one and an unbiased report
        # on the other. Tracking a metric across a training trajectory needs
        # neither, and pays for the ceremony in statistical power: 83 dev rows
        # give a binomial SE near .05, which is wider than every arm difference
        # measured so far. Pooling both held-out carves is 176 rows and ~.035.
        # Off by default so existing arms keep their historical dev split.
        if pool_held_out:
            self.dev_rows = self.dev_rows + self.test_rows
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
    reward_sample = MathEnv.reward_sample


class AimeFamily(TaskFamily):
    def source(self, ds: dict) -> Env:
        reject_unknown_keys(
            ds,
            {
                "seed",
                "eval_subset_size",
                "prompt_file",
                "think_overshoot_penalty",
                "correct_reward",
                "format_reward",
                "relaxed_correct_bonus",
                "pool_held_out",
            },
            "aime",
        )
        seed = int(ds.get("seed", 0))
        eval_subset_size = int(ds.get("eval_subset_size", 512))
        prompt_path = resolve_prompt_file(
            (str(ds["prompt_file"]) if ds.get("prompt_file") else None), PROMPT_FILE
        )
        env = AimeEnv(
            seed=seed,
            eval_subset_size=eval_subset_size,
            correct_reward=float(ds.get("correct_reward", 1.0)),
            format_reward=float(ds.get("format_reward", 0.1)),
            relaxed_correct_bonus=float(ds.get("relaxed_correct_bonus", 0.1)),
            think_overshoot_penalty=float(ds.get("think_overshoot_penalty", 0.0)),
            prompt_file=str(prompt_path),
            pool_held_out=bool(ds.get("pool_held_out", False)),
        )
        self._protocol_identity = _numeric_protocol_identity(
            env,
            dataset_id=DATASET_ID,
            seed=seed,
            eval_subset_size=eval_subset_size,
            prompt_path=prompt_path,
        )
        return env

    def parse_answers(self, text: str) -> AnswerParse:
        return parse_numeric_answers(text)

    def grade(self, meta: dict[str, Any], solution: Any) -> Optional[bool]:
        gt = meta.get("gt")
        if gt is None or solution is None:
            return None
        try:
            return abs(float(solution) - float(gt)) < 1e-6
        except (TypeError, ValueError):
            return None

    def protocol_identity(self) -> dict[str, str]:
        return dict(getattr(self, "_protocol_identity", {}))
