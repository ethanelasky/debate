"""AMC competition pool: an EVAL-ONLY task family (dataset.type: amc).

Fit check (2026-08-14): Qwen3.5-4B base scores .422 greedy / .432+.578(p8)
sampled on AI-MO/aimo-validation-amc while saturating MATH-L5 dev at ~.97 —
this pool discriminates where L5 cannot. Wired as `eval_dataset` so training
keeps its own pool and only dev/test evals read AMC.
"""

from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset

from infra.envs.answer_parsing import parse_number
from infra.envs.base import Env
from infra.envs.tasks.base import reject_unknown_keys
from infra.envs.tasks.math import MathEnv, MathFamily, PROMPT_FILE
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file

DATASET_ID = "AI-MO/aimo-validation-amc"


def _load_amc(dataset_id: str = DATASET_ID):
    return load_dataset(dataset_id, split="train")


class AmcEnv(MathEnv):
    """MathEnv's reward protocol over the AMC pool. The whole pool is the
    eval set (dev == test rows); train sampling is a hard error so a config
    that points training at an eval pool fails at the first rollout."""

    def __init__(
        self,
        seed: int = 0,
        dataset_id: str = DATASET_ID,
        prompt_file: str | None = None,
        think_overshoot_penalty: float = 0.0,
    ):
        self.rng = random.Random(seed)
        self.prompts = load_generation_prompts(resolve_prompt_file(prompt_file, PROMPT_FILE))
        self.correct_reward = 1.0
        self.format_reward = 0.1
        self.relaxed_correct_bonus = 0.1
        self.shaped_reward = 0.0
        self.think_overshoot_penalty = float(think_overshoot_penalty)
        if self.think_overshoot_penalty < 0:
            raise ValueError(
                f"amc: think_overshoot_penalty must be >= 0, got {think_overshoot_penalty}"
            )

        rows = []
        for ex in _load_amc(dataset_id):
            problem = str(ex.get("problem", "")).strip()
            gt = parse_number(str(ex.get("answer", "")))
            if problem and gt is not None:
                rows.append({"problem": problem, "gt": float(gt), "level": 0})
        if not rows:
            raise RuntimeError(f"amc env: no usable rows in {dataset_id}")
        self.train_rows = []
        self.dev_rows = rows
        self.test_rows = rows

    def tasks(self, n: int, split: str = "train"):
        if split == "train":
            raise RuntimeError("amc is an eval-only pool; it cannot serve train rollouts")
        return super().tasks(n, split)


class AmcFamily(MathFamily):
    def source(self, ds: dict) -> Env:
        reject_unknown_keys(
            ds,
            {"seed", "dataset_id", "prompt_file", "think_overshoot_penalty"},
            "dataset (amc)",
        )
        return AmcEnv(**ds)
