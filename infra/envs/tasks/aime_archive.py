"""AIME archive pool: an EVAL-ONLY task family (dataset.type: aime_archive).

A fixed seeded carve of the 1983-2024 AIME archive, sized for resolution: the
83-row AMC and AIME dev pools put ~1.2pp on every problem flip, and their
informative bands are ~25-30 problems. Decade solve rates (OLMo-32B uncapped,
2026-08-19): 1980s .59 / 1990s .62 / 2000s .80 / 2010s .64 / 2020s .39 — the
default year range stops at 2019 so the hardest, freshest years stay reserved
for a frontier line and clear of aimo-validation-aime (2022-24).
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

from datasets import load_dataset

from infra.envs.answer_parsing import parse_number
from infra.envs.base import Env
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file
from infra.envs.tasks.base import reject_unknown_keys
from infra.envs.tasks.math import MathEnv, MathFamily, PROMPT_FILE

DATASET_ID = "di-zhang-fdu/AIME_1983_2024"


def _load_archive(dataset_id: str = DATASET_ID):
    return load_dataset(dataset_id)["train"]


class AimeArchiveEnv(MathEnv):
    """MathEnv's reward protocol over a seeded AIME-archive carve. The carve
    is the eval set (dev == test rows); train sampling is a hard error so a
    config that points training at an eval pool fails at the first rollout."""

    def __init__(
        self,
        seed: int = 0,
        n: int = 300,
        year_min: int = 1983,
        year_max: int = 2019,
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
                f"aime_archive: think_overshoot_penalty must be >= 0, got {think_overshoot_penalty}"
            )
        if not 1983 <= int(year_min) <= int(year_max):
            raise ValueError(f"aime_archive: bad year range {year_min}..{year_max}")

        rows = []
        for ex in _load_archive(dataset_id):
            problem = str(ex.get("Question", "")).strip()
            gt = parse_number(str(ex.get("Answer", "")).strip())
            year = re.match(r"(\d{4})", str(ex.get("ID", "")))
            if (
                not problem
                or gt is None
                or not math.isfinite(gt)
                or year is None
                or not int(year_min) <= int(year.group(1)) <= int(year_max)
            ):
                continue
            rows.append({"problem": problem, "gt": float(gt), "level": 0, "id": str(ex["ID"])})
        if len(rows) < int(n):
            raise RuntimeError(
                f"aime_archive: only {len(rows)} usable rows in {year_min}..{year_max}, need n={n}"
            )
        carve_rng = random.Random(seed + 33333)
        carve_rng.shuffle(rows)
        pool = rows[: int(n)]
        self.train_rows = []
        self.dev_rows = pool
        self.test_rows = pool

    def tasks(self, n: int, split: str = "train"):
        if split == "train":
            raise RuntimeError("aime_archive is an eval-only pool; it cannot serve train rollouts")
        return super().tasks(n, split)


class AimeArchiveFamily(MathFamily):
    def source(self, ds: dict) -> Env:
        reject_unknown_keys(
            ds,
            {
                "seed",
                "n",
                "year_min",
                "year_max",
                "dataset_id",
                "prompt_file",
                "think_overshoot_penalty",
            },
            "dataset (aime_archive)",
        )
        return AimeArchiveEnv(**ds)
