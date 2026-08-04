"""Math task family: Hendrycks MATH with boxed-number extraction and
numeric-tolerance grading. MathEnv was ported from CS285 HW4 MathHardTask
(slimmed); the dataset-block parsing lived in run_debate.py before the task
registry existed; both moved here verbatim."""

from __future__ import annotations

import math
import random
from typing import Any, Optional

from datasets import get_dataset_config_names, load_dataset

from infra.envs.answer_parsing import (
    extract_last_boxed_content,
    extract_last_number,
    extract_number_from_boxed_answer,
    parse_number,
)
from infra.envs.base import Env, SingleTurnEnv, Task
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file
from infra.envs.tasks.base import TaskFamily, reject_unknown_keys

DATASET_ID = "the-jb/hendrycks-math"
PROMPT_FILE = "math.yaml"


def _load(dataset_id: str = DATASET_ID):
    try:
        return load_dataset(dataset_id)
    except Exception:
        for cfg in get_dataset_config_names(dataset_id):
            try:
                return load_dataset(dataset_id, cfg)
            except Exception:
                continue
        raise


def _rows(split, levels: set[int]) -> list[dict[str, Any]]:
    rows = []
    for ex in split:
        level_txt = str(ex.get("level", ex.get("difficulty", "")))
        digits = "".join(c for c in level_txt if c.isdigit())
        if not digits or int(digits) not in levels:
            continue
        problem = str(ex.get("problem", ex.get("question", ""))).strip()
        solution = str(ex.get("solution", ex.get("answer", ""))).strip()
        if not problem or not solution:
            continue
        gt = extract_number_from_boxed_answer(solution)
        if gt is None:
            gt = parse_number(solution)
        if gt is None or not math.isfinite(gt):
            continue
        rows.append({"problem": problem, "gt": float(gt), "level": int(digits)})
    return rows


class MathEnv(SingleTurnEnv):
    def __init__(
        self,
        levels: tuple[int, ...] = (5,),
        seed: int = 0,
        eval_subset_size: int = 512,
        correct_reward: float = 1.0,
        format_reward: float = 0.1,
        relaxed_correct_bonus: float = 0.1,
        shaped_reward: float = 0.0,
        prompt_file: str | None = None,
    ):
        self.rng = random.Random(seed)
        self.prompts = load_generation_prompts(resolve_prompt_file(prompt_file, PROMPT_FILE))
        self.correct_reward = correct_reward
        self.format_reward = format_reward
        self.relaxed_correct_bonus = relaxed_correct_bonus
        self.shaped_reward = shaped_reward

        ds = _load()
        level_set = set(levels)
        self.train_rows = _rows(ds["train"], level_set)
        test_split = next((ds[s] for s in ("test", "validation") if s in ds), None)
        if test_split is not None:
            self.test_rows = _rows(test_split, level_set)
        else:
            rng = random.Random(seed + 22222)
            rows = list(self.train_rows)
            rng.shuffle(rows)
            n_test = max(64, len(rows) // 10)
            self.test_rows, self.train_rows = rows[:n_test], rows[n_test:]
        self.test_rows = self.test_rows[:eval_subset_size]
        if len(self.train_rows) < 2 or not self.test_rows:
            raise RuntimeError(
                f"math env: too few rows after filtering (train={len(self.train_rows)}, test={len(self.test_rows)})"
            )

    def _task(self, row: dict[str, Any], split: str) -> Task:
        return Task(
            messages=self.prompts.messages(row["problem"]),
            meta={"gt": row["gt"], "level": row["level"], "split": split, "question": row["problem"]},
        )

    def tasks(self, n: int, split: str = "train") -> list[Task]:
        if split == "train":
            return [self._task(self.rng.choice(self.train_rows), split) for _ in range(n)]
        return [self._task(row, split) for row in self.test_rows[:n]]

    def reward(self, task: Task, text: str) -> tuple[float, dict[str, Any]]:
        gt = task.meta["gt"]
        pred_boxed = extract_number_from_boxed_answer(text)
        pred_relaxed = pred_boxed if pred_boxed is not None else extract_last_number(text)
        exact_boxed = pred_boxed is not None and abs(pred_boxed - gt) < 1e-6
        exact_relaxed = pred_relaxed is not None and abs(pred_relaxed - gt) < 1e-6
        has_boxed = "\\boxed{" in text.lower()

        reward = 0.0
        reward += self.format_reward if has_boxed else 0.0
        reward += self.correct_reward if exact_boxed else 0.0
        if exact_relaxed and not exact_boxed:
            reward += self.relaxed_correct_bonus * self.correct_reward
        if self.shaped_reward > 0.0 and pred_boxed is not None:
            rel_err = abs(pred_boxed - gt) / max(1.0, abs(gt))
            reward += self.shaped_reward * math.exp(-4.0 * rel_err)

        info = {
            "correct": float(exact_boxed),
            "correct_relaxed": float(exact_relaxed),
            "has_boxed": float(has_boxed and extract_last_boxed_content(text) is not None),
        }
        return reward, info


def strict_extract(text: str) -> Optional[float]:
    return extract_number_from_boxed_answer(text)


def relaxed_extract(text: str) -> Optional[float]:
    v = extract_number_from_boxed_answer(text)
    return v if v is not None else extract_last_number(text)


def _parse_levels(spec) -> tuple[int, ...]:
    """int (5), range string ("3-4"), or list. Scalars/strings survive
    _extends cleanly; lists merge BY INDEX (child [5] over parent [3,4]
    becomes [5,4]) — prefer the scalar/range forms in configs."""
    if isinstance(spec, int):
        return (spec,)
    if isinstance(spec, str):
        if "-" in spec:
            lo, hi = spec.split("-", 1)
            return tuple(range(int(lo), int(hi) + 1))
        return (int(spec),)
    return tuple(int(x) for x in spec)


class MathFamily(TaskFamily):
    def source(self, ds: dict) -> Env:
        reject_unknown_keys(ds, {"levels", "seed", "eval_subset_size", "prompt_file"}, "math")
        return MathEnv(
            seed=int(ds.get("seed", 0)),
            levels=_parse_levels(ds.get("levels", 5)),
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
        # strict format flag, independent of the (possibly relaxed) extractor
        # used for position binding
        return {"strict_boxed": float(extract_number_from_boxed_answer(text) is not None)}
