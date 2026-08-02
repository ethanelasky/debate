"""RLVR math environment: Hendrycks MATH (level 5 by default) with boxed-answer
verification. Ported from CS285 HW4 MathHardTask, slimmed."""

from __future__ import annotations

import math
import random
from typing import Any

from datasets import get_dataset_config_names, load_dataset

from infra.envs.answer_parsing import (
    extract_last_boxed_content,
    extract_last_number,
    extract_number_from_boxed_answer,
    parse_number,
)
from infra.envs.base import Env, Policy, Task, Trajectory, datum_from_sample

DATASET_ID = "the-jb/hendrycks-math"

SYSTEM_PROMPT = (
    "Solve the competition-level math problem.\n"
    "Return the final answer in this exact format: \\boxed{NUMBER}\n"
    "Do not output XML, ####, or extra prose.\n"
    "Output only the boxed final answer."
)


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


class MathEnv(Env):
    def __init__(
        self,
        levels: tuple[int, ...] = (5,),
        seed: int = 0,
        eval_subset_size: int = 512,
        correct_reward: float = 1.0,
        format_reward: float = 0.1,
        relaxed_correct_bonus: float = 0.1,
        shaped_reward: float = 0.0,
    ):
        self.rng = random.Random(seed)
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["problem"] + "\n\nReturn only: \\boxed{NUMBER}"},
        ]
        return Task(messages=messages, meta={"gt": row["gt"], "level": row["level"], "split": split})

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

    def rollout(self, tasks: list[Task], policy: Policy, group_size: int) -> list[list[Trajectory]]:
        results = policy.predict([t.messages for t in tasks], n=group_size)
        groups: list[list[Trajectory]] = []
        n_dropped = 0
        for task, samples in zip(tasks, results):
            trajs = []
            for s in samples:
                if not s.fidelity_ok():
                    n_dropped += 1
                    continue
                reward, info = self.reward(task, s.text)
                trajs.append(Trajectory(datums=[datum_from_sample(s)], reward=reward, info=info))
            groups.append(trajs)
        if n_dropped and groups and groups[0]:
            groups[0][0].info["samples_dropped_fidelity"] = float(n_dropped)
        return groups
