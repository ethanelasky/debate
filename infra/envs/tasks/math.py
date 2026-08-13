"""Math task family: Hendrycks MATH with boxed-number extraction and
numeric-tolerance grading. MathEnv was ported from CS285 HW4 MathHardTask
(slimmed); the dataset-block parsing lived in run_debate.py before the task
registry existed; both moved here verbatim."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Optional

from datasets import get_dataset_config_names, load_dataset

from infra.envs.answer_parsing import (
    extract_last_number,
    extract_number_from_boxed_answer,
    parse_number,
)
from infra.envs.base import Env, SingleTurnEnv, Task
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file
from infra.envs.tasks.base import AnswerParse, TaskFamily, reject_unknown_keys

DATASET_ID = "the-jb/hendrycks-math"
PROMPT_FILE = "math.yaml"


def parse_numeric_answers(text: str) -> AnswerParse:
    """Extract numeric MATH/AIME candidates with strict-first precedence."""
    strict = extract_number_from_boxed_answer(text)
    relaxed = strict if strict is not None else extract_last_number(text)
    return AnswerParse(strict=strict, relaxed=relaxed)


def _split_digest(env: Any) -> str:
    """Hash the exact ordered cohorts that this source will expose."""
    payload = {
        split: [[row["problem"], row["gt"]] for row in getattr(env, f"{split}_rows")]
        for split in ("train", "dev", "test")
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _numeric_protocol_identity(
    env: Any,
    *,
    dataset_id: str,
    seed: int,
    eval_subset_size: int,
    prompt_path: Path,
    levels: tuple[int, ...] | None = None,
) -> dict[str, str]:
    identity = {
        "grading_protocol": "numeric_box_v1",
        "dataset_id": dataset_id,
        "dataset_revision": "unpinned_legacy",
        "seed": str(seed),
        "eval_subset_size": str(eval_subset_size),
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "split_sha256": _split_digest(env),
        "train_count": str(len(env.train_rows)),
        "dev_count": str(len(env.dev_rows)),
        "test_count": str(len(env.test_rows)),
        # Store the resolved values from the source env, not the spelling used
        # in config, so equivalent inputs (for example 1 and 1.0) have one
        # stable identity while every reward-affecting coefficient is pinned.
        "correct_reward": repr(float(env.correct_reward)),
        "format_reward": repr(float(env.format_reward)),
        "relaxed_correct_bonus": repr(float(env.relaxed_correct_bonus)),
        "think_overshoot_penalty": repr(float(env.think_overshoot_penalty)),
    }
    if levels is not None:
        identity["levels"] = ",".join(str(level) for level in levels)
    return identity


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
        think_overshoot_penalty: float = 0.0,
        prompt_file: str | None = None,
    ):
        self.rng = random.Random(seed)
        self.prompts = load_generation_prompts(resolve_prompt_file(prompt_file, PROMPT_FILE))
        self.correct_reward = correct_reward
        self.format_reward = format_reward
        self.relaxed_correct_bonus = relaxed_correct_bonus
        self.shaped_reward = shaped_reward
        # Think-overshoot penalty (dataset.think_overshoot_penalty): priced in
        # reward_sample, not reward() — the overshoot fact lives on the
        # Sample's regions, which reward() (text-only) never sees. Validated
        # BEFORE the dataset download so a bad config fails in milliseconds.
        self.think_overshoot_penalty = float(think_overshoot_penalty)
        if self.think_overshoot_penalty < 0:
            raise ValueError(
                f"math: think_overshoot_penalty must be >= 0 (it is SUBTRACTED "
                f"from the reward on overshoot), got {think_overshoot_penalty}"
            )

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
        # Dev split (Ethan, 2026-08-11): carved EAGERLY from the train side —
        # test_rows stay byte-identical to every historical eval, and no
        # training step can ever sample a dev row (a lazy carve would leak
        # dev rows into the steps before the first dev eval). dev/ steers
        # decisions during training; test/ is touched once at the end.
        k = min(len(self.test_rows), max(1, len(self.train_rows) // 10))
        self.dev_rows, self.train_rows = self.train_rows[:k], self.train_rows[k:]
        if len(self.train_rows) < 2 or not self.test_rows:
            raise RuntimeError(
                f"math env: too few rows after filtering (train={len(self.train_rows)}, test={len(self.test_rows)})"
            )

    def _task(self, row: dict[str, Any], split: str) -> Task:
        return Task(
            messages=self.prompts.render({"PROBLEM": row["problem"]}),
            meta={"gt": row["gt"], "level": row["level"], "split": split, "question": row["problem"]},
        )

    def tasks(self, n: int, split: str = "train") -> list[Task]:
        if split == "train":
            return [self._task(self.rng.choice(self.train_rows), split) for _ in range(n)]
        if split == "dev":
            return [self._task(row, split) for row in self.dev_rows[:n]]
        return [self._task(row, split) for row in self.test_rows[:n]]

    def reward(self, task: Task, text: str) -> tuple[float, dict[str, Any]]:
        gt = task.meta["gt"]
        parsed = parse_numeric_answers(text)
        pred_boxed = parsed.strict
        pred_relaxed = parsed.relaxed
        exact_boxed = pred_boxed is not None and abs(pred_boxed - gt) < 1e-6
        exact_relaxed = pred_relaxed is not None and abs(pred_relaxed - gt) < 1e-6
        # Preserve the established reward protocol exactly. This deliberately
        # differs from the observational answer_format_valid metric: malformed
        # raw box starts earn the historical bonus, while spaced ``\boxed {``
        # forms do not.
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
            "correct_strict": float(exact_boxed),
            "correct_relaxed": float(exact_relaxed),
            "answer_format_valid": float(parsed.answer_format_valid),
        }
        return reward, info

    def reward_sample(self, task: Task, sample) -> tuple[float, dict[str, Any]]:
        """reward() plus the think-overshoot penalty: a flat coefficient off
        the reward when the sample's think phase was FORCE-CLOSED at its cap —
        i.e. the Sample carries a "forced_close" region, which only
        budget_forced_sample (the think_tokens arms) ever writes, so the knob
        is inert on single-phase rollouts. 0.0 (the default) short-circuits to
        the plain reward() path: byte-identical rewards AND info. When active,
        think_overshoot (0.0/1.0) is present on EVERY branch (reward()'s
        every-branch rule), so eval-time means are over all samples."""
        reward, info = self.reward(task, sample.text)
        coeff = self.think_overshoot_penalty
        if coeff == 0.0:
            return reward, info
        overshoot = any(
            r.kind == "forced_close" for r in (getattr(sample, "regions", None) or ())
        )
        if overshoot:
            reward -= coeff
        return reward, {**info, "think_overshoot": float(overshoot)}


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
        reject_unknown_keys(
            ds,
            {
                "levels",
                "seed",
                "eval_subset_size",
                "prompt_file",
                "think_overshoot_penalty",
                "correct_reward",
                "format_reward",
                "relaxed_correct_bonus",
            },
            "math",
        )
        seed = int(ds.get("seed", 0))
        levels = _parse_levels(ds.get("levels", 5))
        eval_subset_size = int(ds.get("eval_subset_size", 512))
        prompt_path = resolve_prompt_file(
            (str(ds["prompt_file"]) if ds.get("prompt_file") else None), PROMPT_FILE
        )
        env = MathEnv(
            seed=seed,
            levels=levels,
            eval_subset_size=eval_subset_size,
            correct_reward=float(ds.get("correct_reward", 1.0)),
            format_reward=float(ds.get("format_reward", 0.1)),
            relaxed_correct_bonus=float(ds.get("relaxed_correct_bonus", 0.1)),
            think_overshoot_penalty=float(ds.get("think_overshoot_penalty", 0.0)),
            prompt_file=str(prompt_path),
        )
        self._protocol_identity = _numeric_protocol_identity(
            env,
            dataset_id=DATASET_ID,
            levels=levels,
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
