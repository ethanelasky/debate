"""Math task family: Hendrycks MATH via MathEnv, boxed-number extraction,
numeric-tolerance grading. The dataset-block parsing lived in run_debate.py
before the task registry existed; moved here verbatim."""

from __future__ import annotations

from typing import Any, Optional

from infra.envs.answer_parsing import extract_last_number, extract_number_from_boxed_answer
from infra.envs.base import Env
from infra.envs.math_env import MathEnv
from infra.envs.tasks.base import TaskFamily, reject_unknown_keys


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
        reject_unknown_keys(ds, {"levels", "seed", "eval_subset_size"}, "math")
        return MathEnv(
            seed=int(ds.get("seed", 0)),
            levels=_parse_levels(ds.get("levels", 5)),
            eval_subset_size=int(ds.get("eval_subset_size", 512)),
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
