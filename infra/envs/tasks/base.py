"""Task family: everything task-specific the runners need, in one object.

DebateEnv and the RLVR runner stay domain-blind; adding a task domain
(math, codecontests, ...) means one module implementing TaskFamily plus a
registry entry in infra/envs/tasks/__init__.py.

Families may hold state: get_family() builds a fresh instance per run, and
source() already sets per-config state on it (the codecontests grading
timeout). A family backed by a learned verifier (an ORM, or an
LLM-equivalence check) holds heavier state — a model client, lazily opened
on first grade — and overrides grade_batch() to score the whole batch in
one call instead of len(items) pooled grade() calls. Callers go through
grade_batch(); grade() is the per-pair primitive programmatic verifiers
implement. The judge-only reward rule is untouched by any of this: grades
feed metrics and transcripts, never a debate reward cell (rewards.py).
"""

from __future__ import annotations

import concurrent.futures
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from infra.config import reject_unknown_keys as _reject_unknown_keys
from infra.envs.base import Env
from infra.envs.task_prompts import (  # re-exported for family modules
    GenerationPrompts,
    load_generation_prompts,
    resolve_prompt_file,
)


class TaskFamily(ABC):
    @abstractmethod
    def source(self, ds: dict) -> Env:
        """Dataset config block (minus 'type'/'relaxed_extraction') -> the
        task-source Env. Doubles as the RLVR env: its rollout() carries the
        verifiable reward, its tasks() feeds DebateEnv. Task.meta must carry
        {"question": str} plus whatever grade() needs; assigned-position
        debates additionally need {"gold", "distractor"}. Task.messages is the
        RLVR prompt VERBATIM (optional system, then user/assistant alternation
        ending on a user message — the eliciting cue); solo-context debates
        (first_speech_non_debate_aware) render it as-is, byte-identical to the
        RLVR arm."""

    @abstractmethod
    def extractor(self, relaxed: bool) -> Callable[[str], Any]:
        """Visible slot text -> parsed solution (None = unparseable). Binds
        debate POSITIONs; `relaxed` is the dataset.relaxed_extraction knob."""

    @abstractmethod
    def grade(self, meta: dict[str, Any], solution: Any) -> Optional[bool]:
        """Is an extracted solution correct for the task with this meta?
        None = ungradeable (no ground truth / malformed solution)."""

    #: grade_batch's default pool width. Meaningful for graders that block per
    #: call (codecontests' subprocess verifier); harmless for pure-python ones.
    grade_workers: int = 8

    #: Per-item failures swallowed by the last grade_batch() call. Read by
    #: DebateEnv for the grade_errors rollout metric.
    last_grade_errors: int = 0

    def grade_batch(self, items: list[tuple[dict[str, Any], Any]]) -> list[Optional[bool]]:
        """(meta, solution) pairs -> grades, positionally. THE seam callers
        grade through; grade() is the per-pair primitive behind it.

        Contract: never raises — a grade failure must not kill a rollout — so
        a per-item exception grades as None and is counted in
        last_grade_errors. Callers dedup before calling (grading can be
        expensive); implementations may assume items are distinct but not
        rely on it.

        Override this for verifiers that want the batch whole: a learned
        verifier scores one batched GPU/API call rather than len(items)
        pooled grade() calls. The default runs grade() in a thread pool of
        grade_workers (subprocess graders overlap; pure-python ones are
        unharmed)."""
        self.last_grade_errors = 0
        if not items:
            return []

        def _one(item: tuple[dict[str, Any], Any]) -> tuple[Optional[bool], bool]:
            try:
                return self.grade(*item), False
            except Exception:  # noqa: BLE001
                return None, True

        if len(items) == 1 or self.grade_workers <= 1:
            scored = [_one(item) for item in items]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.grade_workers) as pool:
                scored = list(pool.map(_one, items))
        self.last_grade_errors = sum(err for _, err in scored)
        return [g for g, _ in scored]

    def format_flags(self, text: str) -> dict[str, float]:
        """Strict-format flags on solution slots, consumed by shaping terms
        (e.g. format_reward on "strict_boxed")."""
        return {}


def reject_unknown_keys(ds: dict, known: set[str], family: str) -> None:
    """Dataset-block flavour of infra.config.reject_unknown_keys."""
    _reject_unknown_keys(ds, known, f"dataset block for task family {family!r}")


__all__ = [
    "TaskFamily",
    "reject_unknown_keys",
    "GenerationPrompts",
    "load_generation_prompts",
    "resolve_prompt_file",
]
