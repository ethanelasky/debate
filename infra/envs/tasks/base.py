"""Task family: everything task-specific the runners need, in one object.

DebateEnv and the RLVR runner stay domain-blind; adding a task domain
(math, codecontests, ...) means one module implementing TaskFamily plus a
registry entry in infra/envs/tasks/__init__.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from infra.envs.base import Env


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

    def format_flags(self, text: str) -> dict[str, float]:
        """Strict-format flags on solution slots, consumed by shaping terms
        (e.g. format_reward on "strict_boxed")."""
        return {}


def reject_unknown_keys(ds: dict, known: set[str], family: str) -> None:
    unknown = sorted(set(ds) - known)
    if unknown:
        raise ValueError(f"dataset keys {unknown} unknown to task family {family!r} (known: {sorted(known)})")
