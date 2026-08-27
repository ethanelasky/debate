"""Flag-gated reward shaping — the machinery both arms use.

Both arms price the same SHAPE of thing: a flat coefficient times a per-datum
feature. They used to spell it three times — ``MathEnv.reward_sample`` and
``SymbolicMathEnv.reward_sample`` inlined the arithmetic, and the debate env
kept a class per term with the loop copied between them. When the two arms
were meant to price budget overshoot identically, nothing structural made them
agree, and a config comment asserted the parity instead (see
configs/_qwen35_training.yaml).

What lives here is the machinery only. The FEATURE stays per-family and is
meant to: ``MathEnv`` pays its historical raw-``\\boxed{`` test while
``SymbolicMathEnv`` pays ``answer_format_valid``, and collapsing those would
silently restate an established reward protocol. ``think_overshoot`` is the one
feature that genuinely has a single definition, so it is defined once below.

Sign convention: a term states a POSITIVE ``coeff`` and carries its own
``sign``, so ``dataset.think_overshoot_penalty: 0.1`` and
``{kind: think_overshoot_penalty, coeff: 0.1}`` read the same way and price the
same amount.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional


def think_overshoot(sample: Any) -> bool:
    """The single definition of "this slot's think phase was FORCE-CLOSED at
    its cap": the Sample carries a ``forced_close`` region.

    Only ``budget_forced_sample`` writes that region (infra/envs/base.py:481),
    which runs only when a slot declares a think/visible cap — so this is False
    on single-phase rollouts, and a term gating on it is inert there rather
    than wrong. Frozen API seats cannot enforce caps at all and return no
    regions (infra/envs/debate/round.py:252); callers that PRICE this feature
    must reject that case rather than read the absence as "did not overshoot".
    """
    return any(r.kind == "forced_close" for r in (getattr(sample, "regions", None) or ()))


def truncated(sample: Any) -> bool:
    """The single definition of "this generation ran out of budget": the
    backend normalized its stop reason to "length" rather than "stop".

    Unlike ``think_overshoot`` this needs no regions, so it is knowable on any
    seat that reports a stop reason. It says the speech did not finish, which
    is the feature a per-speech budget term prices: a speech cut at its cap
    reaches the judge mid-sentence.
    """
    return getattr(sample, "stop_reason", None) == "length"


#: Features computable from a Sample alone, shared by both arms. Keyed by the
#: flag name a term gates on.
SAMPLE_FLAGS: dict[str, Callable[[Any], bool]] = {
    "think_overshoot": think_overshoot,
    "truncated": truncated,
}


@dataclass(frozen=True)
class FlagTerm:
    """``sign * coeff * flags[flag]``, optionally restricted to named slots.

    ``slots=None`` matches every unit, which is also how the single-datum RLVR
    path (no slot vocabulary) matches.
    """

    coeff: float
    flag: str
    sign: int = 1
    slots: Optional[Iterable[str]] = None

    def matches(self, slot: Optional[str]) -> bool:
        return self.slots is None or (slot is not None and slot in self.slots)

    def delta(self, flags: Mapping[str, float], slot: Optional[str] = None) -> float:
        if not self.coeff or not self.matches(slot):
            return 0.0
        return self.sign * self.coeff * float(flags.get(self.flag) or 0.0)


@dataclass(frozen=True)
class BudgetTerm:
    """``sign * coeff * excess(n, limit)``, optionally restricted to named slots.

    The magnitude-valued twin of ``FlagTerm``, and it lives here for the same
    reason that one does. Both arms price "this generation ran past its length
    budget", and until this class they priced it with two implementations on two
    config surfaces: ``soft_token_budget``/``overshoot_penalty`` in
    ``SingleTurnEnv.rollout`` and a ``scoring.shaping`` entry on the debate
    side. Nothing structural made them agree, and each was silently INERT on
    the other arm -- ``SingleTurnEnv.rollout`` never executes for a debate
    (DebateEnv extends Env with its own rollout), so a proposal budget set
    there was accepted by config and did nothing at all. That is the same
    value-shaped void ``dataset.think_overshoot_penalty`` fell into before the
    flag predicates were hoisted here.

    ``mode`` is the shape:

    ``proportional`` charges per unit past ``limit`` -- Kenton et al.'s "soft
        additive penalty proportional to the excess" (2608.17776 3.4).
    ``flat`` charges ``coeff`` once for any overrun, however small.

    Prefer ``proportional`` for anything a group-relative optimizer trains on.
    A flat term reaches a CISPO advantage only through its variance INSIDE the
    group, so on a group whose samples all sit on one side of the budget it is
    a constant shift the baseline removes -- exactly zero signal. Its pressure
    peaks near a 50% trip rate and switches off as the policy complies, which
    is how the flat ``think_overshoot_penalty`` drove the RLVR arm's length to
    collapse on 2026-08-27 and then went inert at step 32, leaving the policy
    short with nothing holding it there.

    ``limit`` is what keeps either shape from becoming a shortness reward:
    below it the charge is exactly zero, so an empty generation earns nothing a
    compliant one does not. A per-unit price with no limit is optimised by
    saying nothing.
    """

    coeff: float
    limit: int
    mode: str = "proportional"
    sign: int = -1
    slots: Optional[Iterable[str]] = None

    def __post_init__(self) -> None:
        # A typo'd mode must not silently pick a shape: the two differ by a
        # factor of the excess, and that difference decided a run.
        if self.mode not in ("proportional", "flat"):
            raise ValueError(
                f"budget term mode must be 'proportional' or 'flat', got {self.mode!r}"
            )
        if self.limit < 0:
            raise ValueError(f"budget term limit must be >= 0, got {self.limit}")

    def matches(self, slot: Optional[str]) -> bool:
        return self.slots is None or (slot is not None and slot in self.slots)

    def excess(self, n: float) -> float:
        """Units past the budget, which is the number a metric should report.
        A boolean `over_budget` saturates at 1.0 the moment the limit is
        crossed, so a batch drifting from barely-over to far-over looks
        identical to a stable one."""
        return float(max(0.0, float(n) - self.limit))

    def delta(self, n: float, slot: Optional[str] = None) -> float:
        if not self.coeff or not self.matches(slot):
            return 0.0
        over = self.excess(n)
        if not over:
            return 0.0
        return self.sign * self.coeff * (1.0 if self.mode == "flat" else over)


def sample_flags(sample: Any, names: Iterable[str]) -> dict[str, float]:
    """The named sample-level features as 0.0/1.0.

    Unknown names raise: a flag nobody can compute must not read as 0.0, which
    is the failure mode this module exists to remove.
    """
    out: dict[str, float] = {}
    for name in names:
        predicate = SAMPLE_FLAGS.get(name)
        if predicate is None:
            raise KeyError(
                f"no sample-level predicate for flag {name!r}; known: {sorted(SAMPLE_FLAGS)}"
            )
        out[name] = float(predicate(sample))
    return out


def shaped_sample_reward(
    reward: float,
    info: dict[str, Any],
    sample: Any,
    terms: Iterable[FlagTerm],
) -> tuple[float, dict[str, Any]]:
    """``(reward, info)`` from a task env's ``reward()``, plus its sample-level
    terms. This is the whole body of a ``reward_sample`` implementation.

    With no priced term this returns the caller's own ``(reward, info)``
    objects untouched — byte-identical rewards AND info, which is what keeps
    ``think_overshoot_penalty: 0.0`` (the default) a true no-op. When a term is
    priced, its flag is present on EVERY branch, so eval-time means are over
    all samples rather than over the ones that tripped it.
    """
    priced = [t for t in terms if t.coeff]
    if not priced:
        return reward, info
    flags = sample_flags(sample, {t.flag for t in priced})
    delta = sum(t.delta(flags) for t in priced)
    return reward + delta, {**info, **flags}
