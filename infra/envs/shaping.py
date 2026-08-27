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
