"""Verdict -> per-seat reward: the ladder, continuous scoring, shaping.

The ladder is the only place a verdict becomes a number. Ground-truth
correctness deliberately never enters a reward cell — this is a judge-only
environment; gt rides along in info for metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional, Protocol

from pydantic import BaseModel, Field

from infra.envs.debate.judge import LogitStatus, SeatVerdict, Verdict
from infra.envs.shaping import FlagTerm
from infra.models.base import ModelSettings, resolved_sampling_profile

RewardMode = Literal["competitive", "collaborative"]


class LadderValues(BaseModel):
    solo_win: float = 1.0
    solo_loss: float = -1.0
    double_win: float = 0.0
    double_loss: float = -1.0
    tie: float = 0.0


@dataclass(frozen=True)
class SeatReward:
    value: float
    scoreable: bool
    cell: str
    source: Literal["binary", "json", "logit", "binary_fallback"] = "binary"


def _unscoreable(seats: list[str], cell: str) -> dict[str, SeatReward]:
    return {s: SeatReward(0.0, False, cell) for s in seats}


def ladder(verdict: Verdict, mode: RewardMode, values: Optional[LadderValues] = None) -> dict[str, SeatReward]:
    v = values or LadderValues()
    seats = list(verdict.seats)
    if not verdict.ok:
        return _unscoreable(seats, "failed")
    rulings = verdict.seats

    if len(seats) == 1:
        s = seats[0]
        r = rulings[s]
        if r == SeatVerdict.CORRECT:
            return {s: SeatReward(v.solo_win, True, "win")}
        if r == SeatVerdict.INCORRECT:
            return {s: SeatReward(v.solo_loss, True, "loss")}
        if r == SeatVerdict.TIE:
            return {s: SeatReward(v.tie, True, "tie")}
        return _unscoreable(seats, "unknown")

    if any(r == SeatVerdict.TIE for r in rulings.values()):
        return {s: SeatReward(v.tie, True, "tie") for s in seats}
    if any(r == SeatVerdict.UNKNOWN for r in rulings.values()):
        return _unscoreable(seats, "unknown")

    a, b = seats[0], seats[1]
    ra, rb = rulings[a], rulings[b]
    if ra == SeatVerdict.CORRECT and rb == SeatVerdict.CORRECT:
        # Only reachable collaboratively; competitively there is one winner.
        return {s: SeatReward(v.double_win, True, "double_win") for s in seats}
    if ra == SeatVerdict.INCORRECT and rb == SeatVerdict.INCORRECT:
        return {s: SeatReward(v.double_loss, True, "double_loss") for s in seats}
    if ra == SeatVerdict.CORRECT:
        return {a: SeatReward(v.solo_win, True, "a_win"), b: SeatReward(v.solo_loss, True, "a_win")}
    return {a: SeatReward(v.solo_loss, True, "b_win"), b: SeatReward(v.solo_win, True, "b_win")}


# ---------------------------------------------------------------- scoring


class ScoringConfig(BaseModel):
    scoring: Literal["binary", "continuous"] = "binary"
    confidence_source: Literal["json", "logit"] = "json"
    ladder: LadderValues = Field(default_factory=LadderValues)
    shaping: list[dict] = Field(default_factory=list)


def _confidence(verdict: Verdict, seat: str, source: str) -> Optional[float]:
    c = verdict.confidence.get(seat)
    if c is None:
        return None
    if source == "json":
        return c.json if c.json_provenance == "elicited" else None
    return c.logit if c.logit_status == LogitStatus.SCORED else None


def score(verdict: Verdict, mode: RewardMode, cfg: Optional[ScoringConfig] = None) -> dict[str, SeatReward]:
    cfg = cfg or ScoringConfig()
    base = ladder(verdict, mode, cfg.ladder)
    if cfg.scoring == "binary":
        return base

    out: dict[str, SeatReward] = {}
    for seat, reward in base.items():
        ruling = verdict.seats.get(seat)
        directional = reward.scoreable and ruling in (SeatVerdict.CORRECT, SeatVerdict.INCORRECT)
        # 'neither' is INCORRECT/INCORRECT with no named winner and no
        # directional confidence: it keeps the table's double_loss.
        if directional and reward.cell == "double_loss" and mode == "competitive":
            directional = False
        if not directional:
            out[seat] = reward
            continue
        c = _confidence(verdict, seat, cfg.confidence_source)
        if c is None:
            # No cross-source fallback: a missing json confidence never reads
            # the logit channel, or vice versa.
            out[seat] = SeatReward(reward.value, reward.scoreable, reward.cell, "binary_fallback")
            continue
        if mode == "competitive":
            # confidence[seat] is P(this seat correct), so one formula covers
            # winner (2p-1) and loser (1-2p).
            value = 2.0 * c - 1.0
        else:
            value = (2.0 * c - 1.0) * (1.0 if ruling == SeatVerdict.CORRECT else -1.0)
        out[seat] = SeatReward(value, reward.scoreable, reward.cell, cfg.confidence_source)
    return out


def validate_scoring(cfg: ScoringConfig, judge_settings: ModelSettings) -> None:
    """Fail at construction, not mid-run: the logit channel is only P(sampled
    continuation) if the judge samples from the unmodified distribution."""
    if cfg.confidence_source != "logit":
        return
    prof = resolved_sampling_profile(judge_settings, "eval")
    bad: list[str] = []
    if prof.temperature != 1.0:
        bad.append(f"temperature={prof.temperature} (need 1.0)")
    if prof.top_p != 1.0:
        bad.append(f"top_p={prof.top_p} (need 1.0)")
    if prof.top_k not in (None, 0):
        bad.append(f"top_k={prof.top_k}")
    if prof.min_p not in (None, 0.0):
        bad.append(f"min_p={prof.min_p}")
    for name in ("presence_penalty", "frequency_penalty"):
        val = getattr(prof, name)
        if val not in (None, 0.0):
            bad.append(f"{name}={val}")
    if prof.repetition_penalty not in (None, 1.0):
        bad.append(f"repetition_penalty={prof.repetition_penalty}")
    if not judge_settings.capture_token_logprobs:
        bad.append("capture_token_logprobs=False")
    if bad:
        raise ValueError(
            f"confidence_source='logit' requires an untouched sampling distribution on judge "
            f"{judge_settings.alias!r}: " + ", ".join(bad)
        )


# ---------------------------------------------------------------- shaping


@dataclass(frozen=True)
class SlotTokenCounts:
    think: int
    visible: int
    total: int
    cap_total: Optional[int] = None
    # Env-computed per-slot features (e.g. answer_format_valid=1.0 when the
    # solution slot yielded the task family's strict answer candidate); read
    # by flag-gated shaping terms like format_reward.
    flags: dict[str, float] = field(default_factory=dict)


@dataclass
class RoundTokenReport:
    counts: dict[tuple[str, str], SlotTokenCounts] = field(default_factory=dict)


@dataclass
class ShapingDelta:
    per_seat: dict[str, float] = field(default_factory=dict)
    per_slot: dict[tuple[str, str], float] = field(default_factory=dict)


class ShapingTerm(Protocol):
    def apply(self, scored: dict[str, SeatReward], report: RoundTokenReport) -> ShapingDelta: ...


@dataclass
class SlotTerm:
    """One pass over the round's scored slots.

    Every shaping term here is "a coefficient times a per-slot quantity",
    differing only in WHICH quantity and where the delta lands. Subclasses
    supply value(); the loop lives here once, and with it the rule that
    shaping NEVER touches unscoreable data.

    ``kind`` is the config spelling (``{kind: ...}``) and is what construction
    errors name, because that is what a config author actually wrote. ``sign``
    and ``per_slot`` are ClassVars on terms that fix them, so a config cannot
    reach in and flip a penalty into a bonus."""

    coeff: float = 0.0
    slots: Optional[list[str]] = None  # None = every trained-seat slot

    kind: ClassVar[str] = "slot_term"
    per_slot: ClassVar[bool] = True

    def value(self, counts: SlotTokenCounts) -> float:
        raise NotImplementedError

    def apply(self, scored: dict[str, SeatReward], report: RoundTokenReport) -> ShapingDelta:
        delta = ShapingDelta()
        for (seat, slot), c in report.counts.items():
            reward = scored.get(seat)
            if reward is None or not reward.scoreable:  # shaping never touches unscoreable data
                continue
            if self.slots is not None and slot not in self.slots:
                continue
            d = self.value(c)
            if self.per_slot:
                delta.per_slot[(seat, slot)] = delta.per_slot.get((seat, slot), 0.0) + d
            else:
                delta.per_seat[seat] = delta.per_seat.get(seat, 0.0) + d
        return delta


@dataclass
class LengthPenalty(SlotTerm):
    counts: Literal["think", "visible", "total"] = "total"
    per_slot: bool = True  # instance field here: this term alone can broadcast
    normalize: Optional[Literal["cap"]] = None

    kind: ClassVar[str] = "length_penalty"

    def value(self, counts: SlotTokenCounts) -> float:
        n = float(getattr(counts, self.counts))
        if self.normalize == "cap":
            if not counts.cap_total:
                return 0.0
            n /= counts.cap_total
        return -self.coeff * n


@dataclass
class FlagShaping(SlotTerm):
    """A flat coefficient gated on an env-computed per-slot flag.

    Shares FlagTerm with the RLVR arm (infra/envs/shaping.py), so a feature
    priced on both arms cannot pick up a different sign or a different
    coefficient convention on one of them: both state a POSITIVE coeff and let
    the KIND decide whether it is paid or charged."""

    flag: str = "answer_format_valid"

    sign: ClassVar[int] = 1

    def value(self, counts: SlotTokenCounts) -> float:
        # slots are filtered by the loop, so this term matches every slot
        return FlagTerm(self.coeff, self.flag, self.sign).delta(counts.flags)


@dataclass
class FormatReward(FlagShaping):
    """Per-slot bonus gated on an env-computed flag — e.g.
    {kind: format_reward, coeff: 0.1, slots: [proposal],
     flag: answer_format_valid}
    pays +0.1 on the proposal datum iff the task family's strict answer
    format was present."""

    kind: ClassVar[str] = "format_reward"
    sign: ClassVar[int] = 1


@dataclass
class ThinkOvershootPenalty(FlagShaping):
    """Per-slot penalty when the slot's think phase was FORCE-CLOSED at its
    cap — the debate-side spelling of ``dataset.think_overshoot_penalty``.

    A debate's reward is ladder + shaping and never reaches
    ``MathEnv.reward_sample``, so the dataset key is structurally inert on this
    arm at any value; pricing overshoot here is how the two arms are made
    comparable on it. Both spellings resolve to the same FlagTerm over the same
    ``think_overshoot`` predicate (infra/envs/shaping.py). See
    configs/_qwen35_training.yaml."""

    flag: str = "think_overshoot"

    kind: ClassVar[str] = "think_overshoot_penalty"
    sign: ClassVar[int] = -1


SHAPING_REGISTRY = {
    cls.kind: cls for cls in (LengthPenalty, FormatReward, ThinkOvershootPenalty)
}


def build_shaping(cfgs: list[dict]) -> list[ShapingTerm]:
    terms: list[ShapingTerm] = []
    for cfg in cfgs or []:
        spec = dict(cfg)
        kind = spec.pop("kind", None)
        if kind not in SHAPING_REGISTRY:
            raise ValueError(f"unknown shaping kind {kind!r}; known: {sorted(SHAPING_REGISTRY)}")
        terms.append(SHAPING_REGISTRY[kind](**spec))
    return terms


def final_reward(
    seat: str, slot: str, scored: dict[str, SeatReward], deltas: list[ShapingDelta]
) -> float:
    reward = scored[seat]
    total = reward.value
    for d in deltas:
        total += d.per_seat.get(seat, 0.0)
        total += d.per_slot.get((seat, slot), 0.0)
    return total
