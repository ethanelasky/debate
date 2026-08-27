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
from infra.envs.shaping import BudgetTerm, FlagTerm
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
    # Whitespace-delimited words in the slot's VISIBLE speech — the unit
    # Kenton et al. state their limit in, and not derivable from `visible`:
    # LaTeX-dense speech ran ~1.6 tokens/word in a live rollout while prose
    # runs near 1.3, so a token-count proxy prices the same 150 words
    # differently depending on how much math the speech happens to contain.
    # Defaults to 0 so a term that does not read it is unaffected, and so the
    # many test fixtures that build this by keyword keep working.
    visible_words: int = 0
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
    """A per-token price on the whole slot, charged from the first token.

    There is no threshold: every speech pays, and the cheapest speech is the
    empty one. When what you want is a stated limit that is free to reach, the
    term is ``BudgetPenalty`` below — the two are otherwise easy to
    confuse, since both are linear.
    """

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


@dataclass
class SpeechOvershootPenalty(FlagShaping):
    """Per-slot penalty when the slot ran out of budget instead of finishing:
    the sample's stop reason is "length", so the speech reaches the judge cut
    mid-sentence.

    This is the speech-side twin of ThinkOvershootPenalty, and it is priced the
    same way for the same reason: a hard cap alone only truncates, and the
    policy has no gradient telling it to fit. Kenton et al. (2608.17776 §3.4)
    hold the non-solution turns to a word limit "via prompting and reward
    penalties" for this reason — a limit stated in the cue and paid for in the
    reward. The solution turn is exempt there and here: its budget is the RLVR
    arm's, and its overshoot is already priced as think_overshoot."""

    flag: str = "truncated"

    kind: ClassVar[str] = "speech_overshoot_penalty"
    sign: ClassVar[int] = -1


@dataclass
class BudgetPenalty(SlotTerm):
    """A per-slot length budget: free up to ``limit``, charged past it.

    What separates this from ``LengthPenalty`` is the THRESHOLD, not the shape
    — that term is already linear. It prices length itself from the first
    token, so every speech pays and the cheapest speech is the empty one. This
    one prices only the excess past ``limit``, so complying is free and the
    term is silent on slots that fit.

    ``counts`` picks the quantity, and the two uses are different enough to be
    worth naming:

    ``words``  — the Kenton et al. (2608.17776 §3.4) word limit on the reply
        slots, a budget stated in the prompt cue and charged for in the reward.
        Words and not tokens because that is the unit the cue states and the
        one the model can count; LaTeX-dense speech ran ~1.6 tokens/word here
        against ~1.3 for prose, so a token proxy prices the same 150 words
        differently depending on how much math a speech happens to contain.
    ``think`` / ``visible`` / ``total`` — token budgets, for the SOLUTION slot.
        The equivalent knob on the RLVR path is SingleTurnEnv's
        ``soft_token_budget``, which a debate never reaches (DebateEnv extends
        Env with its own rollout), so pricing a proposal's length has to happen
        here or not at all — the same trap ``dataset.think_overshoot_penalty``
        fell into, where the key was accepted and silently inert.

    ``mode`` is the shape, and it is a config toggle because the choice between
    the two is an empirical question this repo has now answered once, the
    expensive way:

    ``proportional`` (default, and what the paper specifies — "a soft additive
        penalty proportional to the excess") charges ``coeff`` per word over.
    ``flat`` charges ``coeff`` once for any overrun, however small.

    Prefer ``proportional`` unless you are deliberately reproducing the flat
    behaviour, because a flat term is largely invisible to a group-relative
    optimizer. It reaches a CISPO advantage only through its variance INSIDE
    the group, so on any group whose samples all land on one side of the
    threshold it is a constant shift that the baseline removes — exactly zero
    signal. Its pressure therefore peaks near a 50% trip rate and switches off
    as the policy approaches either extreme. That is what the flat
    ``think_overshoot_penalty`` did to the RLVR arm on 2026-08-27: it drove
    length down until ``train/think_overshoot`` reached 0.000 at step 32, went
    inert, and left the policy in a short regime with nothing holding it there.
    Held-out correctness fell .699 -> .530 over the next ten steps. Charging
    the excess keeps a per-sample gradient at every trip rate and removes the
    cliff that made sitting far below the limit the safe play.

    The excess is bounded by the slot's own hard token cap, so ``proportional``
    needs no clamp; set ``coeff`` so a speech at that cap costs about what one
    ``flat`` trip would.
    """

    limit: int = 150
    mode: Literal["proportional", "flat"] = "proportional"
    counts: Literal["words", "think", "visible", "total"] = "words"

    kind: ClassVar[str] = "budget_penalty"

    def __post_init__(self) -> None:
        if self.counts not in ("words", "think", "visible", "total"):
            raise ValueError(
                f"{self.kind}: counts must be words/think/visible/total, "
                f"got {self.counts!r}"
            )
        # Construct once here purely to VALIDATE, so a bad mode or limit is a
        # config-time error naming the config's own spelling rather than a
        # mid-rollout one. The shared term owns both validations, so the arms
        # cannot drift on what they accept either.
        try:
            self._term()
        except ValueError as exc:
            raise ValueError(f"{self.kind}: {exc}") from exc

    def _term(self) -> BudgetTerm:
        """Built per call from the live fields rather than cached, matching
        SingleTurnEnv._budget_term: `coeff` is a plain dataclass field and a
        cached term would go on pricing a coefficient the config no longer
        says."""
        return BudgetTerm(coeff=self.coeff, limit=self.limit, mode=self.mode, sign=-1)

    def value(self, counts: SlotTokenCounts) -> float:
        n = counts.visible_words if self.counts == "words" else getattr(counts, self.counts)
        # SlotTerm.apply has already filtered by slot, so this prices
        # unconditionally and leaves slot matching to the one caller that owns it.
        return self._term().delta(n)


SHAPING_REGISTRY = {
    cls.kind: cls
    for cls in (
        LengthPenalty,
        FormatReward,
        ThinkOvershootPenalty,
        SpeechOvershootPenalty,
        BudgetPenalty,
    )
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
