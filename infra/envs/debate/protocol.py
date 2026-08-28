"""Turn protocol: the declarative debate structure.

Five rules, restated as code:
  1. A protocol is ordered turns; each turn maps speakers to ordered slots.
  2. Visibility classes: public (released to others at the next turn
     boundary), private (author-only, forever), ephemeral (author-only, and
     only for the rest of the same turn).
  3. Context for a slot = all public slots from strictly-earlier turns +
     the author's own public/private slots from any earlier position +
     the author's own ephemeral slots from earlier in the SAME turn.
  4. Rendering is per-speaker (round.py's job, not this module's).
  5. Nothing else — blindness is readable off the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Kind(Enum):
    SPEECH = "speech"        # free text
    SOLUTION = "solution"    # answer extracted + graded; binds <POSITION> in fresh mode
    DECISION = "decision"    # verdict extracted + compared to solutions


class Visibility(Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    EPHEMERAL = "ephemeral"


@dataclass(frozen=True)
class Slot:
    name: str
    kind: Kind = Kind.SPEECH
    visibility: Visibility = Visibility.PUBLIC
    max_think_tokens: Optional[int] = None
    max_visible_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None
    # Per-slot chat-template thinking, overriding the seat's own
    # enable_thinking. A seat speaks in several slots and does not need the
    # same reasoning budget in each: the opening speech can think while the
    # reply is a short no-think move. None leaves the seat setting alone.
    enable_thinking: Optional[bool] = None

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "Slot":
        if "name" not in raw:
            raise ValueError(f"slot missing 'name': {raw}")
        known = {
            "name",
            "kind",
            "visibility",
            "max_think_tokens",
            "max_visible_tokens",
            "max_total_tokens",
            "enable_thinking",
        }
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"slot {raw['name']!r}: unknown key(s) {sorted(unknown)}")
        enable_thinking = raw.get("enable_thinking")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise ValueError(
                f"slot {raw['name']!r}: enable_thinking must be a bool, got {enable_thinking!r}"
            )
        slot = cls(
            name=str(raw["name"]),
            kind=Kind(raw.get("kind", "speech")),
            visibility=Visibility(raw.get("visibility", "public")),
            max_think_tokens=raw.get("max_think_tokens"),
            max_visible_tokens=raw.get("max_visible_tokens"),
            max_total_tokens=raw.get("max_total_tokens"),
            enable_thinking=enable_thinking,
        )
        for attr in ("max_think_tokens", "max_visible_tokens", "max_total_tokens"):
            v = getattr(slot, attr)
            if v is not None and v <= 0:
                raise ValueError(f"slot {slot.name!r}: {attr} must be positive")
        if (
            slot.max_total_tokens is not None
            and slot.max_think_tokens is not None
            and slot.max_think_tokens > slot.max_total_tokens
        ):
            raise ValueError(f"slot {slot.name!r}: max_think_tokens > max_total_tokens")
        return slot


@dataclass(frozen=True)
class CompiledSlot:
    """One generation step: position in the flat rollout order."""

    index: int   # global generation order
    turn: int
    speaker: str
    seq: int     # position within this speaker's slot list this turn
    slot: Slot


@dataclass
class Protocol:
    turns: list[dict[str, list[Slot]]] = field(default_factory=list)

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "Protocol":
        turns_raw = raw.get("turns")
        if turns_raw is None:
            raise ValueError("protocol missing 'turns'")
        turns: list[dict[str, list[Slot]]] = []
        for t, turn_raw in enumerate(turns_raw or []):
            if not isinstance(turn_raw, dict) or not turn_raw:
                raise ValueError(f"turn {t} must be a non-empty mapping of speaker -> slots")
            turn: dict[str, list[Slot]] = {}
            for speaker, slots_raw in turn_raw.items():
                if not isinstance(slots_raw, list) or not slots_raw:
                    raise ValueError(f"turn {t} speaker {speaker!r}: slots must be a non-empty list")
                # bare-string shorthand: `- speech` == `{name: speech}`
                turn[str(speaker)] = [Slot.parse({"name": s} if isinstance(s, str) else s) for s in slots_raw]
            turns.append(turn)
        proto = cls(turns=turns)
        proto.validate()
        return proto

    def validate(self) -> None:
        slots = self.compile()
        for i, cs in enumerate(slots):
            if cs.index != i:
                raise AssertionError("compile ordering broken")

        decisions = [cs for cs in slots if cs.slot.kind == Kind.DECISION]
        if len(decisions) > 1:
            raise ValueError("at most one decision slot per protocol")
        if decisions:
            d = decisions[0]
            if d.slot.visibility != Visibility.PUBLIC:
                raise ValueError(f"decision slot {d.slot.name!r} must be public (reward reads it)")
            if d.index != slots[-1].index:
                raise ValueError("the decision slot must be the last slot in flat order")

        # One solution slot per speaker, and it must precede that speaker's
        # first PUBLIC slot (fresh-answer binding must exist before anything
        # others can react to). Private slots (e.g. a planning scratchpad) may
        # come first — prompt bindability validation guards placeholder timing.
        first_public: dict[str, int] = {}
        solutions_by_speaker: dict[str, list[CompiledSlot]] = {}
        for cs in slots:
            if cs.slot.visibility == Visibility.PUBLIC:
                first_public.setdefault(cs.speaker, cs.index)
            if cs.slot.kind == Kind.SOLUTION:
                solutions_by_speaker.setdefault(cs.speaker, []).append(cs)
        for speaker, sols in solutions_by_speaker.items():
            if len(sols) > 1:
                raise ValueError(f"speaker {speaker!r} has multiple solution slots")
            if sols[0].index > first_public.get(speaker, sols[0].index):
                raise ValueError(
                    f"speaker {speaker!r}: solution slot must precede its first public slot"
                )

    @property
    def decision_slot(self) -> Optional[CompiledSlot]:
        for cs in self.compile():
            if cs.slot.kind == Kind.DECISION:
                return cs
        return None

    @property
    def solution_slots(self) -> dict[str, CompiledSlot]:
        return {cs.speaker: cs for cs in self.compile() if cs.slot.kind == Kind.SOLUTION}

    @property
    def speakers(self) -> list[str]:
        seen: dict[str, None] = {}
        for turn in self.turns:
            for s in turn:
                seen.setdefault(s, None)
        return list(seen)

    def compile(self) -> list[CompiledSlot]:
        """Flat generation order: turn-major, speaker insertion order, then the
        speaker's slot order. Cross-speaker order within a turn is semantically
        irrelevant (rule 3 makes co-occupants blind) but kept deterministic."""
        out: list[CompiledSlot] = []
        for t, turn in enumerate(self.turns):
            for speaker, slots in turn.items():
                for k, slot in enumerate(slots):
                    out.append(CompiledSlot(index=len(out), turn=t, speaker=speaker, seq=k, slot=slot))
        return out


def visible_to(prior: CompiledSlot, current: CompiledSlot) -> bool:
    """Rule 3: is `prior`'s content in `current`'s generation context?"""
    if prior.index >= current.index:
        return False
    if prior.speaker == current.speaker:
        if prior.slot.visibility == Visibility.EPHEMERAL:
            return prior.turn == current.turn
        return True  # own public/private: any earlier position
    return prior.slot.visibility == Visibility.PUBLIC and prior.turn < current.turn
