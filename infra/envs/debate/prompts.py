"""Prompt library: YAML entries -> rendered system / preamble / cue text.

Loading reuses debate/config.py's _includes/_extends resolver unchanged — a
prompt file is just a config file whose entries happen to be prompt sets. This
module adds a typed view, `<PLACEHOLDER>` substitution, the compile-time
prompt/protocol contract, and the adapter implementing round.py's
PromptLibrary Protocol.

SCHEMA — an entry names the parts of each seat's message sequence directly
(design 2026-08-04). Roles come from the PROTOCOL, not from the pack: the
speaker of the first solution slot is the proposer, the decision slot's
speaker is the judge, and a remaining debater is the critic —
load_prompt_library therefore requires the protocol.

    <entry>:
      _extends: <parent>            # optional
      vars: {KEY: value}            # static substitutions, merged over parent
      overall_system:               # every speaker's system card, first
      debater_system:               # both debaters, after overall_system
      debater_system_proposer:      # proposer only, after debater_system
      debater_system_critic:        # critic only, after debater_system
      judge_system:                 # judge only, after overall_system
      pre_debate:                   # SHARED leading user message (see below)
      pre_debate_proposer:          # proposer's individual leading user message
      pre_debate_critic:            # critic's individual leading user message
      pre_debate_judge:             # judge's individual leading user message
      blind_system:                 # SOLO blind view's system card (choice mode)
      attribution:                  # optional, see below
        <reader>: {<author>: <template>}
      <protocol slot name>: ...     # that slot's per-turn cue

Every seat's rendered view is:

    [system card]                    overall_system + role card, ONE message
    [pre_debate]                     shared user message (if defined)
    [pre_debate_<role>]              individual user message (if defined)
    ...transcript...                 others' speeches (via attribution) and the
                                     slot cues buffer into user messages that
                                     strictly alternate with the seat's own
                                     assistant turns, exactly as before

The system card composes as overall_system + debater_system + role card (one
system message, blocks joined by a blank line), or overall_system +
judge_system for the judge — the old repo's consecutive-SYSTEM collapse.

pre_debate / pre_debate_<role> render as SEPARATE user messages, in that
order, right after the system card. Separate messages are deliberate cache
design: providers that cache at message boundaries (DashScope) can only reuse
a prefix of whole messages, so the byte-stable shared content (the problem /
trajectory) must END a message, with seat-varying content starting a new one.
Consecutive user messages at the context head are legal (frozen API seats;
the strict user/assistant alternation applies from the first
transcript-derived message on).

PER-TURN CUES are keyed by the PROTOCOL SLOT NAME itself — any entry key that
is not a known stage is a cue, and the loader checks the cue inventory
against the protocol both ways: a compiled slot without a cue and a cue key
matching no compiled slot are both errors (dead prompt text is exactly how
the old repo's pre_debate went dark for months).

The PROPOSER's opening cue is the task family's answer-generation user
message spliced in whole (<ANSWER_GEN_USER>), and that rendered message must
stay byte-identical to the RLVR arm's (tests pin it).

SOLO FIRST SPEECH (first_speech_non_debate_aware / choice mode): the first
compiled slot is GENERATED under the task source's own messages verbatim (no
debate framing). The author's LATER views render normally — debate framing
(system + preamble messages), then slot 0 like any other own slot: its cue as
user content, its answer as the first assistant turn — so the answer is
presented as if it had been produced under the debate framing (deliberate,
2026-08-04). Packs keep that honest by making the slot-0 cue byte-identical
to the solo eliciting message (MB's blind instructions; the math packs'
<ANSWER_GEN_USER>), which single-sourcing gives by construction.
pre_debate_proposer typically carries just the author's debate identity
("You are <NAME>, a debater…"): the identity the solo view deliberately
withheld.

`attribution` is how a READER speaker sees an AUTHOR speaker's public
speeches. The template is rendered with the READER's bindings and the speech
text follows after a blank line. Pairs without a template fall back to the
hard-coded default "X said:\n<text>".

Every stage/cue may be a single string OR a LIST of blocks joined with a
blank line. Blocks compose under _extends: the config resolver merges lists
BY INDEX, so a child can override one block of a parent's stage (or append
blocks past the end) without restating the rest — the old repo's
adjacent-block concatenation, kept.

No Jinja: substitution is a plain replace of `<KEY>`. The placeholder pattern
is UPPERCASE-only, so literal lowercase tags (`<problem>`) pass through.
To see where every stage lands in the final message sequences, use
`preview(lib, protocol)`.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from infra.config import load_config_with_includes, resolve_all_experiments
from infra.envs.debate.protocol import CompiledSlot, Kind, Protocol, Visibility

PLACEHOLDER = re.compile(r"<([A-Z_][A-Z0-9_]*)>")

_DEFERRED = ("POSITION", "OPPONENT_POSITION")  # fresh mode: bound by extraction

PROPOSER, CRITIC, JUDGE = "proposer", "critic", "judge"

#: System-card stages, in composition order per role. The old repo's
#: _pre_debate_block (debate/speech_format.py) + its consecutive-SYSTEM collapse.
_SYSTEM_STAGES: dict[str, tuple[str, ...]] = {
    PROPOSER: ("overall_system", "debater_system", "debater_system_proposer"),
    CRITIC: ("overall_system", "debater_system", "debater_system_critic"),
    JUDGE: ("overall_system", "judge_system"),
}
#: Individual leading-user-message stage per role, rendered after the shared
#: pre_debate message.
_PRE_DEBATE_STAGES: dict[str, str] = {
    PROPOSER: "pre_debate_proposer",
    CRITIC: "pre_debate_critic",
    JUDGE: "pre_debate_judge",
}
_STAGE_KEYS = frozenset(
    s for stages in _SYSTEM_STAGES.values() for s in stages
) | frozenset(_PRE_DEBATE_STAGES.values()) | {"pre_debate", "blind_system"}
_ENTRY_KEYS = frozenset({"vars", "attribution"}) | _STAGE_KEYS


@dataclass
class PromptLibrary:
    system: dict[str, str] = field(default_factory=dict)
    #: speaker -> ordered leading user messages (shared pre_debate first, then
    #: the role's pre_debate_<role>); [] = none.
    preamble: dict[str, list[str]] = field(default_factory=dict)
    #: protocol slot name -> per-turn cue template.
    slots: dict[str, str] = field(default_factory=dict)
    vars: dict[str, str] = field(default_factory=dict)
    #: reader -> author -> template replacing the "X said:" default.
    attribution: dict[str, dict[str, str]] = field(default_factory=dict)
    #: the raw shared pre_debate template (also the trajectory message of a
    #: solo first speech built by a task source — run_eval.blind_message_templates).
    shared_pre_debate: str = ""
    #: system card of the SOLO blind view only (choice mode): the RLVR-style
    #: role card ("you're monitoring...") the blind generation runs under. No
    #: debate seat ever renders it.
    blind_system: str = ""


def speaker_roles(protocol: Protocol) -> dict[str, str]:
    """Which seat plays which role. Derived from the PROTOCOL so a pack never
    has to name seats: the decision slot's speaker judges, the first solution
    slot's speaker proposes, any other debater critiques."""
    decision = protocol.decision_slot
    judge = decision.speaker if decision is not None else None
    proposer = next(
        (cs.speaker for cs in protocol.compile() if cs.slot.kind == Kind.SOLUTION and cs.speaker != judge),
        None,
    )
    roles: dict[str, str] = {}
    for s in protocol.speakers:
        roles[s] = JUDGE if s == judge else (PROPOSER if s == proposer else CRITIC)
    return roles


def load_prompt_library(
    path: str | Path, entry: str, protocol: Optional[Protocol] = None
) -> PromptLibrary:
    """File + entry name -> resolved library (_extends/_includes applied), with
    the role stages composed into per-speaker system cards and preamble
    messages, and every non-stage key checked against the protocol's slot
    inventory as that slot's cue.

    `protocol` decides which seat is proposer/critic/judge AND which keys are
    valid cues, so it is REQUIRED."""
    data = resolve_all_experiments(load_config_with_includes(path))
    cfg = data.get(entry)
    if not isinstance(cfg, dict):
        available = ", ".join(n for n in data if not n.startswith("_")) or "<none>"
        raise KeyError(f"prompt entry {entry!r} not in {path} (available: {available})")
    if protocol is None:
        raise ValueError(
            f"prompt entry {entry!r}: load_prompt_library needs the protocol — roles and "
            "the cue inventory both come from it (proposer/critic/judge from the solution "
            "and decision slots; cue keys are protocol slot names)"
        )

    slot_names = {cs.slot.name for cs in protocol.compile()}
    cues = {str(k): _join(v) for k, v in cfg.items() if k not in _ENTRY_KEYS}
    unused = set(cues) - slot_names
    if unused:
        # A pack serves a FAMILY of protocols, so a cue for a slot this
        # protocol does not compile is legal (e.g. an optional scratchpad
        # turn) — but it is also exactly what a typo'd stage name looks like,
        # and silently-dead prompt text is how the old repo's pre_debate went
        # dark for months. Warn, loudly, every load.
        warnings.warn(
            f"prompt entry {entry!r}: cue key(s) {sorted(unused)} match no slot of this "
            f"protocol (slots: {sorted(slot_names)}) and will not render. Fine if they "
            "serve another protocol variant; a typo'd stage name looks exactly like "
            f"this (stages: {sorted(_STAGE_KEYS)}).",
            stacklevel=2,
        )

    stages = {k: _join(cfg[k]) for k in _STAGE_KEYS if k in cfg}
    attribution: dict[str, dict[str, str]] = {}
    for reader, authors in (cfg.get("attribution") or {}).items():
        if not isinstance(authors, dict):
            raise ValueError(
                f"prompt entry {entry!r}: attribution[{reader!r}] must be a "
                f"{{author: template}} map, got {type(authors).__name__}"
            )
        attribution[str(reader)] = {str(a): _join(t) for a, t in authors.items()}

    roles = speaker_roles(protocol)
    shared = stages.get("pre_debate", "").strip()
    preamble: dict[str, list[str]] = {}
    for s, r in roles.items():
        msgs = [m for m in (shared, stages.get(_PRE_DEBATE_STAGES[r], "").strip()) if m]
        preamble[s] = msgs
    return PromptLibrary(
        system={s: _compose(stages, _SYSTEM_STAGES[r]) for s, r in roles.items()},
        preamble=preamble,
        slots=cues,
        vars={str(k): str(v) for k, v in (cfg.get("vars") or {}).items()},
        attribution=attribution,
        shared_pre_debate=shared,
        blind_system=stages.get("blind_system", "").strip(),
    )


def _compose(stages: dict[str, str], names: tuple[str, ...]) -> str:
    """Concatenate the stages that exist, blank-line separated. A stage the pack
    omits is simply skipped, exactly as the old repo skipped a missing tag."""
    return "\n\n".join(t for n in names if (t := stages.get(n, "").strip()))


def _join(template) -> str:
    """A template is a string or a list of blocks joined with a blank line.
    Joining happens AFTER _extends resolution, so children override blocks by
    index or append past the parent's end."""
    if isinstance(template, list):
        return "\n\n".join(str(b).strip() for b in template if str(b).strip())
    return str(template)


def slot_template(lib: PromptLibrary, slot_name: str, speaker: str) -> str:
    """Cue template for a slot. `speaker` is accepted for call-site symmetry;
    cues are per-slot, and per-seat differences come from distinct protocol
    slot names."""
    if slot_name not in lib.slots:
        raise KeyError(f"no cue for slot {slot_name!r} (have: {sorted(lib.slots)})")
    return lib.slots[slot_name]


def render(template: str, bindings: dict[str, str], vars: Optional[dict[str, str]] = None) -> str:
    """Substitute `<KEY>`. `vars` apply first and `bindings` override them (one
    pass over the merged mapping, so substituted text is never re-scanned).
    Lowercase tags pass through; any UPPERCASE placeholder with no binding is a
    hard error rather than a silently half-rendered prompt."""
    subs = {**(vars or {}), **bindings}
    used = {m.group(1) for m in PLACEHOLDER.finditer(template)}
    missing = sorted(used - set(subs))
    if missing:
        raise ValueError(f"unbound placeholder(s) {missing} in template: {template[:120]!r}")
    return PLACEHOLDER.sub(lambda m: str(subs[m.group(1)]), template).strip()


# ------------------------------------------------- compile-time validation


def _entry_templates(
    lib: PromptLibrary, cs: CompiledSlot, first_slot_verbatim: bool = False
) -> Iterator[tuple[str, str]]:
    # Verbatim-first-slot modes (first_speech_non_debate_aware / choice): the
    # speaker's system/preamble never render in slot 0's context, but the slot
    # CUE is still checked — the task source renders it (with task bindings
    # only) into the solo messages, so a deferred placeholder there is still an
    # error.
    solo_slot0 = first_slot_verbatim and cs.index == 0
    if cs.speaker in lib.system and not solo_slot0:
        yield f"system[{cs.speaker}]", lib.system[cs.speaker]
    if not solo_slot0:
        for i, tmpl in enumerate(lib.preamble.get(cs.speaker, [])):
            yield f"preamble[{i}][{cs.speaker}]", tmpl
    try:
        yield f"slot {cs.slot.name}", slot_template(lib, cs.slot.name, cs.speaker)
    except KeyError:
        pass  # reported separately


def _attribution_templates(
    lib: PromptLibrary, slots: list[CompiledSlot]
) -> Iterator[tuple[CompiledSlot, str, str]]:
    """(slot, label, template) for attribution templates at the EARLIEST slot
    each can render: the reader's first slot that can see another speaker's
    earlier public slot. Pairs the protocol never exercises yield nothing."""
    for reader, authors in lib.attribution.items():
        first_reading = next(
            (
                cs
                for cs in slots
                if cs.speaker == reader
                and any(
                    o.index < cs.index
                    and o.speaker != reader
                    and o.slot.visibility == Visibility.PUBLIC
                    and o.turn < cs.turn
                    for o in slots
                )
            ),
            None,
        )
        if first_reading is None:
            continue
        for author, tmpl in authors.items():
            yield first_reading, f"attribution[{reader}][{author}]", tmpl


def _bindability_errors(
    lib: PromptLibrary, slots: list[CompiledSlot], choice_positions: bool = False
) -> list[str]:
    """Fresh mode: <POSITION>/<OPPONENT_POSITION> are bound by extraction from a
    solution slot, so they may not appear in anything rendered at or before the
    solution slot that binds them. round.py binds a solver's own POSITION and
    every other speaker's OPPONENT_POSITION the moment its solution lands; a
    speaker with no solution slot of its own (the judge) reads POSITION once any
    solution has landed. Static lib.vars count as bound. Attribution templates
    are checked at the earliest slot they can render.

    Choice mode (position_binder + first_speech_non_debate_aware): the single
    solution slot binds BOTH deferred names for EVERY speaker the moment it
    lands — so both are usable at any strictly-later slot, and neither at the
    solution slot itself. Slot 0 renders the task messages verbatim, so the
    speaker's system/preamble are exempt there (the slot cue is not: the task
    source renders it into those messages). The solo AUTHOR's system/preamble
    first render at its next slot after 0, by which point the choice has
    landed."""
    sols = [cs for cs in slots if cs.slot.kind == Kind.SOLUTION]
    own = {cs.speaker: cs.index for cs in sols}
    first = min((cs.index for cs in sols), default=None)
    errors: list[str] = []
    by_slot: dict[int, list[tuple[str, str]]] = {}
    for cs, label, tmpl in _attribution_templates(lib, slots):
        by_slot.setdefault(cs.index, []).append((label, tmpl))
    for cs in slots:
        if choice_positions:
            pos = opp = first
        else:
            opp = min((c.index for c in sols if c.speaker != cs.speaker), default=None)
            pos = own.get(cs.speaker, first)
        unbound = {
            name
            for name, at in zip(_DEFERRED, (pos, opp))
            if at is None or at >= cs.index
        } - set(lib.vars)
        entry_templates = list(
            _entry_templates(lib, cs, first_slot_verbatim=choice_positions)
        )
        for label, tmpl in entry_templates + by_slot.get(cs.index, []):
            hit = sorted({m.group(1) for m in PLACEHOLDER.finditer(tmpl)} & unbound)
            if hit:
                errors.append(
                    f"{label}: {', '.join(hit)} not bound yet at slot {cs.index} "
                    f"({cs.speaker}/{cs.slot.name}) in "
                    + ("choice_positions" if choice_positions else "fresh_positions")
                    + " mode"
                )
    return list(dict.fromkeys(errors))


def validate_prompts(
    lib: PromptLibrary,
    protocol: Protocol,
    fresh_positions: bool = False,
    choice_positions: bool = False,
) -> None:
    """The prompt/protocol contract, checked before any generation: every
    compiled slot has a cue, every speaker has a system prompt, attribution
    names real speakers, and (fresh/choice mode) no template needs a position
    that has not been generated yet. Raises ValueError listing every problem
    at once."""
    slots = protocol.compile()
    errors: list[str] = []
    speakers = set(protocol.speakers)
    roles = speaker_roles(protocol)
    for speaker in protocol.speakers:
        if speaker not in lib.system:
            errors.append(f"no system prompt for speaker {speaker!r} (have: {sorted(lib.system)})")
        elif not lib.system[speaker].strip():
            # Composed from stages, so an empty card means the entry defines
            # none of them for this role — silently unprompted otherwise.
            errors.append(
                f"empty system prompt for speaker {speaker!r} (role {roles.get(speaker)!r}): the "
                f"entry defines none of {list(_SYSTEM_STAGES.get(roles.get(speaker, ''), ()))}"
            )
    for cs in slots:
        try:
            slot_template(lib, cs.slot.name, cs.speaker)
        except KeyError as e:
            errors.append(str(e.args[0]))
    for reader, authors in lib.attribution.items():
        if reader not in speakers:
            errors.append(f"attribution reader {reader!r} not in protocol (have: {sorted(speakers)})")
        for author in authors:
            if author not in speakers:
                errors.append(
                    f"attribution[{reader!r}] author {author!r} not in protocol (have: {sorted(speakers)})"
                )
            elif author == reader:
                errors.append(
                    f"attribution[{reader!r}][{author!r}]: a speaker never reads its own "
                    "speeches via attribution (own speeches render as assistant turns)"
                )
    if fresh_positions and choice_positions:
        errors.append("fresh_positions and choice_positions are mutually exclusive")
    elif fresh_positions or choice_positions:
        errors += _bindability_errors(lib, slots, choice_positions=choice_positions)
    if errors:
        raise ValueError("prompt/protocol mismatch:\n  " + "\n  ".join(dict.fromkeys(errors)))


# ------------------------------------------------------------- round adapter


class RenderedPrompts:
    """Implements round.PromptLibrary over a loaded PromptLibrary."""

    def __init__(self, lib: PromptLibrary):
        self.lib = lib

    def system(self, speaker: str, bindings: dict[str, str]) -> str:
        if speaker not in self.lib.system:
            raise KeyError(f"no system prompt for {speaker!r} (have: {sorted(self.lib.system)})")
        return render(self.lib.system[speaker], bindings, self.lib.vars)

    def instruction(self, slot_name: str, speaker: str, bindings: dict[str, str]) -> str:
        return render(slot_template(self.lib, slot_name, speaker), bindings, self.lib.vars)

    def preamble_messages(self, speaker: str, bindings: dict[str, str]) -> list[str]:
        """The speaker's leading user messages (shared pre_debate, then its
        role's pre_debate_<role>), each rendered as its own message."""
        return [
            render(tmpl, bindings, self.lib.vars)
            for tmpl in self.lib.preamble.get(speaker, [])
        ]

    def attributed(
        self,
        author_name: str,
        slot_name: str,
        text: str,
        *,
        reader: Optional[str] = None,
        author: Optional[str] = None,
        reader_bindings: Optional[dict[str, str]] = None,
    ) -> str:
        """How `reader` sees `author`'s speech. A configured (reader, author)
        attribution template is rendered with the READER's bindings and the
        speech text follows after a blank line; without one (or without
        reader/author identity, e.g. legacy positional calls) the hard-coded
        default applies."""
        tmpl = None
        if reader is not None and author is not None:
            tmpl = self.lib.attribution.get(reader, {}).get(author)
        if tmpl is None:
            return f"{author_name} said:\n{text}"
        return render(tmpl, reader_bindings or {}, self.lib.vars) + "\n\n" + text


def load_rendered_prompts(
    path: str | Path, entry: str, protocol: Optional[Protocol] = None
) -> RenderedPrompts:
    return RenderedPrompts(load_prompt_library(path, entry, protocol))


# ------------------------------------------------------------------ preview


def preview(lib: PromptLibrary, protocol: Protocol, speakers: Optional[list[str]] = None) -> str:
    """Render each speaker's final-slot message sequence with placeholders kept
    visible (<TOPIC> stays <TOPIC>) and other speakers' outputs stubbed as
    <alice/proposal output>. Answers "where does my prompt text land" without
    running anything: python -m debate.envs.debate.prompts <file> <entry> <proto.yaml>
    """
    from infra.envs.debate.round import DebateState, SlotRecord, render_context

    slots = protocol.compile()
    placeholder_names = {
        m.group(1)
        for tmpl in list(lib.system.values())
        + [t for msgs in lib.preamble.values() for t in msgs]
        + list(lib.slots.values())
        + [t for authors in lib.attribution.values() for t in authors.values()]
        for m in PLACEHOLDER.finditer(tmpl)
    }
    placeholder_names -= set(lib.vars)  # var-bound placeholders render for real
    identity = {name: f"<{name}>" for name in placeholder_names}  # single-pass sub: safe
    bindings = {s: {**identity, "NAME": f"<{s}:NAME>"} for s in protocol.speakers}

    out: list[str] = []
    for speaker in speakers or protocol.speakers:
        own = [cs for cs in slots if cs.speaker == speaker]
        if not own:
            continue
        final = own[-1]
        state = DebateState(bindings=bindings)
        state.records = [
            SlotRecord(slot=cs, text=f"<{cs.speaker}/{cs.slot.name} output>")
            for cs in slots
            if cs.index < final.index
        ]
        out.append(f"{'=' * 70}\n== {speaker} — context for its final slot ({final.slot.name})\n{'=' * 70}")
        for m in render_context(state, final, RenderedPrompts(lib)):
            out.append(f"--- [{m['role']}] " + "-" * (56 - len(m["role"])))
            out.append(m["content"])
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    import yaml as _yaml

    from infra.envs.debate.protocol import Protocol as _T

    proto_ = _T.parse(_yaml.safe_load(open(sys.argv[3])))
    lib_ = load_prompt_library(sys.argv[1], sys.argv[2], proto_)
    print(preview(lib_, proto_))
