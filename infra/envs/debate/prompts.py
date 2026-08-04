"""Prompt library: YAML entries -> rendered system / preamble / instruction text.

Loading reuses debate/config.py's _includes/_extends resolver unchanged — a
prompt file is just a config file whose entries happen to be prompt sets. This
module adds a typed view, `<PLACEHOLDER>` substitution, the compile-time
prompt/protocol contract, and the adapter implementing round.py's
PromptLibrary Protocol.

TWO ENTRY SCHEMAS, dispatched per entry by its keys:

STAGE SCHEMA (`slot_stages` present) — the old repo's stage vocabulary
(prompts/parser.py PromptTag there), not per-speaker blobs. An entry names
stages; `slot_stages` maps this repo's protocol slot names onto the cue
stages, so protocols stay declarative and the old-name/new-name correspondence
is per-pack DATA rather than renderer magic:

    <entry>:
      _extends: <parent>                 # optional
      vars: {KEY: value}                 # static substitutions, merged over parent
      overall_system:                    # every speaker's system card, first
      debater_system:                    # both debaters, after overall_system
      debater_system_proposer:           # proposer only, after debater_system
      debater_system_critic:             # critic only, after debater_system
      judge_system:                      # judge only, after overall_system
      pre_debate:                        # critic's preamble (see below)
      pre_debate_judge:                  # judge's preamble
      debater_standard_grading_details:  # tail of the critic's preamble
      judge_standard_grading_details:    # tail of the judge's preamble
      <cue stages>: ...                  # pre_opening_speech_proposer, ...
      slot_stages: {<protocol slot>: <cue stage>}

Composition, matching the old repo's speech_format.py + its SYSTEM-collapse:
  system card = overall_system + debater_system + role card  (ONE system
                message, blocks joined by a blank line), or
                overall_system + judge_system for the judge.
  stage_preamble = pre_debate + debater_standard_grading_details  (critic), or
                pre_debate_judge + judge_standard_grading_details (judge).
                Prefixed onto the FIRST user content of that speaker's context.

The PROPOSER gets NO preamble, deliberately: its opening cue is the task
family's answer-generation user message spliced in whole, and that rendered
message must stay byte-identical to the RLVR arm's (tests pin it). Its grading
framing lives in answer_gen_system instead.

Roles come from the PROTOCOL, not from the pack: the speaker of the first
solution slot is the proposer, the decision slot's speaker is the judge, and a
remaining debater is the critic. load_prompt_library therefore needs the
protocol whenever an entry uses the stage schema.

DIRECT SCHEMA (`slots` present, no `slot_stages`) — per-speaker templates
addressed by protocol slot name (see prompt_configs/monitoringbench.yaml):

    <entry>:
      _extends: <parent>        # optional
      vars: {KEY: value}        # static substitutions, merged over parent
      system: {<speaker>: <template>}
      preamble: [<item>, ...]                           # optional, see below
      attribution: {<reader>: {<author>: <template>}}   # optional, see below
      slots:
        <name>: <template>                # string = any speaker
        <name>: {<speaker>: <template>}   # map = per-speaker (e.g. closing)

Two optional, additive direct-schema keys:
- `preamble`: ordered list of items; each item renders as its OWN user message
  at the head of every context of its speaker — after the system message,
  before any transcript-derived content. An item is a single template string
  (every speaker gets that message) or a {<speaker>: <template>} map (listed
  speakers only; others skip that message). Rendered with the reader speaker's
  bindings. SEPARATE messages are deliberate cache design (2026-08-03):
  providers that cache at message boundaries (DashScope) can only reuse a
  prefix of whole messages, so the byte-stable task content (the trajectory)
  must END a message, with seat-varying content (positions) starting a new
  one. Consecutive user messages are legal (frozen API seats; the strict
  user/assistant alternation applies from the first transcript-derived message
  on). This is DISTINCT from the stage schema's prefix preamble, which joins
  the first user message precisely so the alternation stays untouched.
- `attribution`: how a READER speaker sees an AUTHOR speaker's public
  speeches. The template is rendered with the READER's bindings and the
  speech text follows after a blank line. Pairs without a template fall back
  to the hard-coded default "X said:\n<text>".

Choice mode (mb_debate_choice, 2026-08-03) has no schema of its own here:
the blind first slot renders the task source's own messages verbatim
(round.py first_slot_messages / env first_speech_non_debate_aware), and the
task source builds those messages from the shared trajectory preamble item
plus the blind slot's SLOT TEMPLATE. The slot-0 author never renders preamble
messages (its solo exchange replays instead). Only the bindability check
knows about it — see validate_prompts(choice_positions).

Every stage/template may be a single string OR a LIST of blocks joined with a
blank line. Blocks compose under _extends: the config resolver merges lists BY
INDEX, so a child can override one block of a parent's stage (or append blocks
past the end) without restating the rest — the old repo's adjacent-block
concatenation, kept.

No Jinja: substitution is a plain replace of `<KEY>`. The placeholder pattern
is UPPERCASE-only, so literal lowercase tags (`<problem>`) pass through.
To see where every stage lands in the final message sequences, use
`preview(lib, protocol)`.
"""

from __future__ import annotations

import re
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
#: Preamble stages, in order. Prefixed onto the speaker's FIRST user content.
#: The proposer is absent BY DESIGN — see the module docstring.
_PREAMBLE_STAGES: dict[str, tuple[str, ...]] = {
    CRITIC: ("pre_debate", "debater_standard_grading_details"),
    JUDGE: ("pre_debate_judge", "judge_standard_grading_details"),
}
_ROLE_STAGES = frozenset(
    s for stages in (*_SYSTEM_STAGES.values(), *_PREAMBLE_STAGES.values()) for s in stages
)
_STAGE_ENTRY_KEYS = frozenset({"vars", "slot_stages"}) | _ROLE_STAGES
_DIRECT_ENTRY_KEYS = frozenset({"vars", "system", "slots", "preamble", "attribution"})


@dataclass
class PromptLibrary:
    system: dict[str, str] = field(default_factory=dict)
    slots: dict[str, str | dict[str, str]] = field(default_factory=dict)
    vars: dict[str, str] = field(default_factory=dict)
    #: STAGE schema: speaker -> text prefixed onto its first user message
    #: ("" / absent = none).
    stage_preamble: dict[str, str] = field(default_factory=dict)
    #: DIRECT schema: ordered preamble items; each renders as its own leading
    #: user message. str item = every speaker; map item = listed speakers only.
    preamble: list[str | dict[str, str]] = field(default_factory=list)
    #: DIRECT schema: reader -> author -> template replacing "X said:".
    attribution: dict[str, dict[str, str]] = field(default_factory=dict)


def speaker_roles(protocol: Protocol) -> dict[str, str]:
    """Which seat plays which old-repo role. Derived from the PROTOCOL so a pack
    never has to name seats: the decision slot's speaker judges, the first
    solution slot's speaker proposes, any other debater critiques."""
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
    """File + entry name -> resolved library (_extends/_includes applied).

    An entry with `slot_stages` uses the STAGE schema (role-conditioned, needs
    `protocol` to decide which seat is proposer/critic/judge); an entry with
    `slots` uses the DIRECT schema (per-speaker templates, protocol unused)."""
    data = resolve_all_experiments(load_config_with_includes(path))
    cfg = data.get(entry)
    if not isinstance(cfg, dict):
        available = ", ".join(n for n in data if not n.startswith("_")) or "<none>"
        raise KeyError(f"prompt entry {entry!r} not in {path} (available: {available})")
    if "slot_stages" in cfg:
        return _load_stage_entry(cfg, entry, protocol)
    if "slots" in cfg:
        return _load_direct_entry(cfg, entry)
    raise ValueError(
        f"prompt entry {entry!r}: neither `slot_stages` (stage schema) nor `slots` "
        "(direct schema) present — nothing would render"
    )


def _load_stage_entry(cfg: dict, entry: str, protocol: Optional[Protocol]) -> PromptLibrary:
    """The old-repo stages composed into the per-speaker system cards, prefix
    preambles and slot cues this repo renders."""
    if protocol is None:
        raise ValueError(
            f"prompt entry {entry!r}: load_prompt_library needs the protocol — stages are "
            "role-conditioned (proposer/critic/judge come from the protocol's solution and "
            "decision slots)"
        )
    slot_stages = {str(k): str(v) for k, v in (cfg.get("slot_stages") or {}).items()}
    if not slot_stages:
        raise ValueError(
            f"prompt entry {entry!r}: no slot_stages — every protocol slot needs the cue "
            f"stage that serves it, e.g. {{proposal: pre_opening_speech_proposer}}"
        )
    unknown = set(cfg) - _STAGE_ENTRY_KEYS - set(slot_stages.values())
    if unknown:
        # Also catches a cue stage that slot_stages stopped naming: dead prompt
        # text is exactly how the old repo's pre_debate went dark for months.
        raise ValueError(
            f"prompt entry {entry!r}: stage(s) {sorted(unknown)} are neither a known role stage "
            f"nor named by slot_stages, so nothing would render them. Known role stages: "
            f"{sorted(_ROLE_STAGES)}"
        )
    stages = {k: _join(v) for k, v in cfg.items() if k not in ("vars", "slot_stages")}
    missing_cues = sorted({s for s in slot_stages.values() if s not in stages})
    if missing_cues:
        raise ValueError(
            f"prompt entry {entry!r}: slot_stages names stage(s) {missing_cues} that the entry "
            "does not define"
        )

    roles = speaker_roles(protocol)
    return PromptLibrary(
        system={s: _compose(stages, _SYSTEM_STAGES[r]) for s, r in roles.items()},
        stage_preamble={
            s: text
            for s, r in roles.items()
            if (text := _compose(stages, _PREAMBLE_STAGES.get(r, ())))
        },
        slots={slot: stages[stage] for slot, stage in slot_stages.items()},
        vars={str(k): str(v) for k, v in (cfg.get("vars") or {}).items()},
    )


def _load_direct_entry(cfg: dict, entry: str) -> PromptLibrary:
    unknown = set(cfg) - _DIRECT_ENTRY_KEYS
    if unknown:
        raise ValueError(f"prompt entry {entry!r}: unknown key(s) {sorted(unknown)}")
    slots: dict[str, str | dict[str, str]] = {}
    for name, tmpl in (cfg.get("slots") or {}).items():
        slots[str(name)] = (
            {str(s): _join(t) for s, t in tmpl.items()} if isinstance(tmpl, dict) else _join(tmpl)
        )
    attribution: dict[str, dict[str, str]] = {}
    for reader, authors in (cfg.get("attribution") or {}).items():
        if not isinstance(authors, dict):
            raise ValueError(
                f"prompt entry {entry!r}: attribution[{reader!r}] must be a "
                f"{{author: template}} map, got {type(authors).__name__}"
            )
        attribution[str(reader)] = {str(a): _join(t) for a, t in authors.items()}
    raw_preamble = cfg.get("preamble") or []
    if not isinstance(raw_preamble, list):
        raise ValueError(
            f"prompt entry {entry!r}: preamble must be a LIST of message items "
            f"(string = every speaker, map = per-speaker), got {type(raw_preamble).__name__}"
        )
    preamble: list[str | dict[str, str]] = []
    for i, item in enumerate(raw_preamble):
        if isinstance(item, dict):
            preamble.append({str(s): _join(t) for s, t in item.items()})
        elif isinstance(item, (str, int, float)):
            preamble.append(str(item))
        else:
            raise ValueError(
                f"prompt entry {entry!r}: preamble[{i}] must be a template string or "
                f"{{speaker: template}} map, got {type(item).__name__}"
            )
    return PromptLibrary(
        system={str(k): _join(v) for k, v in (cfg.get("system") or {}).items()},
        slots=slots,
        vars={str(k): str(v) for k, v in (cfg.get("vars") or {}).items()},
        preamble=preamble,
        attribution=attribution,
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
    """Template for (slot, speaker). A string entry serves every speaker; a map
    entry is per-speaker. Missing either level is a KeyError naming both."""
    if slot_name not in lib.slots:
        raise KeyError(f"no template for slot {slot_name!r} (have: {sorted(lib.slots)})")
    tmpl = lib.slots[slot_name]
    if isinstance(tmpl, str):
        return tmpl
    if speaker not in tmpl:
        raise KeyError(
            f"no template for slot {slot_name!r} speaker {speaker!r} (have: {sorted(tmpl)})"
        )
    return tmpl[speaker]


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
    # speaker's system never renders in slot 0's context, but the slot TEMPLATE
    # is still checked — the task source renders it (with task bindings only)
    # into the solo messages, so a deferred placeholder there is still an error.
    if cs.speaker in lib.system and not (first_slot_verbatim and cs.index == 0):
        yield f"system[{cs.speaker}]", lib.system[cs.speaker]
    if lib.stage_preamble.get(cs.speaker) and not (first_slot_verbatim and cs.index == 0):
        # The stage preamble rides the speaker's FIRST user message in EVERY
        # context it renders, so it must be bindable at that speaker's
        # earliest rendered slot.
        yield f"preamble[{cs.speaker}]", lib.stage_preamble[cs.speaker]
    try:
        yield f"slot {cs.slot.name}[{cs.speaker}]", slot_template(lib, cs.slot.name, cs.speaker)
    except KeyError:
        pass  # reported separately


def _extra_templates(
    lib: PromptLibrary, slots: list[CompiledSlot], first_slot_verbatim: bool = False
) -> Iterator[tuple[CompiledSlot, str, str]]:
    """(slot, label, template) for preamble / attribution templates at the
    EARLIEST slot each can render: a preamble item renders in every context of
    the speakers it serves EXCEPT slot 0 in verbatim-first-slot modes (whose
    context is the task messages) and never for the slot-0 AUTHOR in those
    modes (its solo exchange replays instead of preamble messages); an
    attribution template first renders at the reader's first slot that can see
    another speaker's earlier public slot. Pairs the protocol never exercises
    yield nothing."""
    solo_author = slots[0].speaker if (first_slot_verbatim and slots) else None
    first_rendering: dict[str, CompiledSlot] = {}
    for cs in slots:
        if first_slot_verbatim and cs.index == 0:
            continue  # slot 0 renders the task messages; no preamble there
        if cs.speaker != solo_author:
            first_rendering.setdefault(cs.speaker, cs)
    for i, item in enumerate(lib.preamble):
        serves = list(first_rendering) if isinstance(item, str) else [
            s for s in item if s in first_rendering
        ]
        for speaker in serves:
            tmpl = item if isinstance(item, str) else item[speaker]
            yield first_rendering[speaker], f"preamble[{i}][{speaker}]", tmpl
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
    solution has landed. Static lib.vars count as bound. Preamble /
    attribution templates are checked at the earliest slot they can render.

    Choice mode (position_binder + first_speech_non_debate_aware): the single
    solution slot binds BOTH deferred names for EVERY speaker the moment it
    lands — so both are usable at any strictly-later slot, and neither at the
    solution slot itself. Slot 0 renders the task messages verbatim, so the
    speaker's system/preamble are exempt there (the slot template is not: the
    task source renders it into those messages)."""
    sols = [cs for cs in slots if cs.slot.kind == Kind.SOLUTION]
    own = {cs.speaker: cs.index for cs in sols}
    first = min((cs.index for cs in sols), default=None)
    errors: list[str] = []
    by_slot: dict[int, list[tuple[str, str]]] = {}
    for cs, label, tmpl in _extra_templates(lib, slots, first_slot_verbatim=choice_positions):
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
    (slot, speaker) the protocol will ask for resolves, every speaker has a
    system prompt, and (fresh/choice mode) no template needs a position that
    has not been generated yet. Raises ValueError listing every problem at
    once."""
    slots = protocol.compile()
    errors: list[str] = []
    speakers = set(protocol.speakers)
    roles = speaker_roles(protocol)
    for speaker in protocol.speakers:
        if speaker not in lib.system:
            errors.append(f"no system prompt for speaker {speaker!r} (have: {sorted(lib.system)})")
        elif not lib.system[speaker].strip():
            # A stage entry composes the card, so an empty one means the entry
            # defines none of the role's stages — silently unprompted otherwise.
            errors.append(
                f"empty system prompt for speaker {speaker!r} (role {roles.get(speaker)!r}): a "
                f"stage entry must define one of {list(_SYSTEM_STAGES.get(roles.get(speaker, ''), ()))}"
            )
    for cs in slots:
        try:
            slot_template(lib, cs.slot.name, cs.speaker)
        except KeyError as e:
            errors.append(str(e.args[0]))
    for i, item in enumerate(lib.preamble):
        if isinstance(item, dict):
            for sp in item:
                if sp not in speakers:
                    errors.append(
                        f"preamble[{i}] speaker {sp!r} not in protocol (have: {sorted(speakers)})"
                    )
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

    def preamble(self, speaker: str, bindings: dict[str, str]) -> Optional[str]:
        """STAGE schema: old-repo pre_debate[_judge] + grading details,
        prefixed onto this speaker's first user content. None for a speaker
        with no preamble (the proposer, whose opening cue must stay
        byte-identical to the RLVR arm; every direct-schema speaker)."""
        tmpl = self.lib.stage_preamble.get(speaker)
        return render(tmpl, bindings, self.lib.vars) if tmpl else None

    def instruction(self, slot_name: str, speaker: str, bindings: dict[str, str]) -> str:
        return render(slot_template(self.lib, slot_name, speaker), bindings, self.lib.vars)

    def preamble_messages(self, speaker: str, bindings: dict[str, str]) -> list[str]:
        """DIRECT schema: one rendered string per leading user message for this
        speaker, in item order; items whose map lacks the speaker are skipped."""
        out: list[str] = []
        for item in self.lib.preamble:
            tmpl = item if isinstance(item, str) else item.get(speaker)
            if tmpl is None:
                continue
            out.append(render(tmpl, bindings, self.lib.vars))
        return out

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
        + list(lib.stage_preamble.values())
        + [t for v in lib.slots.values() for t in (v.values() if isinstance(v, dict) else [v])]
        + [t for item in lib.preamble for t in (item.values() if isinstance(item, dict) else [item])]
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
