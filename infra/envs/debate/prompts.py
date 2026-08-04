"""Prompt library: YAML stages -> rendered system / preamble / instruction text.

Loading reuses debate/config.py's _includes/_extends resolver unchanged — a
prompt file is just a config file whose entries happen to be prompt sets. This
module adds a typed view, `<PLACEHOLDER>` substitution, the compile-time
prompt/protocol contract, and the adapter implementing round.py's
PromptLibrary Protocol.

SCHEMA — the old repo's stage vocabulary (prompts/parser.py PromptTag there),
not per-speaker blobs. An entry names stages; `slot_stages` maps this repo's
protocol slot names onto the cue stages, so protocols stay declarative and the
old-name/new-name correspondence is per-pack DATA rather than renderer magic:

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
  preamble    = pre_debate + debater_standard_grading_details  (critic), or
                pre_debate_judge + judge_standard_grading_details (judge).
                Prefixed onto the FIRST user content of that speaker's context.

The PROPOSER gets NO preamble, deliberately: its opening cue is the task
family's answer-generation user message spliced in whole, and that rendered
message must stay byte-identical to the RLVR arm's (tests pin it). Its grading
framing lives in answer_gen_system instead.

Roles come from the PROTOCOL, not from the pack: the speaker of the first
solution slot is the proposer, the decision slot's speaker is the judge, and a
remaining debater is the critic. load_prompt_library therefore needs the
protocol whenever an entry uses role-conditioned stages.

Every stage may be a single string OR a LIST of blocks joined with a blank
line. Blocks compose under _extends: the config resolver merges lists BY INDEX,
so a child can override one block of a parent's stage (or append blocks past
the end) without restating the rest — the old repo's adjacent-block
concatenation, kept.

No Jinja: substitution is a plain replace of `<KEY>`. The placeholder pattern
is UPPERCASE-only, so literal lowercase tags (`<problem>`) pass through.
Attribution of others' speeches is NOT in the schema; it lives in
RenderedPrompts.attributed. To see where every stage lands in the final message
sequences, use `preview(lib, protocol)`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from infra.config import load_config_with_includes, resolve_all_experiments
from infra.envs.debate.protocol import CompiledSlot, Kind, Protocol

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
_ENTRY_KEYS = frozenset({"vars", "slot_stages"}) | _ROLE_STAGES


@dataclass
class PromptLibrary:
    system: dict[str, str] = field(default_factory=dict)
    slots: dict[str, str | dict[str, str]] = field(default_factory=dict)
    vars: dict[str, str] = field(default_factory=dict)
    #: speaker -> text prefixed onto its first user message ("" / absent = none)
    preamble: dict[str, str] = field(default_factory=dict)


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
    """File + entry name -> resolved library (_extends/_includes applied), with
    the old-repo stages composed into the per-speaker system cards, preambles
    and slot cues this repo renders.

    `protocol` decides which seat is proposer/critic/judge and is REQUIRED: the
    stage vocabulary is role-conditioned, so there is nothing to compose
    without it."""
    data = resolve_all_experiments(load_config_with_includes(path))
    cfg = data.get(entry)
    if not isinstance(cfg, dict):
        available = ", ".join(n for n in data if not n.startswith("_")) or "<none>"
        raise KeyError(f"prompt entry {entry!r} not in {path} (available: {available})")
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
    unknown = set(cfg) - _ENTRY_KEYS - set(slot_stages.values())
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
        preamble={
            s: text
            for s, r in roles.items()
            if (text := _compose(stages, _PREAMBLE_STAGES.get(r, ())))
        },
        slots={slot: stages[stage] for slot, stage in slot_stages.items()},
        vars={str(k): str(v) for k, v in (cfg.get("vars") or {}).items()},
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


def _entry_templates(lib: PromptLibrary, cs: CompiledSlot) -> Iterator[tuple[str, str]]:
    if cs.speaker in lib.system:
        yield f"system[{cs.speaker}]", lib.system[cs.speaker]
    if lib.preamble.get(cs.speaker):
        # The preamble rides the speaker's FIRST user message in EVERY context
        # it renders, so it must be bindable at that speaker's earliest slot.
        yield f"preamble[{cs.speaker}]", lib.preamble[cs.speaker]
    try:
        yield f"slot {cs.slot.name}[{cs.speaker}]", slot_template(lib, cs.slot.name, cs.speaker)
    except KeyError:
        pass  # reported separately


def _bindability_errors(lib: PromptLibrary, slots: list[CompiledSlot]) -> list[str]:
    """Fresh mode: <POSITION>/<OPPONENT_POSITION> are bound by extraction from a
    solution slot, so they may not appear in anything rendered at or before the
    solution slot that binds them. round.py binds a solver's own POSITION and
    every other speaker's OPPONENT_POSITION the moment its solution lands; a
    speaker with no solution slot of its own (the judge) reads POSITION once any
    solution has landed. Static lib.vars count as bound."""
    sols = [cs for cs in slots if cs.slot.kind == Kind.SOLUTION]
    own = {cs.speaker: cs.index for cs in sols}
    first = min((cs.index for cs in sols), default=None)
    errors: list[str] = []
    for cs in slots:
        opp = min((c.index for c in sols if c.speaker != cs.speaker), default=None)
        pos = own.get(cs.speaker, first)
        unbound = {
            name
            for name, at in zip(_DEFERRED, (pos, opp))
            if at is None or at >= cs.index
        } - set(lib.vars)
        for label, tmpl in _entry_templates(lib, cs):
            hit = sorted({m.group(1) for m in PLACEHOLDER.finditer(tmpl)} & unbound)
            if hit:
                errors.append(
                    f"{label}: {', '.join(hit)} not bound yet at slot {cs.index} "
                    f"({cs.speaker}/{cs.slot.name}) in fresh_positions mode"
                )
    return list(dict.fromkeys(errors))


def validate_prompts(lib: PromptLibrary, protocol: Protocol, fresh_positions: bool = False) -> None:
    """The prompt/protocol contract, checked before any generation: every
    (slot, speaker) the protocol will ask for resolves, every speaker has a
    system prompt, and (fresh mode) no template needs a position that has not
    been generated yet. Raises ValueError listing every problem at once."""
    slots = protocol.compile()
    errors: list[str] = []
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
    if fresh_positions:
        errors += _bindability_errors(lib, slots)
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
        """Old-repo pre_debate[_judge] + grading details, prefixed onto this
        speaker's first user content. None for a speaker with no preamble (the
        proposer, whose opening cue must stay byte-identical to the RLVR arm)."""
        tmpl = self.lib.preamble.get(speaker)
        return render(tmpl, bindings, self.lib.vars) if tmpl else None

    def instruction(self, slot_name: str, speaker: str, bindings: dict[str, str]) -> str:
        return render(slot_template(self.lib, slot_name, speaker), bindings, self.lib.vars)

    def attributed(self, author_name: str, slot_name: str, text: str) -> str:
        return f"{author_name} said:\n{text}"


def load_rendered_prompts(path: str | Path, entry: str, protocol: Protocol) -> RenderedPrompts:
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
        + list(lib.preamble.values())
        + [t for v in lib.slots.values() for t in (v.values() if isinstance(v, dict) else [v])]
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
