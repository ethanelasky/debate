"""Prompt library: YAML templates -> rendered system / instruction strings.

Loading reuses debate/config.py's _includes/_extends resolver unchanged — a
prompt file is just a config file whose entries happen to be prompt sets. This
module adds a typed view, `<PLACEHOLDER>` substitution, the compile-time
prompt/topology contract, and the adapter implementing round.py's
PromptLibrary Protocol.

Schema (see prompt_configs/hendrycks_math.yaml):
    <entry>:
      _extends: <parent>        # optional
      vars: {KEY: value}        # static substitutions, merged over parent
      system: {<speaker>: <template>}
      slots:
        <name>: <template>                # string = any speaker
        <name>: {<speaker>: <template>}   # map = per-speaker (e.g. closing)

Every <template> may be a single string OR a LIST of blocks joined with a
blank line. Blocks compose under _extends: the config resolver merges lists
BY INDEX, so a child can override one block of a parent's prompt (or append
blocks past the end) without restating the rest — the old repo's
adjacent-block concatenation, kept.

No Jinja: substitution is a plain replace of `<KEY>`. The placeholder pattern
is UPPERCASE-only, so literal lowercase tags (`<problem>`) pass through.
Attribution of others' speeches is NOT in the schema; it lives in
RenderedPrompts.attributed. To see where every block lands in the final
message sequences, use `preview(lib, topology)`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from infra.config import load_config_with_includes, resolve_all_experiments
from infra.envs.debate.topology import CompiledSlot, Kind, Topology

PLACEHOLDER = re.compile(r"<([A-Z_][A-Z0-9_]*)>")

_ENTRY_KEYS = {"vars", "system", "slots"}
_DEFERRED = ("POSITION", "OPPONENT_POSITION")  # fresh mode: bound by extraction


@dataclass
class PromptLibrary:
    system: dict[str, str] = field(default_factory=dict)
    slots: dict[str, str | dict[str, str]] = field(default_factory=dict)
    vars: dict[str, str] = field(default_factory=dict)


def load_prompt_library(path: str | Path, entry: str) -> PromptLibrary:
    """File + entry name -> resolved library (_extends/_includes applied)."""
    data = resolve_all_experiments(load_config_with_includes(path))
    cfg = data.get(entry)
    if not isinstance(cfg, dict):
        available = ", ".join(n for n in data if not n.startswith("_")) or "<none>"
        raise KeyError(f"prompt entry {entry!r} not in {path} (available: {available})")
    unknown = set(cfg) - _ENTRY_KEYS
    if unknown:
        raise ValueError(f"prompt entry {entry!r}: unknown key(s) {sorted(unknown)}")
    slots: dict[str, str | dict[str, str]] = {}
    for name, tmpl in (cfg.get("slots") or {}).items():
        slots[str(name)] = (
            {str(s): _join(t) for s, t in tmpl.items()} if isinstance(tmpl, dict) else _join(tmpl)
        )
    return PromptLibrary(
        system={str(k): _join(v) for k, v in (cfg.get("system") or {}).items()},
        slots=slots,
        vars={str(k): str(v) for k, v in (cfg.get("vars") or {}).items()},
    )


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


def validate_prompts(lib: PromptLibrary, topology: Topology, fresh_positions: bool = False) -> None:
    """The prompt/topology contract, checked before any generation: every
    (slot, speaker) the topology will ask for resolves, every speaker has a
    system prompt, and (fresh mode) no template needs a position that has not
    been generated yet. Raises ValueError listing every problem at once."""
    slots = topology.compile()
    errors: list[str] = []
    for speaker in topology.speakers:
        if speaker not in lib.system:
            errors.append(f"no system prompt for speaker {speaker!r} (have: {sorted(lib.system)})")
    for cs in slots:
        try:
            slot_template(lib, cs.slot.name, cs.speaker)
        except KeyError as e:
            errors.append(str(e.args[0]))
    if fresh_positions:
        errors += _bindability_errors(lib, slots)
    if errors:
        raise ValueError("prompt/topology mismatch:\n  " + "\n  ".join(dict.fromkeys(errors)))


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

    def attributed(self, author_name: str, slot_name: str, text: str) -> str:
        return f"{author_name} said:\n{text}"


def load_rendered_prompts(path: str | Path, entry: str) -> RenderedPrompts:
    return RenderedPrompts(load_prompt_library(path, entry))


# ------------------------------------------------------------------ preview


def preview(lib: PromptLibrary, topology: Topology, speakers: Optional[list[str]] = None) -> str:
    """Render each speaker's final-slot message sequence with placeholders kept
    visible (<TOPIC> stays <TOPIC>) and other speakers' outputs stubbed as
    <alice/proposal output>. Answers "where does my prompt text land" without
    running anything: python -m debate.envs.debate.prompts <file> <entry> <topo.yaml>
    """
    from infra.envs.debate.round import DebateState, SlotRecord, render_context

    slots = topology.compile()
    placeholder_names = {
        m.group(1)
        for tmpl in list(lib.system.values())
        + [t for v in lib.slots.values() for t in (v.values() if isinstance(v, dict) else [v])]
        for m in PLACEHOLDER.finditer(tmpl)
    }
    placeholder_names -= set(lib.vars)  # var-bound placeholders render for real
    identity = {name: f"<{name}>" for name in placeholder_names}  # single-pass sub: safe
    bindings = {s: {**identity, "NAME": f"<{s}:NAME>"} for s in topology.speakers}

    out: list[str] = []
    for speaker in speakers or topology.speakers:
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

    from infra.envs.debate.topology import Topology as _T

    lib_ = load_prompt_library(sys.argv[1], sys.argv[2])
    topo_ = _T.parse(_yaml.safe_load(open(sys.argv[3])))
    print(preview(lib_, topo_))
