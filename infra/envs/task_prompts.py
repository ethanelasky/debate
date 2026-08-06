"""Loader for task-family answer-generation prompts (infra/prompts/tasks/*).

Lives outside the tasks package so env modules can import it without pulling in
the family registry (infra.envs.tasks.__init__ imports the env modules).

A prompt config is the family's SOLO answer-generation context, written as the
message list it renders to — one schema for every family:

    format_notes      optional; a shared block substituted into message
                      contents wherever <FORMAT_NOTES> appears, so wording
                      used in more than one place exists exactly ONCE
    messages          required; the solo context, in order. Each item is
                      {role, content, name?}: role is system|user, content is
                      the message template, and a message that a debate pack
                      may splice in whole carries a `name` from
                      TASK_SUPPLIED_TEMPLATES (see below).
    plan              optional; the private plan-turn cue (splice name
                      PLAN_USER), NOT part of the solo messages. Plan-then-
                      answer rollouts (envs/planned.py) render it as the
                      user message of a scratchpad turn BEFORE the solo
                      context's eliciting message; debate packs splice it as
                      a pre-solution scratchpad slot's cue. It must carry the
                      family's row placeholder — the plan turn is where the
                      problem first reaches the speaker.

The row's content binds via a placeholder the FAMILY owns — <PROBLEM> for
math/codecontests (the problem text, mid-message), <BACKGROUND_TEXT> for
monitoringbench (the rendered trajectory, a whole message of its own so the
debate layer can reuse it as every seat's shared preamble). The family's
source passes its placeholder as `require_placeholder`, checked eagerly at
load.

These messages are the RLVR prompt VERBATIM, and under
first_speech_non_debate_aware they are also the debate first-slot context — so
every failure mode here is loud.

A debate pack splices a NAMED message in as a template via `<NAME>` (see
supplied_templates), which makes the two arms render the same composition
byte-for-byte: math/codecontests packs splice <ANSWER_GEN_USER> as the
proposal slot cue; the MB packs splice <ANSWER_GEN_USER> as the
blind_assessment cue and <TRAJECTORY_USER> as the shared pre_debate trajectory
message. One prompt per family across both arms — editing wording here changes
every arm together, which is the point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

PROMPT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "infra" / "prompts" / "tasks"
_REPO_ROOT = Path(__file__).resolve().parents[2]
PROBLEM_PLACEHOLDER = "<PROBLEM>"
BACKGROUND_PLACEHOLDER = "<BACKGROUND_TEXT>"
_NOTES_PLACEHOLDER = "<FORMAT_NOTES>"

_PLACEHOLDER = re.compile(r"<([A-Z_][A-Z0-9_]*)>")
_ROLES = ("system", "user")
_MESSAGE_KEYS = {"role", "content", "name"}

# Registered names a message may expose itself under for pack splicing.
# DebateEnv substitutes a supplied template into slot/system templates BEFORE
# rendering (so placeholders it carries are bound normally), and raises at
# construction if a pack references one of these names and the task source
# supplied no message with it — no silent fallback, no failure deferred to
# generation time. The registry is what makes that check possible: env code
# can tell "task-supplied template" from an ordinary binding placeholder.
TASK_SUPPLIED_TEMPLATES = ("ANSWER_GEN_USER", "TRAJECTORY_USER", "PLAN_USER")

#: The `plan` key's fixed splice name. Unlike the message-list names, the plan
#: cue is its own top-level key: it is NOT part of the solo context (plain
#: single-turn RLVR must keep rendering exactly `messages`), so it cannot live
#: in the messages list without changing that arm's prompt.
PLAN_TEMPLATE_NAME = "PLAN_USER"


@dataclass(frozen=True)
class GenerationPrompts:
    #: the solo context templates, in order: [{"role", "content"}].
    messages: list[dict[str, str]]
    #: splice name -> that message's content (messages that carried `name`).
    named: dict[str, str] = field(default_factory=dict)

    def render(self, bindings: dict[str, str]) -> list[dict[str, str]]:
        """The solo context with `<KEY>` placeholders substituted. Strict, in
        one pass: every UPPERCASE placeholder a template carries must be bound
        (a typo'd or deliberately-unbound tag hard-errors here, before any
        generation is paid for), substituted text is never re-scanned, and
        each rendered message is stripped — the same contract as the debate
        prompt layer's render()."""
        out = []
        for m in self.messages:
            used = set(_PLACEHOLDER.findall(m["content"]))
            missing = sorted(used - set(bindings))
            if missing:
                raise ValueError(
                    f"unbound placeholder(s) {missing} in task prompt message: "
                    f"{m['content'][:120]!r}"
                )
            content = _PLACEHOLDER.sub(lambda mo: str(bindings[mo.group(1)]), m["content"])
            out.append({"role": m["role"], "content": content.strip()})
        return out

    def supplied_templates(self) -> dict[str, str]:
        """The named messages, for pack splicing — a pack's slot or stage can
        BE one of these messages rather than a re-typed copy of it. Contents
        are still templates: a spliced message may carry <PROBLEM>, which the
        caller rebinds to its own topic placeholder."""
        return dict(self.named)


def resolve_prompt_file(path: str | Path | None, default_name: str) -> Path:
    """Absolute path, repo-relative path, or the packaged default when None."""
    if path is None:
        return PROMPT_CONFIG_DIR / default_name
    p = Path(path)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def load_generation_prompts(
    path: str | Path, require_placeholder: str = PROBLEM_PLACEHOLDER
) -> GenerationPrompts:
    """Load and validate one family's config. `require_placeholder` is the
    family's row-content placeholder (<PROBLEM>, <BACKGROUND_TEXT>): it must
    appear in at least one message, so a typo'd tag fails at load, not as a
    prompt with the row content silently missing."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"task prompt config not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of prompt keys")

    unknown = sorted(set(data) - {"messages", "format_notes", "plan"})
    if unknown or "messages" not in data:
        raise ValueError(
            f"{path}: bad prompt keys (missing {sorted({'messages'} - set(data))}, "
            f"unknown {unknown}); required ['messages'], optional ['format_notes', 'plan']"
        )
    notes = data.get("format_notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"{path}: format_notes must be a string, got {type(notes).__name__}")

    raw = data["messages"]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: messages must be a non-empty list")
    messages: list[dict[str, str]] = []
    named: dict[str, str] = {}
    for i, m in enumerate(raw):
        if not isinstance(m, dict) or (set(m) - _MESSAGE_KEYS) or not {"role", "content"} <= set(m):
            raise ValueError(
                f"{path}: messages[{i}] must be a mapping with keys role/content "
                f"(optional name), got {m!r:.120}"
            )
        role, content = m["role"], m["content"]
        if role not in _ROLES:
            raise ValueError(f"{path}: messages[{i}] role must be one of {_ROLES}, got {role!r}")
        if not isinstance(content, str):
            raise ValueError(
                f"{path}: messages[{i}] content must be a string, got {type(content).__name__}"
            )
        content = _fill_notes(content, notes, path, f"messages[{i}]")
        if "name" in m:
            name = m["name"]
            if name not in TASK_SUPPLIED_TEMPLATES or name == PLAN_TEMPLATE_NAME:
                raise ValueError(
                    f"{path}: messages[{i}] name {name!r} is not a registered splice "
                    f"name {TASK_SUPPLIED_TEMPLATES}"
                    + (
                        " (PLAN_USER comes from the top-level `plan` key — a plan cue "
                        "inside `messages` would change the plain single-turn prompt)"
                        if name == PLAN_TEMPLATE_NAME
                        else ""
                    )
                )
            if name in named:
                raise ValueError(f"{path}: duplicate message name {name!r}")
            named[name] = content
        messages.append({"role": role, "content": content})

    plan = data.get("plan")
    if plan is not None:
        if not isinstance(plan, str):
            raise ValueError(f"{path}: plan must be a string, got {type(plan).__name__}")
        plan = _fill_notes(plan, notes, path, "plan")
        if require_placeholder not in plan:
            raise ValueError(
                f"{path}: plan does not contain the {require_placeholder} placeholder — "
                "the plan turn is where the problem first reaches the speaker"
            )
        named[PLAN_TEMPLATE_NAME] = plan

    if not any(require_placeholder in m["content"] for m in messages):
        raise ValueError(
            f"{path}: no message contains the {require_placeholder} placeholder — "
            "the row's content could not be bound into the prompt"
        )
    return GenerationPrompts(messages=messages, named=named)


def _fill_notes(text: str, notes: Optional[str], path: Path, key: str) -> str:
    if _NOTES_PLACEHOLDER not in text:
        return text
    if notes is None:
        raise ValueError(f"{path}: {key} uses {_NOTES_PLACEHOLDER} but format_notes is not set")
    return text.replace(_NOTES_PLACEHOLDER, notes)
