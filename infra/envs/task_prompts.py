"""Loader for task-family answer-generation prompts (infra/envs/tasks/prompt_configs/*).

Lives outside the tasks package so env modules can import it without pulling in
the family registry (infra.envs.tasks.__init__ imports the env modules).

A prompt config is a flat mapping (key names follow the old repo's
answer_generation_* naming):

    format_notes      optional; a shared block substituted into the other
                      fields wherever <FORMAT_NOTES> appears, so wording used
                      in more than one place exists exactly ONCE
    answer_gen_system required; the RLVR system card
    answer_gen_user   required; the RLVR user turn, carrying <PROBLEM>

answer_gen_system/answer_gen_user are the RLVR prompt VERBATIM, and under
first_speech_non_debate_aware they are also the debate proposal context — so
every failure mode here is loud.

A debate pack may also splice answer_gen_user in as a slot template via
<ANSWER_GEN_USER> (see supplied_templates), which makes the two arms render the
same composition byte-for-byte. Codecontests does exactly that. Math does NOT:
its debate format instruction ("EXACTLY one \\boxed{...}") is deliberately
different wording from its RLVR prompt ("\\boxed{NUMBER}"), so the math debate
pack writes its own proposal slot and nothing is unified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

PROMPT_CONFIG_DIR = Path(__file__).resolve().parent / "tasks" / "prompt_configs"
_REPO_ROOT = Path(__file__).resolve().parents[2]
PROBLEM_PLACEHOLDER = "<PROBLEM>"
_NOTES_PLACEHOLDER = "<FORMAT_NOTES>"

_REQUIRED_KEYS = {"answer_gen_system", "answer_gen_user"}
_OPTIONAL_KEYS = {"format_notes"}

# Templates a debate pack can splice in from the task config instead of
# restating. DebateEnv substitutes these into slot/system templates BEFORE
# rendering (so placeholders they carry are bound normally), and raises at
# construction if a pack references one and the task source supplied nothing —
# no silent fallback, no failure deferred to generation time.
TASK_SUPPLIED_TEMPLATES = ("ANSWER_GEN_USER",)


@dataclass(frozen=True)
class GenerationPrompts:
    answer_gen_system: str
    answer_gen_user: str

    def messages(self, problem: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.answer_gen_system},
            {
                "role": "user",
                "content": self.answer_gen_user.replace(PROBLEM_PLACEHOLDER, problem),
            },
        ]

    def supplied_templates(self) -> dict[str, str]:
        """Templates a debate pack may splice in, so its proposal slot can BE
        this answer prompt rather than a re-typed copy of it. Still carries
        <PROBLEM>: the caller rebinds that to its own topic placeholder."""
        return {"ANSWER_GEN_USER": self.answer_gen_user}


def resolve_prompt_file(path: str | Path | None, default_name: str) -> Path:
    """Absolute path, repo-relative path, or the packaged default when None."""
    if path is None:
        return PROMPT_CONFIG_DIR / default_name
    p = Path(path)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def load_generation_prompts(path: str | Path) -> GenerationPrompts:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"task prompt config not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of prompt keys")

    missing = sorted(_REQUIRED_KEYS - set(data))
    unknown = sorted(set(data) - _REQUIRED_KEYS - _OPTIONAL_KEYS)
    if missing or unknown:
        raise ValueError(
            f"{path}: bad prompt keys (missing {missing}, unknown {unknown}); "
            f"required {sorted(_REQUIRED_KEYS)}, optional {sorted(_OPTIONAL_KEYS)}"
        )
    for key, value in data.items():
        if not isinstance(value, str):
            raise ValueError(f"{path}: {key} must be a string, got {type(value).__name__}")

    notes = data.get("format_notes")
    fields = {k: _fill_notes(data[k], notes, path, k) for k in data if k != "format_notes"}

    if PROBLEM_PLACEHOLDER not in fields["answer_gen_user"]:
        raise ValueError(
            f"{path}: answer_gen_user must contain the {PROBLEM_PLACEHOLDER} placeholder"
        )
    return GenerationPrompts(**fields)


def _fill_notes(text: str, notes: Optional[str], path: Path, key: str) -> str:
    if _NOTES_PLACEHOLDER not in text:
        return text
    if notes is None:
        raise ValueError(f"{path}: {key} uses {_NOTES_PLACEHOLDER} but format_notes is not set")
    return text.replace(_NOTES_PLACEHOLDER, notes)
