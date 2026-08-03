"""Loader for task-family answer-generation prompts (infra/envs/tasks/prompt_configs/*).

Lives outside the tasks package so env modules can import it without pulling in
the family registry (infra.envs.tasks.__init__ imports the env modules).

A prompt config is a flat mapping (key names follow the old repo's
answer_generation_* naming):

    format_notes              optional; the shared format block, substituted
                              into the other fields wherever <FORMAT_NOTES>
                              appears, so the wording exists exactly ONCE
    answer_gen_system         required; the RLVR system card
    answer_gen_user           required; the RLVR user turn, carrying <PROBLEM>
    answer_format_instruction optional; the debate-side rendering of the same
                              format wording, injected by DebateEnv as the
                              ANSWER_FORMAT_INSTRUCTION prompt var

answer_gen_system/answer_gen_user are the RLVR prompt VERBATIM, and under
first_speech_non_debate_aware they are also the debate proposal context — so
every failure mode here is loud.

Only codecontests sets answer_format_instruction. Math deliberately does NOT:
its debate format instruction ("EXACTLY one \\boxed{...}") and its RLVR system
prompt ("\\boxed{NUMBER}") are different battle-tested wordings for different
formats, so the math debate yaml keeps its own var and nothing is unified.
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
_OPTIONAL_KEYS = {"format_notes", "answer_format_instruction"}


@dataclass(frozen=True)
class GenerationPrompts:
    answer_gen_system: str
    answer_gen_user: str
    answer_format_instruction: Optional[str] = None

    def messages(self, problem: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.answer_gen_system},
            {
                "role": "user",
                "content": self.answer_gen_user.replace(PROBLEM_PLACEHOLDER, problem),
            },
        ]

    def prompt_vars(self) -> dict[str, str]:
        """Debate prompt vars this task family supplies, so a debate yaml can
        reference the wording instead of restating it."""
        if self.answer_format_instruction is None:
            return {}
        return {"ANSWER_FORMAT_INSTRUCTION": self.answer_format_instruction}


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
