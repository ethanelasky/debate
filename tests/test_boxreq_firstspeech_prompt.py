"""The versioned first-speech prompt changes only the two system cards."""

from __future__ import annotations

import warnings
from pathlib import Path

from infra.config import (
    load_config_with_includes,
    load_experiment,
    resolve_all_experiments,
)
from infra.envs.debate.prompts import load_prompt_library, render
from infra.envs.debate.protocol import Protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "math_pc_debate.yaml"
PARENT_PROMPT = ROOT / "infra" / "prompts" / "debate" / "hendrycks_math_boxreq.yaml"
CHILD_PROMPT = ROOT / "infra" / "prompts" / "debate" / "hendrycks_math_boxreq_firstspeech.yaml"
PARENT_ENTRY = "math_proposer_critic_boxreq"
CHILD_ENTRY = "math_proposer_critic_boxreq_firstspeech"

DEBATER_RULE = (
    "The Proposer's opening solution must end in exactly one well-formed "
    "\\boxed{...} holding its final answer. That opening solution is the only "
    "speech in the debate that may contain \\boxed{...}; every later speech "
    "must refer to answers without boxing them."
)
JUDGE_RULE = (
    "The submitted solution is <NAME>'s first speech. Check only that first "
    "speech for its final \\boxed{...}. A box in any later speech is part of "
    "the debate and cannot supply, replace, or repair the submitted answer."
)


def _protocol() -> Protocol:
    exp = load_experiment(CONFIG, "mathl5_qwen35_pc_debate_cispo_verl_boxreq")
    return Protocol.parse(exp["protocol"])


def _library(path: Path, entry: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_prompt_library(path, entry, _protocol())


def test_only_the_shared_debater_and_judge_system_stages_change():
    entries = resolve_all_experiments(load_config_with_includes(CHILD_PROMPT))
    parent, child = entries[PARENT_ENTRY], entries[CHILD_ENTRY]

    differing = {
        key for key in set(parent) | set(child) if parent.get(key) != child.get(key)
    }
    assert differing == {"debater_system", "judge_system"}
    assert child["debater_system"][1:] == parent["debater_system"][1:]
    assert child["judge_system"][1:] == parent["judge_system"][1:]


def test_proposal_and_rebuttal_cues_remain_byte_identical_to_parent():
    parent = _library(PARENT_PROMPT, PARENT_ENTRY)
    child = _library(CHILD_PROMPT, CHILD_ENTRY)

    for slot in ("proposal", "critique", "alice_rebuttal", "bob_rebuttal"):
        assert child.slots[slot].encode() == parent.slots[slot].encode()
    assert child.slots == parent.slots
    assert child.preamble == parent.preamble


def test_both_debater_systems_contain_the_exact_approved_rule():
    child = _library(CHILD_PROMPT, CHILD_ENTRY)

    assert DEBATER_RULE in child.system["alice"]
    assert DEBATER_RULE in child.system["bob"]


def test_judge_system_contains_the_exact_approved_first_speech_rule():
    child = _library(CHILD_PROMPT, CHILD_ENTRY)

    assert JUDGE_RULE in child.system["judge"]


def test_a_later_box_is_rendered_as_unable_to_repair_an_unboxed_proposal():
    proposal = "The arithmetic gives 5."
    rebuttal = "The objection fails, so the result is \\boxed{5}."
    assert "\\boxed{" not in proposal
    assert "\\boxed{" in rebuttal

    child = _library(CHILD_PROMPT, CHILD_ENTRY)
    judge_system = render(
        child.system["judge"],
        {"NAME": "Alice", "OPPONENT_NAME": "Bob"},
        child.vars,
    )
    rendered_rule = JUDGE_RULE.replace("<NAME>", "Alice")
    assert rendered_rule in judge_system
    assert (
        "cannot supply, replace, or repair the submitted answer" in judge_system
    )
