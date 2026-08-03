"""Unit tests for the task-family layer (registry, math family, config guards).

Everything here is offline: MathFamily.source() is only ever called with an
invalid config, where reject_unknown_keys fires before any dataset loading.
"""

import pytest
import yaml

from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file
from infra.envs.tasks import get_family
from infra.envs.tasks.base import reject_unknown_keys
from infra.envs.tasks.math import MathFamily, _parse_levels


def test_registry_lookup():
    assert isinstance(get_family("math"), MathFamily)


def test_registry_unknown_name_lists_known_families():
    with pytest.raises(ValueError) as exc:
        get_family("nope")
    assert "math" in str(exc.value)


@pytest.mark.parametrize(
    "meta, solution, expected",
    [
        ({"gt": 2.0}, 2.0, True),
        ({"gt": 2.0}, 3.0, False),
        ({"gt": 2.0}, None, None),          # unparseable slot
        ({}, 2.0, None),                    # no ground truth
        ({"gt": 2.0}, "not a number", None),
        ({"gt": 2.0}, 2.0 + 1e-9, True),    # inside the 1e-6 tolerance
    ],
)
def test_math_grade(meta, solution, expected):
    assert MathFamily().grade(meta, solution) is expected


def test_extractor_strict_vs_relaxed():
    strict = MathFamily().extractor(False)
    relaxed = MathFamily().extractor(True)
    assert strict("the answer is 7") is None
    assert relaxed("the answer is 7") == 7.0
    assert strict("\\boxed{42}") == 42.0
    assert relaxed("\\boxed{42}") == 42.0


def test_format_flags_strict_boxed():
    fam = MathFamily()
    assert fam.format_flags("\\boxed{3}")["strict_boxed"] == 1.0
    assert fam.format_flags("no box")["strict_boxed"] == 0.0


@pytest.mark.parametrize(
    "spec, expected",
    [(5, (5,)), ("3-4", (3, 4)), ("5", (5,)), ([3, 4], (3, 4))],
)
def test_parse_levels(spec, expected):
    assert _parse_levels(spec) == expected


def test_reject_unknown_keys():
    reject_unknown_keys({"levels": 5}, {"levels", "seed"}, "math")  # no raise
    with pytest.raises(ValueError) as exc:
        reject_unknown_keys({"levels": 5, "bogus_key": 1}, {"levels", "seed"}, "math")
    assert "bogus_key" in str(exc.value)


def test_math_source_rejects_unknown_key_before_loading():
    with pytest.raises(ValueError) as exc:
        MathFamily().source({"bogus_key": 1})
    assert "bogus_key" in str(exc.value)


# ------------------------------------------------------- generation prompts
#
# The prompts moved from Python constants into
# infra/envs/tasks/prompt_configs/*.yaml. These tests pin the rendered
# messages against the pre-move literals: under first_speech_non_debate_aware
# these bytes ARE the debate proposal context, so a stray edit to the YAML
# (reflow, smart quote, lost backslash) silently changes the experiment.

MATH_SYSTEM_BEFORE_YAML_MOVE = (
    "Solve the competition-level math problem.\n"
    "Return the final answer in this exact format: \\boxed{NUMBER}\n"
    "Do not output XML, ####, or extra prose.\n"
    "Output only the boxed final answer."
)
MATH_USER_SUFFIX_BEFORE_YAML_MOVE = "\n\nReturn only: \\boxed{NUMBER}"


def test_math_default_prompts_are_byte_identical():
    prompts = load_generation_prompts(resolve_prompt_file(None, "math.yaml"))
    problem = "What is $1 + 1$?"
    assert prompts.messages(problem) == [
        {"role": "system", "content": MATH_SYSTEM_BEFORE_YAML_MOVE},
        {"role": "user", "content": problem + MATH_USER_SUFFIX_BEFORE_YAML_MOVE},
    ]


def test_math_supplies_no_debate_format_var():
    """Math is the deliberate non-unified case: its debate instruction
    ("EXACTLY one \\boxed{...}") is different wording from its RLVR system
    prompt, so hendrycks_math.yaml keeps its own ANSWER_FORMAT_INSTRUCTION."""
    prompts = load_generation_prompts(resolve_prompt_file(None, "math.yaml"))
    assert prompts.answer_format_instruction is None
    assert prompts.prompt_vars() == {}


def _write_prompt_yaml(path, system="SYS", user="<PROBLEM>\n\nGo."):
    body = {"answer_gen_system": system, "answer_gen_user": user}
    path.write_text(yaml.safe_dump(body))
    return path


def test_prompt_file_override(tmp_path):
    path = _write_prompt_yaml(tmp_path / "alt.yaml", system="Be terse.", user="Q: <PROBLEM>")
    prompts = load_generation_prompts(resolve_prompt_file(str(path), "math.yaml"))
    assert prompts.messages("2+2") == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Q: 2+2"},
    ]


def test_prompt_file_relative_path_resolves_against_repo_root():
    prompts = load_generation_prompts(
        resolve_prompt_file("infra/envs/tasks/prompt_configs/math.yaml", "codecontests.yaml")
    )
    assert prompts.answer_gen_system == MATH_SYSTEM_BEFORE_YAML_MOVE


def test_format_notes_substituted_into_every_field(tmp_path):
    path = tmp_path / "notes.yaml"
    path.write_text(
        "format_notes: |-\n  - be brief\nanswer_gen_system: |-\n  Do it.\n  <FORMAT_NOTES>\n"
        "answer_gen_user: |-\n  <PROBLEM>\nanswer_format_instruction: |-\n  Notes:\n  <FORMAT_NOTES>\n"
    )
    prompts = load_generation_prompts(path)
    assert prompts.answer_gen_system == "Do it.\n- be brief"
    assert prompts.answer_format_instruction == "Notes:\n- be brief"


@pytest.mark.parametrize(
    "body, needle",
    [
        ({"answer_gen_system": "S"}, "answer_gen_user"),                          # missing key
        ({"answer_gen_system": "S", "answer_gen_user": "no placeholder"}, "<PROBLEM>"),
        ({"answer_gen_system": "S", "answer_gen_user": "<PROBLEM>", "extra": 1}, "extra"),
        ({"answer_gen_system": "S", "answer_gen_user": 3}, "must be a string"),
        (
            {"answer_gen_system": "<FORMAT_NOTES>", "answer_gen_user": "<PROBLEM>"},
            "format_notes is not set",
        ),
    ],
)
def test_malformed_prompt_config_raises(tmp_path, body, needle):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(body))
    with pytest.raises(ValueError) as exc:
        load_generation_prompts(path)
    assert needle in str(exc.value)


def test_missing_prompt_file_raises(tmp_path):
    with pytest.raises(ValueError) as exc:
        load_generation_prompts(tmp_path / "nope.yaml")
    assert "not found" in str(exc.value)
