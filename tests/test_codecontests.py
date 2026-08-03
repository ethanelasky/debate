"""CodeContests task family: loading/filtering, code extraction, the stdin
verifier (real subprocesses), reward, and grading."""

from __future__ import annotations

import json

import pytest

from infra.envs.tasks import get_family
from infra.envs.tasks.codecontests import (
    CodeContestsEnv,
    CodeContestsFamily,
    extract_code,
    run_stdin_tests,
)

SUM_SOLUTION = "a, b = map(int, input().split())\nprint(a + b)"
ECHO_SOLUTION = "print(input().strip())"
TIMEOUT = 15

ROWS = [
    {
        "name": "sum",
        "description": "Read two integers on one line and print their sum.",
        "inputs": ["1 2", "3 4"],
        "outputs": ["3", "7"],
        "num_corner_cases": 2,
    },
    {
        "name": "echo",
        "description": "Read one line and print it back.",
        "inputs": ["hello", "world"],
        "outputs": ["hello", "world"],
        "num_corner_cases": 2,
    },
    {
        "name": "multi",
        "description": "Find a valid pairing; if several exist print any of them.",
        "inputs": ["1"],
        "outputs": ["1"],
        "num_corner_cases": 1,
    },
    {
        "name": "unverifiable",
        "description": "Compute something interesting.",
        "inputs": [],
        "outputs": [],
        "num_corner_cases": 0,
    },
]


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


@pytest.fixture
def env(tmp_path):
    train = _write_jsonl(tmp_path / "train.jsonl", ROWS)
    test = _write_jsonl(tmp_path / "test.jsonl", ROWS)
    return CodeContestsEnv(path=train, test_path=test, timeout_seconds=TIMEOUT)


# ------------------------------------------------------------------- loading


def test_loader_filters_ineligible_rows(env):
    assert [r["name"] for r in env.train_rows] == ["sum", "echo"]
    assert [r["name"] for r in env.test_rows] == ["sum", "echo"]


def test_split_off_test_rows_when_no_test_path(tmp_path):
    path = _write_jsonl(tmp_path / "train.jsonl", ROWS * 4)
    env = CodeContestsEnv(path=path, timeout_seconds=TIMEOUT)
    assert len(env.train_rows) >= 2 and env.test_rows
    assert len(env.train_rows) + len(env.test_rows) == 8


def test_too_few_rows_raises(tmp_path):
    path = _write_jsonl(tmp_path / "train.jsonl", ROWS[2:])  # both ineligible
    with pytest.raises(RuntimeError):
        CodeContestsEnv(path=path, timeout_seconds=TIMEOUT)


def test_tasks_carry_meta_and_hide_test_io(env):
    tasks = env.tasks(3, split="train") + env.tasks(2, split="test")
    for t in tasks:
        assert t.meta["question"] and t.meta["inputs"] and t.meta["outputs"]
        prompt = "\n".join(m["content"] for m in t.messages)
        for case in t.meta["inputs"] + t.meta["outputs"]:
            assert case not in prompt  # verifier cases must never be prompted
    assert [t.meta["name"] for t in env.tasks(2, split="test")] == ["sum", "echo"]


# ------------------------------------------------------- generation prompts
#
# Pinned against the pre-YAML-move Python literal: under
# first_speech_non_debate_aware these bytes ARE the debate proposal context.

SYSTEM_BEFORE_YAML_MOVE = (
    "Solve the competitive programming problem.\n"
    "- Write a standalone Python script that reads from stdin and prints to stdout.\n"
    "- Write your answer in ``` blocks. Example:\n"
    "```python\n"
    "# your code here\n"
    "```\n"
    "- Read input from stdin using input() or sys.stdin.\n"
    "- Print output to stdout using print().\n"
    "- Do NOT wrap your solution in a function or class unless the problem requires it.\n"
    "- Import any libraries you use.\n"
    "Your goal is to write code that is consistent with the specification in "
    "order to pass the hidden test cases. Do not include functionality or "
    "error handling beyond what is described."
)


def test_default_prompts_are_byte_identical(env):
    task = env.tasks(1, split="test")[0]
    assert task.messages == [
        {"role": "system", "content": SYSTEM_BEFORE_YAML_MOVE},
        {"role": "user", "content": task.meta["question"] + "\n\nProvide your answer:"},
    ]


def test_prompt_file_override(tmp_path):
    alt = tmp_path / "alt.yaml"
    alt.write_text("answer_gen_system: |-\n  Terse.\nanswer_gen_user: |-\n  Q: <PROBLEM>\n")
    path = _write_jsonl(tmp_path / "train.jsonl", ROWS)
    env = CodeContestsEnv(
        path=path, test_path=path, timeout_seconds=TIMEOUT, prompt_file=str(alt)
    )
    task = env.tasks(1, split="test")[0]
    assert task.messages == [
        {"role": "system", "content": "Terse."},
        {"role": "user", "content": "Q: " + task.meta["question"]},
    ]


PROPOSAL_INSTRUCTION_BEFORE_SINGLE_SOURCING = (
    "Context:\n"
    "\n"
    "<question>Read two integers on one line and print their sum.</question>\n"
    "\n"
    "Notes:\n"
    "- Write a standalone Python script that reads from stdin and prints to stdout.\n"
    "- Write your answer in ``` blocks. Example:\n"
    "```python\n"
    "# your code here\n"
    "```\n"
    "- Read input from stdin using input() or sys.stdin.\n"
    "- Print output to stdout using print().\n"
    "- Do NOT wrap your solution in a function or class unless the problem requires it.\n"
    "- Import any libraries you use.\n"
    "\n"
    "Provide your answer:"
)


def _pc_topology():
    from infra.envs.debate.topology import Topology

    return Topology.parse(
        {
            "turns": [
                {"alice": [{"name": "proposal", "kind": "solution"}]},
                {"bob": [{"name": "critique"}]},
                {"alice": [{"name": "defense"}]},
                {"bob": [{"name": "rebuttal"}]},
                {"judge": [{"name": "verdict", "kind": "decision"}]},
            ]
        }
    )


def test_debate_proposal_slot_unchanged_by_single_sourcing(env):
    """ANSWER_FORMAT_INSTRUCTION moved out of the debate yaml and is now
    injected by DebateEnv from this family's answer-generation config. The
    rendered proposal slot must be byte-identical to what the duplicated var
    produced."""
    from infra.envs.debate.env import DebateEnv, DebateEnvConfig
    from infra.envs.debate.judge import JudgeConfig

    debate = DebateEnv(
        DebateEnvConfig(
            topology=_pc_topology(),
            prompt_file="infra/envs/debate/prompt_configs/codecontests.yaml",
            prompt_entry="codecontests_proposer_critic",
            trained_speakers=["alice"],
            frozen_models={"bob": object(), "judge": object()},
            judge=JudgeConfig(schema_name="collaborative"),
            fresh_positions=True,
        ),
        env,
        CodeContestsFamily(),
    )
    rendered = debate.prompts.instruction(
        "proposal", "alice", {"TOPIC": env.test_rows[0]["problem"]}
    )
    assert rendered == PROPOSAL_INSTRUCTION_BEFORE_SINGLE_SOURCING


def test_debate_raises_when_task_source_supplies_no_format_var(env, monkeypatch):
    """No silent fallback: if the task config stops supplying
    ANSWER_FORMAT_INSTRUCTION, DebateEnv fails at construction rather than
    letting a run reach generation and die on an unbound placeholder."""
    from infra.envs.debate.env import DebateEnv, DebateEnvConfig
    from infra.envs.debate.judge import JudgeConfig

    monkeypatch.setattr(type(env.prompts), "prompt_vars", lambda self: {})
    with pytest.raises(ValueError) as exc:
        DebateEnv(
            DebateEnvConfig(
                topology=_pc_topology(),
                prompt_file="infra/envs/debate/prompt_configs/codecontests.yaml",
                prompt_entry="codecontests_proposer_critic",
                trained_speakers=["alice"],
                frozen_models={"bob": object(), "judge": object()},
                judge=JudgeConfig(schema_name="collaborative"),
                fresh_positions=True,
            ),
            env,
            CodeContestsFamily(),
        )
    assert "ANSWER_FORMAT_INSTRUCTION" in str(exc.value)


def test_source_accepts_prompt_file_key(tmp_path):
    alt = tmp_path / "alt.yaml"
    alt.write_text("answer_gen_system: |-\n  Terse.\nanswer_gen_user: |-\n  Q: <PROBLEM>\n")
    path = _write_jsonl(tmp_path / "train.jsonl", ROWS)
    env = CodeContestsFamily().source(
        {"path": path, "test_path": path, "prompt_file": str(alt)}
    )
    assert env.tasks(1, split="test")[0].messages[0]["content"] == "Terse."


# ---------------------------------------------------------------- extraction


def test_extract_closed_python_fence():
    assert extract_code("blah\n```python\nprint(1)\n```\ndone") == "print(1)"


def test_extract_generic_fence():
    assert extract_code("```\nprint(2)\n```") == "print(2)"


def test_extract_takes_last_fence():
    text = "```python\nprint(1)\n```\nactually:\n```python\nprint(2)\n```"
    assert extract_code(text) == "print(2)"


def test_unclosed_fence_only_when_relaxed():
    text = "here it is:\n```python\nprint(1)\nprint(2)"
    assert extract_code(text, relaxed=True) == "print(1)\nprint(2)"
    assert extract_code(text, relaxed=False) is None


def test_extract_none_without_code():
    assert extract_code("no code here at all") is None
    assert extract_code("") is None
    assert extract_code("```python\n```") is None


# ------------------------------------------------------------------ verifier


def test_correct_solution_passes_all_cases():
    r = run_stdin_tests(SUM_SOLUTION, ["1 2", "3 4"], ["3", "7"], timeout=TIMEOUT)
    assert r["passed"] and r["tests_passed"] == 2 and r["tests_total"] == 2


def test_wrong_solution_reports_first_failure():
    r = run_stdin_tests("print(0)", ["1 2", "3 4"], ["3", "7"], timeout=TIMEOUT)
    assert not r["passed"] and r["status"] == "failed"
    assert r["first_failure"]["test_idx"] == 0
    assert r["first_failure"]["expected"] == "3" and r["first_failure"]["actual"] == "0"


def test_syntax_error_solution_is_an_error():
    r = run_stdin_tests("def (:", ["1 2"], ["3"], timeout=TIMEOUT)
    assert r["status"] == "error" and not r["passed"]
    assert "SyntaxError" in r["first_failure"]["stderr"]


def test_float_normalization():
    r = run_stdin_tests("print(0.5)", ["x"], ["0.5000"], timeout=TIMEOUT)
    assert r["passed"]


def test_runtime_error_is_a_failure_not_a_crash():
    r = run_stdin_tests("raise ValueError('boom')", ["x"], ["1"], timeout=TIMEOUT)
    assert not r["passed"]
    assert "ValueError" in r["first_failure"]["stderr"]


# -------------------------------------------------------------------- reward


def test_reward_correct_solution(env):
    task = env.tasks(1, split="test")[0]  # the sum problem
    reward, info = env.reward(task, f"sure:\n```python\n{SUM_SOLUTION}\n```")
    assert reward == pytest.approx(1.1)
    assert info["correct"] == 1.0 and info["has_code"] == 1.0
    assert info["tests_passed_frac"] == pytest.approx(1.0)


def test_reward_unfenced_text(env):
    task = env.tasks(1, split="test")[0]
    reward, info = env.reward(task, "I think the answer is a + b.")
    assert reward == 0.0 and info["has_code"] == 0.0 and info["correct"] == 0.0


def test_reward_fenced_wrong_code(env):
    task = env.tasks(1, split="test")[0]
    reward, info = env.reward(task, "```python\nprint(0)\n```")
    assert reward == pytest.approx(0.1)
    assert info["correct"] == 0.0 and info["tests_passed_frac"] == 0.0


def test_reward_cpp_code_gets_format_credit_only(env):
    task = env.tasks(1, split="test")[0]
    reward, info = env.reward(task, "```\n#include <iostream>\nint main(){ std::cout<<1; }\n```")
    assert reward == pytest.approx(0.1)
    assert info["correct"] == 0.0 and info["cpp_code"] == 1.0


# -------------------------------------------------------------------- family


@pytest.fixture
def family():
    return CodeContestsFamily(timeout_seconds=TIMEOUT)


def _meta():
    return {"inputs": ["1 2", "3 4"], "outputs": ["3", "7"]}


def test_grade_passing_solution(family):
    assert family.grade(_meta(), SUM_SOLUTION) is True


def test_grade_failing_solution(family):
    assert family.grade(_meta(), "print(0)") is False


def test_grade_none_solution(family):
    assert family.grade(_meta(), None) is None


def test_grade_without_test_cases(family):
    assert family.grade({"inputs": [], "outputs": []}, SUM_SOLUTION) is None


def test_grade_cpp_solution(family):
    assert family.grade(_meta(), "#include <iostream>\nint main(){ std::cout<<1; }") is False


def test_extractor_relaxed_knob(family):
    text = "```python\nprint(1)"
    assert family.extractor(relaxed=True)(text) == "print(1)"
    assert family.extractor(relaxed=False)(text) is None


def test_format_flags(family):
    assert family.format_flags("```python\nprint(1)\n```") == {"code_fence": 1.0}
    assert family.format_flags("just prose") == {"code_fence": 0.0}
    assert family.format_flags("```python\nprint(1)") == {"code_fence": 0.0}


def test_source_requires_path(family):
    with pytest.raises(ValueError, match="dataset.path"):
        family.source({})


def test_source_rejects_unknown_keys(family, tmp_path):
    path = _write_jsonl(tmp_path / "train.jsonl", ROWS)
    with pytest.raises(ValueError, match="unknown"):
        family.source({"path": path, "levels": 5})


def test_source_builds_env(family, tmp_path):
    path = _write_jsonl(tmp_path / "train.jsonl", ROWS)
    test = _write_jsonl(tmp_path / "test.jsonl", ROWS)
    env = family.source({"path": path, "test_path": test, "timeout_seconds": TIMEOUT})
    assert isinstance(env, CodeContestsEnv)
    assert len(env.train_rows) == 2 and family.timeout_seconds == TIMEOUT


def test_family_is_registered():
    assert isinstance(get_family("codecontests"), CodeContestsFamily)
