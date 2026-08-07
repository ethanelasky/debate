"""CodeContests task family: loading/filtering, code extraction, the stdin
verifier (real subprocesses), reward, and grading."""

from __future__ import annotations

import json
import logging
import runpy
from pathlib import Path

import pytest

from infra.config import resolve_experiments_from_file
from infra.envs.tasks import get_family
from infra.envs.tasks.codecontests import (
    CodeContestsEnv,
    CodeContestsFamily,
    extract_code,
    is_cpp_code,
    run_stdin_tests,
)

SUM_SOLUTION = "a, b = map(int, input().split())\nprint(a + b)"
ECHO_SOLUTION = "print(input().strip())"
TIMEOUT = 15

# Built-dataset schema: two disjoint suites per row. `multi` and
# `unverifiable` are dropped at BUILD time now (multi-answer phrasing, no
# tests), so the loader only has to reject a missing/ragged reward suite.
ROWS = [
    {
        "name": "sum",
        "problem": "Read two integers on one line and print their sum.",
        "rlvr_inputs": ["1 2", "3 4"],
        "rlvr_outputs": ["3", "7"],
        "truth_inputs": ["5 6"],
        "truth_outputs": ["11"],
    },
    {
        "name": "echo",
        "problem": "Read one line and print it back.",
        "rlvr_inputs": ["hello", "world"],
        "rlvr_outputs": ["hello", "world"],
        "truth_inputs": ["again"],
        "truth_outputs": ["again"],
    },
    {
        "name": "no_truth",
        "problem": "Read one line and print it back.",
        "rlvr_inputs": ["only"],
        "rlvr_outputs": ["only"],
        "truth_inputs": [],
        "truth_outputs": [],
    },
    {
        "name": "no_rlvr",
        "problem": "Compute something interesting.",
        "rlvr_inputs": [],
        "rlvr_outputs": [],
        "truth_inputs": [],
        "truth_outputs": [],
    },
]


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def _build_paired_rows(gdm_rows, cco_rows):
    repo = Path(__file__).resolve().parents[1]
    builder = runpy.run_path(str(repo / "scripts" / "build_codecontests_paired_eval.py"))
    return builder["build_paired_rows"](gdm_rows, cco_rows)


@pytest.fixture
def env(tmp_path):
    train = _write_jsonl(tmp_path / "train.jsonl", ROWS)
    test = _write_jsonl(tmp_path / "test.jsonl", ROWS)
    return CodeContestsEnv(path=train, test_path=test, timeout_seconds=TIMEOUT)


# ------------------------------------------------------------------- loading


def test_loader_keeps_rows_with_a_reward_suite(env):
    """Eligibility (multi-answer, size caps, stdin/stdout) is applied by the
    BUILD script, so the loader's only job is structural: keep rows that have
    a usable rlvr suite. `no_truth` is kept — an empty truth suite is normal
    (~66% of the real dataset) and just means grade() returns None there."""
    assert [r["name"] for r in env.train_rows] == ["sum", "echo", "no_truth"]
    assert [r["name"] for r in env.test_rows] == ["sum", "echo", "no_truth"]


def test_split_off_test_rows_when_no_test_path(tmp_path):
    path = _write_jsonl(tmp_path / "train.jsonl", ROWS * 4)
    env = CodeContestsEnv(path=path, timeout_seconds=TIMEOUT)
    assert len(env.train_rows) >= 2 and env.test_rows
    assert len(env.train_rows) + len(env.test_rows) == 12


def test_too_few_rows_raises(tmp_path):
    path = _write_jsonl(tmp_path / "train.jsonl", ROWS[3:])  # only the no-rlvr row
    with pytest.raises(RuntimeError):
        CodeContestsEnv(path=path, timeout_seconds=TIMEOUT)


def test_len_mismatch_row_dropped_and_counted(tmp_path, caplog):
    rows = ROWS[:2] + [
        {
            "name": "mismatch",
            "problem": "Read one line and print it back twice.",
            "rlvr_inputs": ["a", "b"],
            "rlvr_outputs": ["a"],  # zip would silently truncate to one case
            "truth_inputs": [],
            "truth_outputs": [],
        }
    ]
    path = _write_jsonl(tmp_path / "train.jsonl", rows)
    with caplog.at_level(logging.INFO, logger="infra.envs.tasks.codecontests"):
        env = CodeContestsEnv(path=path, test_path=path, timeout_seconds=TIMEOUT)
    assert [r["name"] for r in env.train_rows] == ["sum", "echo"]
    messages = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "len_mismatch=1" in messages and "kept=2" in messages


def test_train_test_overlap_warns_but_keeps_rows(tmp_path, caplog):
    path = _write_jsonl(tmp_path / "train.jsonl", ROWS)
    with caplog.at_level(logging.WARNING, logger="infra.envs.tasks.codecontests"):
        env = CodeContestsEnv(path=path, test_path=path, timeout_seconds=TIMEOUT)
    warning = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning and "3 problem name(s)" in warning[0].getMessage()
    assert len(env.train_rows) == 3 and len(env.test_rows) == 3  # nothing dropped


def test_cf_rating_filter_is_inclusive_and_excludes_unrated(tmp_path):
    rows = []
    for rating in (0, 800, 1000, 1100, 1200):
        row = {**ROWS[0], "name": f"rated-{rating}", "cf_rating": rating}
        rows.append(row)
    path = _write_jsonl(tmp_path / "rated.jsonl", rows)
    env = CodeContestsEnv(
        path=path,
        test_path=path,
        timeout_seconds=TIMEOUT,
        min_cf_rating=800,
        max_cf_rating=1100,
    )
    assert [row["cf_rating"] for row in env.train_rows] == [800, 1000, 1100]
    assert [row["cf_rating"] for row in env.test_rows] == [800, 1000, 1100]


def test_cf_rating_filter_rejects_inverted_bounds(tmp_path):
    path = _write_jsonl(tmp_path / "train.jsonl", ROWS)
    with pytest.raises(ValueError, match="min_cf_rating must be <= max_cf_rating"):
        CodeContestsEnv(
            path=path,
            test_path=path,
            min_cf_rating=1200,
            max_cf_rating=800,
        )


def test_tasks_carry_meta_and_hide_test_io(env):
    tasks = env.tasks(3, split="train") + env.tasks(2, split="test")
    for t in tasks:
        assert t.meta["question"] and t.meta["rlvr_inputs"] and t.meta["rlvr_outputs"]
        prompt = "\n".join(m["content"] for m in t.messages)
        for case in t.meta["rlvr_inputs"] + t.meta["rlvr_outputs"] + t.meta["truth_inputs"] + t.meta["truth_outputs"]:
            assert case not in prompt  # verifier cases must never be prompted
    assert [t.meta["name"] for t in env.tasks(2, split="test")] == ["sum", "echo"]


# ------------------------------------------------------- generation prompts
#
# Pinned against the OLD repo's `answer_generation_user` composition (the
# "Context:" frame + <question> block + `judge_answer_generation_user` Notes +
# "Provide your answer:" cue), which is the shape both arms now render. Under
# first_speech_non_debate_aware these bytes ARE the debate proposal context.

EXPECTED_SYSTEM = (
    "Solve the programming problem.\n"
    "Your goal is to write code that is consistent with the specification in "
    "order to pass the hidden test cases. Do not include functionality or "
    "error handling beyond what is described."
)

# The bullets live in the user message now, exactly as in the old pack.
EXPECTED_USER = (
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


def test_rlvr_prompt_matches_old_repo_composition(env):
    task = env.tasks(1, split="test")[0]
    assert task.meta["question"] == "Read two integers on one line and print their sum."
    assert task.messages == [
        {"role": "system", "content": EXPECTED_SYSTEM},
        {"role": "user", "content": EXPECTED_USER},
    ]


def test_prompt_file_override(tmp_path):
    alt = tmp_path / "alt.yaml"
    alt.write_text(
        "messages:\n"
        "  - role: system\n"
        "    content: Terse.\n"
        "  - role: user\n"
        "    name: ANSWER_GEN_USER\n"
        "    content: 'Q: <PROBLEM>'\n"
    )
    path = _write_jsonl(tmp_path / "train.jsonl", ROWS)
    env = CodeContestsEnv(
        path=path, test_path=path, timeout_seconds=TIMEOUT, prompt_file=str(alt)
    )
    task = env.tasks(1, split="test")[0]
    assert task.messages == [
        {"role": "system", "content": "Terse."},
        {"role": "user", "content": "Q: " + task.meta["question"]},
    ]


def _pc_protocol():
    from infra.envs.debate.protocol import Protocol

    return Protocol.parse(
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


def _debate_env(env):
    from infra.envs.debate.env import DebateEnv, DebateEnvConfig
    from infra.envs.debate.judge import JudgeConfig

    return DebateEnv(
        DebateEnvConfig(
            protocol=_pc_protocol(),
            prompt_file="infra/prompts/debate/codecontests.yaml",
            prompt_entry="codecontests_proposer_critic",
            trained_speakers=["alice"],
            frozen_models={"bob": object(), "judge": object()},
            judge=JudgeConfig(schema_name="collaborative"),
            fresh_positions=True,
        ),
        env,
        CodeContestsFamily(),
    )


def test_debate_proposal_slot_equals_rlvr_user_message(env):
    """The invariant the single-sourcing exists to guarantee: the rendered
    debate proposal slot and the RLVR user message are the SAME composition,
    byte for byte, for the same problem. Both are answer_gen_user; the debate
    side just binds the problem through <TOPIC> instead of <PROBLEM>."""
    task = env.tasks(1, split="test")[0]
    rendered = _debate_env(env).prompts.instruction(
        "proposal", "alice", {"TOPIC": task.meta["question"]}
    )
    assert rendered == task.messages[1]["content"]
    assert rendered == EXPECTED_USER


def test_debate_raises_when_task_source_supplies_no_template(env, monkeypatch):
    """No silent fallback: if the task config stops supplying ANSWER_GEN_USER,
    DebateEnv fails at construction rather than letting a run reach generation
    and die on an unbound placeholder."""
    monkeypatch.setattr(type(env.prompts), "supplied_templates", lambda self: {})
    with pytest.raises(ValueError) as exc:
        _debate_env(env)
    assert "ANSWER_GEN_USER" in str(exc.value)


def test_source_accepts_prompt_file_key(tmp_path):
    alt = tmp_path / "alt.yaml"
    alt.write_text(
        "messages:\n"
        "  - role: system\n"
        "    content: Terse.\n"
        "  - role: user\n"
        "    name: ANSWER_GEN_USER\n"
        "    content: 'Q: <PROBLEM>'\n"
    )
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


# ------------------------------------------------------------- C++ detection


def test_is_cpp_code_requires_strong_anchor():
    # plain Python with C++-ish identifiers must NOT be classified as C++
    # (it would be graded incorrect without ever being executed)
    assert not is_cpp_code("vector = [0]*n\nnull = None\nprint(min(vector))")
    # two weak-pattern hits (`vector <`, case-insensitive NULL) used to clear
    # the threshold; without a strong anchor they must not
    assert not is_cpp_code("if vector < n:\n    null = None\nprint(vector)")
    assert is_cpp_code(
        "#include <bits/stdc++.h>\nusing namespace std;\n"
        "int main(){int a;cin>>a;cout<<a;}"
    )


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


# Forges a result JSON as the LAST line of the runner's real stdout (atexit
# fires after the runner restores sys.stdout). With the old parse-last-stdout
# scheme this flipped the verdict to passed; the file side-channel must grade
# by the real test outcomes.
FORGED_STDOUT_SOLUTION = (
    "import atexit, json\n"
    "atexit.register(lambda: print(json.dumps({\n"
    "    'status': 'passed', 'passed': True, 'tests_passed': 2,\n"
    "    'tests_total': 2, 'timeout': False, 'first_failure': None,\n"
    "})))\n"
    "print(0)\n"
)


def test_forged_stdout_verdict_is_ignored():
    r = run_stdin_tests(FORGED_STDOUT_SOLUTION, ["1 2", "3 4"], ["3", "7"], timeout=TIMEOUT)
    assert r["passed"] is False and r["status"] == "failed"
    assert r["tests_passed"] == 0 and r["tests_total"] == 2
    fam = CodeContestsFamily(timeout_seconds=TIMEOUT)
    assert fam.grade({"truth_inputs": ["1 2"], "truth_outputs": ["3"]}, FORGED_STDOUT_SOLUTION) is False


def test_timeout_is_reported_and_graded_false():
    r = run_stdin_tests("while True: pass", ["1"], ["1"], timeout=2)
    assert not r["passed"] and r["timeout"] is True and r["status"] == "timeout"
    fam = CodeContestsFamily(timeout_seconds=2)
    assert fam.grade({"truth_inputs": ["1"], "truth_outputs": ["1"]}, "while True: pass") is False


# -------------------------------------------------------------------- reward


def test_reward_correct_solution(env):
    task = env.tasks(1, split="test")[0]  # the sum problem
    reward, info = env.reward(task, f"sure:\n```python\n{SUM_SOLUTION}\n```")
    assert reward == pytest.approx(1.1)
    assert info["correct"] == 1.0 and info["has_code"] == 1.0
    assert info["tests_passed_frac"] == pytest.approx(1.0)
    assert info["gdm_correct"] == info["correct"]
    assert info["gdm_tests_passed_frac"] == info["tests_passed_frac"]


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
    return {"truth_inputs": ["1 2", "3 4"], "truth_outputs": ["3", "7"]}


def test_grade_passing_solution(family):
    assert family.grade(_meta(), SUM_SOLUTION) is True


def test_grade_failing_solution(family):
    assert family.grade(_meta(), "print(0)") is False


def test_grade_none_solution(family):
    assert family.grade(_meta(), None) is None


def test_grade_without_test_cases(family):
    assert family.grade({"truth_inputs": [], "truth_outputs": []}, SUM_SOLUTION) is None


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
    assert len(env.train_rows) == 3 and family.timeout_seconds == TIMEOUT


def test_source_passes_cf_rating_bounds(family, tmp_path):
    rows = [
        {**ROWS[0], "name": "unrated", "cf_rating": 0},
        {**ROWS[1], "name": "easy", "cf_rating": 900},
        {**ROWS[0], "name": "easy-2", "cf_rating": 1100},
        {**ROWS[2], "name": "hard", "cf_rating": 1200},
    ]
    path = _write_jsonl(tmp_path / "rated.jsonl", rows)
    env = family.source(
        {
            "path": path,
            "test_path": path,
            "min_cf_rating": 800,
            "max_cf_rating": 1100,
        }
    )
    assert [row["name"] for row in env.train_rows] == ["easy", "easy-2"]


def test_family_is_registered():
    assert isinstance(get_family("codecontests"), CodeContestsFamily)


# --------------------------------------------------------- soft token budget


def test_flat_overshoot_penalty_is_constant_not_ramped(tmp_path):
    """A sample past the budget loses a CONSTANT amount, however far over it
    went. The old repo ramped this between a soft budget and a hard limit;
    flat is the deliberate change (Frank, 2026-08-04)."""
    from infra.backend.base import Sample

    path = _write_jsonl(tmp_path / "train.jsonl", ROWS)
    env = CodeContestsEnv(path=path, test_path=path, timeout_seconds=TIMEOUT,
                          soft_token_budget=10, overshoot_penalty=0.25)

    class FakePolicy:
        """Two samples: one just over the budget, one far over."""
        def predict(self, messages, n):
            def mk(k):
                return Sample(
                    tokens=[0] * k, logprobs=[0.0] * k,
                    text="```python\n" + SUM_SOLUTION + "\n```",
                    stop_reason="stop", prompt_tokens=[0],
                )
            return [[mk(11), mk(500)] for _ in messages]

    groups = env.rollout(env.tasks(1, split="train"), FakePolicy(), group_size=2)
    rewards = sorted(t.reward for t in groups[0])
    assert len(set(rewards)) == 1, f"penalty scaled with length: {rewards}"
    assert all(t.info["over_budget"] == 1.0 for t in groups[0])

    env.soft_token_budget = None
    unpenalised = sorted(t.reward for t in env.rollout(
        env.tasks(1, split="train"), FakePolicy(), group_size=2)[0])
    assert unpenalised[0] - rewards[0] == pytest.approx(0.25)


# --------------------------------------------------------- paired eval suite

# Self-contained rows: the GDM and CCO suites are deliberately disjoint. A
# runtime sidecar join is neither needed nor accepted.
PAIRED_ROWS = [
    {
        "name": "sum",
        "problem": ROWS[0]["problem"],
        "gdm_inputs": ["1 2"],
        "gdm_outputs": ["3"],
        "cco_inputs": ["10 20", "0 0"],
        "cco_outputs": ["30", "0"],
        "cf_rating": 900,
    },
    {
        "name": "echo",
        "problem": ROWS[1]["problem"],
        "gdm_inputs": ["hello"],
        "gdm_outputs": ["hello"],
        "cco_inputs": ["zzz"],
        "cco_outputs": ["zzz"],
        "cf_rating": 1000,
    },
]


def _paired_env(tmp_path, paired_rows=PAIRED_ROWS):
    train = _write_jsonl(tmp_path / "train.jsonl", ROWS)
    paired = _write_jsonl(tmp_path / "paired.jsonl", paired_rows)
    return CodeContestsEnv(
        path=train, paired_test_path=paired, timeout_seconds=TIMEOUT
    )


def test_paired_artifact_defines_the_eval_pool(tmp_path):
    env = _paired_env(tmp_path)
    assert [r["name"] for r in env.test_rows] == ["sum", "echo"]
    assert {r["name"] for r in env.train_rows} >= {"sum", "echo", "no_truth"}
    assert all(r["gdm_inputs"] and r["cco_inputs"] for r in env.test_rows)


def test_paired_builder_preserves_gdm_order_and_stable_metadata():
    gdm = [
        {
            **ROWS[1],
            "cf_contest_id": 2,
            "cf_index": "B",
            "cf_rating": 1000,
            "difficulty": 3,
            "source": 2,
        },
        {
            **ROWS[0],
            "cf_contest_id": 1,
            "cf_index": "A",
            "cf_rating": 900,
            "difficulty": 2,
            "source": 2,
        },
    ]
    cco = [
        {"name": "sum", "cco_inputs": ["10 20"], "cco_outputs": ["30"]},
        {"name": "echo", "cco_inputs": ["zzz"], "cco_outputs": ["zzz"]},
    ]
    rows = _build_paired_rows(gdm, cco)
    assert [row["name"] for row in rows] == ["echo", "sum"]
    assert rows[0]["gdm_inputs"] == ROWS[1]["rlvr_inputs"]
    assert rows[0]["cco_inputs"] == ["zzz"]
    assert rows[0]["cf_contest_id"] == 2 and rows[0]["cf_index"] == "B"


def test_paired_builder_rejects_ragged_or_duplicate_sources():
    with pytest.raises(ValueError, match="malformed CCO"):
        _build_paired_rows(
            ROWS[:1],
            [{"name": "sum", "cco_inputs": ["x"], "cco_outputs": []}],
        )
    with pytest.raises(ValueError, match="duplicate CCO"):
        _build_paired_rows(
            ROWS[:1],
            [
                {"name": "sum", "cco_inputs": ["x"], "cco_outputs": ["y"]},
                {"name": "sum", "cco_inputs": ["a"], "cco_outputs": ["b"]},
            ],
        )


def test_same_heldout_completion_reports_both_named_suites(tmp_path):
    from infra.envs.base import Trajectory
    from infra.train import _aggregate

    env = _paired_env(tmp_path)
    test_task = next(t for t in env.tasks(2, split="test") if t.meta["name"] == "sum")
    reward, info = env.reward(test_task, f"```python\n{SUM_SOLUTION}\n```")
    assert info["gdm_correct"] == info["correct"] == 1.0
    assert info["gdm_tests_passed_frac"] == info["tests_passed_frac"] == 1.0
    assert info["cco_correct"] == 1.0
    assert info["cco_tests_passed_frac"] == 1.0
    logged = _aggregate([Trajectory(datums=[], reward=reward, info=info)], "eval")
    assert logged["eval/gdm_correct"] == 1.0
    assert logged["eval/gdm_tests_passed_frac"] == 1.0
    assert logged["eval/cco_correct"] == 1.0
    assert logged["eval/cco_tests_passed_frac"] == 1.0

    # Train tasks carry no CCO suite and retain the GDM-only reward path.
    train_task = env.tasks(1, split="train")[0]
    _, train_info = env.reward(train_task, f"```python\n{SUM_SOLUTION}\n```")
    assert "cco_correct" not in train_info


def test_cco_verdict_does_not_change_the_gdm_reward(tmp_path):
    """The second suite is a MEASUREMENT. If it leaked into reward the RLVR arm
    would be training on CCO, and the two curves would stop being independent."""
    env = _paired_env(tmp_path)
    # hardcodes the one rlvr case; passes it, fails the unseen CCO case
    cheat = "print(3)"
    reward, info = env.reward(env.tasks(1, split="test")[0], f"```python\n{cheat}\n```")
    assert reward == pytest.approx(1.1)
    assert info["gdm_correct"] == info["correct"] == 1.0
    assert info["cco_correct"] == 0.0


def test_paired_loader_drops_rows_missing_either_suite(tmp_path):
    malformed = [
        {**PAIRED_ROWS[0], "cco_outputs": []},
        {**PAIRED_ROWS[1], "gdm_inputs": []},
    ]
    train = _write_jsonl(tmp_path / "train.jsonl", ROWS)
    paired = _write_jsonl(tmp_path / "paired.jsonl", malformed)
    with pytest.raises(RuntimeError, match="test=0"):
        CodeContestsEnv(path=train, paired_test_path=paired, timeout_seconds=TIMEOUT)


def test_runtime_rejects_paired_artifact_plus_side_test_file(tmp_path):
    train = _write_jsonl(tmp_path / "train.jsonl", ROWS)
    paired = _write_jsonl(tmp_path / "paired.jsonl", PAIRED_ROWS)
    with pytest.raises(ValueError, match="only one"):
        CodeContestsEnv(path=train, test_path=train, paired_test_path=paired)


def test_expected_paired_eval_size_fails_loudly(tmp_path):
    train = _write_jsonl(tmp_path / "train.jsonl", ROWS)
    paired = _write_jsonl(tmp_path / "paired.jsonl", PAIRED_ROWS)
    with pytest.raises(RuntimeError, match="expected 3, got 2"):
        CodeContestsEnv(
            path=train,
            paired_test_path=paired,
            expected_eval_size=3,
            timeout_seconds=TIMEOUT,
        )


def test_resolved_production_config_reaches_family_and_paired_env(tmp_path):
    """End-to-end regression for the exact boundary that dropped CCO before."""
    repo = Path(__file__).resolve().parents[1]
    exp = resolve_experiments_from_file(
        repo / "configs" / "codecontests_rlvr_olmo.yaml"
    )["codecontests_rlvr_olmo_easy1000_b16_seed0_50"]
    assert exp["dataset"]["eval_subset_size"] == 55
    assert exp["dataset"]["expected_eval_size"] == 55
    assert exp["training"]["eval_n"] == 55

    rated_train_rows = [{**row, "cf_rating": 900} for row in ROWS]
    train = _write_jsonl(tmp_path / "train.jsonl", rated_train_rows)
    paired = _write_jsonl(tmp_path / "paired.jsonl", PAIRED_ROWS)
    ds = dict(exp["dataset"])
    family = get_family(ds.pop("type"))
    ds.pop("relaxed_extraction", None)
    ds["path"] = train
    ds["paired_test_path"] = paired
    ds["expected_eval_size"] = 2
    env = family.source(ds)
    tasks = env.tasks(55, split="test")
    assert len(tasks) == 2
    assert all(task.meta["gdm_inputs"] and task.meta["cco_inputs"] for task in tasks)
