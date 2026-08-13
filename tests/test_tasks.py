"""Unit tests for the task-family layer (registry, math family, config guards).

Everything here is offline: MathFamily.source() is only ever called with an
invalid config, where reject_unknown_keys fires before any dataset loading.
"""

import json
from copy import deepcopy

import pytest
import yaml

from infra.envs.answer_parsing import extract_last_number, extract_number_from_boxed_answer
from infra.envs.base import Task
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file
from infra.envs.tasks import TASK_FAMILIES, get_family
from infra.envs.tasks.base import reject_unknown_keys
from infra.envs.tasks.math import MathEnv, MathFamily, _parse_levels, parse_numeric_answers


def test_registry_lookup():
    from infra.envs.tasks.monitoringbench import MonitoringBenchFamily

    assert isinstance(get_family("math"), MathFamily)
    assert isinstance(get_family("monitoringbench"), MonitoringBenchFamily)


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


def test_parse_answers_exposes_strict_and_relaxed_candidates():
    fam = MathFamily()
    for text, expected_strict, expected_relaxed in [
        ("the answer is 7", None, 7.0),
        ("\\boxed{42}", 42.0, 42.0),
        ("\\boxed{41}; correction: 42", 41.0, 41.0),
    ]:
        parsed = fam.parse_answers(text)
        assert parsed == parse_numeric_answers(text)
        assert parsed.strict == expected_strict
        assert parsed.relaxed == expected_relaxed


def test_answer_format_valid_comes_from_strict_parse():
    fam = MathFamily()
    assert fam.parse_answers("\\boxed{3}").answer_format_valid is True
    assert fam.parse_answers("\\boxed{x}").answer_format_valid is False
    assert fam.parse_answers("no box").answer_format_valid is False


def _offline_math_env(**rewards):
    env = object.__new__(MathEnv)
    env.correct_reward = float(rewards.get("correct_reward", 1.0))
    env.format_reward = float(rewards.get("format_reward", 0.1))
    env.relaxed_correct_bonus = float(rewards.get("relaxed_correct_bonus", 0.1))
    env.shaped_reward = 0.0
    return env


def test_math_reward_uses_generic_metrics_without_old_names():
    task = Task(messages=[], meta={"gt": 2.0})
    env = _offline_math_env()

    reward, info = env.reward(task, "work\n\\boxed{2}")
    assert reward == pytest.approx(1.1)
    assert info == {
        "correct_strict": 1.0,
        "correct_relaxed": 1.0,
        "answer_format_valid": 1.0,
    }
    assert not ({"correct", "has_boxed"} & set(info))


def _legacy_numeric_reward(text, gt):
    """Independent copy of the pre-migration numeric reward oracle."""
    pred_boxed = extract_number_from_boxed_answer(text)
    pred_relaxed = pred_boxed if pred_boxed is not None else extract_last_number(text)
    exact_boxed = pred_boxed is not None and abs(pred_boxed - gt) < 1e-6
    exact_relaxed = pred_relaxed is not None and abs(pred_relaxed - gt) < 1e-6
    reward = 0.1 if "\\boxed{" in text.lower() else 0.0
    reward += 1.0 if exact_boxed else 0.0
    reward += 0.1 if exact_relaxed and not exact_boxed else 0.0
    return reward


@pytest.mark.parametrize(
    "text, expected_format_valid",
    [
        ("\\boxed{oops} but the answer is 2", False),  # malformed raw box
        ("work then \\boxed{2", False),  # unclosed raw box
        ("work then \\boxed {2}", True),  # parseable, but no exact raw substring
        ("<think>\\boxed{2}</think> answer 3", False),  # hidden from parsing, not reward gate
        ("work then \\boxed{2}", True),  # normal strict
        ("work then answer 2", False),  # normal relaxed-only
    ],
)
def test_math_reward_matches_legacy_raw_box_gate(text, expected_format_valid):
    task = Task(messages=[], meta={"gt": 2.0})
    reward, info = _offline_math_env().reward(task, text)

    assert reward == pytest.approx(_legacy_numeric_reward(text, 2.0))
    assert info["answer_format_valid"] == float(expected_format_valid)
    assert set(info) == {"correct_strict", "correct_relaxed", "answer_format_valid"}


def test_math_reward_allows_zero_format_reward_without_changing_metrics():
    task = Task(messages=[], meta={"gt": 2.0})
    malformed = "\\boxed{oops} but the answer is 2"
    reward, info = _offline_math_env(format_reward=0).reward(task, malformed)
    assert reward == pytest.approx(0.1)
    assert info == {
        "correct_strict": 0.0,
        "correct_relaxed": 1.0,
        "answer_format_valid": 0.0,
    }


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


def _fake_math_dataset(problem_suffix=""):
    def rows(n, offset):
        return [
            {
                "problem": f"problem {offset + i}{problem_suffix}",
                "solution": f"\\boxed{{{offset + i}}}",
                "level": "Level 5",
            }
            for i in range(n)
        ]

    return {"train": rows(40, 0), "test": rows(10, 100)}


def test_math_protocol_identity_is_stable_complete_and_defensive(monkeypatch):
    import infra.envs.tasks.math as math_module

    monkeypatch.setattr(math_module, "_load", lambda: deepcopy(_fake_math_dataset()))
    config = {
        "levels": 5,
        "seed": 3,
        "eval_subset_size": 6,
        "correct_reward": 2,
        "format_reward": 0,
        "relaxed_correct_bonus": 0.25,
        "think_overshoot_penalty": 0.125,
    }
    first = MathFamily()
    env = first.source(config)
    second = MathFamily()
    second.source(config)

    identity = first.protocol_identity()
    assert identity == second.protocol_identity()
    assert identity["grading_protocol"] == "numeric_box_v1"
    assert identity["dataset_id"] == "the-jb/hendrycks-math"
    assert identity["dataset_revision"] == "unpinned_legacy"
    assert identity["levels"] == "5"
    assert identity["seed"] == "3"
    assert identity["eval_subset_size"] == "6"
    assert identity["train_count"] == str(len(env.train_rows))
    assert identity["dev_count"] == str(len(env.dev_rows))
    assert identity["test_count"] == str(len(env.test_rows))
    assert identity["correct_reward"] == "2.0"
    assert identity["format_reward"] == "0.0"
    assert identity["relaxed_correct_bonus"] == "0.25"
    assert identity["think_overshoot_penalty"] == "0.125"
    assert len(identity["prompt_sha256"]) == 64
    assert len(identity["split_sha256"]) == 64
    assert env.correct_reward == 2.0
    assert env.format_reward == 0.0
    assert env.relaxed_correct_bonus == 0.25
    assert env.think_overshoot_penalty == 0.125

    identity["seed"] = "mutated"
    assert first.protocol_identity()["seed"] == "3"


def test_math_protocol_identity_changes_with_config_and_cohort(monkeypatch):
    import infra.envs.tasks.math as math_module

    dataset = _fake_math_dataset()
    monkeypatch.setattr(math_module, "_load", lambda: deepcopy(dataset))
    baseline = MathFamily()
    baseline.source({"seed": 0, "eval_subset_size": 6})

    different_seed = MathFamily()
    different_seed.source({"seed": 1, "eval_subset_size": 6})
    assert different_seed.protocol_identity()["seed"] != baseline.protocol_identity()["seed"]

    dataset["train"][0]["problem"] += " changed"
    different_cohort = MathFamily()
    different_cohort.source({"seed": 0, "eval_subset_size": 6})
    assert (
        different_cohort.protocol_identity()["split_sha256"]
        != baseline.protocol_identity()["split_sha256"]
    )


@pytest.mark.parametrize(
    ("knob", "changed_value", "default_identity", "changed_identity"),
    [
        ("correct_reward", 2, "1.0", "2.0"),
        ("format_reward", 0, "0.1", "0.0"),
        ("relaxed_correct_bonus", 0.25, "0.1", "0.25"),
        ("think_overshoot_penalty", 0.125, "0.0", "0.125"),
    ],
)
def test_math_protocol_identity_tracks_each_reward_knob(
    monkeypatch, knob, changed_value, default_identity, changed_identity
):
    import infra.envs.tasks.math as math_module

    monkeypatch.setattr(math_module, "_load", lambda: deepcopy(_fake_math_dataset()))
    baseline = MathFamily()
    baseline.source({"seed": 0, "eval_subset_size": 6})
    changed = MathFamily()
    changed.source({"seed": 0, "eval_subset_size": 6, knob: changed_value})

    assert baseline.protocol_identity()[knob] == default_identity
    assert changed.protocol_identity()[knob] == changed_identity
    assert changed.protocol_identity()[knob] != baseline.protocol_identity()[knob]


# ------------------------------------------------------- generation prompts
#
# Pinned against the OLD repo's hendrycks_math.yaml `answer_generation_user`,
# which is the shape both arms now render. Under
# first_speech_non_debate_aware these bytes ARE the debate proposal context, so
# a stray edit to the YAML (reflow, smart quote, lost backslash) silently
# changes the experiment.

MATH_SYSTEM = "You are an expert mathematician."

MATH_USER = (
    "Solve this mathematics problem.\n"
    "\n"
    "<problem>What is $1 + 1$?</problem>\n"
    "\n"
    "Give your answer directly, with a concise but rigorous justification, as "
    "EXACTLY one \\boxed{...} on the last line. The boxed answer must be the "
    "very last thing you output, with nothing after it."
)


def test_math_prompt_matches_old_repo_composition():
    prompts = load_generation_prompts(resolve_prompt_file(None, "math.yaml"))
    assert prompts.render({"PROBLEM": "What is $1 + 1$?"}) == [
        {"role": "system", "content": MATH_SYSTEM},
        {"role": "user", "content": MATH_USER},
    ]


def test_math_debate_pack_splices_the_rlvr_prompt():
    """Math is single-sourced like codecontests: the debate pack references
    <ANSWER_GEN_USER> instead of restating the wording, and no longer carries
    its own ANSWER_FORMAT_INSTRUCTION var (that sentence is now part of the
    shared composition). The byte-identity of the two arms is enforced in
    test_math_debate_proposal_slot_equals_rlvr_user_message below."""
    from infra.envs.debate.prompts import load_prompt_library, slot_template
    from infra.envs.debate.protocol import Protocol

    lib = load_prompt_library(
        "infra/prompts/debate/hendrycks_math.yaml",
        "math_proposer_critic",
        Protocol.parse(
            {
                "turns": [
                    {"alice": [{"name": "proposal", "kind": "solution"}]},
                    {"bob": [{"name": "critique"}]},
                    {"judge": [{"name": "verdict", "kind": "decision"}]},
                ]
            }
        ),
    )
    assert slot_template(lib, "proposal", "alice").strip() == "<ANSWER_GEN_USER>"
    assert "ANSWER_FORMAT_INSTRUCTION" not in lib.vars


def test_math_debate_proposal_slot_equals_rlvr_user_message():
    """The invariant the single-sourcing exists to guarantee, math side: the
    rendered debate proposal slot and the RLVR user message are the SAME
    composition, byte for byte, for the same problem. Built through a real
    DebateEnv so the splice path is what gets exercised. (The codecontests
    equivalent lives in tests/test_codecontests.py.)"""
    from infra.envs.base import Task
    from infra.envs.debate.env import DebateEnv, DebateEnvConfig
    from infra.envs.debate.judge import JudgeConfig
    from infra.envs.debate.protocol import Protocol

    problem = "What is $1 + 1$?"

    class Source:
        prompts = load_generation_prompts(resolve_prompt_file(None, "math.yaml"))

        def tasks(self, n, split="train"):
            return [
                Task(
                    messages=self.prompts.render({"PROBLEM": problem}),
                    meta={"question": problem},
                )
            ] * n

    debate = DebateEnv(
        DebateEnvConfig(
            protocol=Protocol.parse(
                {
                    "turns": [
                        {"alice": [{"name": "proposal", "kind": "solution"}]},
                        {"bob": [{"name": "critique"}]},
                        {"alice": [{"name": "defense"}]},
                        {"judge": [{"name": "verdict", "kind": "decision"}]},
                    ]
                }
            ),
            prompt_file="infra/prompts/debate/hendrycks_math.yaml",
            prompt_entry="math_proposer_critic",
            trained_speakers=["alice"],
            frozen_models={"bob": object(), "judge": object()},
            judge=JudgeConfig(),
            fresh_positions=True,
        ),
        Source(),
        MathFamily(),
    )
    task = Source().tasks(1)[0]
    rendered = debate.prompts.instruction("proposal", "alice", {"TOPIC": problem})
    assert rendered == task.messages[1]["content"]
    assert rendered == MATH_USER


def _messages_body(system="SYS", user="<PROBLEM>\n\nGo."):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "name": "ANSWER_GEN_USER", "content": user},
        ]
    }


def _write_prompt_yaml(path, system="SYS", user="<PROBLEM>\n\nGo."):
    path.write_text(yaml.safe_dump(_messages_body(system, user), sort_keys=False))
    return path


def test_prompt_file_override(tmp_path):
    path = _write_prompt_yaml(tmp_path / "alt.yaml", system="Be terse.", user="Q: <PROBLEM>")
    prompts = load_generation_prompts(resolve_prompt_file(str(path), "math.yaml"))
    assert prompts.render({"PROBLEM": "2+2"}) == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Q: 2+2"},
    ]


def test_prompt_file_relative_path_resolves_against_repo_root():
    prompts = load_generation_prompts(
        resolve_prompt_file("infra/prompts/tasks/math.yaml", "codecontests.yaml")
    )
    assert prompts.messages[0] == {"role": "system", "content": MATH_SYSTEM}


def test_format_notes_substituted_into_every_message(tmp_path):
    path = tmp_path / "notes.yaml"
    path.write_text(
        "format_notes: |-\n  - be brief\n"
        "messages:\n"
        "  - role: system\n    content: |-\n      Do it.\n      <FORMAT_NOTES>\n"
        "  - role: user\n    content: |-\n      <PROBLEM>\n\n      Notes:\n      <FORMAT_NOTES>\n"
    )
    prompts = load_generation_prompts(path)
    assert prompts.messages[0]["content"] == "Do it.\n- be brief"
    assert prompts.messages[1]["content"] == "<PROBLEM>\n\nNotes:\n- be brief"


@pytest.mark.parametrize(
    "body, needle",
    [
        ({}, "messages"),                                                       # missing key
        ({"messages": []}, "non-empty list"),
        ({"messages": [{"role": "user", "content": "no placeholder"}]}, "<PROBLEM>"),
        ({"messages": [{"role": "user", "content": "<PROBLEM>"}], "extra": 1}, "extra"),
        ({"messages": [{"role": "user", "content": 3}]}, "must be a string"),
        ({"messages": [{"role": "tool", "content": "<PROBLEM>"}]}, "role"),
        ({"messages": [{"content": "<PROBLEM>"}]}, "role/content"),
        (
            {"messages": [{"role": "user", "content": "<PROBLEM>", "name": "NOT_REGISTERED"}]},
            "registered splice name",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "<PROBLEM>", "name": "ANSWER_GEN_USER"},
                    {"role": "user", "content": "x", "name": "ANSWER_GEN_USER"},
                ]
            },
            "duplicate",
        ),
        (
            {"messages": [{"role": "user", "content": "<PROBLEM>\n<FORMAT_NOTES>"}]},
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


# --------------------------------------------------- family task conformance
# Every family's tasks must carry meta["question"] (the TaskFamily contract in
# infra/envs/tasks/base.py): DebateEnv binds it as the debate TOPIC and raises
# if the key is ABSENT. An explicit "" is legal (env.py _build_state) and is
# exactly monitoringbench's shape — MB prompts never render <TOPIC>.


@pytest.mark.parametrize("name", sorted(TASK_FAMILIES))
def test_family_tasks_carry_question(name, tmp_path):
    if name == "codecontests":
        rows = [
            {
                "name": f"p{i}",
                "problem": f"Read one line and print {i}.",
                "rlvr_inputs": ["x"],
                "rlvr_outputs": [str(i)],
                "truth_inputs": ["y"],
                "truth_outputs": [str(i)],
            }
            for i in range(4)
        ]
        path = tmp_path / "cc.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        env = get_family(name).source(
            {"path": str(path), "test_path": str(path), "timeout_seconds": 5}
        )
        tasks = env.tasks(2, split="train") + env.tasks(2, split="test")
    elif name == "math":
        # MathEnv.__init__ downloads the dataset, which these offline tests
        # never do (module docstring); exercise task construction directly.
        from infra.envs.tasks.math import PROMPT_FILE, MathEnv

        env = object.__new__(MathEnv)  # skip __init__ (dataset download)
        env.prompts = load_generation_prompts(resolve_prompt_file(None, PROMPT_FILE))
        row = {"problem": "What is 1+1?", "gt": 2.0, "level": 5}
        tasks = [MathEnv._task(env, row, "train")]
    elif name == "math_symbolic":
        # Skip both the pinned dataset load and Math-Verify worker startup;
        # this generic contract only exercises task construction.
        from infra.envs.tasks.math_symbolic import PROMPT_FILE, SymbolicMathEnv

        env = object.__new__(SymbolicMathEnv)
        env.prompts = load_generation_prompts(resolve_prompt_file(None, PROMPT_FILE))
        row = {
            "problem": "What is x+x?",
            "gt": "2x",
            "category": "Algebra",
            "level": 5,
            "row_id": "synthetic-row",
        }
        tasks = [SymbolicMathEnv._task(env, row, "train")]
    elif name == "aime":
        # Same shape as the math recipe: skip __init__ (dataset download),
        # exercise task construction directly.
        from infra.envs.tasks.aime import PROMPT_FILE, AimeEnv

        env = object.__new__(AimeEnv)  # skip __init__ (dataset download)
        env.prompts = load_generation_prompts(resolve_prompt_file(None, PROMPT_FILE))
        row = {"problem": "What is 1+1?", "gt": 2.0, "id": "1983-1"}
        tasks = [AimeEnv._task(env, row, "train")]
    elif name == "monitoringbench":
        # SYNTHETIC rows only — the real data files are never read in tests.
        rows = [
            {
                "id": f"r{i}",
                "label": "attack" if i % 2 else "honest",
                "steps": [{"action": "ls", "responses": "ok"}],
            }
            for i in range(4)
        ]
        path = tmp_path / "mb.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        env = get_family(name).source({"files": [str(path)], "seed": 0, "test_size": 1})
        tasks = env.tasks(2, split="train") + env.tasks(1, split="test")
    else:
        pytest.fail(f"no offline conformance recipe for task family {name!r}; add one here")
    for task in tasks:
        question = task.meta.get("question")
        assert isinstance(question, str), (
            f"{name}: Task.meta['question'] missing (TaskFamily contract)"
        )
        if name != "monitoringbench":  # MB's question is an explicit ""
            assert question.strip(), f"{name}: Task.meta['question'] empty"


def test_grade_batch_default_is_positional_grade():
    fam = MathFamily()
    grades = fam.grade_batch([({"gt": 2.0}, 2.0), ({"gt": 2.0}, 3.0), ({}, 2.0)])
    assert grades == [True, False, None]
    assert fam.last_grade_errors == 0
    assert fam.grade_batch([]) == []


def test_grade_batch_serial_path_matches_pooled():
    fam = MathFamily()
    fam.grade_workers = 1
    assert fam.grade_batch([({"gt": 2.0}, 2.0), ({"gt": 2.0}, 3.0)]) == [True, False]


def test_grade_batch_failure_grades_none_and_is_counted():
    class ExplodingFamily(MathFamily):
        def grade(self, meta, solution):
            if meta.get("boom"):
                raise RuntimeError("verifier fell over")
            return super().grade(meta, solution)

    fam = ExplodingFamily()
    grades = fam.grade_batch([({"gt": 1.0}, 1.0), ({"boom": True}, 1.0), ({"gt": 1.0}, 2.0)])
    assert grades == [True, None, False]
    assert fam.last_grade_errors == 1
    # a clean follow-up call resets the counter
    assert fam.grade_batch([({"gt": 1.0}, 1.0)]) == [True]
    assert fam.last_grade_errors == 0
