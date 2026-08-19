"""AIME family: loads the 1983-2024 pool, renders the SAME single-sourced math
answer prompt, grades integers with the shared extractor. Network-dependent
loading is exercised once via the HF cache; everything else is offline."""

from copy import deepcopy

import pytest

from infra.envs.tasks import get_family
from _dataset_tqdm_monitor import dataset_tqdm_monitor_owner


@pytest.fixture(scope="module")
def env():
    with dataset_tqdm_monitor_owner() as monitor_call:
        environment = monitor_call(get_family("aime").source, {"seed": 0})
        yield environment


def test_pool_loads_and_splits(env):
    # three-way since 2026-08-11: dev is carved from the train side
    assert len(env.train_rows) + len(env.dev_rows) + len(env.test_rows) > 900
    assert len(env.test_rows) >= 64
    assert len(env.dev_rows) >= 64
    train_ids = {r["id"] for r in env.train_rows}
    dev_ids = {r["id"] for r in env.dev_rows}
    test_ids = {r["id"] for r in env.test_rows}
    assert not train_ids & test_ids and not train_ids & dev_ids and not dev_ids & test_ids


def test_answers_are_aime_integers(env):
    for r in env.train_rows[:50] + env.test_rows[:50]:
        assert r["gt"] == int(r["gt"])
        assert 0 <= r["gt"] <= 999


def test_task_renders_math_prompt(env):
    t = env.tasks(1, split="test")[0]
    roles = [m["role"] for m in t.messages]
    assert roles == ["system", "user"]
    assert t.meta["question"] in t.messages[-1]["content"]
    assert "EXACTLY one \\boxed{...}" in t.messages[-1]["content"]


def test_split_is_seed_deterministic():
    a = get_family("aime").source({"seed": 0})
    b = get_family("aime").source({"seed": 0})
    assert [r["id"] for r in a.test_rows[:10]] == [r["id"] for r in b.test_rows[:10]]


def test_reward_grades_boxed_integers(env):
    t = env.tasks(1, split="test")[0]
    gt = int(t.meta["gt"])
    r_good, info_good = env.reward(t, f"Derivation.\n\\boxed{{{gt}}}")
    r_bad, info_bad = env.reward(t, f"Derivation.\n\\boxed{{{gt + 1}}}")
    assert info_good == {
        "correct_strict": 1.0,
        "correct_relaxed": 1.0,
        "answer_format_valid": 1.0,
    }
    assert info_bad == {
        "correct_strict": 0.0,
        "correct_relaxed": 0.0,
        "answer_format_valid": 1.0,
    }
    assert not ({"correct", "has_boxed"} & set(info_good))
    assert r_good > r_bad


def test_family_grade_and_parse_answers(env):
    fam = get_family("aime")
    assert fam.grade({"gt": 60.0}, 60) is True
    assert fam.grade({"gt": 60.0}, 61) is False
    parsed = fam.parse_answers("no box, answer 60")
    assert parsed.strict is None and parsed.relaxed == 60.0


def test_aime_numeric_parser_does_not_accept_symbolic_arithmetic():
    parsed = get_family("aime").parse_answers("\\boxed{2+2}")
    assert parsed.strict is None
    assert parsed.relaxed == 2.0  # numeric fallback, not symbolic evaluation
    assert parsed.answer_format_valid is False


def _fake_aime_dataset(problem_suffix=""):
    return {
        "train": [
            {
                "Question": f"AIME problem {i}{problem_suffix}",
                "Answer": str(i % 1000),
                "ID": f"fake-{i}",
            }
            for i in range(100)
        ]
    }


def test_aime_protocol_identity_is_stable_and_sensitive(monkeypatch):
    import infra.envs.tasks.aime as aime_module

    dataset = _fake_aime_dataset()
    monkeypatch.setattr(aime_module, "load_dataset", lambda _dataset_id: deepcopy(dataset))

    first = get_family("aime")
    first_env = first.source(
        {
            "seed": 7,
            "eval_subset_size": 20,
            "correct_reward": 2,
            "format_reward": 0,
            "relaxed_correct_bonus": 0.25,
            "think_overshoot_penalty": 0.125,
        }
    )
    second = get_family("aime")
    second.source(
        {
            "seed": 7,
            "eval_subset_size": 20,
            "correct_reward": 2,
            "format_reward": 0,
            "relaxed_correct_bonus": 0.25,
            "think_overshoot_penalty": 0.125,
        }
    )
    identity = first.protocol_identity()
    assert identity == second.protocol_identity()
    assert identity["grading_protocol"] == "numeric_box_v1"
    assert identity["dataset_id"] == "di-zhang-fdu/AIME_1983_2024"
    assert identity["dataset_revision"] == "unpinned_legacy"
    assert identity["seed"] == "7"
    assert identity["eval_subset_size"] == "20"
    assert identity["train_count"] == str(len(first_env.train_rows))
    assert identity["dev_count"] == str(len(first_env.dev_rows))
    assert identity["test_count"] == str(len(first_env.test_rows))
    assert identity["correct_reward"] == "2.0"
    assert identity["format_reward"] == "0.0"
    assert identity["relaxed_correct_bonus"] == "0.25"
    assert identity["think_overshoot_penalty"] == "0.125"
    assert len(identity["prompt_sha256"]) == len(identity["split_sha256"]) == 64
    assert first_env.format_reward == 0.0

    dataset["train"][0]["Question"] += " changed"
    changed = get_family("aime")
    changed.source(
        {
            "seed": 7,
            "eval_subset_size": 20,
            "correct_reward": 2,
            "format_reward": 0,
            "relaxed_correct_bonus": 0.25,
            "think_overshoot_penalty": 0.125,
        }
    )
    assert changed.protocol_identity()["split_sha256"] != identity["split_sha256"]


@pytest.mark.parametrize(
    ("knob", "changed_value", "default_identity", "changed_identity"),
    [
        ("correct_reward", 2, "1.0", "2.0"),
        ("format_reward", 0, "0.1", "0.0"),
        ("relaxed_correct_bonus", 0.25, "0.1", "0.25"),
        ("think_overshoot_penalty", 0.125, "0.0", "0.125"),
    ],
)
def test_aime_protocol_identity_tracks_each_reward_knob(
    monkeypatch, knob, changed_value, default_identity, changed_identity
):
    import infra.envs.tasks.aime as aime_module

    monkeypatch.setattr(
        aime_module, "load_dataset", lambda _dataset_id: deepcopy(_fake_aime_dataset())
    )
    baseline = get_family("aime")
    baseline.source({"seed": 7, "eval_subset_size": 20})
    changed = get_family("aime")
    changed.source({"seed": 7, "eval_subset_size": 20, knob: changed_value})

    assert baseline.protocol_identity()[knob] == default_identity
    assert changed.protocol_identity()[knob] == changed_identity
    assert changed.protocol_identity()[knob] != baseline.protocol_identity()[knob]
