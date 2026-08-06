"""AIME family: loads the 1983-2024 pool, renders the SAME single-sourced math
answer prompt, grades integers with the shared extractor. Network-dependent
loading is exercised once via the HF cache; everything else is offline."""

import pytest

from infra.envs.tasks import get_family


@pytest.fixture(scope="module")
def env():
    return get_family("aime").source({"seed": 0})


def test_pool_loads_and_splits(env):
    assert len(env.train_rows) + len(env.test_rows) > 900
    assert len(env.test_rows) >= 64
    train_ids = {r["id"] for r in env.train_rows}
    assert not train_ids & {r["id"] for r in env.test_rows}


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
    assert info_good["correct"] == 1.0 and info_bad["correct"] == 0.0
    assert r_good > r_bad


def test_family_grade_and_extractor(env):
    fam = get_family("aime")
    assert fam.grade({"gt": 60.0}, 60) is True
    assert fam.grade({"gt": 60.0}, 61) is False
    assert fam.extractor(relaxed=True)("no box, answer 60") == 60.0
    assert fam.extractor(relaxed=False)("no box, answer 60") is None
