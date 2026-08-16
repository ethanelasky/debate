"""AMC eval-only family: pool loading, eval-only contract, reward reuse."""

import pytest

import infra.envs.tasks.amc as amc_mod
from infra.envs.tasks import get_family
from infra.envs.tasks.amc import AmcEnv


FAKE_ROWS = [
    {"problem": "What is 2+2?", "answer": "4.0"},
    {"problem": "What is 10*10?", "answer": "100"},
    {"problem": "unusable", "answer": "not-a-number"},
]


@pytest.fixture
def fake_load(monkeypatch):
    monkeypatch.setattr(amc_mod, "_load_amc", lambda dataset_id=None: FAKE_ROWS)


def test_pool_is_eval_only(fake_load):
    env = AmcEnv()
    assert len(env.dev_rows) == 2  # unusable row dropped
    assert env.dev_rows == env.test_rows
    assert env.train_rows == []
    with pytest.raises(RuntimeError, match="eval-only"):
        env.tasks(4, split="train")


def test_dev_tasks_carry_numeric_gt(fake_load):
    env = AmcEnv()
    tasks = env.tasks(10, split="dev")
    assert len(tasks) == 2
    assert tasks[0].meta["gt"] == 4.0
    assert tasks[1].meta["gt"] == 100.0


def test_reward_protocol_is_math_env(fake_load):
    env = AmcEnv()
    task = env.tasks(1, split="dev")[0]
    reward, info = env.reward(task, "The answer is \\boxed{4}")
    assert info["correct_strict"] == 1.0
    assert reward == pytest.approx(1.1)  # correct + format bonus
    reward_wrong, info_wrong = env.reward(task, "\\boxed{5}")
    assert info_wrong["correct_strict"] == 0.0


def test_family_registered_and_rejects_unknown_keys(fake_load):
    family = get_family("amc")
    env = family.source({"seed": 0})
    assert env.dev_rows
    with pytest.raises(ValueError, match="amc"):
        family.source({"levels": 5})


def test_eval_dataset_key_in_rlvr_allowlist():
    from infra.run_rlvr import EXPERIMENT_KEYS

    assert "eval_dataset" in EXPERIMENT_KEYS
