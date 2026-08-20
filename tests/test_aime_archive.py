"""AIME archive eval-only family: carve size, year window, eval-only contract."""

import pytest

import infra.envs.tasks.aime_archive as arch_mod
from infra.envs.tasks import get_family
from infra.envs.tasks.aime_archive import AimeArchiveEnv


FAKE_ROWS = [
    {"ID": f"{year}-I-{i}", "Question": f"problem {year}-{i}", "Answer": str((year + i) % 1000)}
    for year in (1984, 1999, 2005, 2019, 2023)
    for i in range(1, 5)
]
FAKE_ROWS.append({"ID": "2001-II-9", "Question": "unusable", "Answer": "not-a-number"})


@pytest.fixture
def fake_load(monkeypatch):
    monkeypatch.setattr(arch_mod, "_load_archive", lambda dataset_id=None: FAKE_ROWS)


def test_eval_only_contract(fake_load):
    env = AimeArchiveEnv(n=10)
    assert len(env.dev_rows) == 10
    assert env.dev_rows == env.test_rows
    assert env.train_rows == []
    with pytest.raises(RuntimeError, match="eval-only"):
        env.tasks(4, split="train")


def test_year_window_excludes_2020s(fake_load):
    env = AimeArchiveEnv(n=16)
    years = {int(r["id"][:4]) for r in env.dev_rows}
    assert 2023 not in years
    assert years <= {1984, 1999, 2005, 2019}


def test_year_window_override(fake_load):
    env = AimeArchiveEnv(n=4, year_min=2020, year_max=2024)
    assert {int(r["id"][:4]) for r in env.dev_rows} == {2023}


def test_carve_is_seed_stable(fake_load):
    a = AimeArchiveEnv(n=8, seed=0)
    b = AimeArchiveEnv(n=8, seed=0)
    c = AimeArchiveEnv(n=8, seed=1)
    assert a.dev_rows == b.dev_rows
    assert a.dev_rows != c.dev_rows


def test_too_small_pool_raises(fake_load):
    with pytest.raises(RuntimeError, match="usable rows"):
        AimeArchiveEnv(n=100)


def test_family_registered_and_rejects_unknown_keys(fake_load):
    family = get_family("aime_archive")
    env = family.source({"seed": 0, "n": 6})
    assert len(env.dev_rows) == 6
    reward, info = env.reward(env.tasks(1, split="dev")[0], "\\boxed{" + str(int(env.dev_rows[0]["gt"])) + "}")
    assert info["correct_strict"] == 1.0
    with pytest.raises(ValueError, match="aime_archive"):
        family.source({"levels": 5})
