"""train_filter_file: shrinks only the training pool, never the eval carve."""

import json

import pytest

import infra.envs.tasks.math as math_mod
from infra.envs.tasks import get_family
from infra.envs.tasks.math import MathEnv, problem_key


def _fake_dataset():
    train = [
        {"level": "Level 5", "problem": f"train problem {i}", "solution": f"\\boxed{{{i}}}"}
        for i in range(30)
    ]
    test = [
        {"level": "Level 5", "problem": f"test problem {i}", "solution": f"\\boxed{{{i}}}"}
        for i in range(6)
    ]
    return {"train": train, "test": test}


@pytest.fixture
def fake_load(monkeypatch):
    monkeypatch.setattr(math_mod, "_load", _fake_dataset)


def test_filter_shrinks_train_only(fake_load, tmp_path):
    plain = MathEnv(levels=(5,), seed=0)
    keep_rows = plain.train_rows[5:9]
    ff = tmp_path / "keep.json"
    ff.write_text(json.dumps([problem_key(r["problem"]) for r in keep_rows]))

    filtered = MathEnv(levels=(5,), seed=0, train_filter_file=str(ff))
    assert filtered.train_rows == keep_rows
    assert filtered.dev_rows == plain.dev_rows
    assert filtered.test_rows == plain.test_rows


def test_filter_preserves_train_order(fake_load, tmp_path):
    plain = MathEnv(levels=(5,), seed=0)
    keep_rows = [plain.train_rows[7], plain.train_rows[2], plain.train_rows[11]]
    ff = tmp_path / "keep.json"
    ff.write_text(json.dumps([problem_key(r["problem"]) for r in keep_rows]))

    filtered = MathEnv(levels=(5,), seed=0, train_filter_file=str(ff))
    assert filtered.train_rows == [plain.train_rows[2], plain.train_rows[7], plain.train_rows[11]]


def test_zero_match_raises(fake_load, tmp_path):
    ff = tmp_path / "keep.json"
    ff.write_text(json.dumps([problem_key("no such problem")]))
    with pytest.raises(RuntimeError, match="matched 0"):
        MathEnv(levels=(5,), seed=0, train_filter_file=str(ff))


def test_family_accepts_filter_key(fake_load, tmp_path):
    plain = MathEnv(levels=(5,), seed=0)
    ff = tmp_path / "keep.json"
    ff.write_text(json.dumps([problem_key(r["problem"]) for r in plain.train_rows[:3]]))
    env = get_family("math").source({"levels": 5, "seed": 0, "train_filter_file": str(ff)})
    assert len(env.train_rows) == 3


def test_dev_rows_cannot_pass_filter(fake_load, tmp_path):
    plain = MathEnv(levels=(5,), seed=0)
    ff = tmp_path / "keep.json"
    keys = [problem_key(r["problem"]) for r in plain.dev_rows] + [
        problem_key(r["problem"]) for r in plain.train_rows[:2]
    ]
    ff.write_text(json.dumps(keys))
    filtered = MathEnv(levels=(5,), seed=0, train_filter_file=str(ff))
    assert filtered.train_rows == plain.train_rows[:2]
    assert filtered.dev_rows == plain.dev_rows
