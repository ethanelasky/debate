"""Runner-installed Docent callbacks at the required local evidence boundary.

These tests let each real runner assemble its Config, then exercise that
Config through the public training entry point with production-compatible
retained rollout data. Topology and identity ordering stay covered separately
in test_protocol_identity.py.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

import infra.train as train_mod
from infra.envs.base import Policy, Trajectory
from infra.backend.base import SamplingParams


def _main_args():
    return SimpleNamespace(
        experiment_file="unused.yaml",
        experiment="experiment",
        levels=None,
        wandb_resume=None,
        no_wandb=True,
        load=None,
        lr=None,
        group_size=None,
        batch_size=None,
        steps=None,
        start_step=None,
        wandb_entity=None,
        wandb_project=None,
    )


def _capture_rlvr_config(monkeypatch, tmp_path, namespace):
    import infra.run_rlvr as run_rlvr

    monkeypatch.chdir(tmp_path)
    captured = []
    family = SimpleNamespace(
        source=lambda ds: SimpleNamespace(),
        close=lambda: None,
    )
    backend = SimpleNamespace(
        config=SimpleNamespace(checkpoint_dir=None), tokenizer=object()
    )
    exp = {
        "model": "org/model",
        "max_completion_tokens": 32,
        "dataset": {"type": "math"},
        "training": {},
    }
    monkeypatch.setattr(
        run_rlvr, "runner_parser", lambda doc: SimpleNamespace(parse_args=_main_args)
    )
    monkeypatch.setattr(run_rlvr, "load_experiment", lambda *args: exp)
    monkeypatch.setattr(run_rlvr, "get_family", lambda kind: family)
    monkeypatch.setattr(run_rlvr, "resolve_launch_namespace", lambda: namespace)
    monkeypatch.setattr(run_rlvr, "resolve_topology", lambda: {})
    monkeypatch.setattr(run_rlvr, "rlvr_protocol_identity", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_rlvr, "build_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(
        run_rlvr, "train", lambda env, backend, cfg, eval_env=None: captured.append(cfg)
    )

    run_rlvr._main([])

    assert len(captured) == 1
    return captured[0]


def _capture_debate_config(monkeypatch, tmp_path, namespace):
    import infra.run_debate as run_debate
    from infra.models.base import ModelSettings

    monkeypatch.chdir(tmp_path)
    captured = []
    family = SimpleNamespace(close=lambda: None)
    trained = {
        "alice": ModelSettings(
            model_type="local", model_file_path="org/model", alias="Alice"
        )
    }
    runner_env = SimpleNamespace(
        family=family,
        protocol=object(),
        task_source=object(),
    )
    backend = SimpleNamespace(config=SimpleNamespace(checkpoint_dir=None))
    exp = {"dataset": {"type": "math"}, "training": {}}
    monkeypatch.setattr(
        run_debate,
        "runner_parser",
        lambda doc: SimpleNamespace(parse_args=_main_args),
    )
    monkeypatch.setattr(run_debate, "load_experiment", lambda *args: exp)
    monkeypatch.setattr(run_debate, "validate_experiment", lambda exp: None)
    monkeypatch.setattr(run_debate, "split_agents", lambda exp: (trained, {}))
    monkeypatch.setattr(run_debate, "validate_trained_seats", lambda *args: None)
    monkeypatch.setattr(run_debate, "build_env", lambda *args: runner_env)
    monkeypatch.setattr(run_debate, "debate_gen_budgets", lambda *args: {})
    monkeypatch.setattr(
        run_debate,
        "resolve_launch_namespace",
        lambda value=None: namespace if value is None else value,
    )
    monkeypatch.setattr(run_debate, "resolve_topology", lambda: {})
    monkeypatch.setattr(
        run_debate, "debate_protocol_identity", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(run_debate, "build_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(
        run_debate,
        "train",
        lambda env, backend, cfg, eval_env=None: captured.append(cfg),
    )

    run_debate._main([])

    assert len(captured) == 1
    return captured[0]


class _EvidenceLoopBackend:
    tokenizer = object()

    def __init__(self):
        self.saved = []

    def sync_sampler(self):
        pass

    def save(self, name):
        self.saved.append(name)


class _RlvREvidenceEnv:
    def __init__(self, records):
        self._next_records = copy.deepcopy(records)
        self.last_rollout_records = []
        self.last_rollout_info = {}

    def tasks(self, n, split="train"):
        return [object()]

    def rollout(self, tasks, policy, group_size):
        self.last_rollout_records = copy.deepcopy(self._next_records)
        return [[Trajectory(datums=[], reward=1.0)]]


def _debate_evidence_env(monkeypatch, sentinel):
    from test_debate_env import GOOD_VERDICT, ScriptedBackend, TaskSource, make_env

    env = make_env(["alice"], [GOOD_VERDICT])
    policy = Policy(
        ScriptedBackend(
            [f"{sentinel}: I compute. \\boxed{{2}}", "My defense stands."]
        ),
        SamplingParams(max_tokens=128),
    )
    env.rollout(TaskSource().tasks(1), policy, group_size=1)
    next_states = copy.deepcopy(env.last_states)
    env.last_states = []
    monkeypatch.setattr(env, "tasks", lambda n, split="train": [object()])

    def rollout(tasks, policy, group_size):
        env.last_states = copy.deepcopy(next_states)
        return [[Trajectory(datums=[], reward=1.0)]]

    monkeypatch.setattr(
        env,
        "rollout",
        rollout,
    )
    env.last_rollout_info = {}
    return env


def _one_step(cfg, step):
    cfg.start_step = step
    cfg.steps = step + 1
    cfg.batch_size = 1
    cfg.group_size = 1
    cfg.eval_every = 0
    cfg.final_test_eval = False
    cfg.save_every = 0
    assert cfg.log_transcripts is True


def _without_generated_ids(rows):
    if isinstance(rows, dict):
        return {
            key: _without_generated_ids(value)
            for key, value in rows.items()
            if key != "id"
        }
    if isinstance(rows, list):
        return [_without_generated_ids(value) for value in rows]
    return rows


def _rlvr_records_with_sentinel(records, sentinel):
    current = copy.deepcopy(records)
    current[0]["completion"] = f"{sentinel}: the answer is \\boxed{{42}}"
    current[0]["meta"]["question"] = sentinel
    return current


def _assert_rlvr_science(row, sentinel):
    assert row["name"] == "task-0 [train] -> strict-correct"
    assert row["metadata"]["task"]["question"] == sentinel
    assert row["metadata"]["task"]["gt"] == 42.0
    assert row["metadata"]["task"]["split"] == "train"
    assert row["metadata"]["reward"] == 1.1
    assert row["metadata"]["stop_reason"] == "stop"
    assert row["metadata"]["info"] == {
        "correct_strict": 1.0,
        "correct_relaxed": 1.0,
        "answer_format_valid": 1.0,
    }
    assert row["transcripts"][0]["messages"][-1]["content"] == (
        f"{sentinel}: the answer is \\boxed{{42}}"
    )


def _assert_debate_science(row, sentinel):
    from infra.envs.debate.judge_accuracy import judge_was_right

    meta = row["metadata"]
    assert meta["verdict"]["winner"] == "Debater_A"
    assert meta["bindings"]["alice"] == "Debater_A"
    assert meta["bindings"]["bob"] == "Debater_B"
    assert meta["grades"] == {"alice": True}
    assert judge_was_right(meta) == (True, "decidable")
    assert sentinel in json.dumps(row["transcripts"])


def test_rlvr_runner_persists_both_local_evidence_views(monkeypatch, tmp_path):
    from test_singleturn_docent import RECORDS

    sentinel = "RLVR-CURRENT-STEP-7"
    cfg = _capture_rlvr_config(monkeypatch, tmp_path, "rlvr-evidence")
    env = _RlvREvidenceEnv(_rlvr_records_with_sentinel(RECORDS, sentinel))
    assert env.last_rollout_records == []
    _one_step(cfg, 7)
    backend = _EvidenceLoopBackend()

    train_mod.train(env, backend, cfg)

    docent = tmp_path / "docent/experiment/rlvr-evidence/step-00007.jsonl"
    transcript = (
        tmp_path / "transcripts/experiment/rlvr-evidence/train-step-00007.jsonl"
    )
    docent_rows = [json.loads(line) for line in docent.read_text().splitlines()]
    transcript_rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    assert len(docent_rows) == len(RECORDS) == len(transcript_rows)
    _assert_rlvr_science(docent_rows[0], sentinel)
    _assert_rlvr_science(transcript_rows[0], sentinel)
    # Each independent AgentRun build receives a fresh opaque UUID. Everything
    # scientific in the two local views must otherwise be identical.
    assert _without_generated_ids(docent_rows) == _without_generated_ids(
        transcript_rows
    )
    assert backend.saved == ["final"]

    env.last_rollout_records = []
    with pytest.raises(RuntimeError, match="all-fidelity-dropped"):
        cfg.on_rollout(8, env)
    assert not (docent.parent / "step-00008.jsonl").exists()


def test_rlvr_docent_overwrite_aborts_public_train(monkeypatch, tmp_path):
    from test_singleturn_docent import RECORDS

    cfg = _capture_rlvr_config(monkeypatch, tmp_path, "rlvr-collision")
    env = _RlvREvidenceEnv(RECORDS)
    assert env.last_rollout_records == []
    env.rollout([object()], policy=None, group_size=1)
    cfg.on_rollout(7, env)
    docent = tmp_path / "docent/experiment/rlvr-collision/step-00007.jsonl"
    original = docent.read_bytes()
    env.last_rollout_records = []
    _one_step(cfg, 7)
    backend = _EvidenceLoopBackend()

    with pytest.raises(FileExistsError, match="refusing existing launch output"):
        train_mod.train(env, backend, cfg)

    assert docent.read_bytes() == original
    assert not (
        tmp_path / "transcripts/experiment/rlvr-collision/train-step-00007.jsonl"
    ).exists()
    assert backend.saved == []


def test_debate_runner_persists_both_local_evidence_views(monkeypatch, tmp_path):
    # Build the real state while its repository-relative prompt fixture is
    # reachable, before runner setup moves the process into the output root.
    sentinel = "DEBATE-CURRENT-STEP-8"
    env = _debate_evidence_env(monkeypatch, sentinel)
    assert env.last_states == []
    cfg = _capture_debate_config(monkeypatch, tmp_path, "debate-evidence")
    _one_step(cfg, 8)
    backend = _EvidenceLoopBackend()

    train_mod.train(env, backend, cfg)

    docent = tmp_path / "docent/experiment/debate-evidence/step-00008.jsonl"
    transcript = (
        tmp_path
        / "transcripts/experiment/debate-evidence/train-step-00008.jsonl"
    )
    rows = [json.loads(line) for line in docent.read_text().splitlines()]
    transcript_rows = [
        json.loads(line) for line in transcript.read_text().splitlines()
    ]
    _assert_debate_science(rows[0], sentinel)
    _assert_debate_science(transcript_rows[0], sentinel)
    from infra.envs.debate.judge_accuracy import from_jsonl

    for path in (docent, transcript):
        accuracy = from_jsonl(str(path))
        assert (accuracy.total, accuracy.decidable, accuracy.correct) == (1, 1, 1)
    assert _without_generated_ids(rows) == _without_generated_ids(transcript_rows)
    assert backend.saved == ["final"]

    env.last_states = []
    with pytest.raises(RuntimeError, match="retained no states"):
        cfg.on_rollout(9, env)
    assert not (docent.parent / "step-00009.jsonl").exists()


def test_debate_docent_overwrite_aborts_public_train(monkeypatch, tmp_path):
    env = _debate_evidence_env(monkeypatch, "DEBATE-COLLISION-STEP-8")
    assert env.last_states == []
    cfg = _capture_debate_config(monkeypatch, tmp_path, "debate-collision")
    env.rollout([object()], policy=None, group_size=1)
    cfg.on_rollout(8, env)
    docent = tmp_path / "docent/experiment/debate-collision/step-00008.jsonl"
    original = docent.read_bytes()
    env.last_states = []
    _one_step(cfg, 8)
    backend = _EvidenceLoopBackend()

    with pytest.raises(FileExistsError, match="refusing existing launch output"):
        train_mod.train(env, backend, cfg)

    assert docent.read_bytes() == original
    assert not (
        tmp_path
        / "transcripts/experiment/debate-collision/train-step-00008.jsonl"
    ).exists()
    assert backend.saved == []
