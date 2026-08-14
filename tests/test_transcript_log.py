"""Transcript capture -> wandb (infra/transcript_log.py) and the SingleTurnEnv
record retention it reads. Offline: wandb is never initialized here, so the
logger's no-op path is what runs; the builders are tested as pure functions.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infra.backend.base import SamplingParams
from infra.envs.base import Policy, SingleTurnEnv, Task
from infra.transcript_log import (
    SAMPLE_CHARS,
    flatten_messages,
    log_rollout_transcripts,
    singleturn_sample_rows,
)
from test_single_turn_env import ScriptedBackend


class MetaEnv(SingleTurnEnv):
    """Tasks carry MB-shaped meta including the big 'bindings' payload."""

    def tasks(self, n, split="train"):
        return [
            Task(
                messages=[{"role": "user", "content": f"q{i}"}],
                meta={
                    "task_id": f"row_{i}",
                    "label": "attack",
                    "bindings": {"PROBLEM": "SECRET-TRAJECTORY" * 1000},
                },
            )
            for i in range(n)
        ]

    def reward(self, task, text):
        return 1.0, {
            "correct_strict": 1.0,
            "correct_relaxed": 1.0,
            "answer_format_valid": 1.0,
        }


def _policy(script):
    return Policy(ScriptedBackend(script), SamplingParams(max_tokens=32))


def test_rollout_retains_records_without_bindings():
    env = MetaEnv()
    env.rollout(env.tasks(2), _policy(["yes", "no"]), group_size=1)
    recs = env.last_rollout_records
    assert [r["completion"] for r in recs] == ["yes", "no"]
    assert [r["meta"]["task_id"] for r in recs] == ["row_0", "row_1"]
    # bindings duplicate the rendered prompt (and can be a 60k-token
    # trajectory); the retained meta must not carry them.
    assert all("bindings" not in r["meta"] for r in recs)
    # reward() 's own keys survive into the record. Subset rather than equality:
    # rollout() also stamps the length metrics (tokens, over_budget) that every
    # run reports, and pinning the exact dict here would make adding a metric
    # look like a regression in a test about bindings.
    assert all(
        r["reward"] == 1.0 and r["info"]["correct_strict"] == 1.0
        for r in recs
    )
    # each rollout overwrites, never appends
    env.rollout(env.tasks(1), _policy(["third"]), group_size=1)
    assert [r["completion"] for r in env.last_rollout_records] == ["third"]


def test_dropped_samples_leave_no_record():
    env = MetaEnv()
    env.rollout(env.tasks(2), _policy(["ok", ""]), group_size=1)  # "" = fidelity drop
    assert [r["completion"] for r in env.last_rollout_records] == ["ok"]


def test_singleturn_sample_rows_shape_and_truncation():
    records = [
        {
            "task_index": 0,
            "meta": {"task_id": "row_0"},
            "messages": [{"role": "user", "content": "x" * (2 * SAMPLE_CHARS)}],
            "completion": "ans",
            "stop_reason": "stop",
            "reward": 1.1,
            "info": {
                "correct_strict": 1.0,
                "correct_relaxed": 1.0,
                "answer_format_valid": 1.0,
            },
        }
    ]
    (row,) = singleturn_sample_rows(records, step=7)
    step, task, reward, info, transcript = row
    assert (step, task, reward) == (7, "row_0", 1.1)
    assert "truncated" in transcript and len(transcript) < 2 * SAMPLE_CHARS
    # cap rows at SAMPLE_ROWS
    assert len(singleturn_sample_rows(records * 10, step=0)) == 4


def test_flatten_messages():
    assert flatten_messages(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    ) == "[system]\ns\n\n[user]\nu"


def test_logger_is_noop_without_wandb_run(tmp_path, monkeypatch):
    # No wandb.init in tests: wandb.run is None and the logger must return
    # before touching the filesystem.
    monkeypatch.chdir(tmp_path)
    env = MetaEnv()
    env.rollout(env.tasks(1), _policy(["a"]), group_size=1)
    log_rollout_transcripts(0, env, "train")
    assert not (tmp_path / "transcripts").exists()


def test_transcript_every_gates_upload_never_local_write(monkeypatch):
    import infra.train as train_mod
    import infra.transcript_log as tl

    calls = []
    monkeypatch.setattr(
        tl, "log_rollout_transcripts",
        lambda step, env, split, run_name=None, upload=True: calls.append((step, split, upload)),
    )
    cfg = train_mod.Config(wandb_project="p", transcript_every=10, run_name="r")
    for s in (1, 10, 20):
        train_mod._log_transcripts(cfg, s, env=None, split="train")
    train_mod._log_transcripts(cfg, 7, env=None, split="eval")
    # every round reaches the writer (local persistence); upload is the only
    # thing the cadence gates
    assert calls == [(1, "train", False), (10, "train", True), (20, "train", True), (7, "eval", True)]

    calls.clear()
    nowandb = train_mod.Config(wandb_project=None, transcript_every=10, run_name="r")
    train_mod._log_transcripts(nowandb, 3, env=None, split="train")
    assert calls == [(3, "train", False)]  # no wandb: local write still happens
