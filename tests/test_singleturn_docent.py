"""Single-turn docent export: RLVR rollout records -> AgentRuns, same output
path as the debate export so both run types ingest into docent identically."""

import json

from infra.envs.debate.docent_export import export_jsonl
from infra.envs.singleturn_docent import agent_runs

RECORDS = [
    {
        "task_index": 0,
        "meta": {"gt": 42.0, "level": 5, "split": "train", "question": "q?"},
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "solve q"},
        ],
        "completion": "the answer is \\boxed{42}",
        "stop_reason": "stop",
        "reward": 1.1,
        "info": {"correct": 1.0, "correct_relaxed": 1.0, "has_boxed": 1.0},
    },
    {
        "task_index": 1,
        "meta": {"gt": 7.0, "level": 5, "split": "train", "question": "r?"},
        "messages": [{"role": "user", "content": "solve r"}],
        "completion": "\\boxed{8}",
        "stop_reason": "length",
        "reward": 0.1,
        "info": {"correct": 0.0, "correct_relaxed": 0.0, "has_boxed": 1.0},
    },
]


def test_one_run_per_record_with_completion_appended():
    runs = agent_runs(RECORDS)
    assert len(runs) == 2
    msgs = runs[0].transcripts[0].messages
    assert [m.role for m in msgs] == ["system", "user", "assistant"]
    assert "boxed{42}" in msgs[-1].text


def test_metadata_carries_grading_and_status_in_name():
    runs = agent_runs(RECORDS)
    assert runs[0].metadata["reward"] == 1.1
    assert runs[0].metadata["task"]["gt"] == 42.0
    assert runs[0].name.endswith("-> correct")
    assert runs[1].name.endswith("-> incorrect")
    assert runs[1].metadata["stop_reason"] == "length"


def test_export_jsonl_roundtrips(tmp_path):
    path = str(tmp_path / "st.jsonl")
    export_jsonl(agent_runs(RECORDS), path)
    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == 2
    assert rows[0]["transcripts"][0]["messages"][-1]["role"] == "assistant"
    assert rows[0]["metadata"]["info"]["correct_relaxed"] == 1.0
