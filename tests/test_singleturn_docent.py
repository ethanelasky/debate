"""Scientific single-turn Docent encoding and immutable local evidence."""

import json
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

from infra.envs.debate.docent_export import (
    collection_name_for_launch,
    export_jsonl,
    export_jsonl_claimed,
)
from infra.envs.singleturn_docent import agent_runs
from infra.launch_namespace import claim_directory


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
        "info": {
            "correct_strict": 1.0,
            "correct_relaxed": 1.0,
            "answer_format_valid": 1.0,
        },
    },
    {
        "task_index": 1,
        "meta": {"gt": 7.0, "level": 5, "split": "train", "question": "r?"},
        "messages": [{"role": "user", "content": "solve r"}],
        "completion": "the answer is 7",
        "stop_reason": "stop",
        "reward": 0.1,
        "info": {
            "correct_strict": 0.0,
            "correct_relaxed": 1.0,
            "answer_format_valid": 0.0,
        },
    },
    {
        "task_index": 2,
        "meta": {"gt": 9.0, "level": 5, "split": "train", "question": "s?"},
        "messages": [{"role": "user", "content": "solve s"}],
        "completion": "\\boxed{8}",
        "stop_reason": "length",
        "reward": 0.1,
        "info": {
            "correct_strict": 0.0,
            "correct_relaxed": 0.0,
            "answer_format_valid": 1.0,
        },
    },
]


def test_one_run_per_record_with_completion_appended():
    runs = agent_runs(RECORDS)
    assert len(runs) == 3
    messages = runs[0].transcripts[0].messages
    assert [message.role for message in messages] == ["system", "user", "assistant"]
    assert "boxed{42}" in messages[-1].text


def test_metadata_carries_grading_and_mutually_exclusive_status_in_name():
    runs = agent_runs(RECORDS)
    assert runs[0].metadata["reward"] == 1.1
    assert runs[0].metadata["info"]["correct_strict"] == 1.0
    assert runs[0].metadata["info"]["correct_relaxed"] == 1.0
    assert runs[0].metadata["info"]["answer_format_valid"] == 1.0
    assert runs[0].name.endswith("strict-correct")
    assert runs[1].name.endswith("relaxed-only-correct")
    assert runs[2].name.endswith("incorrect")


def test_export_jsonl_roundtrips_exact_canonical_bytes(tmp_path):
    runs = agent_runs(RECORDS)
    destination = tmp_path / "runs.jsonl"
    export_jsonl(runs, str(destination))
    assert destination.read_bytes() == b"".join(
        run.model_dump_json().encode("utf-8") + b"\n" for run in runs
    )
    assert [json.loads(line) for line in destination.read_text().splitlines()]


def test_claimed_export_preserves_exact_agent_run_bytes(tmp_path):
    runs = agent_runs(RECORDS)
    directory = claim_directory(tmp_path / "attempt")
    export_jsonl_claimed(runs, directory, "docent.jsonl")
    assert (directory / "docent.jsonl").read_bytes() == b"".join(
        run.model_dump_json().encode("utf-8") + b"\n" for run in runs
    )


def test_collection_name_is_stable_and_distinct_per_namespace():
    assert collection_name_for_launch("math-pc-rl", "run-A") == (
        "math-pc-rl--launch-run-A"
    )
    assert collection_name_for_launch("math-pc-rl", "run-A") == (
        "math-pc-rl--launch-run-A"
    )
    assert collection_name_for_launch("math-pc-rl", "run-B") != (
        "math-pc-rl--launch-run-A"
    )


@pytest.mark.parametrize("namespace", ["", "../attempt", "attempt/child", "two words"])
def test_collection_name_refuses_invalid_namespace(namespace):
    with pytest.raises(ValueError, match="DEBATE_LAUNCH_NAMESPACE"):
        collection_name_for_launch("math-pc-rl", namespace)


def test_docent_version_is_exactly_pinned_in_all_install_surfaces():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]
    assert "docent-python==0.1.77" in dependencies
    for script_name in ("provision_sm90.sh", "provision_blackwell.sh"):
        text = (root / "scripts" / script_name).read_text()
        assert "docent-python==0.1.77" in text
        assert "docent-python>=" not in text
    lock_text = (root / "uv.lock").read_text()
    assert 'name = "docent-python"\nversion = "0.1.77"' in lock_text


def test_runtime_docent_version_matches_reviewed_pin():
    assert version("docent") == "0.1.77"
    assert version("docent-python") == "0.1.77"
