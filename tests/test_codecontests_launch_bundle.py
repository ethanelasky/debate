"""Launch-contract regression tests for the 32B CodeContests cap arms.

These tests intentionally resolve the full inheritance chain and then apply
the shipped hardware table, matching the two config stages used at launch.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from infra.config import load_experiment
from infra.run_common import apply_topology

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "codecontests_rlvr_olmo.yaml"
TOPOLOGIES = ROOT / "configs" / "topologies.yaml"
MODEL_ARTIFACT = "/workspace/models/olmo32-bf16"

CAP_ARMS = {
    1024: {
        "long": "codecontests_rlvr_olmo31_32b_cap1024_long",
        "smoke": "codecontests_rlvr_olmo31_32b_cap1024_smoke1",
        "soft": 512,
    },
    2048: {
        "long": "codecontests_rlvr_olmo31_32b_cap2048_long",
        "smoke": "codecontests_rlvr_olmo31_32b_cap2048_smoke1",
        "soft": 1024,
    },
}

B200X2_ARMS = {
    1024: {
        "long": "codecontests_rlvr_olmo31_32b_cap1024_b200x2_long",
        "smoke": "codecontests_rlvr_olmo31_32b_cap1024_b200x2_smoke1",
    },
    2048: {
        "long": "codecontests_rlvr_olmo31_32b_cap2048_b200x2_long",
        "smoke": "codecontests_rlvr_olmo31_32b_cap2048_b200x2_smoke1",
    },
}


@pytest.fixture(scope="module")
def topology_table() -> dict:
    return yaml.safe_load(TOPOLOGIES.read_text())


@pytest.mark.parametrize("cap", sorted(CAP_ARMS))
def test_long_cap_arms_keep_science_and_launch_shape(cap: int, topology_table: dict):
    spec = CAP_ARMS[cap]
    exp = load_experiment(CONFIG, spec["long"])

    assert exp["model"] == MODEL_ARTIFACT
    assert exp["max_completion_tokens"] == cap
    assert exp["dataset"]["soft_token_budget"] == spec["soft"]
    assert exp["dataset"]["overshoot_penalty"] == pytest.approx(0.1)
    assert exp["dataset"]["path"] == "data/codecontests/train.jsonl"
    assert exp["dataset"]["paired_test_path"] == "data/codecontests/paired_test.jsonl"

    tr = exp["training"]
    assert (tr["steps"], tr["batch_size"], tr["group_size"]) == (100, 8, 8)
    assert (tr["eval_every"], tr["eval_n"], tr["save_every"]) == (10, 256, 25)
    assert tr["eval_max_tokens"] == cap
    assert tr["verl"]["response_length"] == cap
    assert tr["verl"]["max_token_len_per_gpu"] == 8192

    # Arm keys deliberately win topology defaults. Both currently available
    # four-card targets therefore retain the workload-proven 4/TP4/.40 shape.
    for topology_name in ("4xH100", "4xH200"):
        effective = apply_topology(tr["verl"], topology_table[topology_name])
        assert effective["n_gpus"] == 4
        assert effective["rollout_tp"] == 4
        assert effective["gpu_memory_utilization"] == pytest.approx(0.40)


@pytest.mark.parametrize("cap", sorted(CAP_ARMS))
def test_one_step_smoke_changes_only_runtime_cadence(cap: int):
    spec = CAP_ARMS[cap]
    long = load_experiment(CONFIG, spec["long"])
    smoke = load_experiment(CONFIG, spec["smoke"])

    expected = copy.deepcopy(long)
    expected["training"].update({"steps": 1, "eval_every": 0, "save_every": 0})
    assert smoke == expected
    assert (smoke["training"]["batch_size"], smoke["training"]["group_size"]) == (8, 8)


def test_four_gpu_hopper_topology_gates(topology_table: dict):
    assert topology_table["4xH200"] == {
        "n_gpus": 4,
        "rollout_tp": 4,
        "gpu_memory_utilization": 0.45,
    }
    assert topology_table["4xH100"]["n_gpus"] == 4
    assert topology_table["4xH100"]["rollout_tp"] == 4
    assert topology_table["4xH100"]["gpu_memory_utilization"] == pytest.approx(0.40)


@pytest.mark.parametrize("cap", sorted(CAP_ARMS))
def test_b200x2_capacity_fallback_changes_only_hardware(
    cap: int, topology_table: dict
):
    primary = load_experiment(CONFIG, CAP_ARMS[cap]["long"])
    fallback = load_experiment(CONFIG, B200X2_ARMS[cap]["long"])

    expected = copy.deepcopy(primary)
    expected["training"]["verl"].update(
        {"n_gpus": 2, "rollout_tp": 2, "gpu_memory_utilization": 0.42}
    )
    assert fallback == expected

    effective = apply_topology(
        fallback["training"]["verl"], topology_table["2xB200"]
    )
    assert effective["n_gpus"] == 2
    assert effective["rollout_tp"] == 2
    assert effective["gpu_memory_utilization"] == pytest.approx(0.42)
    assert any(
        "disable_custom_all_reduce=true" in item
        for item in effective["extra_overrides"]
    )


@pytest.mark.parametrize("cap", sorted(CAP_ARMS))
def test_b200x2_smoke_changes_only_runtime_cadence(cap: int):
    long = load_experiment(CONFIG, B200X2_ARMS[cap]["long"])
    smoke = load_experiment(CONFIG, B200X2_ARMS[cap]["smoke"])

    expected = copy.deepcopy(long)
    expected["training"].update({"steps": 1, "eval_every": 0, "save_every": 0})
    assert smoke == expected
    assert (
        smoke["training"]["batch_size"],
        smoke["training"]["group_size"],
    ) == (8, 8)


REQUIRED_CODECONTESTS_DATA = (
    "train.jsonl",
    "test.jsonl",
    "manifest.json",
    "cco_eval.jsonl",
    "cco_eval.manifest.json",
    "paired_test.jsonl",
    "paired_test.manifest.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_complete_dataset_bundle(data: Path) -> None:
    payloads = {
        "train.jsonl": b'{"split":"train"}\n',
        "test.jsonl": b'{"split":"test"}\n',
        "cco_eval.jsonl": b'{"split":"cco"}\n',
        "paired_test.jsonl": b'{"split":"paired"}\n',
    }
    for name, payload in payloads.items():
        (data / name).write_bytes(payload)

    (data / "manifest.json").write_text(
        json.dumps(
            {
                "outputs": {
                    split: {
                        "file": f"{split}.jsonl",
                        "size_bytes": (data / f"{split}.jsonl").stat().st_size,
                        "sha256": _sha256(data / f"{split}.jsonl"),
                    }
                    for split in ("train", "test")
                }
            }
        )
    )
    (data / "cco_eval.manifest.json").write_text(
        json.dumps(
            {
                "bytes": (data / "cco_eval.jsonl").stat().st_size,
                "sha256": _sha256(data / "cco_eval.jsonl"),
            }
        )
    )
    (data / "paired_test.manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "gdm": {"sha256": _sha256(data / "test.jsonl")},
                    "cco": {"sha256": _sha256(data / "cco_eval.jsonl")},
                },
                "output": {
                    "file": "paired_test.jsonl",
                    "size_bytes": (data / "paired_test.jsonl").stat().st_size,
                    "sha256": _sha256(data / "paired_test.jsonl"),
                },
            }
        )
    )


def _run_sync_preflight(tmp_path: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    return subprocess.run(
        [
            "bash",
            str(tmp_path / "scripts" / "pod_sync.sh"),
            "not-contacted.invalid",
            "1",
            "--with-data",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize("missing", REQUIRED_CODECONTESTS_DATA)
def test_pod_sync_rejects_incomplete_data_before_remote_contact(
    tmp_path: Path, missing: str
):
    """Exercise the real shell preflight, with no ssh/rsync test double."""

    scripts = tmp_path / "scripts"
    data = tmp_path / "data" / "codecontests"
    scripts.mkdir()
    data.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "pod_sync.sh", scripts / "pod_sync.sh")
    for name in REQUIRED_CODECONTESTS_DATA:
        if name != missing:
            (data / name).write_text("fixture\n")

    proc = _run_sync_preflight(tmp_path)

    assert proc.returncode == 2
    assert f"data/codecontests/{missing} missing locally" in proc.stderr
    assert "== repo ->" not in proc.stdout


@pytest.mark.parametrize(
    "corrupt",
    ("train.jsonl", "test.jsonl", "cco_eval.jsonl", "paired_test.jsonl"),
)
def test_pod_sync_rejects_same_size_sha_corruption_before_remote_contact(
    tmp_path: Path, corrupt: str
):
    scripts = tmp_path / "scripts"
    data = tmp_path / "data" / "codecontests"
    scripts.mkdir()
    data.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "pod_sync.sh", scripts / "pod_sync.sh")
    _write_complete_dataset_bundle(data)

    original = (data / corrupt).read_bytes()
    replacement = bytes([original[0] ^ 1]) + original[1:]
    assert len(replacement) == len(original)
    (data / corrupt).write_bytes(replacement)

    proc = _run_sync_preflight(tmp_path)

    assert proc.returncode == 2
    assert f"data/codecontests/{corrupt} sha256 mismatch" in proc.stderr
    assert "== repo ->" not in proc.stdout


def test_pod_sync_rejects_size_corruption_before_remote_contact(tmp_path: Path):
    scripts = tmp_path / "scripts"
    data = tmp_path / "data" / "codecontests"
    scripts.mkdir()
    data.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "pod_sync.sh", scripts / "pod_sync.sh")
    _write_complete_dataset_bundle(data)
    with (data / "paired_test.jsonl").open("ab") as handle:
        handle.write(b"x")

    proc = _run_sync_preflight(tmp_path)

    assert proc.returncode == 2
    assert "data/codecontests/paired_test.jsonl size mismatch" in proc.stderr
    assert "== repo ->" not in proc.stdout
