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
from safetensors.torch import save_file
import torch
import yaml

from infra.config import load_experiment
from infra.run_common import (
    CANONICAL_BF16_CONVERTER,
    CANONICAL_BF16_MARKER,
    CANONICAL_OLMO32_REPO,
    CANONICAL_OLMO32_REVISION,
    apply_topology,
    validate_local_policy_artifact,
)
from scripts.convert_hf_safetensors_bf16 import convert_repository

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "codecontests_rlvr_olmo.yaml"
TOPOLOGIES = ROOT / "configs" / "topologies.yaml"
MODEL_ARTIFACT = "/workspace/models/olmo32-bf16-legacy-audited-fc84a4f"

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
    assert tr["lr"] == pytest.approx(1.0e-4)
    assert tr["warmup_steps"] == 10
    assert tr["lr_schedule"] == "cosine"
    assert tr["min_lr_ratio"] == pytest.approx(0.2)
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


def test_pod_sync_uses_macos_compatible_rsync_progress_flag():
    script = (ROOT / "scripts" / "pod_sync.sh").read_text()
    assert "rsync -a --info=progress2" not in script
    assert "rsync -a --progress" in script
    assert script.count("--no-owner --no-group") == 2


def _write_tiny_complete_canonical_model(root: Path) -> dict:
    root.mkdir()
    support = {
        "config.json": json.dumps(
            {
                "architectures": ["Olmo3ForCausalLM"],
                "torch_dtype": "bfloat16",
                "dtype": "bfloat16",
            }
        ).encode(),
        "tokenizer_config.json": b'{"model_max_length":4096}\n',
        "tokenizer.json": b'{"version":"1.0"}\n',
    }
    for name, payload in support.items():
        (root / name).write_bytes(payload)

    shard_payloads = {
        "model-00001-of-00002.safetensors": b"tiny-bf16-shard-one",
        "model-00002-of-00002.safetensors": b"tiny-bf16-shard-two",
    }
    tensor_sizes = {
        "model-00001-of-00002.safetensors": 12,
        "model-00002-of-00002.safetensors": 14,
    }
    for name, payload in shard_payloads.items():
        (root / name).write_bytes(payload)
    weight_map = {
        "model.embed.weight": "model-00001-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors",
    }
    index = {
        "metadata": {"total_size": sum(tensor_sizes.values())},
        "weight_map": weight_map,
    }
    index_path = root / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index))

    records = {
        name: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "file_bytes": len(payload),
            "tensor_bytes": tensor_sizes[name],
        }
        for name, payload in shard_payloads.items()
    }
    marker = {
        "schema_version": 1,
        "converter": CANONICAL_BF16_CONVERTER,
        "complete": True,
        "source": {
            "repo": CANONICAL_OLMO32_REPO,
            "revision": CANONICAL_OLMO32_REVISION,
            "index_sha256": hashlib.sha256(b"pinned-fp32-source-index").hexdigest(),
            "support_sha256": {
                name: hashlib.sha256(b"source-" + payload).hexdigest()
                for name, payload in support.items()
            },
        },
        "support_files": list(support),
        "weight_shards": list(shard_payloads),
        "weight_shard_identity": {
            name: {
                "size": len(payload) * 2,
                "sha256": hashlib.sha256(b"source-" + payload).hexdigest(),
            }
            for name, payload in shard_payloads.items()
        },
        "shards": records,
        "weight_file_bytes": sum(len(payload) for payload in shard_payloads.values()),
        "tensor_bytes": sum(tensor_sizes.values()),
        "output_index_sha256": _sha256(index_path),
        "output_support_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in support.items()
        },
    }
    (root / CANONICAL_BF16_MARKER).write_text(json.dumps(marker))
    return marker


def _validate_tiny_canonical(root: Path) -> str:
    return validate_local_policy_artifact(
        str(root),
        canonical_olmo32_path=str(root),
        canonical_size_range=(1, 1_000_000),
    )


def test_canonical_model_preflight_requires_complete_converter_commit(tmp_path: Path):
    root = tmp_path / "olmo32-bf16"
    _write_tiny_complete_canonical_model(root)
    assert _validate_tiny_canonical(root) == str(root)

    (root / "model.safetensors.index.json").unlink()
    with pytest.raises(ValueError, match="safetensors index"):
        _validate_tiny_canonical(root)

    for spelling in (f"{root}/", f"{root.parent}/./{root.name}"):
        with pytest.raises(ValueError, match="safetensors index"):
            validate_local_policy_artifact(
                spelling,
                canonical_olmo32_path=str(root),
                canonical_size_range=(1, 1_000_000),
            )


def test_canonical_model_preflight_rejects_symlink_alias(tmp_path: Path):
    root = tmp_path / "olmo32-bf16"
    _write_tiny_complete_canonical_model(root)
    alias = tmp_path / "model-alias"
    alias.symlink_to(root, target_is_directory=True)

    with pytest.raises(ValueError, match="aliases and symlink components"):
        validate_local_policy_artifact(
            str(alias),
            canonical_olmo32_path=str(root),
            canonical_size_range=(1, 1_000_000),
        )


def test_canonical_preflight_accepts_real_converter_output(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    save_file(
        {"model.weight": torch.arange(8, dtype=torch.float32)},
        source / shard_name,
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 32},
                "weight_map": {"model.weight": shard_name},
            }
        )
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Olmo3ForCausalLM"],
                "torch_dtype": "float32",
                "dtype": "float32",
            }
        )
    )
    (source / "tokenizer_config.json").write_text('{"model_max_length":4096}\n')
    (source / "tokenizer.json").write_text('{"version":"1.0"}\n')
    output = tmp_path / "converted"
    convert_repository(
        repo=CANONICAL_OLMO32_REPO,
        revision=CANONICAL_OLMO32_REVISION,
        source_dir=source,
        output_dir=output,
        staging_dir=tmp_path / "staging",
        expected_size_bytes=(1, 1_000_000),
        progress=lambda _message: None,
    )

    assert validate_local_policy_artifact(
        str(output),
        canonical_olmo32_path=str(output),
        canonical_size_range=(1, 1_000_000),
    ) == str(output)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("complete", False),
        ("converter", "wrong-converter"),
        ("schema_version", 0),
    ],
)
def test_canonical_model_preflight_rejects_nonfinal_marker(
    tmp_path: Path, field: str, value: object
):
    root = tmp_path / "olmo32-bf16"
    marker = _write_tiny_complete_canonical_model(root)
    marker[field] = value
    (root / CANONICAL_BF16_MARKER).write_text(json.dumps(marker))

    with pytest.raises(ValueError, match="no complete conversion marker"):
        _validate_tiny_canonical(root)


def test_canonical_model_preflight_rejects_changed_shard_bytes(tmp_path: Path):
    root = tmp_path / "olmo32-bf16"
    _write_tiny_complete_canonical_model(root)
    shard = root / "model-00001-of-00002.safetensors"
    original = shard.read_bytes()
    shard.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    with pytest.raises(ValueError, match="shard failed integrity check"):
        _validate_tiny_canonical(root)


def test_pod_run_delegates_local_model_preflight_to_validated_helper():
    script = (ROOT / "scripts" / "pod_run.sh").read_text()
    assert "validate_local_policy_artifact(model)" in script
    assert "if indices" not in script


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
