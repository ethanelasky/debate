"""Topology auto-detection and merge (infra/run_common.resolve_topology):
hardware plumbing lives in configs/topologies.yaml keyed by "<count>x<GPU>";
arms keep science knobs and win any key collision, except extra_overrides
which concatenates so topology engine flags survive arm additions."""

import os
from unittest import mock

import pytest

import infra.run_common as rc
from infra.run_common import (
    VERL_KEYS,
    _short_gpu_name,
    apply_topology,
    detect_topology_key,
    resolve_topology,
)


def test_short_gpu_name_strips_vendor_and_suffix():
    assert _short_gpu_name("NVIDIA B200") == "B200"
    assert _short_gpu_name("NVIDIA H100 80GB HBM3") == "H100"
    assert _short_gpu_name("NVIDIA H200") == "H200"
    assert _short_gpu_name("Tesla V100-SXM2-16GB") == "V100-SXM2-16GB"


def _fake_smi(names: list[str]):
    out = mock.Mock(returncode=0, stdout="".join(f"{n}\n" for n in names))
    return mock.patch("subprocess.run", return_value=out)


def test_detect_key_counts_and_names():
    with _fake_smi(["NVIDIA B200", "NVIDIA B200"]):
        assert detect_topology_key() == "2xB200"


def test_detect_none_without_nvidia_smi():
    with mock.patch("subprocess.run", side_effect=FileNotFoundError):
        assert detect_topology_key() is None


def test_resolve_empty_on_gpuless_machine(tmp_path):
    with mock.patch.object(rc, "detect_topology_key", return_value=None):
        assert resolve_topology(str(tmp_path / "absent.yaml")) == {}


def test_resolve_known_key_sets_provenance(tmp_path):
    p = tmp_path / "topo.yaml"
    p.write_text("2xB200: {n_gpus: 2, rollout_tp: 2}\n")
    with mock.patch.object(rc, "detect_topology_key", return_value="2xB200"):
        os.environ.pop("DEBATE_TOPOLOGY", None)
        assert resolve_topology(str(p)) == {"n_gpus": 2, "rollout_tp": 2}
        assert os.environ.get("DEBATE_TOPOLOGY") == "2xB200"


def test_resolve_unknown_key_is_hard_error(tmp_path):
    p = tmp_path / "topo.yaml"
    p.write_text("2xB200: {n_gpus: 2}\n")
    with mock.patch.object(rc, "detect_topology_key", return_value="4xH100"):
        with pytest.raises(RuntimeError, match="4xH100"):
            resolve_topology(str(p))


def test_resolve_rejects_keys_outside_verl_contract(tmp_path):
    p = tmp_path / "topo.yaml"
    p.write_text("2xB200: {n_gpus: 2, learning_rate: 1}\n")
    with mock.patch.object(rc, "detect_topology_key", return_value="2xB200"):
        with pytest.raises(RuntimeError, match="learning_rate"):
            resolve_topology(str(p))


def test_shipped_topologies_file_is_valid():
    import yaml

    with open(rc.TOPOLOGY_FILE) as fh:
        table = yaml.safe_load(fh)
    assert table, "shipped topologies.yaml must not be empty"
    for key, entry in table.items():
        assert set(entry) <= VERL_KEYS, f"{key} carries non-verl keys"
        assert "n_gpus" in entry and "rollout_tp" in entry, key


def test_apply_topology_arm_wins_scalar_keys():
    merged = apply_topology({"gpu_memory_utilization": 0.5}, {"gpu_memory_utilization": 0.42, "n_gpus": 2})
    assert merged["gpu_memory_utilization"] == 0.5
    assert merged["n_gpus"] == 2


def test_apply_topology_concatenates_extra_overrides():
    topo = {"extra_overrides": ["++a=1"]}
    arm = {"extra_overrides": ["++b=2", "++a=1"]}
    assert apply_topology(arm, topo)["extra_overrides"] == ["++a=1", "++b=2"]


def test_apply_topology_empty_topology_is_identity():
    arm = {"n_gpus": 2, "extra_overrides": ["++a=1"]}
    assert apply_topology(dict(arm), {}) == arm
