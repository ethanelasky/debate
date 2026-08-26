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


# -- the GPU split is one decision, not two ---------------------------------
#
# n_gpus and rollout_tp together say how a box's GPUs are divided between the
# FSDP training group and the rollout engine's tensor-parallel replicas. verl
# asserts the relationship in TinkerActorRolloutRefWorker.init_model(), which
# is reached only after the env restore, Ray startup and model shard have all
# been paid for. These tests move that assertion to launch time, where it is
# free.


def test_the_half_override_that_cost_two_b200s():
    """qwen35_fmt_warmup_l3, 2026-08-26: the arm pinned n_gpus=1 and said
    nothing about rollout_tp, so the 2xB200 topology supplied TP=2. Nineteen
    minutes later verl said "rollout world_size: 1 is not divisible by
    infer_world_size: 2" and the run exited 1."""
    with pytest.raises(RuntimeError) as exc:
        apply_topology(
            {"n_gpus": 1, "gpu_memory_utilization": 0.45},
            {"n_gpus": 2, "rollout_tp": 2, "gpu_memory_utilization": 0.42},
        )
    assert "n_gpus=1" in str(exc.value) and "rollout_tp=2" in str(exc.value)


def test_the_error_says_which_side_supplied_which_number():
    """Without this the reader goes hunting for a `rollout_tp: 2` that appears
    in no arm anywhere — it came from the machine, not the experiment."""
    with pytest.raises(RuntimeError) as exc:
        apply_topology({"n_gpus": 1}, {"n_gpus": 2, "rollout_tp": 2})
    message = str(exc.value)
    arm_at = message.index("n_gpus=1")
    topo_at = message.index("rollout_tp=2")
    assert "the arm" in message[arm_at:topo_at]
    assert "topology" in message[topo_at:]


def test_a_matched_pair_passes_whatever_the_topology_says():
    merged = apply_topology(
        {"n_gpus": 1, "rollout_tp": 1}, {"n_gpus": 2, "rollout_tp": 2}
    )
    assert (merged["n_gpus"], merged["rollout_tp"]) == (1, 1)


def test_tp_dividing_the_group_is_fine():
    """4 GPUs as two TP2 rollout replicas is the 4xB200 plan, not an error."""
    assert apply_topology({}, {"n_gpus": 4, "rollout_tp": 2})["n_gpus"] == 4


def test_absent_keys_fall_back_to_one_and_pass():
    assert apply_topology({}, {}) == {}


def test_absent_keys_match_the_backend_defaults():
    """check_gpu_split writes 1 out rather than importing VerlBackendConfig,
    which drags in torch and vllm. This is what keeps the two honest."""
    verl = pytest.importorskip("infra.backend.verl")
    assert verl.VerlBackendConfig.n_gpus == 1
    assert verl.VerlBackendConfig.rollout_tp == 1


def test_zero_tp_is_refused_rather_than_dividing_by_zero():
    with pytest.raises(RuntimeError):
        apply_topology({"rollout_tp": 0}, {"n_gpus": 2})


def test_every_shipped_arm_resolves_on_every_shipped_topology():
    """The regression net. The failure was not a typo in one arm — the sweep
    that found it counted 132 broken (arm, topology) pairs across six files,
    because every arm tuned for one GPU inherited TP from whatever box it
    landed on. This is what stops the next arm reintroducing it: a new
    n_gpus without a rollout_tp fails here, on a laptop, for free.
    """
    import glob

    import yaml

    from infra.config import resolve_experiments_from_file, runnable_experiments

    with open(rc.TOPOLOGY_FILE) as fh:
        topologies = yaml.safe_load(fh)

    broken, checked = [], 0
    for path in sorted(glob.glob("configs/*.yaml")):
        experiments = resolve_experiments_from_file(path)
        for name in runnable_experiments(experiments):
            verl = ((experiments[name].get("training") or {}).get("verl")) or {}
            for key, topology in topologies.items():
                checked += 1
                try:
                    apply_topology(dict(verl), dict(topology))
                except RuntimeError as exc:
                    broken.append(f"{path}::{name} on {key}: {exc}")

    # Guard the guard: a refactor that stops resolving configs would otherwise
    # turn this into a test that passes by checking nothing.
    assert checked > 100, f"only {checked} combinations checked; the sweep broke"
    assert not broken, "\n".join(broken[:10])
