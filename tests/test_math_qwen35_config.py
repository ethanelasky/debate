import copy
from pathlib import Path

from infra.config import resolve_experiments_from_file


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "math_qwen35.yaml"


def test_qwen9b_science_arm_changes_only_model_and_required_runtime_bounds():
    experiments = resolve_experiments_from_file(CONFIG_PATH)
    parent = experiments["mathl5_qwen35_cispo_sub8"]
    science = experiments["mathl5_qwen9b_cispo_qwen4b_sub8"]

    expected = copy.deepcopy(parent)
    expected["model"] = "Qwen/Qwen3.5-9B"
    expected["training"]["verl"]["extra_overrides"].append(
        "actor_rollout_ref.rollout.max_num_seqs=256"
    )
    expected["training"]["verl"]["gpu_memory_utilization"] = 0.38

    assert science == expected
    assert science["dataset"]["train_filter_file"] == (
        "configs/filters/l5_sub8_qwen35.json"
    )
    assert science["training"]["verl"]["extra_overrides"][-1] == (
        "actor_rollout_ref.rollout.max_num_seqs=256"
    )
    assert science["training"]["verl"]["gpu_memory_utilization"] == 0.38


def test_qwen9b_smoke_arm_changes_only_smoke_training_settings():
    experiments = resolve_experiments_from_file(CONFIG_PATH)
    science = experiments["mathl5_qwen9b_cispo_qwen4b_sub8"]
    smoke = experiments["mathl5_qwen9b_cispo_qwen4b_sub8_smoke"]

    expected = copy.deepcopy(science)
    expected["training"].update(
        {
            "steps": 2,
            "batch_size": 2,
            "group_size": 4,
            "eval_every": 0,
            "save_every": 0,
        }
    )

    assert smoke == expected
