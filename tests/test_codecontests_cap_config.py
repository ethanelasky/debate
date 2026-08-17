"""Launch-contract checks for the paired long CodeContests cap runs."""

from __future__ import annotations

import pytest
import yaml

from infra.config import load_experiment


@pytest.mark.parametrize(
    ("experiment", "cap", "soft_cap"),
    [
        ("codecontests_rlvr_olmo31_32b_cap1024_long", 1024, 512),
        ("codecontests_rlvr_olmo31_32b_cap2048_long", 2048, 1024),
    ],
)
def test_long_cap_runs_use_the_decaying_lr_recipe(experiment, cap, soft_cap):
    config = load_experiment("configs/codecontests_rlvr_olmo.yaml", experiment)
    training = config["training"]

    assert config["model"] == (
        "/workspace/models/olmo32-bf16-legacy-audited-fc84a4f"
    )
    assert "think_tokens" not in config
    assert config["max_completion_tokens"] == cap
    assert config["dataset"]["soft_token_budget"] == soft_cap
    assert config["dataset"]["verifier"] == "piston"
    assert config["dataset"]["piston_url"] == "http://127.0.0.1:2000"
    assert config["dataset"]["piston_python_version"] == "3.12.0"
    assert training["lr"] == pytest.approx(1.0e-4)
    assert training["warmup_steps"] == 10
    assert training["lr_schedule"] == "cosine"
    assert training["min_lr_ratio"] == pytest.approx(0.2)
    assert training["steps"] == 100
    assert training["verl"]["response_length"] == cap


@pytest.mark.parametrize("cap", [1024, 2048])
def test_cap_smoke_preserves_the_long_run_contract(cap):
    long = load_experiment(
        "configs/codecontests_rlvr_olmo.yaml",
        f"codecontests_rlvr_olmo31_32b_cap{cap}_long",
    )
    smoke = load_experiment(
        "configs/codecontests_rlvr_olmo.yaml",
        f"codecontests_rlvr_olmo31_32b_cap{cap}_smoke1",
    )

    assert smoke["model"] == long["model"]
    assert smoke["dataset"] == long["dataset"]
    assert smoke["max_completion_tokens"] == long["max_completion_tokens"]
    assert smoke["training"]["batch_size"] == long["training"]["batch_size"]
    assert smoke["training"]["group_size"] == long["training"]["group_size"]
    assert smoke["training"]["lr_schedule"] == long["training"]["lr_schedule"]
    assert smoke["training"]["steps"] == 1
    assert smoke["training"]["eval_every"] == 0
    assert smoke["training"]["save_every"] == 0


@pytest.mark.parametrize("topology", ["2xB200", "2xH200"])
def test_tp2_topologies_use_the_run_proven_memory_safe_sync(topology):
    with open("configs/topologies.yaml") as stream:
        settings = yaml.safe_load(stream)[topology]

    assert settings["n_gpus"] == 2
    assert settings["rollout_tp"] == 2
    assert "actor_rollout_ref.rollout.load_format=safetensors" in settings[
        "extra_overrides"
    ]
    assert "actor_rollout_ref.rollout.layered_summon=true" in settings[
        "extra_overrides"
    ]
