"""Experiment-config key validation: a typo must fail at launch, not train the
wrong arm in silence. The regression net at the bottom is the point — every
shipped config must survive the allowlists.
"""

import copy
from pathlib import Path

import pytest

from infra.config import resolve_experiments_from_file, runnable_experiments
from infra.run_debate import validate_experiment as validate_debate
from infra.run_eval import validate_experiment as validate_eval
from infra.run_rlvr import validate_experiment as validate_rlvr

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _debate_exp() -> dict:
    return {
        "protocol": {"turns": [{"alice": [{"name": "proposal", "kind": "solution"}]}]},
        "prompt_config": {"file_path": "x.yaml", "entry": "e"},
        "agents": {"alice": {"trained": True, "model_settings": {"model_type": "tinker"}}},
        "dataset": {"type": "math"},
        "training": {"steps": 2, "lr": 1e-5, "verl": {"n_gpus": 1}},
        "first_speech_non_debate_aware": True,
    }


def _rlvr_exp() -> dict:
    return {
        "model": "allenai/Olmo-3-7B-Instruct-DPO",
        "max_completion_tokens": 2000,
        "dataset": {"type": "math"},
        "training": {"steps": 2, "lr": 1e-5, "verl": {"n_gpus": 1}},
    }


def test_valid_experiments_pass():
    validate_debate(_debate_exp())
    validate_rlvr(_rlvr_exp())


def test_misspelled_top_level_key_raises():
    exp = _debate_exp()
    exp["first_speech_non_debate_awre"] = exp.pop("first_speech_non_debate_aware")
    with pytest.raises(ValueError) as exc:
        validate_debate(exp)
    assert "first_speech_non_debate_awre" in str(exc.value)


def test_misspelled_training_key_raises():
    for exp, validate in ((_debate_exp(), validate_debate), (_rlvr_exp(), validate_rlvr)):
        exp["training"]["kl_coeff"] = 0.02
        with pytest.raises(ValueError) as exc:
            validate(exp)
        assert "kl_coeff" in str(exc.value)


def test_misspelled_verl_key_raises():
    for exp, validate in ((_debate_exp(), validate_debate), (_rlvr_exp(), validate_rlvr)):
        exp["training"]["verl"]["gpu_mem_utilization"] = 0.6
        with pytest.raises(ValueError) as exc:
            validate(exp)
        assert "gpu_mem_utilization" in str(exc.value)


def test_unknown_agent_key_raises():
    exp = _debate_exp()
    exp["agents"]["alice"]["train"] = True  # meant `trained`
    with pytest.raises(ValueError) as exc:
        validate_debate(exp)
    assert "train" in str(exc.value) and "agents.alice" in str(exc.value)


def test_private_keys_pass():
    exp = _debate_exp()
    exp["_protocols"] = {"pc": {"turns": []}}
    exp["_extends"] = "_base"
    validate_debate(exp)
    rlvr = _rlvr_exp()
    rlvr["_anything"] = 1
    validate_rlvr(rlvr)


def test_rlvr_rejects_debate_only_keys():
    exp = _rlvr_exp()
    exp["agents"] = {}
    with pytest.raises(ValueError) as exc:
        validate_rlvr(exp)
    assert "agents" in str(exc.value)


def test_rlvr_rejects_prompt_config_and_protocol():
    # Debate-layer keys: RLVR builds its prompt from the family's
    # answer-generation config, so these are typos here (they were legal
    # before MB's blind view moved to infra/prompts/tasks/monitoringbench.yaml).
    for key, value in (("prompt_config", {"x": 1}), ("protocol", {"turns": []})):
        exp = _rlvr_exp()
        exp[key] = value
        with pytest.raises(ValueError, match=key):
            validate_rlvr(exp)


def _shipped_experiments():
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        if path.name == "topologies.yaml":  # hardware table, not experiments
            continue                        # (validated by test_topology.py)
        resolved = resolve_experiments_from_file(path)
        for name in runnable_experiments(resolved):
            yield pytest.param(resolved[name], id=f"{path.name}::{name}")


@pytest.mark.parametrize("exp", list(_shipped_experiments()))
def test_shipped_configs_validate(exp):
    """Regression net: the allowlists must not reject anything we ship.

    Routing: no agents -> RLVR; agents + training -> debate trainer; agents
    without training -> the eval-only runner (run_eval), whose allowlist
    carries the eval-only keys (speech_token_limit, choice_positions, ...)."""
    if "agents" not in exp:
        validate = validate_rlvr
    elif "training" in exp:
        validate = validate_debate
    else:
        validate = validate_eval
    validate(copy.deepcopy(exp))
