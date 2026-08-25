"""The Qwen3.5 pair shares one objective, and no verl arm divides the
advantage by completion length on top of verl's own token-mean."""

from pathlib import Path

import pytest
import yaml

from infra.config import (
    load_experiment,
    resolve_experiments_from_file,
    runnable_experiments,
)
from infra.train import Config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SHARED_FILE = CONFIG_DIR / "_qwen35_training.yaml"
DEBATE_ARM = ("math_pc_debate.yaml", "mathl5_qwen35_pc_debate_cispo_verl")
RLVR_ARM = ("math_qwen35.yaml", "mathl5_qwen35_cispo")

# adv_length_norm modes whose scale carries a 1/completion_tokens factor.
PER_TOKEN_NORMS = {"datum", "trajectory"}
VERL_DEFAULT_AGG = "token-mean"


def _shared_objective_keys() -> list[str]:
    raw = yaml.safe_load(SHARED_FILE.read_text())
    return sorted(raw["_qwen35_train_base"]["training"])


def _effective_loss_agg_mode(training: dict) -> str:
    """Last loss_agg_mode in extra_overrides wins over the backend's default."""
    mode = VERL_DEFAULT_AGG
    for override in (training.get("verl") or {}).get("extra_overrides") or []:
        key, sep, value = str(override).partition("=")
        if sep and key.lstrip("+").endswith("actor.loss_agg_mode"):
            mode = value
    return mode


def _verl_arms() -> list:
    params = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        experiments = resolve_experiments_from_file(path)
        for name in runnable_experiments(experiments):
            training = (experiments[name] or {}).get("training") or {}
            if training.get("backend") == "verl":
                params.append(pytest.param(path.name, name, id=f"{path.stem}::{name}"))
    return params


@pytest.mark.parametrize("key", _shared_objective_keys())
def test_shipped_qwen35_arms_agree_on_every_shared_training_key(key):
    debate = load_experiment(CONFIG_DIR / DEBATE_ARM[0], DEBATE_ARM[1])["training"]
    rlvr = load_experiment(CONFIG_DIR / RLVR_ARM[0], RLVR_ARM[1])["training"]
    assert debate[key] == rlvr[key], (
        f"{key} differs: debate={debate[key]!r} rlvr={rlvr[key]!r}. "
        f"A key declared in {SHARED_FILE.name} must reach both arms; an "
        "override that wins over it is a silent objective divergence."
    )


def test_the_debate_arm_takes_adv_length_norm_from_the_shared_base():
    base = load_experiment(CONFIG_DIR / DEBATE_ARM[0], "_base_math_pc")["training"]
    arm = load_experiment(CONFIG_DIR / DEBATE_ARM[0], DEBATE_ARM[1])["training"]
    assert base["adv_length_norm"] in PER_TOKEN_NORMS
    assert arm["adv_length_norm"] == "none"


@pytest.mark.parametrize("filename,name", _verl_arms())
def test_verl_arms_do_not_length_normalize_advantages_twice(filename, name):
    training = load_experiment(CONFIG_DIR / filename, name)["training"]
    norm = training.get("adv_length_norm", Config.adv_length_norm)
    if norm not in PER_TOKEN_NORMS:
        return
    assert _effective_loss_agg_mode(training) != VERL_DEFAULT_AGG, (
        f"{name} runs verl at loss_agg_mode={VERL_DEFAULT_AGG} with "
        f"adv_length_norm={norm!r}: the engine already divides by the batch's "
        "token count, so the advantage scale is applied twice."
    )
