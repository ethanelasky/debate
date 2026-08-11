"""Config plumbing must fail loudly, not fall through to defaults.

Each guard here closes a verified silent-wrong path: a verl block ignored
because backend defaulted to tinker; a generation budget the engine would
truncate at response_length without a word; clip overrides emitted to loss
kinds that never read them; over-length prompts clipped server-side; quoted
YAML "false" reading as True; loss metrics skewed by unequal micro-batches.
"""

import argparse
import types

import pytest

from infra.backend.base import LossSpec, SamplingParams
from infra.backend.verl import (
    VerlBackend,
    VerlBackendConfig,
    _reject_over_length_prompts,
    _token_weighted_loss_means,
)
from infra.run_common import (
    CONFIG_FIELD_NAMES,
    TRAINING_KEYS,
    _strict_bool,
    build_backend,
    training_config_kwargs,
)
from infra.train import Config


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        steps=None, batch_size=None, group_size=None, lr=None,
        start_step=None, load=None, wandb_resume=None,
    )


def _verl_training(ckpt_root: str, **verl_extra) -> dict:
    return {
        "backend": "verl",
        "lora_rank": 32,
        "lr": 1e-5,
        "verl": {"n_gpus": 1, "checkpoint_dir": ckpt_root, **verl_extra},
    }


@pytest.fixture
def fake_verl_backend(monkeypatch):
    import infra.backend.verl as verl_mod

    monkeypatch.setattr(verl_mod, "VerlBackend", lambda config: config)


# ------------------------------------------- verl block vs backend selection


def test_verl_block_with_defaulted_backend_raises(tmp_path):
    tr = _verl_training(str(tmp_path))
    del tr["backend"]  # resolves to the "tinker" default
    with pytest.raises(RuntimeError) as exc:
        build_backend(tr, "some/model", "run")
    message = str(exc.value)
    assert "training.verl" in message and "'tinker'" in message


def test_verl_block_with_explicit_other_backend_raises(tmp_path):
    tr = _verl_training(str(tmp_path)) | {"backend": "tinker"}
    with pytest.raises(RuntimeError, match="training.verl"):
        build_backend(tr, "some/model", "run")


def test_verl_block_with_verl_backend_passes(tmp_path, fake_verl_backend):
    build_backend(_verl_training(str(tmp_path)), "some/model", "run")


def test_build_backend_cannot_bypass_local_artifact_validation(monkeypatch):
    import infra.run_common as run_common

    class ArtifactRejected(RuntimeError):
        pass

    def reject(model_path: str):
        raise ArtifactRejected(model_path)

    monkeypatch.setattr(run_common, "validate_local_policy_artifact", reject)
    with pytest.raises(ArtifactRejected, match="olmo32-bf16"):
        run_common.build_backend(
            {"backend": "tinker"},
            "/workspace/models/olmo32-bf16",
            "direct-runner-launch",
        )


# ------------------------------------------------- generation budget checks


def test_budget_over_response_length_raises_with_all_three_numbers(tmp_path):
    tr = _verl_training(str(tmp_path), response_length=2000)
    with pytest.raises(RuntimeError) as exc:
        build_backend(
            tr, "some/model", "run",
            gen_budgets={
                "max_completion_tokens": 4096,
                "training.eval_max_tokens": 3000,
            },
        )
    message = str(exc.value)
    assert "2000" in message and "4096" in message and "3000" in message
    assert "response_length" in message
    assert "max_completion_tokens" in message and "training.eval_max_tokens" in message


def test_budget_checked_against_default_response_length(tmp_path):
    # No response_length in the YAML: the cap is the dataclass default, and
    # the check must see through to it rather than skip.
    tr = _verl_training(str(tmp_path))
    with pytest.raises(RuntimeError, match=str(VerlBackendConfig.response_length)):
        build_backend(
            tr, "some/model", "run",
            gen_budgets={"max_completion_tokens": VerlBackendConfig.response_length + 1},
        )


def test_budget_at_or_below_response_length_passes(tmp_path, fake_verl_backend):
    tr = _verl_training(str(tmp_path), response_length=2000)
    build_backend(
        tr, "some/model", "run",
        gen_budgets={"max_completion_tokens": 2000, "training.eval_max_tokens": None},
    )


def test_omitted_budgets_leave_build_backend_unchanged(tmp_path, fake_verl_backend):
    # gen_budgets is optional; a caller that omits it (or a future runner that
    # has no budgets to declare) must get the unchecked legacy behavior.
    cfg = build_backend(_verl_training(str(tmp_path)), "some/model", "run")
    # <root>/<run name>-<run id>; the run id is per launch, so the assertion is
    # on the prefix rather than the whole path.
    assert cfg.checkpoint_dir.startswith(str(tmp_path / "run") + "-")


# ----------------------------------------------------- clip override emission


def _overrides(kind: str, **loss_kw) -> list[str]:
    return VerlBackendConfig(
        model_path="m", loss=LossSpec(kind=kind, **loss_kw)
    ).hydra_overrides()


def test_ppo_emits_clip_ratios():
    overrides = _overrides("ppo", clip_low=0.8, clip_high=1.2)
    assert "actor_rollout_ref.actor.clip_ratio_low=0.2" in overrides
    assert "actor_rollout_ref.actor.clip_ratio_high=0.2" in overrides


def test_importance_sampling_emits_effectively_unclipped_ratios():
    overrides = _overrides("importance_sampling")
    assert "actor_rollout_ref.actor.clip_ratio_low=1.0" in overrides
    assert "actor_rollout_ref.actor.clip_ratio_high=1000.0" in overrides


@pytest.mark.parametrize("kind", ["reinforce"])
def test_non_consuming_loss_kinds_omit_clip_ratios(kind):
    # gpg does not read actor.clip_ratio_* on this path; emitting the keys
    # would advertise a clip range the loss never applies.
    overrides = _overrides(kind)
    assert not any("clip_ratio" in o for o in overrides)


def test_cispo_emits_clip_epsilons():
    # verl's cispo clamps the IS weight to [1-eps_low, 1+eps_high]; omitting
    # the keys would hand it the generic clip_ratio default silently.
    overrides = _overrides("cispo", clip_low=0.8, clip_high=1.28)
    assert "actor_rollout_ref.actor.clip_ratio_low=0.2" in overrides
    assert "actor_rollout_ref.actor.clip_ratio_high=0.28" in overrides


def test_gspo_emits_paper_scale_epsilons():
    # verl's gspo reads clip_ratio_low/high as 1±eps and would otherwise fall
    # back to the PPO-scale default — effectively unclipped for seq ratios.
    overrides = _overrides("gspo", clip_low=0.9997, clip_high=1.0004)
    assert "actor_rollout_ref.actor.clip_ratio_low=0.0003" in overrides
    assert "actor_rollout_ref.actor.clip_ratio_high=0.0004" in overrides
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=gspo" in overrides


def test_gspo_rejects_ppo_scale_bounds():
    with pytest.raises(ValueError, match="paper-scale"):
        _overrides("gspo", clip_low=0.8, clip_high=1.2)


def test_gspo_rejects_lossspec_defaults():
    # LossSpec defaults are PPO bounds; a gspo config that forgets to set
    # clips must fail at build, not train unclipped.
    with pytest.raises(ValueError, match="paper-scale"):
        _overrides("gspo")


# ------------------------------------------------------- prompt-length guard


def test_over_length_prompt_raises_with_length_and_cap():
    with pytest.raises(ValueError) as exc:
        _reject_over_length_prompts([[1] * 10], prompt_length=8)
    message = str(exc.value)
    assert "10" in message and "8" in message


def test_prompts_at_the_cap_pass():
    _reject_over_length_prompts([[1] * 8, [1] * 3], prompt_length=8)


def test_sample_rejects_over_length_prompts_before_touching_the_engine():
    # No VerlBackend instantiation (needs verl/ray/GPUs): call the unbound
    # method on a stand-in carrying only .config. The guard must fire before
    # any engine attribute is touched — a stand-in without them proves it.
    dummy = types.SimpleNamespace(config=VerlBackendConfig(model_path="m", prompt_length=4))
    with pytest.raises(ValueError, match="prompt_length"):
        VerlBackend.sample(dummy, [[1] * 5], SamplingParams(max_tokens=16))


# ------------------------------------------------------------- exposed knobs


def test_norm_adv_by_std_and_rl_seed_reach_config_kwargs():
    kw = training_config_kwargs({"norm_adv_by_std": False, "rl_seed": 7}, _args())
    assert kw == {"norm_adv_by_std": False, "seed": 7}


def test_new_knobs_are_in_the_training_allowlist():
    assert {"norm_adv_by_std", "rl_seed"} <= TRAINING_KEYS
    # the YAML key is rl_seed precisely so `seed` stays dataset-only
    assert "seed" not in TRAINING_KEYS
    assert CONFIG_FIELD_NAMES["rl_seed"] == "seed"


def test_defaults_unchanged_when_yaml_omits_the_knobs():
    assert training_config_kwargs({}, _args()) == {}
    assert Config().norm_adv_by_std is True
    assert Config().seed == 0


# ------------------------------------------------------------- strict bools


def test_strict_bool_accepts_real_bools_and_01_ints():
    assert _strict_bool(True) is True
    assert _strict_bool(False) is False
    assert _strict_bool(1) is True
    assert _strict_bool(0) is False


@pytest.mark.parametrize("value", ["false", "true", "False", "yes", 2, 1.0, None])
def test_strict_bool_rejects_everything_else(value):
    with pytest.raises(ValueError):
        _strict_bool(value)


def test_quoted_false_log_transcripts_raises():
    with pytest.raises(ValueError, match="false"):
        training_config_kwargs({"log_transcripts": "false"}, _args())


def test_quoted_norm_adv_by_std_raises():
    with pytest.raises(ValueError, match="true"):
        training_config_kwargs({"norm_adv_by_std": "true"}, _args())


def test_quoted_use_remove_padding_raises(tmp_path):
    tr = _verl_training(str(tmp_path), use_remove_padding="false")
    with pytest.raises(ValueError, match="false"):
        build_backend(tr, "some/model", "run")


# ------------------------------------------------- token-weighted loss means


def test_loss_means_are_token_weighted():
    metrics = _token_weighted_loss_means([({"pg_loss": 1.0}, 100), ({"pg_loss": 3.0}, 300)])
    # unweighted mean would be 2.0 — the 300-token micro-batch must dominate
    assert metrics == {"loss/pg_loss": pytest.approx(2.5)}


def test_loss_means_skip_non_numeric_values():
    metrics = _token_weighted_loss_means([({"pg_loss": 2.0, "note": "text"}, 10)])
    assert metrics == {"loss/pg_loss": pytest.approx(2.0)}


def test_loss_means_weight_each_key_over_the_batches_that_report_it():
    metrics = _token_weighted_loss_means(
        [({"pg_loss": 1.0, "kl": 0.5}, 100), ({"pg_loss": 3.0}, 300)]
    )
    assert metrics["loss/pg_loss"] == pytest.approx(2.5)
    assert metrics["loss/kl"] == pytest.approx(0.5)


def test_loss_means_empty_input():
    assert _token_weighted_loss_means([]) == {}


# --------------------------------------- sdpa fallback when flash-attn absent

import importlib.util  # noqa: E402  (appended section; stdlib, test-only)

_SDPA = "+actor_rollout_ref.model.override_config.attn_implementation=sdpa"


def test_missing_flash_attn_emits_sdpa_override(monkeypatch):
    # verl's model config defaults attn_implementation to flash_attention_2
    # and hard-errors at model load when the package is absent; the config
    # must steer transformers to sdpa instead.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    assert _SDPA in VerlBackendConfig(model_path="m").hydra_overrides()


def test_present_flash_attn_emits_no_attn_override(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    overrides = VerlBackendConfig(model_path="m").hydra_overrides()
    assert not any("attn_implementation" in o for o in overrides)


def test_user_pinned_attn_implementation_is_not_duplicated(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    user = "+actor_rollout_ref.model.override_config.attn_implementation=eager"
    overrides = VerlBackendConfig(
        model_path="m", extra_overrides=(user,)
    ).hydra_overrides()
    assert [o for o in overrides if "attn_implementation" in o] == [user]
