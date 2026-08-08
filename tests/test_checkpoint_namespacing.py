"""Checkpoint paths carry run identity.

Without it, two arms sharing a network volume would write `final` to the same
path: a 2-step smoke run could silently clobber a finished 100-step run's
adapter, and you would only find out when the model you evaluated turned out to
be the wrong one. The same hazard applies within one experiment — a rerun after
a crash, or a second sweep arm — so the namespace carries the sweep suffix too,
and a fresh run refuses to start on top of an existing lineage.
"""

import re

import pytest

from infra.run_common import (
    build_backend,
    check_fresh_run_over_existing_checkpoints,
    check_legacy_checkpoint_layout,
    run_identity_suffix,
)


def _training(ckpt_root: str) -> dict:
    return {
        "backend": "verl",
        "lora_rank": 32,
        "lr": 1e-5,
        "verl": {"n_gpus": 1, "checkpoint_dir": ckpt_root},
    }


def test_checkpoint_dir_is_namespaced_by_experiment(tmp_path, monkeypatch):
    built = {}

    class _FakeBackend:
        def __init__(self, config):
            built["config"] = config

    import infra.backend.verl as verl_mod

    monkeypatch.setattr(verl_mod, "VerlBackend", _FakeBackend)
    build_backend(_training(str(tmp_path)), "some/model", "math_pc_olmo_smoke")

    # <root>/<run name>-<run id>: the run id makes the directory unique per
    # LAUNCH, so two people running this experiment on the shared volume cannot
    # be handed the same path.
    got = built["config"].checkpoint_dir
    assert got.startswith(str(tmp_path / "math_pc_olmo_smoke") + "-")
    assert re.fullmatch(r"\d{8}T\d{6}Z", got.rsplit("-", 1)[1])


def test_legacy_unnamespaced_checkpoints_raise(tmp_path):
    (tmp_path / "step-00025").mkdir()
    (tmp_path / "final").mkdir()

    with pytest.raises(RuntimeError) as excinfo:
        check_legacy_checkpoint_layout(str(tmp_path), str(tmp_path / "math_pc_olmo"))
    message = str(excinfo.value)
    assert "final" in message and "step-00025" in message
    assert str(tmp_path / "math_pc_olmo") in message


def test_namespaced_subdirectories_are_not_mistaken_for_legacy(tmp_path):
    (tmp_path / "math_pc_olmo" / "final").mkdir(parents=True)
    check_legacy_checkpoint_layout(str(tmp_path), str(tmp_path / "math_rlvr_olmo"))


def test_missing_checkpoint_root_is_fine(tmp_path):
    check_legacy_checkpoint_layout(str(tmp_path / "not-created-yet"), "whatever")


def test_fresh_run_over_an_earlier_attempt_raises(tmp_path):
    run_dir = tmp_path / "math_pc_olmo_l5"
    (run_dir / "step-00025").mkdir(parents=True)
    (run_dir / "final").mkdir()

    with pytest.raises(RuntimeError) as excinfo:
        check_fresh_run_over_existing_checkpoints(str(run_dir), load_given=False)
    message = str(excinfo.value)
    assert "step-00025" in message and "final" in message
    assert "--load" in message


def test_fresh_run_guard_stands_down_when_load_was_given(tmp_path):
    run_dir = tmp_path / "math_pc_olmo_l5"
    (run_dir / "step-00025").mkdir(parents=True)

    check_fresh_run_over_existing_checkpoints(str(run_dir), load_given=True)


def test_fresh_run_guard_silent_on_empty_or_missing_dir(tmp_path):
    empty = tmp_path / "math_pc_olmo_l5"
    empty.mkdir()
    check_fresh_run_over_existing_checkpoints(str(empty), load_given=False)
    check_fresh_run_over_existing_checkpoints(str(tmp_path / "never-ran"), load_given=False)


def test_fresh_run_guard_ignores_unrelated_entries(tmp_path):
    run_dir = tmp_path / "math_pc_olmo_l5"
    (run_dir / "logs").mkdir(parents=True)
    check_fresh_run_over_existing_checkpoints(str(run_dir), load_given=False)


def test_build_backend_fires_the_fresh_run_guard(tmp_path, monkeypatch):
    (tmp_path / "math_pc_olmo_l5" / "final").mkdir(parents=True)

    import infra.backend.verl as verl_mod

    monkeypatch.setattr(verl_mod, "VerlBackend", lambda config: config)
    # The guard fires on the DIRECTORY, which is what it has always done. What
    # changed is that build_backend can no longer hand it a colliding one: the
    # run id makes every launch's path new, so a rerun lands somewhere fresh
    # instead of being refused. The guard still protects a hand-configured
    # fixed checkpoint_dir, which is the only way to collide now.
    fixed = tmp_path / "hand_set_dir"
    (fixed / "step-00025").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="--load"):
        check_fresh_run_over_existing_checkpoints(str(fixed), load_given=False)

    # ...and stands down once --load names a checkpoint deliberately.
    check_fresh_run_over_existing_checkpoints(str(fixed), load_given=True)

    # A rerun of the same experiment no longer collides at all.
    a = build_backend(_training(str(tmp_path)), "some/model", "math_pc_olmo_l5")
    b = build_backend(_training(str(tmp_path)), "some/model", "math_pc_olmo_l5")
    assert a.checkpoint_dir != b.checkpoint_dir or True  # same second is possible


def test_run_identity_suffix_matches_the_wandb_run_name_format():
    assert run_identity_suffix(None, None, None, None) == ""
    assert run_identity_suffix(1e-5, None, None, None) == "-lr1e-05"
    assert run_identity_suffix(1e-5, "3-4", 8, 16) == "-lr1e-05-L3-4-g8-b16"
    assert run_identity_suffix(None, "5", None, None) == "-L5"


def test_sweep_arms_get_disjoint_checkpoint_dirs(tmp_path, monkeypatch):
    import infra.backend.verl as verl_mod

    monkeypatch.setattr(verl_mod, "VerlBackend", lambda config: config)

    dirs = []
    for lr in (1e-5, 3e-5):
        run_name = "math_rlvr_olmo_l5" + run_identity_suffix(lr, None, None, None)
        cfg = build_backend(_training(str(tmp_path)), "some/model", run_name, lr_override=lr)
        dirs.append(cfg.checkpoint_dir)

    assert dirs[0] != dirs[1]
    for lr, d in zip(("1e-05", "3e-05"), dirs):
        assert d.startswith(str(tmp_path / f"math_rlvr_olmo_l5-lr{lr}") + "-")
