"""Checkpoint paths carry run identity.

Without it, two arms sharing a network volume would write `final` to the same
path: a 2-step smoke run could silently clobber a finished 100-step run's
adapter, and you would only find out when the model you evaluated turned out to
be the wrong one. The same hazard applies within one experiment — a rerun after
a crash, or a second sweep arm — so the namespace carries the sweep suffix too,
and a fresh run refuses to start on top of an existing lineage.
"""

import uuid

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


def test_checkpoint_dir_is_namespaced_by_run_and_exact_launch(tmp_path, monkeypatch):
    built = {}

    class _FakeBackend:
        def __init__(self, config):
            assert (tmp_path / "math_pc_olmo_smoke" / "attempt-7").is_dir()
            built["config"] = config

    import infra.backend.verl as verl_mod

    monkeypatch.setattr(verl_mod, "VerlBackend", _FakeBackend)
    build_backend(
        _training(str(tmp_path)),
        "some/model",
        "math_pc_olmo_smoke",
        launch_namespace="attempt-7",
    )

    # The scheduler/manual launch namespace is a separate exact component,
    # shared byte-for-byte with every other sink.
    got = built["config"].checkpoint_dir
    assert got == str(tmp_path / "math_pc_olmo_smoke" / "attempt-7")


def test_legacy_unnamespaced_checkpoints_raise(tmp_path):
    (tmp_path / "step-00025").mkdir()
    (tmp_path / "final").mkdir()

    with pytest.raises(RuntimeError) as excinfo:
        check_legacy_checkpoint_layout(str(tmp_path), str(tmp_path / "math_pc_olmo"))
    message = str(excinfo.value)
    assert "final" in message and "step-00025" in message
    assert str(tmp_path / "math_pc_olmo") in message
    assert "moving these into the future launch path" in message
    assert "different empty root" in message


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


def test_load_never_authorizes_adopting_existing_launch_leaf(tmp_path):
    run_dir = tmp_path / "math_pc_olmo_l5"
    (run_dir / "step-00025").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="does not authorize this destination"):
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
    existing = tmp_path / "math_pc_olmo_l5" / "attempt-existing"
    (existing / "final").mkdir(parents=True)

    import infra.backend.verl as verl_mod

    monkeypatch.setattr(verl_mod, "VerlBackend", lambda config: config)
    # The low-level guard remains useful for explicit paths.
    fixed = tmp_path / "hand_set_dir"
    (fixed / "step-00025").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="--load"):
        check_fresh_run_over_existing_checkpoints(str(fixed), load_given=False)

    # --load is only a read source and never weakens destination refusal.
    with pytest.raises(RuntimeError, match="does not authorize this destination"):
        check_fresh_run_over_existing_checkpoints(str(fixed), load_given=True)

    # A scheduler-supplied attempt namespace is a reservation: reuse refuses,
    # even when --load was supplied. Manual UUID fallback makes this rare on
    # the manual path; it is future-facing protection for scheduler attempts.
    with pytest.raises(RuntimeError, match="does not authorize this destination"):
        build_backend(
            _training(str(tmp_path)),
            "some/model",
            "math_pc_olmo_l5",
            load_given=True,
            launch_namespace="attempt-existing",
        )


def test_compat_backend_and_config_share_one_generated_namespace(
    tmp_path, monkeypatch
):
    import infra.backend.verl as verl_mod
    from infra.train import Config, train

    class _FakeBackend:
        tokenizer = None

        def __init__(self, config):
            self.config = config
            self.saved = []

        def save(self, name):
            self.saved.append(name)

    monkeypatch.setattr(verl_mod, "VerlBackend", _FakeBackend)
    backend = build_backend(_training(str(tmp_path)), "some/model", "run")
    cfg = Config(
        steps=0,
        eval_every=0,
        save_every=0,
        log_transcripts=False,
        launch_namespace=None,
    )

    train(object(), backend, cfg)

    namespace = backend.config.launch_namespace
    parsed = uuid.UUID(namespace)
    assert parsed.version == 4 and str(parsed) == namespace
    assert cfg.launch_namespace == namespace
    assert backend.config.checkpoint_dir == str(tmp_path / "run" / namespace)
    assert backend.saved == ["final"]


def test_compat_backend_config_namespace_mismatch_refuses(tmp_path, monkeypatch):
    import infra.backend.verl as verl_mod
    from infra.train import Config, train

    class _FakeBackend:
        tokenizer = None

        def __init__(self, config):
            self.config = config

        def save(self, name):
            raise AssertionError("mismatch must refuse before training")

    monkeypatch.setattr(verl_mod, "VerlBackend", _FakeBackend)
    backend = build_backend(
        _training(str(tmp_path)),
        "some/model",
        "run",
        launch_namespace="backend-attempt",
    )

    with pytest.raises(ValueError, match="different launch namespaces"):
        train(
            object(),
            backend,
            Config(
                steps=0,
                eval_every=0,
                save_every=0,
                log_transcripts=False,
                launch_namespace="config-attempt",
            ),
        )


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
        cfg = build_backend(
            _training(str(tmp_path)),
            "some/model",
            run_name,
            lr_override=lr,
            launch_namespace=f"attempt-{lr}",
        )
        dirs.append(cfg.checkpoint_dir)

    assert dirs[0] != dirs[1]
    for lr, d in zip(("1e-05", "3e-05"), dirs):
        assert d.startswith(str(tmp_path / f"math_rlvr_olmo_l5-lr{lr}"))


def test_same_run_gets_disjoint_scheduler_attempt_directories(tmp_path, monkeypatch):
    import infra.backend.verl as verl_mod

    monkeypatch.setattr(verl_mod, "VerlBackend", lambda config: config)
    first = build_backend(
        _training(str(tmp_path)), "some/model", "run", launch_namespace="attempt-a"
    )
    second = build_backend(
        _training(str(tmp_path)), "some/model", "run", launch_namespace="attempt-b"
    )

    assert first.checkpoint_dir == str(tmp_path / "run" / "attempt-a")
    assert second.checkpoint_dir == str(tmp_path / "run" / "attempt-b")
