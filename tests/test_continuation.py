"""Continuation plumbing: --start-step / --wandb-resume -> Config.

A continuation must keep the original lineage's step numbering (saves, eval
cadence, wandb x-axis) — the whole point of the flags.
"""

import infra.train as train_mod
from infra.run_common import (
    acquire_wandb_resume_cli_lease,
    apply_wandb_resume_cli_authorization,
    release_wandb_resume_cli_lease,
    resolved_start_step,
    runner_parser,
    training_config_kwargs,
)
from infra.train import Config


def _args(extra: list[str]):
    parser = runner_parser(None)
    return parser.parse_args(["--experiment-file", "f.yaml", "--experiment", "e", *extra])


def test_start_step_inferred_from_load_path():
    args = _args(["--load", "/workspace/checkpoints/exp/step-00025"])
    assert resolved_start_step(args) == 25


def test_start_step_inferred_with_trailing_slash():
    args = _args(["--load", "/workspace/checkpoints/exp/step-00050/"])
    assert resolved_start_step(args) == 50


def test_explicit_start_step_beats_inference():
    args = _args(["--load", "/workspace/checkpoints/exp/step-00025", "--start-step", "40"])
    assert resolved_start_step(args) == 40


def test_no_load_no_flag_starts_at_zero():
    assert resolved_start_step(_args([])) == 0


def test_non_step_load_path_starts_at_zero():
    assert resolved_start_step(_args(["--load", "/workspace/checkpoints/exp/final"])) == 0


def test_config_kwargs_carry_continuation_fields():
    args = _args(
        ["--load", "/workspace/checkpoints/exp/step-00025", "--wandb-resume", "abc123de"]
    )
    kw = training_config_kwargs({}, args)
    assert kw["start_step"] == 25
    assert kw["wandb_run_id"] == "abc123de"
    cfg = Config(**kw)  # the fields exist on Config under these exact names
    assert (cfg.start_step, cfg.wandb_run_id) == (25, "abc123de")


def test_runner_applies_opaque_running_override_only_when_explicit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        train_mod, "_wandb_resume_lock_root", lambda: str(tmp_path / "locks")
    )
    ordinary_args = _args(
        [
            "--load",
            "/workspace/checkpoints/exp/step-00025",
            "--wandb-resume",
            "abc123de",
        ]
    )
    ordinary = training_config_kwargs(
        {},
        ordinary_args,
    )
    assert "wandb_resume_running_override" not in ordinary
    ordinary_cfg = Config(**ordinary)
    ordinary_handoff = acquire_wandb_resume_cli_lease(ordinary_args)
    apply_wandb_resume_cli_authorization(
        ordinary_cfg,
        ordinary_args,
        resume_lease_handoff=ordinary_handoff,
    )
    assert ordinary_cfg._wandb_resume_running_capability is None
    assert ordinary_cfg._wandb_resume_lease_handoff is ordinary_handoff
    release_wandb_resume_cli_lease(ordinary_handoff)

    explicit_args = _args(
        [
            "--load",
            "/workspace/checkpoints/exp/step-00025",
            "--wandb-resume",
            "abc123de",
            "--wandb-resume-running-override",
        ]
    )
    explicit = training_config_kwargs(
        {},
        explicit_args,
    )
    assert "wandb_resume_running_override" not in explicit
    explicit_cfg = Config(**explicit)
    explicit_handoff = acquire_wandb_resume_cli_lease(explicit_args)
    apply_wandb_resume_cli_authorization(
        explicit_cfg,
        explicit_args,
        resume_lease_handoff=explicit_handoff,
    )
    capability = explicit_cfg._wandb_resume_running_capability
    assert capability is not None
    assert capability.run_id == "abc123de"
    assert not isinstance(capability, bool)
    assert explicit_cfg._wandb_resume_lease_handoff is explicit_handoff
    release_wandb_resume_cli_lease(explicit_handoff)


def test_config_kwargs_omit_fields_on_fresh_runs():
    kw = training_config_kwargs({}, _args([]))
    assert "start_step" not in kw and "wandb_run_id" not in kw


def test_loop_range_runs_remaining_steps_only():
    cfg = Config(start_step=25, steps=100)
    assert list(range(cfg.start_step, cfg.steps))[0] == 25
    assert len(range(cfg.start_step, cfg.steps)) == 75


def test_continuation_first_step_does_not_resave_loaded_checkpoint():
    """Step 25 must not clobber the loaded checkpoint; later save- and
    eval-cadence boundaries remain recovery points."""
    cfg = Config(start_step=25, steps=100, save_every=25)
    saves = [
        step
        for step in range(cfg.start_step, cfg.steps)
        if cfg.save_every
        and step > cfg.start_step
        and (
            step % cfg.save_every == 0
            or (cfg.eval_every and step % cfg.eval_every == 0)
        )
    ]
    assert saves == [40, 50, 60, 75, 80]


def test_lora_rank_reaches_config():
    kw = training_config_kwargs({"lora_rank": 64}, _args([]))
    assert Config(**kw).lora_rank == 64
