"""Continuation plumbing: --start-step / --wandb-resume -> Config.

A continuation must keep the original lineage's step numbering (saves, eval
cadence, wandb x-axis) — the whole point of the flags.
"""

from infra.run_common import resolved_start_step, runner_parser, training_config_kwargs
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


def test_config_kwargs_omit_fields_on_fresh_runs():
    kw = training_config_kwargs({}, _args([]))
    assert "start_step" not in kw and "wandb_run_id" not in kw


def test_loop_range_runs_remaining_steps_only():
    cfg = Config(start_step=25, steps=100)
    assert list(range(cfg.start_step, cfg.steps))[0] == 25
    assert len(range(cfg.start_step, cfg.steps)) == 75
