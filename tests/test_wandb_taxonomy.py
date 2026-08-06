"""Canonical W&B taxonomy and logger initialization contracts."""

from __future__ import annotations

import sys

import pytest

from infra.run_common import wandb_config_kwargs
from infra.train import Config, _make_logger


class _Run:
    def __init__(self) -> None:
        self.logged: list[tuple[dict, int]] = []

    def log(self, metrics: dict, *, step: int) -> None:
        self.logged.append((metrics, step))


class _Wandb:
    def __init__(self) -> None:
        self.init_calls: list[dict] = []
        self.run = _Run()

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


def test_wandb_config_kwargs_builds_canonical_full_run_metadata():
    assert wandb_config_kwargs(
        domain="math",
        run_type="pc",
        experiment="math_pc_olmo_l5",
        model_path="allenai/Olmo-3-7B-Instruct-DPO",
    ) == {
        "wandb_entity": "palaestra-research",
        "wandb_project": "debate-rebuild",
        "wandb_group": "math/pc/math_pc_olmo_l5",
        "wandb_job_type": "train",
        "wandb_tags": [
            "domain:math",
            "run_type:pc",
            "experiment:math_pc_olmo_l5",
            "model:allenai/Olmo-3-7B-Instruct-DPO",
            "scope:full",
        ],
    }


def test_wandb_config_kwargs_marks_smoke_and_can_disable_logging():
    kwargs = wandb_config_kwargs(
        domain="monitoringbench",
        run_type="rlvr",
        experiment="mb_rlvr_olmo_smoke",
        model_path="model/path",
        enabled=False,
    )
    assert kwargs["wandb_project"] is None
    assert kwargs["wandb_group"] == "monitoringbench/rlvr/mb_rlvr_olmo_smoke"
    assert kwargs["wandb_tags"][-1] == "scope:smoke"


def test_wandb_config_kwargs_rejects_noncanonical_project():
    with pytest.raises(ValueError, match="debate-rebuild"):
        wandb_config_kwargs(
            domain="math",
            run_type="rlvr",
            experiment="arm",
            model_path="model/path",
            configured_project="personal-scratch",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("domain", ""),
        ("domain", "   "),
        ("domain", "math/algebra"),
        ("run_type", "pc/train"),
        ("experiment", "family/arm"),
    ],
)
def test_wandb_config_kwargs_rejects_invalid_group_components(field: str, value: str):
    values = {"domain": "math", "run_type": "pc", "experiment": "experiment"}
    values[field] = value
    with pytest.raises(ValueError, match=field):
        wandb_config_kwargs(**values, model_path="model/path")


def test_logger_fresh_init_passes_all_metadata(monkeypatch):
    wandb = _Wandb()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    cfg = Config(
        run_name="arm-lr1e-05",
        **wandb_config_kwargs(
            domain="math",
            run_type="pc",
            experiment="arm",
            model_path="model/path",
        ),
    )

    logger = _make_logger(cfg)

    assert wandb.init_calls == [
        {
            "entity": "palaestra-research",
            "project": "debate-rebuild",
            "name": "arm-lr1e-05",
            "config": vars(cfg),
            "group": "math/pc/arm",
            "job_type": "train",
            "tags": [
                "domain:math",
                "run_type:pc",
                "experiment:arm",
                "model:model/path",
                "scope:full",
            ],
        }
    ]
    logger(7, {"metric": 1.0})
    assert wandb.run.logged == [({"metric": 1.0}, 7)]


def test_logger_resume_passes_only_lineage_fields(monkeypatch):
    wandb = _Wandb()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    cfg = Config(
        wandb_run_id="abc123",
        **wandb_config_kwargs(
            domain="math",
            run_type="rlvr",
            experiment="arm",
            model_path="model/path",
        ),
    )

    _make_logger(cfg)

    assert wandb.init_calls == [
        {
            "entity": "palaestra-research",
            "project": "debate-rebuild",
            "id": "abc123",
            "resume": "must",
        }
    ]


def test_logger_off_does_not_initialize_wandb(monkeypatch):
    wandb = _Wandb()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    cfg = Config(
        **wandb_config_kwargs(
            domain="math",
            run_type="rlvr",
            experiment="arm",
            model_path="model/path",
            enabled=False,
        )
    )

    logger = _make_logger(cfg)
    logger(0, {"metric": 1.0})

    assert wandb.init_calls == []
    assert wandb.run.logged == []
