"""RLVR runner: experiment YAML -> task-source env (direct verifiable reward)
-> train loop. The judge-free baseline for the debate experiments: same model,
same backend, same GRPO machinery — reward comes from answer verification
instead of a judge, so it isolates optimization from reward quality.

Config shape:

    math_rlvr_olmo_l5:
      model: allenai/Olmo-3-7B-Instruct-DPO
      max_completion_tokens: 2000
      dataset: {type: math, levels: 5, seed: 0}
      training: {backend: verl, verl: {...}, lora_rank: 32, loss: {...},
                 steps: 100, batch_size: 8, group_size: 8, lr: 1e-4, ...}

MB blind choice (dataset.type: monitoringbench) needs nothing extra: the task
source renders its RLVR prompt from the family's answer-generation config
(infra/prompts/tasks/monitoringbench.yaml, dataset.prompt_file to override) —
the same file the eval arm's debate packs splice their blind view from, so the
RLVR prompt is byte-identical to the eval arm's blind view by construction.
"""

from __future__ import annotations

import os

from infra.backend.base import SamplingParams
from infra.config import load_experiment, reject_unknown_keys
from infra.envs.tasks import get_family
from infra.run_common import (
    TRAINING_KEYS,
    VERL_KEYS,
    build_backend,
    run_identity_suffix,
    runner_parser,
    training_config_kwargs,
    wandb_config_kwargs,
)
from infra.train import Config, train

EXPERIMENT_KEYS = {"model", "max_completion_tokens", "dataset", "training"}


def validate_experiment(exp: dict) -> None:
    """Reject config keys this runner never reads, before anything is built.

    Same contract as run_debate.validate_experiment: a typo must fail at
    launch instead of silently falling back to a default. Adding a new
    `exp.get(...)`/`tr.get(...)` read means adding the key here (or to
    run_common.TRAINING_KEYS / VERL_KEYS, which both runners share).
    """
    reject_unknown_keys(exp, EXPERIMENT_KEYS, "rlvr experiment")
    tr = exp.get("training") or {}
    reject_unknown_keys(tr, TRAINING_KEYS, "training")
    reject_unknown_keys(tr.get("verl") or {}, VERL_KEYS, "training.verl")


def main() -> None:
    args = runner_parser(__doc__).parse_args()

    exp = load_experiment(args.experiment_file, args.experiment)
    validate_experiment(exp)
    if args.levels is not None:
        exp.setdefault("dataset", {})["levels"] = args.levels

    ds = dict(exp.get("dataset") or {})
    family = get_family(ds.pop("type", None))
    ds.pop("relaxed_extraction", None)  # debate-only knob; source envs score both
    env = family.source(ds)

    tr = exp.get("training") or {}
    model_path = str(exp["model"])
    run_name = args.experiment + run_identity_suffix(
        args.lr, args.levels, args.group_size, args.batch_size
    )
    backend = build_backend(
        tr, model_path, run_name, lr_override=args.lr, load_given=bool(args.load)
    )
    if args.load:
        backend.load(args.load)

    max_tokens = exp.get("max_completion_tokens")
    if max_tokens is None:
        raise ValueError("set max_completion_tokens (per-generation budget) in the experiment")

    cfg = Config(
        base_model=model_path,
        sampling=SamplingParams(max_tokens=int(max_tokens), temperature=1.0, top_p=1.0),
        run_name=run_name,
        **wandb_config_kwargs(
            domain=str((exp.get("dataset") or {}).get("type") or ""),
            run_type="rlvr",
            experiment=args.experiment,
            model_path=model_path,
            enabled=not args.no_wandb,
            configured_project=args.wandb_project or tr.get("wandb_project"),
        ),
        **training_config_kwargs(tr, args),
    )

    # Comprehensive docent capture, single-turn twin of run_debate's: every
    # training rollout's kept samples -> docent/step-NNNNN.jsonl.
    def _export_docent(step: int, env_) -> None:
        from infra.envs.debate.docent_export import export_jsonl
        from infra.envs.singleturn_docent import agent_runs

        records = getattr(env_, "last_rollout_records", None)
        if not records:
            return
        os.makedirs("docent", exist_ok=True)
        export_jsonl(agent_runs(records), f"docent/step-{step:05d}.jsonl")

    cfg.on_rollout = _export_docent
    train(env, backend, cfg)


if __name__ == "__main__":
    main()
