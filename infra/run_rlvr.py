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

MB blind choice (dataset.type: monitoringbench) additionally carries
`prompt_config` + `protocol` — the SAME entry and turns as the eval arm
(mb_debate_choice). The blind first-turn templates are built through
run_eval.blind_message_templates, the function the eval arm uses, so the RLVR
prompt is byte-identical to the eval arm's blind view for the same row.
"""

from __future__ import annotations

import argparse

from infra.backend.base import SamplingParams
from infra.config import load_experiment, reject_unknown_keys
from infra.envs.tasks import get_family
from infra.run_debate import (
    TRAINING_KEYS,
    VERL_KEYS,
    build_backend,
    run_identity_suffix,
    training_config_kwargs,
)
from infra.train import Config, train

EXPERIMENT_KEYS = {"model", "max_completion_tokens", "dataset", "training", "prompt_config", "protocol"}


def validate_experiment(exp: dict) -> None:
    """Reject config keys this runner never reads, before anything is built.

    Same contract as run_debate.validate_experiment: a typo must fail at
    launch instead of silently falling back to a default. Adding a new
    `exp.get(...)`/`tr.get(...)` read means adding the key here (or to
    run_debate.TRAINING_KEYS / VERL_KEYS, which this shares).
    """
    reject_unknown_keys(exp, EXPERIMENT_KEYS, "rlvr experiment")
    tr = exp.get("training") or {}
    reject_unknown_keys(tr, TRAINING_KEYS, "training")
    reject_unknown_keys(tr.get("verl") or {}, VERL_KEYS, "training.verl")
    if bool(exp.get("prompt_config")) != bool(exp.get("protocol")):
        raise ValueError(
            "prompt_config and protocol come together: both are needed to build "
            "the MB blind-choice templates (blind_choice_dataset), neither for "
            "plain task sources like math"
        )


def blind_choice_dataset(exp: dict, ds: dict) -> dict:
    """MB choice mode: when the experiment carries prompt_config + protocol,
    render the blind first-turn templates through the SAME function the eval
    arm uses (run_eval.blind_message_templates, same entry + turns), so the
    RLVR prompt is byte-identical to the eval arm's blind view for the same
    row. Plain task sources (no prompt_config) pass through untouched."""
    if not exp.get("prompt_config"):
        return ds
    from infra.run_eval import _resolve_protocol, blind_message_templates

    messages, prompt_vars = blind_message_templates(exp, _resolve_protocol(exp))
    return {**ds, "choice_messages": messages, "choice_prompt_vars": prompt_vars}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-file", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--wandb-project", default=None, help="override training.wandb_project")
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    parser.add_argument("--steps", type=int, default=None, help="override training.steps")
    parser.add_argument("--lr", type=float, default=None, help="override training.lr (sweeps)")
    parser.add_argument("--levels", default=None, help="override dataset.levels (e.g. 4 or 3-4)")
    parser.add_argument("--group-size", type=int, default=None, help="override training.group_size")
    parser.add_argument("--batch-size", type=int, default=None, help="override training.batch_size")
    parser.add_argument("--load", default=None)
    args = parser.parse_args()

    exp = load_experiment(args.experiment_file, args.experiment)
    validate_experiment(exp)
    if args.levels is not None:
        exp.setdefault("dataset", {})["levels"] = args.levels

    ds = dict(exp.get("dataset") or {})
    family = get_family(ds.pop("type", None))
    ds.pop("relaxed_extraction", None)  # debate-only knob; source envs score both
    env = family.source(blind_choice_dataset(exp, ds))

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
        wandb_project=(
            None if args.no_wandb else args.wandb_project or tr.get("wandb_project") or "debate"
        ),
        run_name=run_name,
        **training_config_kwargs(tr, args),
    )
    train(env, backend, cfg)


if __name__ == "__main__":
    main()
