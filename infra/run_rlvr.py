"""RLVR runner: experiment YAML -> MathEnv (direct correctness reward) ->
train loop. The judge-free baseline for the debate experiments: same model,
same backend, same GRPO machinery — reward comes from answer verification
instead of a judge, so it isolates optimization from reward quality.

Config shape:

    math_rlvr_olmo_l5:
      model: allenai/Olmo-3-7B-Instruct-DPO
      max_completion_tokens: 2000
      dataset: {type: math, levels: 5, seed: 0}
      training: {backend: verl, verl: {...}, lora_rank: 32, loss: {...},
                 steps: 100, batch_size: 8, group_size: 8, lr: 1e-4, ...}
"""

from __future__ import annotations

import argparse

from infra.backend.base import LossSpec, SamplingParams
from infra.config import load_experiment
from infra.envs.tasks import get_family
from infra.run_debate import build_backend
from infra.train import Config, train


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
    if args.levels is not None:
        exp.setdefault("dataset", {})["levels"] = args.levels

    ds = dict(exp.get("dataset") or {})
    family = get_family(ds.pop("type", None))
    ds.pop("relaxed_extraction", None)  # debate-only knob; source envs score both
    env = family.source(ds)

    tr = exp.get("training") or {}
    model_path = str(exp["model"])
    backend = build_backend(tr, model_path, lr_override=args.lr)
    if args.load:
        backend.load(args.load)

    max_tokens = exp.get("max_completion_tokens")
    if max_tokens is None:
        raise ValueError("set max_completion_tokens (per-generation budget) in the experiment")

    cfg = Config(
        base_model=model_path,
        steps=args.steps if args.steps is not None else int(tr.get("steps", 100)),
        batch_size=(args.batch_size if args.batch_size is not None else int(tr.get("batch_size", 8))),
        group_size=(args.group_size if args.group_size is not None else int(tr.get("group_size", 8))),
        micro_batch=int(tr.get("micro_batch", 64)),
        lr=float(args.lr if args.lr is not None else tr.get("lr", 1e-5)),
        loss=LossSpec(**(tr.get("loss") or {})),
        ppo_epochs=int(tr.get("ppo_epochs", 1)),
        adv_length_norm=str(tr.get("adv_length_norm", "none")),
        kl_coef=float(tr.get("kl_coef", 0.0)),
        kl_discount_factor=float(tr.get("kl_discount_factor", 0.0)),
        sampling=SamplingParams(max_tokens=int(max_tokens), temperature=1.0, top_p=1.0),
        eval_every=int(tr.get("eval_every", 5)),
        eval_max_tokens=(int(tr["eval_max_tokens"]) if "eval_max_tokens" in tr else None),
        eval_n=int(tr.get("eval_n", 128)),
        save_every=int(tr.get("save_every", 0)),
        wandb_project=(
            None if args.no_wandb else args.wandb_project or tr.get("wandb_project") or "debate"
        ),
        run_name=args.experiment
        + (f"-lr{args.lr:g}" if args.lr is not None else "")
        + (f"-L{args.levels}" if args.levels is not None else "")
        + (f"-g{args.group_size}" if args.group_size is not None else "")
        + (f"-b{args.batch_size}" if args.batch_size is not None else ""),
    )
    train(env, backend, cfg)


if __name__ == "__main__":
    main()
