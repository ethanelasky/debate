"""The training loop. This is the whole thing."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any

from infra.backend.base import Backend, LossSpec, OptimParams, SamplingParams
from infra.envs.base import Env, Policy, Trajectory
from infra.rl.datums import grpo_pack


@dataclass
class Config:
    base_model: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    steps: int = 200
    batch_size: int = 32       # tasks per step
    group_size: int = 8        # samples per task (GRPO group)
    micro_batch: int = 64      # datums per forward_backward call
    lr: float = 1e-5
    loss: LossSpec = field(default_factory=LossSpec)
    optim: OptimParams | None = None
    sampling: SamplingParams = field(default_factory=lambda: SamplingParams(max_tokens=512))
    chat_template_kwargs: dict | None = None
    norm_adv_by_std: bool = True
    kl_coef: float = 0.0            # 0 = no reference-KL penalty
    kl_discount_factor: float = 0.0  # >0 smears future KL onto earlier tokens
    eval_every: int = 20
    eval_n: int = 128
    save_every: int = 50
    wandb_project: str | None = None  # None = no wandb
    run_name: str | None = None
    seed: int = 0


def _aggregate(trajs: list[Trajectory], prefix: str) -> dict[str, float]:
    if not trajs:
        return {f"{prefix}/n": 0.0}
    out: dict[str, float] = {
        f"{prefix}/reward_mean": sum(t.reward for t in trajs) / len(trajs),
        f"{prefix}/n": float(len(trajs)),
    }
    keys = {k for t in trajs for k, v in t.info.items() if isinstance(v, (int, float))}
    for k in sorted(keys):
        vals = [t.info[k] for t in trajs if isinstance(t.info.get(k), (int, float))]
        out[f"{prefix}/{k}"] = sum(vals) / len(vals)
    return out


def _rollout_info_metrics(env: Env, prefix: str) -> dict[str, float]:
    """Flatten env.last_rollout_info (debate env's drop counters) if present."""
    info = getattr(env, "last_rollout_info", None)
    if not info:
        return {}
    out: dict[str, float] = {}
    for k, v in info.items():
        if isinstance(v, (int, float)):
            out[f"{prefix}/{k}"] = float(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    out[f"{prefix}/{k}/{kk}"] = float(vv)
    return out


def evaluate(env: Env, policy: Policy, n: int) -> dict[str, float]:
    groups = env.rollout(env.tasks(n, split="test"), policy.greedy(), group_size=1)
    return _aggregate([t for g in groups for t in g], "eval") | _rollout_info_metrics(env, "eval")


def train(env: Env, backend: Backend, cfg: Config) -> None:
    logger = _make_logger(cfg)
    policy = Policy(backend, cfg.sampling, cfg.chat_template_kwargs)
    optim = cfg.optim or OptimParams(lr=cfg.lr)

    for step in range(cfg.steps):
        t0 = time.monotonic()
        backend.sync_sampler()
        groups = env.rollout(env.tasks(cfg.batch_size, "train"), policy, cfg.group_size)
        metrics = _aggregate([t for g in groups for t in g], "train")
        metrics.update(_rollout_info_metrics(env, "train"))

        datums, pack_stats = grpo_pack(
            groups, norm_by_std=cfg.norm_adv_by_std, shuffle_seed=cfg.seed + step
        )
        metrics.update(pack_stats)

        if datums:
            from infra.rl.kl import apply_kl_penalty, entropy_proxy

            metrics["train/policy_token_entropy"] = entropy_proxy(datums)
            if cfg.kl_coef > 0:
                metrics.update(
                    apply_kl_penalty(backend, datums, cfg.kl_coef, cfg.kl_discount_factor)
                )
            for i in range(0, len(datums), cfg.micro_batch):
                backend.forward_backward(datums[i : i + cfg.micro_batch], cfg.loss)
            metrics.update(backend.optim_step(optim))

        if cfg.eval_every and step % cfg.eval_every == 0:
            metrics.update(evaluate(env, policy, cfg.eval_n))
        if cfg.save_every and step > 0 and step % cfg.save_every == 0:
            metrics["checkpoint_saved"] = 1.0
            backend.save(f"step-{step:05d}")

        metrics["step_seconds"] = time.monotonic() - t0
        logger(step, metrics)

    backend.save("final")


def _make_logger(cfg: Config):
    run = None
    if cfg.wandb_project:
        import wandb

        run = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))

    def log(step: int, metrics: dict[str, Any]) -> None:
        if run is not None:
            run.log(metrics, step=step)
        shown = {k: round(v, 4) for k, v in sorted(metrics.items()) if isinstance(v, float)}
        print(f"[step {step}] {shown}")

    return log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["math"], default="math")
    parser.add_argument("--base-model", default=Config.base_model)
    parser.add_argument("--lora-rank", type=int, default=Config.lora_rank)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--group-size", type=int, default=Config.group_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--loss", choices=["ppo", "importance_sampling"], default="ppo")
    parser.add_argument("--eval-every", type=int, default=Config.eval_every)
    parser.add_argument("--eval-n", type=int, default=Config.eval_n)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--load", default=None, help="checkpoint path to resume from")
    args = parser.parse_args()

    cfg = Config(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        steps=args.steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        lr=args.lr,
        loss=LossSpec(kind=args.loss),
        sampling=SamplingParams(max_tokens=args.max_tokens),
        eval_every=args.eval_every,
        eval_n=args.eval_n,
        wandb_project=args.wandb_project,
    )

    from infra.backend.tinker import TinkerBackend
    from infra.envs.math_env import MathEnv

    backend = TinkerBackend(cfg.base_model, lora_rank=cfg.lora_rank)
    if args.load:
        backend.load(args.load)
    env = MathEnv(seed=cfg.seed)
    train(env, backend, cfg)


if __name__ == "__main__":
    main()
