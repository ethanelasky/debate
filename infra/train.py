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
    steps: int = 100
    batch_size: int = 8        # tasks per step
    group_size: int = 8        # samples per task (GRPO group)
    micro_batch: int = 64      # datums per forward_backward call
    lr: float = 1e-5
    loss: LossSpec = field(default_factory=LossSpec)
    ppo_epochs: int = 1  # passes over each rollout batch (one optim_step per epoch)
    optim: OptimParams | None = None
    sampling: SamplingParams = field(default_factory=lambda: SamplingParams(max_tokens=512))
    chat_template_kwargs: dict | None = None
    norm_adv_by_std: bool = True
    adv_length_norm: str = "none"   # none|datum|trajectory|count — see grpo_pack
    kl_coef: float = 0.0            # 0 = no reference-KL penalty
    kl_discount_factor: float = 0.0  # >0 smears future KL onto earlier tokens
    eval_every: int = 20
    eval_n: int = 128
    eval_max_tokens: int | None = None  # eval env generations need an explicit budget
    save_every: int = 50
    wandb_project: str | None = None  # None = no wandb
    # wandb entity (team). None falls back to the API key's DEFAULT entity,
    # which is a personal namespace — runs then land somewhere the rest of the
    # team cannot see. Set it explicitly in the experiment config.
    wandb_entity: str | None = None
    run_name: str | None = None
    on_rollout: object | None = None  # callback(step, env) after each train rollout
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
    # Per-seat breakdown: in zero-sum self-play the cross-seat means are
    # identities (reward 0, confidence 0.5) — the signal lives per seat.
    seats = {t.info.get("seat") for t in trajs if isinstance(t.info.get("seat"), str)}
    for seat in sorted(s for s in seats if s):
        seat_trajs = [t for t in trajs if t.info.get("seat") == seat]
        out[f"{prefix}/{seat}/reward_mean"] = sum(t.reward for t in seat_trajs) / len(seat_trajs)
        for k in ("judge_conf_json", "judge_conf_logit", "solution_correct"):
            vals = [t.info[k] for t in seat_trajs if isinstance(t.info.get(k), (int, float))]
            if vals:
                out[f"{prefix}/{seat}/{k}"] = sum(vals) / len(vals)
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


def train(env: Env, backend: Backend, cfg: Config, eval_env: Env | None = None) -> None:
    """eval_env: evaluate on a DIFFERENT env than training (e.g. debate
    training with plain-RLVR MathEnv evals). None = eval on `env`."""
    logger = _make_logger(cfg)
    policy = Policy(backend, cfg.sampling, cfg.chat_template_kwargs)
    optim = cfg.optim or OptimParams(lr=cfg.lr)

    for step in range(cfg.steps):
        t0 = time.monotonic()
        backend.sync_sampler()
        groups = env.rollout(env.tasks(cfg.batch_size, "train"), policy, cfg.group_size)
        if cfg.on_rollout is not None:
            try:
                cfg.on_rollout(step, env)
            except Exception as e:  # transcript capture must never kill training
                print(f"[on_rollout] {type(e).__name__}: {e}")
        metrics = _aggregate([t for g in groups for t in g], "train")
        metrics.update(_rollout_info_metrics(env, "train"))

        datums, pack_stats = grpo_pack(
            groups,
            norm_by_std=cfg.norm_adv_by_std,
            length_normalize=cfg.adv_length_norm,
            shuffle_seed=cfg.seed + step,
        )
        metrics.update(pack_stats)

        if datums:
            from infra.rl.kl import apply_kl_penalty, entropy_proxy

            metrics["train/policy_token_entropy"] = entropy_proxy(datums)
            if cfg.kl_coef > 0:
                metrics.update(
                    apply_kl_penalty(backend, datums, cfg.kl_coef, cfg.kl_discount_factor)
                )
            import random as _random

            for epoch in range(max(1, cfg.ppo_epochs)):
                # later epochs go off-policy; the ratio in ppo/importance losses
                # corrects against the stored sampler logprobs
                _random.Random(cfg.seed * 1000 + step * 10 + epoch).shuffle(datums)
                for i in range(0, len(datums), cfg.micro_batch):
                    backend.forward_backward(datums[i : i + cfg.micro_batch], cfg.loss)
                metrics.update(backend.optim_step(optim))

        if cfg.eval_every and step % cfg.eval_every == 0:
            eval_policy = policy
            if cfg.eval_max_tokens is not None:
                from dataclasses import replace as _replace

                eval_policy = Policy(
                    policy.backend,
                    _replace(policy.params, max_tokens=cfg.eval_max_tokens),
                    policy.chat_template_kwargs,
                )
            metrics.update(evaluate(eval_env or env, eval_policy, cfg.eval_n))
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

        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,   # None -> wandb's default entity
            name=cfg.run_name,
            config=vars(cfg),
        )

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
    parser.add_argument("--wandb-entity", default=None)
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
        wandb_entity=args.wandb_entity,
    )

    from infra.backend.tinker import TinkerBackend
    from infra.envs.tasks import get_family

    backend = TinkerBackend(cfg.base_model, lora_rank=cfg.lora_rank)
    if args.load:
        backend.load(args.load)
    env = get_family("math").source({"seed": cfg.seed})
    train(env, backend, cfg)


if __name__ == "__main__":
    main()
