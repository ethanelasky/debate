"""Trajectory groups -> flat training datums with GRPO advantages filled in."""

from __future__ import annotations

import random

from infra.backend.base import Datum
from infra.envs.base import Trajectory
from infra.rl.advantages import compute_grpo_advantages


def grpo_pack(
    groups: list[list[Trajectory]],
    *,
    norm_by_std: bool = True,
    drop_zero_advantage: bool = True,
    length_normalize: str = "none",  # "none" | "datum" | "trajectory"
    shuffle_seed: int | None = None,
) -> tuple[list[Datum], dict[str, float]]:
    """Advantages per datum, normalized per (group, datum position): datum k of
    each trajectory competes with datum k of the other trajectories in its
    group, so slot-specific rewards (Trajectory.datum_rewards) get their own
    baseline. With uniform per-trajectory rewards this is numerically identical
    to per-trajectory normalization (same G values in every position's group).

    Zero-advantage datums (constant reward within their position group)
    contribute no gradient; dropping them saves forward/backward compute.

    length_normalize controls GRADIENT MASS (backend losses are token-sums, so
    unscaled advantages weight long/many generations proportionally more):
      "none"        raw broadcast (token-sum semantics)
      "datum"       advantages /= the datum's unmasked completion length —
                    every datum contributes equal mass (seq-mean-token-mean)
      "trajectory"  additionally /= the trajectory's datum count — every
                    trajectory (= seat, per debate) contributes equal mass;
                    a seat with more slots or bigger budgets gets no extra pull
                    on a shared self-play adapter
      "count"       /= datum count ONLY (no token scaling) — the seat-balance
                    mode for gspo, whose loss is already one term per sequence
    """
    rows: list[tuple[Datum, float, tuple[int, int], float]] = []  # datum, reward, (gid, pos), scale
    n_trajs = 0
    for gid, group in enumerate(groups):
        for traj in group:
            n_trajs += 1
            for k, d in enumerate(traj.datums):
                r = traj.datum_rewards[k] if traj.datum_rewards is not None else traj.reward
                scale = 1.0
                if length_normalize in ("datum", "trajectory"):
                    n_completion = len(d.tokens) - d.prompt_len
                    unmasked = sum(d.mask) if d.mask is not None else n_completion
                    scale /= max(1.0, unmasked)
                if length_normalize in ("trajectory", "count"):
                    scale /= max(1, len(traj.datums))
                rows.append((d, float(r), (gid, k), scale))
    if not rows:
        return [], {"pack/n_trajectories": float(n_trajs), "pack/n_datums": 0.0}

    advantages = compute_grpo_advantages(
        [r for _, r, _, _ in rows], [g for _, _, g, _ in rows], norm_by_std=norm_by_std
    )

    datums: list[Datum] = []
    n_dropped = 0
    for (d, _, _, scale), adv in zip(rows, advantages):
        if drop_zero_advantage and adv == 0.0:
            n_dropped += 1
            continue
        n = len(d.tokens) - d.prompt_len
        datums.append(
            Datum(
                tokens=d.tokens,
                prompt_len=d.prompt_len,
                sampler_logprobs=d.sampler_logprobs,
                advantages=[adv * scale] * n,
                mask=d.mask,
            )
        )

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(datums)

    stats = {
        "pack/n_trajectories": float(n_trajs),
        "pack/n_datums": float(len(datums)),
        "pack/n_datums_dropped_zero_advantage": float(n_dropped),
        "pack/mean_abs_advantage": (
            sum(abs(a) for a in advantages) / len(advantages) if advantages else 0.0
        ),
    }
    return datums, stats
