"""GRPO group-relative advantages, verl-parity-verified against sft-control.

    advantage_i = (reward_i - group_mean) / (group_std + epsilon)

- SAMPLE std (Bessel, / (n-1)) to match torch.std(unbiased=True) / verl.
- Single-member group: advantage = raw reward.
- All-identical group (std < epsilon): advantage = 0 for every member.
- norm_by_std=False is Dr. GRPO de-meaning (arXiv:2503.20783): centering
  without the 1/std rescaling that injects question-difficulty bias.
"""

from __future__ import annotations

from typing import Sequence

GRPO_ADVANTAGE_EPSILON = 1e-6


def compute_grpo_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[object],
    *,
    epsilon: float = GRPO_ADVANTAGE_EPSILON,
    norm_by_std: bool = True,
    population_std: bool = False,
) -> list[float]:
    if len(rewards) != len(group_ids):
        raise ValueError(f"rewards ({len(rewards)}) and group_ids ({len(group_ids)}) must align")

    groups: dict[object, list[tuple[int, float]]] = {}
    for i, (gid, r) in enumerate(zip(group_ids, rewards)):
        groups.setdefault(gid, []).append((i, float(r)))

    advantages = [0.0] * len(rewards)
    for group in groups.values():
        rs = [r for _, r in group]
        n = len(rs)
        if n == 1:
            advantages[group[0][0]] = rs[0]
            continue
        mean_r = sum(rs) / n
        if not norm_by_std:
            for idx, r in group:
                advantages[idx] = r - mean_r
            continue
        # population_std (/n) matches hw4's torch.std(unbiased=False); the
        # default sample std (/(n-1)) is torch.std(unbiased=True) = verl
        # parity. At n=8 the two differ by a uniform 1.069x on every
        # advantage — an effective-lr constant, but a wrong one when the
        # point is reproducing a reference curve.
        denom = n if population_std else (n - 1)
        std_r = (sum((r - mean_r) ** 2 for r in rs) / denom) ** 0.5
        if std_r < epsilon:
            continue  # no gradient signal; leave at 0.0
        for idx, r in group:
            advantages[idx] = (r - mean_r) / (std_r + epsilon)
    return advantages
