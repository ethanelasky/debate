"""Reference-KL penalty, tinker-cookbook style (rl/metrics.py::incorporate_kl_penalty).

Penalty (the gradient path): CENTERED k1 in advantage space —
    adv_t += kl_coef · (avg_kl − k1_t),  k1_t = logπ_θ(a_t) − logπ_ref(a_t)
k1 is the unbiased choice here: in the reward/score-function placement its
gradient is the exact KL gradient in expectation, whereas differentiating k3
directly is biased (k3's low variance is a VALUE property, not a gradient
one). Centering by the batch mean makes the penalty zero-mean: it reshapes
credit toward low-divergence tokens without deflating the reward scale.
Optional kl_discount_factor smears future KL back onto earlier tokens
(credit assignment), as in the cookbook.

Metrics (the value path): k1 mean (the actual estimate, cookbook's
kl_policy_base), plus k3 = exp(δ)−δ−1 and k2 = ½δ² means as low-variance
non-negative monitors. Log-ratios clamp at ±20 before exp.
"""

from __future__ import annotations

import math

from infra.backend.base import Backend, Datum

LOG_RATIO_CLIP = 20.0


def k1_per_token(policy_logprobs: list[float], ref_logprobs: list[float]) -> list[float]:
    return [
        max(-LOG_RATIO_CLIP, min(LOG_RATIO_CLIP, lp - ref))
        for lp, ref in zip(policy_logprobs, ref_logprobs)
    ]


def discounted_future_sum(xs: list[float], gamma: float) -> list[float]:
    out = [0.0] * len(xs)
    acc = 0.0
    for i in range(len(xs) - 1, -1, -1):
        acc = xs[i] + gamma * acc
        out[i] = acc
    return out


def apply_kl_penalty(
    backend: Backend,
    datums: list[Datum],
    kl_coef: float,
    discount_factor: float = 0.0,
) -> dict[str, float]:
    """Mutates packed datums' advantages in place; returns metrics. Policy
    logprobs are the SAMPLER's (behavior policy at rollout); ref logprobs from
    backend.ref_logprobs (frozen base)."""
    if not datums:
        return {}
    ref_lps = backend.ref_logprobs(datums)

    per_datum_k1: list[list[float]] = []
    masks: list[list[float]] = []
    k1_sum = 0.0
    n_tokens = 0
    for d, ref in zip(datums, ref_lps):
        n = len(d.tokens) - d.prompt_len
        if len(ref) != n:
            raise ValueError(f"ref logprobs length {len(ref)} != completion length {n}")
        mask = d.mask if d.mask is not None else [1.0] * n
        k1 = [k if m else 0.0 for k, m in zip(k1_per_token(d.sampler_logprobs, ref), mask)]
        per_datum_k1.append(k1)
        masks.append(mask)
        k1_sum += sum(k1)
        n_tokens += sum(1 for m in mask if m)

    avg_k1 = k1_sum / max(1, n_tokens)
    k2_sum = k3_sum = 0.0
    for d, k1, mask in zip(datums, per_datum_k1, masks):
        penalty = [kl_coef * m * (avg_k1 - k) for k, m in zip(k1, mask)]
        if discount_factor > 0:
            penalty = discounted_future_sum(penalty, discount_factor)
        for t, p in enumerate(penalty):
            d.advantages[t] += p
        for k, m in zip(k1, mask):
            if m:
                k2_sum += 0.5 * k * k
                k3_sum += math.exp(-k) + k - 1.0  # k3 with delta = ref - policy = -k1

    return {
        "kl/policy_vs_ref_k1": avg_k1,               # cookbook's kl_policy_base
        "kl/policy_vs_ref_k2": k2_sum / max(1, n_tokens),
        "kl/policy_vs_ref_k3": k3_sum / max(1, n_tokens),
        "kl/coef": float(kl_coef),
        "kl/discount_factor": float(discount_factor),
    }


def entropy_proxy(datums: list[Datum]) -> float:
    """HW4's logging-only entropy: −mean sampled-token logprob over unmasked
    completion tokens."""
    total, n = 0.0, 0
    for d in datums:
        mask = d.mask if d.mask is not None else [1.0] * len(d.sampler_logprobs)
        for lp, m in zip(d.sampler_logprobs, mask):
            if m:
                total -= lp
                n += 1
    return total / max(1, n)
