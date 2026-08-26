"""Reference-KL logging and the k1 estimator.

KL is applied ONE way: verl's native in-loss term against the adapter-disabled
base (kl_loss_type=low_var_kl, i.e. k3), stamped per step as datum.ref_logprobs
and enabled by training.kl_coef. Everything here is the value path — the
gradient lives in the verl worker.

k3 is the right estimator for that placement. A differentiable k1 term would
have gradient grad log pi_theta, whose expectation under theta is zero: pure
variance, no pull.

Metrics: k1 mean (the estimate), plus k3 = exp(d)-d-1 and k2 = 0.5*d**2 as
low-variance non-negative monitors. Log-ratios clamp at +/-20 before exp.
"""

from __future__ import annotations

import math

from infra.backend.base import Datum

LOG_RATIO_CLIP = 20.0


def k1_per_token(policy_logprobs: list[float], ref_logprobs: list[float]) -> list[float]:
    return [
        max(-LOG_RATIO_CLIP, min(LOG_RATIO_CLIP, lp - ref))
        for lp, ref in zip(policy_logprobs, ref_logprobs)
    ]


def ref_kl_metrics(datums: list[Datum]) -> dict[str, float]:
    """Logging-only KL from the ref_logprobs the loop stamps each step."""
    k1_sum = k2_sum = k3_sum = 0.0
    n_tokens = 0
    for d in datums:
        if d.ref_logprobs is None:
            continue
        mask = d.mask if d.mask is not None else [1.0] * len(d.sampler_logprobs)
        for k, m in zip(k1_per_token(d.sampler_logprobs, d.ref_logprobs), mask):
            if m:
                k1_sum += k
                k2_sum += 0.5 * k * k
                k3_sum += math.exp(-k) + k - 1.0
                n_tokens += 1
    if not n_tokens:
        return {}
    return {
        "kl/policy_vs_ref_k1": k1_sum / n_tokens,
        "kl/policy_vs_ref_k2": k2_sum / n_tokens,
        "kl/policy_vs_ref_k3": k3_sum / n_tokens,
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
