"""Centered-k1 KL penalty (tinker-cookbook semantics) + metrics + entropy."""

import math

import pytest

from infra.backend.base import Datum
from infra.rl.kl import apply_kl_penalty, discounted_future_sum, entropy_proxy, k1_per_token


class FakeRefBackend:
    def __init__(self, ref):
        self._ref = ref

    def ref_logprobs(self, data):
        return [self._ref[: len(d.tokens) - d.prompt_len] for d in data]


def mk(lps, mask=None, adv=None):
    n = len(lps)
    return Datum(
        tokens=list(range(n + 2)),
        prompt_len=2,
        sampler_logprobs=lps,
        advantages=list(adv) if adv else [0.0] * n,
        mask=mask,
    )


def test_k1_clamped():
    assert k1_per_token([-1.0], [-1.0]) == [0.0]
    assert k1_per_token([-0.5], [-1.5]) == [1.0]
    assert k1_per_token([0.0], [-100.0]) == [20.0]


def test_centered_penalty_is_zero_mean_and_signed():
    # two tokens with k1 = [1.0, 0.0]; avg = 0.5
    d = mk([-0.5, -1.0])
    backend = FakeRefBackend([-1.5, -1.0])
    metrics = apply_kl_penalty(backend, [d], kl_coef=0.1)
    # adv += coef*(avg - k1): high-divergence token penalized, low boosted
    assert d.advantages[0] == pytest.approx(0.1 * (0.5 - 1.0))
    assert d.advantages[1] == pytest.approx(0.1 * (0.5 - 0.0))
    assert sum(d.advantages) == pytest.approx(0.0)  # zero-mean across the batch
    assert metrics["kl/policy_vs_ref_k1"] == pytest.approx(0.5)
    assert metrics["kl/policy_vs_ref_k2"] == pytest.approx(0.5 * (1.0 + 0.0) / 2)
    k3 = (math.exp(-1.0) + 1.0 - 1.0) + 0.0
    assert metrics["kl/policy_vs_ref_k3"] == pytest.approx(k3 / 2)


def test_masked_tokens_untouched_and_excluded_from_mean():
    d = mk([-0.5, -9.9], mask=[1.0, 0.0], adv=[1.0, 1.0])
    backend = FakeRefBackend([-1.5, -1.5])
    apply_kl_penalty(backend, [d], kl_coef=0.1)
    # only token 0 counts: avg = its own k1 -> centered penalty 0
    assert d.advantages[0] == pytest.approx(1.0)
    assert d.advantages[1] == 1.0  # masked: untouched


def test_discounted_future_sum():
    assert discounted_future_sum([1.0, 1.0, 1.0], 0.5) == [1.75, 1.5, 1.0]
    assert discounted_future_sum([2.0], 0.9) == [2.0]


def test_entropy_proxy_masked_mean():
    d = mk([-2.0, -4.0], mask=[1.0, 0.0])
    assert entropy_proxy([d]) == pytest.approx(2.0)
