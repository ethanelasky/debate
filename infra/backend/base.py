"""The one abstraction: a Backend owning policy weights, with four compute
endpoints plus weight lifecycle. Nothing sits above this.

Contract:
  - sample() reflects the weights as of the last sync_sampler(), NOT live
    training weights. (Tinker samplers are frozen snapshots; VERL syncs into a
    vLLM server. Making the snapshot explicit keeps both backends honest about
    which policy generated a rollout.)
  - forward_backward() ACCUMULATES gradients; optim_step() applies them and
    clears the accumulator. Microbatching / grad accumulation is N
    forward_backward calls followed by one optim_step. There is no separate
    backward(): Tinker's primitive is fused, and VERL's per-microbatch
    loss.backward() has the same shape.
  - forward_backward() may defer its loss metrics: backends are allowed to
    pipeline the call and fold the metrics into the next optim_step() result.
  - forward() returns per-token logprobs of the completion region with no
    gradient side effects (reference/KL, diagnostics).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

Tokens = list[int]


@dataclass
class SamplingParams:
    # None = no ceiling here; the per-slot protocol caps must then bound every
    # generation (Policy raises if a generation ends up with no budget at all).
    max_tokens: int | None = None
    # Floor on generated tokens (vLLM min_tokens): suppresses EOS for the
    # first N positions. The hw4-parity arms use 8; None = no floor, and an
    # immediate-EOS sample then fails fidelity_ok and is dropped untrained.
    min_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    stop: list[str] | None = None


@dataclass(frozen=True)
class Region:
    """A contiguous span of completion tokens: think | forced_close | visible.
    forced_close tokens are INJECTED (unsampled): logprob 0.0, datum mask 0.0."""

    kind: Literal["think", "forced_close", "visible"]
    start: int
    end: int


@dataclass
class Sample:
    """One sampled completion.

    logprobs are the SAMPLER'S own per-token logprobs — these are
    old_log_probs for the PPO ratio at temperature 1.0. Do not recompute the
    behavior anchor with the trainer casually: on backends where the trainer
    is a different policy than the sampler, that biases the ratio. The ONE
    sanctioned exception is the tempered-sampling re-anchor in train.py
    (temperature != 1.0): there sync_sampler has already pushed the current
    weights, no optimizer step intervenes, and the engine's recompute is the
    only anchor on the same logit scaling as the training-side ratio — vLLM's
    returned logprobs come from RAW logits and would sit off by the
    temperature. Removing that re-anchor reintroduces pre-update clipping on
    every tempered arm.

    stop_reason contract: exactly "stop" (natural/EOS/stop-string) or
    "length" (hit max_tokens). Backends normalize their engine's values.

    regions: set by Policy's budget-forced sampling only; backends leave None.
    """

    tokens: Tokens
    logprobs: list[float]
    text: str
    stop_reason: str  # "stop" | "length"
    prompt_tokens: Tokens | None = None  # set by Policy so envs can build Datums
    regions: tuple[Region, ...] | None = None

    def fidelity_ok(self) -> bool:
        return (
            len(self.tokens) == len(self.logprobs)
            and len(self.tokens) > 0
            and bool(self.stop_reason)
        )


@dataclass
class Datum:
    """One training sequence, unpadded. Backends own padding and layout.

    sampler_logprobs and advantages align with the completion region:
    tokens[prompt_len:]. mask (same alignment) zeroes non-policy tokens in
    multi-turn rollouts; backends fold it into advantages.
    """

    tokens: Tokens
    prompt_len: int
    sampler_logprobs: list[float]
    advantages: list[float]
    mask: list[float] | None = None
    # Frozen-reference logprobs over the completion region, stamped by the
    # train loop under kl_mechanism "loss" (verl consumes them as the batch's
    # ref_log_prob for its differentiable in-loss KL). None = no in-loss KL.
    ref_logprobs: list[float] | None = None

    def __post_init__(self) -> None:
        n = len(self.tokens) - self.prompt_len
        if n <= 0:
            raise ValueError(f"Datum has no completion tokens (prompt_len={self.prompt_len}, len={len(self.tokens)})")
        for name, xs in (("sampler_logprobs", self.sampler_logprobs), ("advantages", self.advantages)):
            if len(xs) != n:
                raise ValueError(f"Datum.{name} has length {len(xs)}, expected {n}")
        if self.mask is not None and len(self.mask) != n:
            raise ValueError(f"Datum.mask has length {len(self.mask)}, expected {n}")

    @property
    def completion_advantages(self) -> list[float]:
        if self.mask is None:
            return self.advantages
        return [a * m for a, m in zip(self.advantages, self.mask)]


@dataclass
class LossSpec:
    """Policy-loss selection. Backend support:
      ppo                  clipped ratio surrogate (GRPO = this + group-normalized
                           advantages from grpo_pack)        [tinker + verl]
      importance_sampling  unclipped ratio * advantage        [tinker + verl(wide clip)]
      reinforce            -logprob * advantage (no ratio)    [tinker via weighted CE,
                                                              verl via loss_mode=gpg]
      cispo                clipped-IS variant                 [tinker + verl]
      gspo                 sequence-level ratio               [verl only]
      cross_entropy        SFT                                [tinker + verl]
    Unsupported (kind, backend) pairs raise at forward_backward time.
    """

    kind: Literal["ppo", "importance_sampling", "reinforce", "cispo", "gspo", "cross_entropy"] = "ppo"
    clip_low: float = 0.8
    clip_high: float = 1.2


@dataclass
class OptimParams:
    lr: float
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0


class Backend(ABC):
    tokenizer: object  # HF tokenizer for the policy model (render + decode)

    @abstractmethod
    def sync_sampler(self) -> None:
        """Make sample() reflect the current training weights."""

    @abstractmethod
    def sample(
        self, prompts: list[Tokens], params: SamplingParams, n: int = 1
    ) -> list[list[Sample]]:
        """n samples per prompt; result[i] are the samples for prompts[i]."""

    @abstractmethod
    def forward(self, data: list[Datum]) -> list[list[float]]:
        """Per-token logprobs of tokens[prompt_len:] under current training weights."""

    def ref_logprobs(self, data: list[Datum]) -> list[list[float]]:
        """Per-token completion logprobs under the FROZEN reference policy
        (the pre-training base). Needed only when kl_coef > 0."""
        raise NotImplementedError(f"{type(self).__name__} has no reference policy")

    @abstractmethod
    def forward_backward(self, data: list[Datum], loss: LossSpec) -> dict[str, float]:
        """Accumulate gradients. Metrics may be deferred to the next optim_step()."""

    @abstractmethod
    def optim_step(self, params: OptimParams) -> dict[str, float]:
        """Apply accumulated gradients and clear them."""

    @abstractmethod
    def save(self, name: str) -> str:
        """Checkpoint weights + optimizer state; returns a path for load()."""

    @abstractmethod
    def load(self, path: str) -> None: ...
