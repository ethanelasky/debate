"""Tinker backend. Near-1:1 mapping onto the Tinker service API.

Pipelining: forward_backward() submits the request and returns immediately;
its loss metrics are folded into the next optim_step() result. This matches
the tinker-cookbook pattern of submitting optim_step before awaiting the
forward/backward future.
"""

from __future__ import annotations

import tinker
from tinker import types

from infra.backend.base import (
    Backend,
    Datum,
    LossSpec,
    OptimParams,
    Sample,
    SamplingParams,
    Tokens,
)


def _to_tinker_datum(d: Datum, loss: LossSpec) -> types.Datum:
    # Cookbook alignment: model_input predicts tokens[1:], so targets in the
    # prompt region are zeroed (they carry zero advantage/weight anyway).
    ob_len = d.prompt_len - 1
    completion = d.tokens[d.prompt_len :]
    model_input = types.ModelInput.from_ints(d.tokens[:-1])
    if loss.kind == "cross_entropy":
        weights = d.mask if d.mask is not None else [1.0] * len(completion)
        loss_fn_inputs = {
            "target_tokens": [0] * ob_len + completion,
            "weights": [0.0] * ob_len + list(weights),
        }
    elif loss.kind == "reinforce":
        # REINFORCE = -sum(adv * logp): weighted cross-entropy with the
        # (mask-folded) advantages as weights. No ratio, no sampler anchor.
        loss_fn_inputs = {
            "target_tokens": [0] * ob_len + completion,
            "weights": [0.0] * ob_len + d.completion_advantages,
        }
    else:
        loss_fn_inputs = {
            "target_tokens": [0] * ob_len + completion,
            "logprobs": [0.0] * ob_len + d.sampler_logprobs,
            "advantages": [0.0] * ob_len + d.completion_advantages,
        }
    return types.Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)


_TINKER_LOSS = {
    "ppo": "ppo",
    "importance_sampling": "importance_sampling",
    "cispo": "cispo",
    "cross_entropy": "cross_entropy",
    "reinforce": "cross_entropy",  # weighted CE (see _to_tinker_datum)
    # "gspo" runs through forward_backward_custom (sequence-level ratio)
}


def make_gspo_loss_fn(metas: list[tuple[int, int, list[float], list[float], list[float]]],
                      clip_low: float, clip_high: float):
    """GSPO (arXiv:2507.18071): sequence-level ratio s_i = exp(mean_t(logp_new - logp_old)),
    loss_i = -min(s_i*A_i, clip(s_i)*A_i). metas: per datum
    (ob_len, n_completion, sampler_logprobs, advantages, mask). The sequence
    advantage is the masked mean of per-token advantages (== the broadcast
    scalar when uniform). NOTE: sequence ratios concentrate near 1 — GSPO wants
    much tighter clips than PPO (paper ~[0.9997, 1.0004]); set them in LossSpec.
    Returns a CustomLossFnV1 for forward_backward_custom (loss summed over
    datums, matching tinker's loss:sum convention)."""
    import torch

    def loss_fn(data, logprobs_list):
        losses = []
        clipped = 0
        for (ob_len, n, q, adv, mask), lp in zip(metas, logprobs_list):
            m = torch.tensor(mask, dtype=lp.dtype)
            denom = m.sum().clamp(min=1.0)
            new_lp = lp[ob_len : ob_len + n]
            s = torch.exp(((new_lp - torch.tensor(q, dtype=lp.dtype)) * m).sum() / denom)
            a = (torch.tensor(adv, dtype=lp.dtype) * m).sum() / denom
            s_clip = torch.clamp(s, clip_low, clip_high)
            losses.append(-torch.min(s * a, s_clip * a))
            clipped += int((s < clip_low) or (s > clip_high))
        loss = torch.stack(losses).sum()
        return loss, {
            "gspo/loss_sum": float(loss.item()),
            "gspo/seq_clip_frac": clipped / max(1, len(metas)),
        }

    return loss_fn


def make_kl_loss_fn(metas: list[tuple[int, int, list[float], list[float]]], coef: float):
    """k3 = exp(d) - d - 1, d = ref - logp_current, summed over unmasked
    completion tokens and scaled by coef. Matches verl's low_var_kl."""
    import torch

    def loss_fn(data, logprobs_list):
        terms = []
        for (ob_len, n, ref, mask), lp in zip(metas, logprobs_list):
            m = torch.tensor(mask, dtype=lp.dtype)
            d = torch.tensor(ref, dtype=lp.dtype) - lp[ob_len : ob_len + n]
            terms.append(((torch.exp(d) - d - 1.0) * m).sum())
        loss = coef * torch.stack(terms).sum()
        return loss, {"kl/in_loss_sum": float(loss.item())}

    return loss_fn


class _Resolved:
    """Adapter so pre-resolved custom-loss results flow through the same
    pending-futures path as normal forward_backward futures."""

    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class TinkerBackend(Backend):
    consumes_ref_logprobs = True

    def __init__(
        self,
        base_model: str,
        lora_rank: int = 32,
        service_client: tinker.ServiceClient | None = None,
        kl_loss_coef: float = 0.0,
    ):
        self.base_model = base_model
        self.kl_loss_coef = float(kl_loss_coef)
        self.service_client = service_client or tinker.ServiceClient()
        self.training_client = self.service_client.create_lora_training_client(
            base_model=base_model, rank=lora_rank
        )
        self.tokenizer = self.training_client.get_tokenizer()
        self._sampling_client = None
        self._pending_fwd_bwd: list = []

    def sync_sampler(self) -> None:
        self._sampling_client = self.training_client.save_weights_and_get_sampling_client()

    def sample(
        self, prompts: list[Tokens], params: SamplingParams, n: int = 1
    ) -> list[list[Sample]]:
        if self._sampling_client is None:
            self.sync_sampler()
        sp = types.SamplingParams(
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop=params.stop,
        )
        futures = [
            self._sampling_client.sample(
                prompt=types.ModelInput.from_ints(p), num_samples=n, sampling_params=sp
            )
            for p in prompts
        ]
        out: list[list[Sample]] = []
        for fut in futures:
            seqs = fut.result().sequences
            out.append(
                [
                    Sample(
                        tokens=list(seq.tokens),
                        logprobs=list(seq.logprobs) if seq.logprobs is not None else [],
                        text=self.tokenizer.decode(seq.tokens),
                        stop_reason=str(seq.stop_reason),
                    )
                    for seq in seqs
                ]
            )
        return out

    def ref_logprobs(self, data: list[Datum]) -> list[list[float]]:
        """Reference = the base model (LoRA-zero == base at init). One
        compute_logprobs call per datum, overlapped via futures.

        Consumed in-loss by _accumulate_kl.
        """
        if getattr(self, "_ref_client", None) is None:
            self._ref_client = self.service_client.create_sampling_client(base_model=self.base_model)
        futures = [
            self._ref_client.compute_logprobs(types.ModelInput.from_ints(d.tokens)) for d in data
        ]
        out = []
        for d, fut in zip(data, futures):
            lps = fut.result()  # index k = logprob of tokens[k]; None where uncomputed
            comp = [lp if lp is not None else 0.0 for lp in lps[d.prompt_len :]]
            out.append(comp)
        return out

    def forward(self, data: list[Datum]) -> list[list[float]]:
        # Tinker's forward requires a loss fn; cross_entropy with zero weights
        # is the idiom for pure logprob extraction.
        tds = []
        for d in data:
            ob_len = d.prompt_len - 1
            completion = d.tokens[d.prompt_len :]
            tds.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(d.tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": [0] * ob_len + completion,
                        "weights": [0.0] * (ob_len + len(completion)),
                    },
                )
            )
        result = self.training_client.forward(tds, "cross_entropy").result()
        out = []
        for d, output in zip(data, result.loss_fn_outputs):
            lp = output["logprobs"].tolist()
            out.append(lp[d.prompt_len - 1 :])
        return out

    def forward_backward(self, data: list[Datum], loss: LossSpec) -> dict[str, float]:
        if loss.kind == "gspo":
            return self._forward_backward_gspo(data, loss)
        if loss.kind not in _TINKER_LOSS:
            raise NotImplementedError(f"loss {loss.kind!r} not supported on tinker (use verl)")
        kind = _TINKER_LOSS[loss.kind]
        config = None
        if loss.kind == "ppo":
            config = {"clip_low_threshold": loss.clip_low, "clip_high_threshold": loss.clip_high}
        tds = [_to_tinker_datum(d, loss) for d in data]
        n_tokens = sum(len(d.tokens) - d.prompt_len for d in data)
        fut = self.training_client.forward_backward(tds, kind, loss_fn_config=config)
        self._pending_fwd_bwd.append((fut, n_tokens))
        self._accumulate_kl(data)
        return {}

    def _accumulate_kl(self, data: list[Datum]) -> None:
        """Gradients accumulate across forward_backward calls until optim_step,
        so the in-loss KL is a second additive pass rather than a rewrite of
        every built-in loss."""
        if self.kl_loss_coef <= 0:
            return
        metas, tds = [], []
        for d in data:
            if d.ref_logprobs is None:
                raise RuntimeError("kl_loss_coef > 0 but datum carries no ref_logprobs")
            ob_len = d.prompt_len - 1
            completion = d.tokens[d.prompt_len :]
            mask = d.mask if d.mask is not None else [1.0] * len(completion)
            metas.append((ob_len, len(completion), d.ref_logprobs, mask))
            tds.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(d.tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": [0] * ob_len + completion,
                        "weights": [0.0] * (ob_len + len(completion)),
                    },
                )
            )
        result = self.training_client.forward_backward_custom(
            tds, make_kl_loss_fn(metas, self.kl_loss_coef)
        )
        if hasattr(result, "result"):
            result = result.result()
        self._pending_fwd_bwd.append((_Resolved(result), 0))

    def _forward_backward_gspo(self, data: list[Datum], loss: LossSpec) -> dict[str, float]:
        """GSPO via forward_backward_custom (any loss differentiable in
        logprobs). Two server round-trips; the call blocks, so its result joins
        the pending list pre-resolved."""
        metas = []
        tds = []
        for d in data:
            ob_len = d.prompt_len - 1
            completion = d.tokens[d.prompt_len :]
            mask = d.mask if d.mask is not None else [1.0] * len(completion)
            metas.append((ob_len, len(completion), d.sampler_logprobs, d.completion_advantages, mask))
            tds.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(d.tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": [0] * ob_len + completion,
                        "weights": [0.0] * (ob_len + len(completion)),
                    },
                )
            )
        loss_fn = make_gspo_loss_fn(metas, loss.clip_low, loss.clip_high)
        result = self.training_client.forward_backward_custom(tds, loss_fn)
        if hasattr(result, "result"):
            result = result.result()
        n_tokens = sum(len(d.tokens) - d.prompt_len for d in data)
        self._pending_fwd_bwd.append((_Resolved(result), n_tokens))
        self._accumulate_kl(data)
        return {}

    def optim_step(self, params: OptimParams) -> dict[str, float]:
        adam = types.AdamParams(
            learning_rate=params.lr,
            beta1=params.betas[0],
            beta2=params.betas[1],
            eps=params.eps,
            weight_decay=params.weight_decay,
            grad_clip_norm=params.grad_clip,
        )
        optim_fut = self.training_client.optim_step(adam)

        loss_sum, n_tokens = 0.0, 0
        extra: dict[str, float] = {}
        for fut, nt in self._pending_fwd_bwd:
            result = fut.result()
            loss_sum += float(result.metrics.get("loss:sum", 0.0))
            n_tokens += nt
            for k, v in result.metrics.items():
                if k.startswith("gspo/"):  # custom-loss metrics pass through
                    extra[f"loss/{k}"] = extra.get(f"loss/{k}", 0.0) + float(v)
        self._pending_fwd_bwd.clear()

        metrics = {"loss/sum": loss_sum, "loss/per_token": loss_sum / max(1, n_tokens), **extra}
        optim_result = optim_fut.result()
        if optim_result.metrics:
            metrics.update({f"optim/{k}": float(v) for k, v in optim_result.metrics.items()})
        return metrics

    def save(self, name: str) -> str:
        return self.training_client.save_state(name, overwrite=True).result().path

    def load(self, path: str) -> None:
        # Tinker only allows load_state on an uninitialized client; resuming
        # means building a fresh training client from the checkpoint.
        self.training_client = self.service_client.create_training_client_from_state_with_optimizer(path)
        self.tokenizer = self.training_client.get_tokenizer()
        self._sampling_client = None
        self._pending_fwd_bwd.clear()
