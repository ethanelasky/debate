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
    else:
        loss_fn_inputs = {
            "target_tokens": [0] * ob_len + completion,
            "logprobs": [0.0] * ob_len + d.sampler_logprobs,
            "advantages": [0.0] * ob_len + d.completion_advantages,
        }
    return types.Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)


class TinkerBackend(Backend):
    def __init__(
        self,
        base_model: str,
        lora_rank: int = 32,
        service_client: tinker.ServiceClient | None = None,
    ):
        self.base_model = base_model
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
        compute_logprobs call per datum, overlapped via futures."""
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
        kind = loss.kind
        config = None
        if kind == "ppo":
            config = {"clip_low_threshold": loss.clip_low, "clip_high_threshold": loss.clip_high}
        tds = [_to_tinker_datum(d, loss) for d in data]
        n_tokens = sum(len(d.tokens) - d.prompt_len for d in data)
        fut = self.training_client.forward_backward(tds, kind, loss_fn_config=config)
        self._pending_fwd_bwd.append((fut, n_tokens))
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
        for fut, nt in self._pending_fwd_bwd:
            result = fut.result()
            loss_sum += float(result.metrics.get("loss:sum", 0.0))
            n_tokens += nt
        self._pending_fwd_bwd.clear()

        metrics = {"loss/sum": loss_sum, "loss/per_token": loss_sum / max(1, n_tokens)}
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
