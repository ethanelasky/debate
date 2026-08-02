"""Tinker sampling as a Model: eval of trained checkpoints (tinker:// URIs)
and tinker-served frozen seats (judges, opponents).

Slim port of the old repo's 1,476-line tinker_model.py. Kept: the two
constructors and their routing, batch predict with overlapped futures, full
token/logprob capture (this wrapper is the token-fidelity path), <think>
extraction. Dropped: the budget-forced two-phase thinking sampler
(thinking_budget_tokens) — enable_thinking only toggles the chat-template
flag here. Training-time sampling does NOT go through this class; that's
backend.Policy over TinkerBackend.
"""

from __future__ import annotations

import re
from typing import Optional

from infra.models.base import DEFAULT_MAX_NEW_TOKENS, Model, ModelInput, ModelResponse

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class TinkerModel(Model):
    def __init__(
        self,
        alias: str,
        sampling_client,
        is_debater: bool = True,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        enable_thinking: Optional[bool] = None,
    ):
        super().__init__(alias=alias, is_debater=is_debater)
        self.sampling_client = sampling_client
        self.tokenizer = sampling_client.get_tokenizer()
        self.temperature = temperature if temperature is not None else 1.0
        self.top_p = top_p if top_p is not None else 1.0
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

    # ------------------------------------------------------------ builders

    @classmethod
    def from_base_model(cls, base_model: str, **kwargs) -> "TinkerModel":
        import tinker

        client = tinker.ServiceClient().create_sampling_client(base_model=base_model)
        return cls(sampling_client=client, **kwargs)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **kwargs) -> "TinkerModel":
        """checkpoint_path is a tinker:// sampler-weights URI
        (save_weights_for_sampler output)."""
        import tinker

        client = tinker.ServiceClient().create_sampling_client(model_path=checkpoint_path)
        return cls(sampling_client=client, **kwargs)

    # ------------------------------------------------------------- predict

    def _render(self, convo: list[ModelInput]) -> list[int]:
        messages = [{"role": mi.role.api_name, "content": mi.content} for mi in convo]
        kwargs = {}
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        out = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, **kwargs
        )
        if not isinstance(out, list):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return list(out)

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens: int = 0,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> list[ModelResponse]:
        from tinker import types

        params = types.SamplingParams(
            max_tokens=max_new_tokens or self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        prompts = [self._render(convo) for convo in inputs]
        futures = [
            self.sampling_client.sample(
                prompt=types.ModelInput.from_ints(p),
                num_samples=num_return_sequences,
                sampling_params=params,
            )
            for p in prompts
        ]

        responses: list[ModelResponse] = []
        for prompt_tokens, fut in zip(prompts, futures):
            try:
                sequences = fut.result().sequences
            except Exception as e:
                responses.extend(
                    ModelResponse(failed=True, raw_response=f"{type(e).__name__}: {e}")
                    for _ in range(num_return_sequences)
                )
                continue
            for seq in sequences:
                raw = self.tokenizer.decode(seq.tokens)
                thinking = None
                speech = raw
                m = THINK_RE.search(raw)
                if m:
                    thinking = m.group(1).strip()
                    speech = raw[m.end() :].strip()
                responses.append(
                    ModelResponse(
                        speech=speech,
                        thinking=thinking,
                        raw_response=raw,
                        prompt_tokens=list(prompt_tokens),
                        response_tokens=list(seq.tokens),
                        response_logprobs=list(seq.logprobs) if seq.logprobs is not None else [],
                        stop_reason=str(seq.stop_reason),
                    )
                )
        return responses

    def decode_tokens(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def supports_server_side_sampling(self) -> bool:
        return True
