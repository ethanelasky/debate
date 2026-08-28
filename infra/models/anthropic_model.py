"""Anthropic (Claude) wrapper.

Covers system extraction (Anthropic takes `system` out of band), the
streaming Messages request, extended thinking, per-model max_tokens caps, and
stop_reason mapping. Deliberately absent: sampling-capability declarations,
tool adapters, best-of-n, ledger stamping.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from infra.models.base import (
    DEFAULT_MAX_NEW_TOKENS,
    Model,
    ModelInput,
    ModelResponse,
    RoleType,
)
from infra.models.provider_gate import call_with_retry, run_batched_predict

logger = logging.getLogger(__name__)

# max_tokens above these caps is a 400 from the API; clamp instead of failing.
MODEL_OUTPUT_TOKEN_CAPS = {
    "claude-opus-4-5-20251101": 64000,
    "claude-opus-4-5": 64000,
    "claude-opus-4-6": 128000,
    "claude-opus-4-7": 128000,
    "claude-sonnet-4-6": 64000,
    "claude-haiku-4-5-20251001": 64000,
    "claude-haiku-4-5": 64000,
}

_STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}


class AnthropicModel(Model):
    PROVIDER = "anthropic"
    DEFAULT_MODEL_ENDPOINT = "claude-opus-4-5"

    def __init__(
        self,
        alias: str,
        is_debater: bool = True,
        endpoint: Optional[str] = None,
        thinking_budget_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """thinking_budget_tokens: None auto-maxes to max_new_tokens - 1024;
        an explicit int overrides; 0 (or <1024) disables extended thinking."""
        super().__init__(alias=alias, is_debater=is_debater)
        import anthropic  # optional dep; ANTHROPIC_API_KEY read by the SDK

        # 10min SDK default is too short for Opus with a large thinking budget.
        self.client = anthropic.Anthropic(timeout=1800.0, **({"api_key": api_key} if api_key else {}))
        self.endpoint = endpoint or self.DEFAULT_MODEL_ENDPOINT
        self.thinking_budget_tokens = thinking_budget_tokens

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> list[ModelResponse]:
        if num_return_sequences != 1:
            raise NotImplementedError("AnthropicModel: duplicate inputs instead of num_return_sequences>1")
        return run_batched_predict(
            provider=self.PROVIDER,
            num_items=len(inputs),
            call_one=lambda i: self.predict_single_input(inputs[i], max_new_tokens, **kwargs),
            logger=logger,
            deadline_result_factory=lambda: ModelResponse(failed=True),
        )

    def predict_single_input(
        self,
        model_input_list: list[ModelInput],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        **kwargs,
    ) -> ModelResponse:
        kwargs.pop("tools", None)
        kwargs.pop("tool_results", None)
        system, messages = self.generate_llm_input_from_model_inputs(model_input_list)
        try:
            completion = call_with_retry(
                self.PROVIDER,
                lambda: self.call_anthropic(system, messages, max_new_tokens, **kwargs),
            )
        except Exception as e:
            logger.warning("Anthropic API error: %s", e)
            return ModelResponse(failed=True)

        return ModelResponse(
            speech=_extract_text(completion),
            thinking=_extract_thinking(completion),
            prompt="\n".join(mi.content for mi in model_input_list),
            stop_reason=_STOP_REASONS.get(getattr(completion, "stop_reason", None) or ""),
        )

    def call_anthropic(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        **kwargs,
    ):
        cap = MODEL_OUTPUT_TOKEN_CAPS.get(self.endpoint)
        if cap is not None and max_new_tokens > cap:
            max_new_tokens = cap
        budget = self.thinking_budget_tokens
        if budget is None:
            budget = max_new_tokens - 1024
        request: dict[str, Any] = {
            "model": self.endpoint,
            "max_tokens": max_new_tokens,
            "system": system,
            "messages": messages,
        }
        temperature = kwargs.pop("temperature", None)
        if temperature is not None:
            request["temperature"] = temperature
        if budget >= 1024:
            request["thinking"] = {"type": "enabled", "budget_tokens": budget}
            request["temperature"] = 1.0  # the API rejects anything else here
        # Anthropic rejects non-streaming requests that may exceed 10 minutes;
        # the aggregated final message has the same shape as messages.create().
        with self.client.messages.stream(**request) as stream:
            return stream.get_final_message()

    def copy(self, alias: Optional[str] = None, is_debater: Optional[bool] = None, **kwargs) -> "AnthropicModel":
        return AnthropicModel(
            alias=alias or self.alias,
            is_debater=self.is_debater if is_debater is None else is_debater,
            endpoint=self.endpoint,
            thinking_budget_tokens=self.thinking_budget_tokens,
        )

    @staticmethod
    def generate_llm_input_from_model_inputs(
        input_list: list[ModelInput],
    ) -> tuple[str, list[dict[str, str]]]:
        """(system prompt, non-system messages). All system turns are joined
        into one prompt — Anthropic takes it as a separate request field."""
        system = "\n".join(mi.content for mi in input_list if mi.role == RoleType.SYSTEM)
        messages = [
            {"role": mi.role.api_name, "content": mi.content}
            for mi in input_list
            if mi.role != RoleType.SYSTEM
        ]
        return system, messages


def _extract_text(completion: Any) -> str:
    """Join `text` blocks; with thinking enabled content interleaves types."""
    content = getattr(completion, "content", None) or []
    if not isinstance(content, list):
        return str(content)
    return "".join(
        getattr(b, "text", "") for b in content if getattr(b, "type", None) in (None, "text")
    )


def _extract_thinking(completion: Any) -> Optional[str]:
    content = getattr(completion, "content", None) or []
    if not isinstance(content, list):
        return None
    parts = [getattr(b, "thinking", "") for b in content if getattr(b, "type", None) == "thinking"]
    return "".join(p for p in parts if p) or None
