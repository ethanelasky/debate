"""Fireworks.ai — OpenAI-compatible inference for Qwen and other open weights."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any, Optional

import openai

from infra.models.base import SpeechStructure
from infra.models.openai_model import OpenAIModel

# Fireworks rejects non-streaming requests above this budget.
_STREAMING_REQUIRED_ABOVE = 5000


class FireworksModel(OpenAIModel):
    # Gate key; budgets in provider_gate (fireworks: max_tries=1 — this
    # wrapper never had retries and that is preserved).
    PROVIDER = "fireworks"

    QWEN_MODELS = {
        "qwen3-8b": "accounts/fireworks/models/qwen3-8b",
        "qwen3-30b": "accounts/fireworks/models/qwen3-30b-a3b",
        "qwen3-235b": "accounts/fireworks/models/qwen3-235b-a22b",
        "qwen3-235b-instruct": "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
        "qwen3-coder-480b": "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
    }

    DEFAULT_MODEL_ENDPOINT = QWEN_MODELS["qwen3-8b"]
    BASE_URL = "https://api.fireworks.ai/inference/v1"

    def __init__(
        self,
        alias: str,
        is_debater: bool = True,
        endpoint: Optional[str] = None,
        model_name: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        from infra.models.base import Model

        Model.__init__(self, alias=alias, is_debater=is_debater)

        resolved_api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "FIREWORKS_API_KEY environment variable not set. "
                "Get your API key from https://fireworks.ai"
            )
        self.client = openai.OpenAI(base_url=base_url or self.BASE_URL, api_key=resolved_api_key)
        # Short names are accepted through either param (model_utils passes
        # the config's model_file_path as `endpoint`).
        self.endpoint = (
            self.QWEN_MODELS.get(model_name or "")
            or self.QWEN_MODELS.get(endpoint or "")
            or endpoint
            or self.DEFAULT_MODEL_ENDPOINT
        )
        self.reasoning_effort = reasoning_effort
        self.logger = logging.getLogger(__name__)

    def _is_reasoning_model(self) -> bool:
        return False

    def _uses_responses_api(self) -> bool:
        return False

    def _supports_decision_logprobs(self) -> bool:
        return True

    def call_openai(
        self,
        messages: list[dict[str, Any]],
        speech_structure: SpeechStructure,
        max_new_tokens: int,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> Any:
        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        wants_logprobs = speech_structure != SpeechStructure.OPEN_ENDED
        request: dict[str, Any] = {
            "model": self.endpoint,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "logprobs": wants_logprobs,
            "top_logprobs": 5 if wants_logprobs else None,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if top_p is not None:
            request["top_p"] = top_p
        if num_return_sequences > 1:
            request["n"] = num_return_sequences

        if max_new_tokens <= _STREAMING_REQUIRED_ABOVE:
            return self.client.chat.completions.create(**request)

        request["stream"] = True
        stream = self.client.chat.completions.create(**request)
        content: list[str] = []
        logprobs_data = None
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                content.append(delta.content)
            if getattr(chunk.choices[0], "logprobs", None):
                logprobs_data = chunk.choices[0].logprobs
        choice = SimpleNamespace(
            message=SimpleNamespace(content="".join(content)),
            logprobs=logprobs_data,
            finish_reason=None,
        )
        return SimpleNamespace(choices=[choice])

    def copy(self, is_debater: Optional[bool] = None, **kwargs) -> "FireworksModel":
        return FireworksModel(
            alias=kwargs.get("alias", self.alias),
            is_debater=is_debater if is_debater is not None else self.is_debater,
            endpoint=self.endpoint,
            reasoning_effort=self.reasoning_effort,
        )
