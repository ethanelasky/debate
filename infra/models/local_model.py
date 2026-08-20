"""Local OpenAI-compatible server (vllm-mlx, LM Studio, llama-server, ...).

Drop-in swap for FireworksModel when iterating on a laptop: a config saying
``model_type: fireworks`` works as ``model_type: local`` provided the server
was launched with ``--served-model-name`` matching ``model_file_path``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx
import openai

from infra.models.base import SpeechStructure
from infra.models.openai_model import OpenAIModel


class LocalModel(OpenAIModel):
    # Gate key; budgets in provider_gate (local: max_tries=1 — this wrapper
    # never had retries and that is preserved).
    PROVIDER = "local"

    DEFAULT_BASE_URL = "http://127.0.0.1:8765/v1"
    DEFAULT_API_KEY = "local"  # ignored unless the server was given --api-key
    DEFAULT_MODEL_ENDPOINT = "qwen3-8b"

    def __init__(
        self,
        alias: str,
        is_debater: bool = True,
        endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs,
    ):
        from infra.models.base import Model

        Model.__init__(self, alias=alias, is_debater=is_debater)

        # 122B-A10B at 4-bit on Apple Silicon emits ~10-15 tok/s, so a
        # 32k-token speech can take ~an hour; the read timeout must sit above
        # that worst case.
        self.client = openai.OpenAI(
            base_url=base_url or os.getenv("LOCAL_LLM_BASE_URL") or self.DEFAULT_BASE_URL,
            api_key=api_key or os.getenv("LOCAL_LLM_API_KEY") or self.DEFAULT_API_KEY,
            timeout=httpx.Timeout(connect=30.0, read=7200.0, write=30.0, pool=30.0),
        )
        self.endpoint = endpoint or self.DEFAULT_MODEL_ENDPOINT
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking
        self.logger = logging.getLogger(__name__)

    def _is_reasoning_model(self) -> bool:
        return False

    def _uses_responses_api(self) -> bool:
        return False

    def _supports_decision_logprobs(self) -> bool:
        return True

    def _extract_thinking_from_choice(self, choice: Any) -> Optional[str]:
        # Servers started with `--reasoning-parser qwen3` put reasoning tokens
        # on message.reasoning_content.
        message = getattr(choice, "message", None)
        for attr in ("reasoning_content", "reasoning"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value:
                return value
        return None

    def build_request_kwargs(
        self,
        messages: list[dict[str, Any]],
        speech_structure: SpeechStructure,
        max_new_tokens: int,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> dict[str, Any]:
        """Wire payload, factored out of call_openai so tests can inspect it
        without a network call."""
        request: dict[str, Any] = {
            "model": self.endpoint,
            "messages": messages,
            "max_tokens": max_new_tokens,
        }
        # top_logprobs is valid only alongside logprobs=true, and strict
        # servers (mlx_lm.server) reject `top_logprobs: None` outright — so
        # omit both on the open-ended path rather than sending nulls.
        if speech_structure != SpeechStructure.OPEN_ENDED:
            request["logprobs"] = True
            request["top_logprobs"] = 5
        for name in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            value = kwargs.get(name)
            if value is not None:
                request[name] = value
        if num_return_sequences > 1:
            request["n"] = num_return_sequences

        # vLLM extensions: the SDK drops unknown top-level kwargs but forwards
        # extra_body verbatim.
        extra_body: dict[str, Any] = {}
        for name in ("top_k", "min_p"):
            value = kwargs.get(name)
            if value is not None:
                extra_body[name] = value
        repetition_penalty = kwargs.get("repetition_penalty")
        if repetition_penalty is not None and repetition_penalty != 1.0:
            # 1.0 is the identity multiplier; sending it only clutters the request.
            extra_body["repetition_penalty"] = repetition_penalty
        if self.enable_thinking is not None:
            # vLLM applies model-specific chat-template options from this
            # extension object. Keep None as "use the server/model default".
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking,
            }
        if extra_body:
            request["extra_body"] = extra_body
        json_schema = kwargs.get("json_schema")
        if json_schema is not None:
            # vLLM grammar-constrained decoding (xgrammar): number bounds are
            # not reliably enforced there, so the schema should carry types and
            # enums only — value clamping stays in the parser.
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": json_schema},
            }
        return request

    def call_openai(
        self,
        messages: list[dict[str, Any]],
        speech_structure: SpeechStructure,
        max_new_tokens: int,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> Any:
        return self.client.chat.completions.create(
            **self.build_request_kwargs(
                messages=messages,
                speech_structure=speech_structure,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                **kwargs,
            )
        )

    def copy(self, is_debater: Optional[bool] = None, **kwargs) -> "LocalModel":
        return LocalModel(
            alias=kwargs.get("alias", self.alias),
            is_debater=is_debater if is_debater is not None else self.is_debater,
            endpoint=self.endpoint,
            base_url=str(self.client.base_url),
            api_key=self.client.api_key,
            reasoning_effort=self.reasoning_effort,
            enable_thinking=self.enable_thinking,
        )
