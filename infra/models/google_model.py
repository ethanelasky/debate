"""Google Gemini wrapper (google-genai SDK).

Covers client setup, the generate_content request/response core,
thought-summary capture, and safety-block detection (blocked prompt or
candidate -> failed=True). Deliberately absent: sampling-capability
declarations, tool adapters, ledger stamping.
"""

from __future__ import annotations

import logging
import os
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

# Short name -> API model id.
GEMINI_MODELS = {
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-flash-latest": "gemini-flash-latest",
    "gemini-pro-latest": "gemini-pro-latest",
}

# finish_reason values that mean the candidate was suppressed, not completed.
BLOCKED_FINISH_REASONS = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}
)
_STOP_REASONS = {"STOP": "stop", "MAX_TOKENS": "length"}


class GoogleModel(Model):
    PROVIDER = "google"
    DEFAULT_MODEL_ENDPOINT = "gemini-3.1-pro"

    def __init__(
        self,
        alias: str,
        is_debater: bool = True,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(alias=alias, is_debater=is_debater)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set (https://aistudio.google.com/app/apikey)")
        endpoint = endpoint or self.DEFAULT_MODEL_ENDPOINT
        self.endpoint = GEMINI_MODELS.get(endpoint, endpoint)
        self._client = None  # lazy: keeps import cost off __init__

    def _get_client(self):
        if self._client is None:
            from google import genai  # optional dep: pip install google-genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> list[ModelResponse]:
        if num_return_sequences != 1:
            raise NotImplementedError("GoogleModel: duplicate inputs instead of num_return_sequences>1")
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
        system_instruction, contents = _generate_gemini_input(model_input_list)
        try:
            response = call_with_retry(
                self.PROVIDER,
                lambda: self._call_google(system_instruction, contents, max_new_tokens, **kwargs),
                giveup=_retry_giveup,
            )
        except Exception as e:
            logger.warning("Google API error: %s", e)
            return ModelResponse(failed=True)

        block = _block_reason(response)
        if block:
            logger.warning("Google blocked the request/response: %s", block)
            return ModelResponse(failed=True)

        message = _extract_text(response)
        if not message:
            logger.warning("Google returned no text")
            return ModelResponse(failed=True)

        return ModelResponse(
            speech=message,
            thinking=_extract_thinking(response),
            prompt="\n".join(mi.content for mi in model_input_list),
            stop_reason=_STOP_REASONS.get(_finish_reason(response) or ""),
        )

    def _call_google(
        self,
        system_instruction: str,
        contents: list[dict],
        max_new_tokens: int,
        **kwargs,
    ):
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_new_tokens,
            "thinking_config": types.ThinkingConfig(include_thoughts=True),
        }
        for key in ("temperature", "top_p"):
            value = kwargs.pop(key, None)
            if value is not None:
                config_kwargs[key] = value
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        return self._get_client().models.generate_content(
            model=self.endpoint,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

    def copy(self, alias: Optional[str] = None, is_debater: Optional[bool] = None, **kwargs) -> "GoogleModel":
        return GoogleModel(
            alias=alias or self.alias,
            is_debater=self.is_debater if is_debater is None else is_debater,
            endpoint=self.endpoint,
            api_key=self.api_key,
        )


def _generate_gemini_input(input_list: list[ModelInput]) -> tuple[str, list[dict]]:
    """(system_instruction, contents). Gemini takes system out of band and
    names the assistant role "model"."""
    system = "\n".join(mi.content for mi in input_list if mi.role == RoleType.SYSTEM)
    contents = [
        {
            "role": "model" if mi.role == RoleType.ASSISTANT else "user",
            "parts": [{"text": mi.content}],
        }
        for mi in input_list
        if mi.role != RoleType.SYSTEM
    ]
    return system, contents


def _retry_giveup(e: Exception) -> bool:
    """Deterministic model/config errors are not worth retrying."""
    text = str(e).lower()
    return "invalid" in text or "not found" in text


def _parts(response: Any):
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []) if content else []:
            yield part


def _enum_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(getattr(value, "name", value)).rsplit(".", 1)[-1].upper()


def _finish_reason(response: Any) -> Optional[str]:
    for candidate in getattr(response, "candidates", None) or []:
        name = _enum_name(getattr(candidate, "finish_reason", None))
        if name:
            return name
    return None


def _block_reason(response: Any) -> Optional[str]:
    """Non-None when the prompt or the candidate was suppressed by safety."""
    if response is None:
        return "empty response"
    feedback = getattr(response, "prompt_feedback", None)
    prompt_block = _enum_name(getattr(feedback, "block_reason", None)) if feedback else None
    if prompt_block and prompt_block not in ("BLOCK_REASON_UNSPECIFIED", "NONE"):
        return f"prompt: {prompt_block}"
    finish = _finish_reason(response)
    if finish in BLOCKED_FINISH_REASONS:
        return f"candidate: {finish}"
    return None


def _extract_text(response: Any) -> str:
    return "".join(
        getattr(p, "text", "") or ""
        for p in _parts(response)
        if not getattr(p, "thought", False)
    )


def _extract_thinking(response: Any) -> Optional[str]:
    """Thought summaries come back as parts with thought=True."""
    return "".join(
        getattr(p, "text", "") or "" for p in _parts(response) if getattr(p, "thought", False)
    ) or None
