"""OpenRouter (https://openrouter.ai) — one OpenAI-compatible API in front of
many providers. ``endpoint`` must be an OpenRouter model slug, e.g.
``openai/gpt-oss-120b``. Key from ``OPENROUTER_API_KEY``; ``OPENROUTER_BASE_URL``
overrides the default host.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import httpx
import openai

from infra.models.base import ModelResponse, SpeechStructure
from infra.models.openai_model import OpenAIModel


class RequiredTokenLogprobsError(RuntimeError, ValueError):
    """Non-transient reward-contract failure; the retry layer re-raises it."""


_HARMONY_FINAL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|\Z)", re.DOTALL)
_HARMONY_ANALYSIS_RE = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|\Z)", re.DOTALL)
_HARMONY_ANY_MARKER_RE = re.compile(r"<\|(?:channel|start|end|return|message)\|>")


def _split_harmony(text: str) -> tuple[str, Optional[str]]:
    """Split gpt-oss Harmony-tagged output into (final, analysis).

    Untagged text passes through as (text, None). If generation was truncated
    before the ``final`` channel opened, returns ("", analysis) so the caller
    can still surface the reasoning.
    """
    if not text or "<|channel|>" not in text:
        return text, None
    final = _HARMONY_FINAL_RE.search(text)
    analysis = _HARMONY_ANALYSIS_RE.search(text)
    analysis_text = analysis.group(1).strip() if analysis else None
    return (final.group(1).strip() if final else ""), analysis_text


class OpenRouterModel(OpenAIModel):
    # Gate key; budgets in provider_gate (openrouter: max_tries=6, factor=4).
    PROVIDER = "openrouter"

    # The old wrapper retried on this explicit tuple with NO giveup predicate,
    # so the inherited predicate must be disabled here.
    _RETRY_EXCEPTION = (
        json.JSONDecodeError,
        openai.APIError,
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.InternalServerError,
        openai.APITimeoutError,
    )
    _retry_giveup = None

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        alias: str,
        is_debater: bool = True,
        endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        capture_token_logprobs: bool = False,
        require_token_logprobs: bool = False,
        provider_order: Optional[list[str]] = None,
        quantizations: Optional[list[str]] = None,
        allow_fallbacks: Optional[bool] = None,
        data_collection: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs,
    ):
        from infra.models.base import Model

        Model.__init__(self, alias=alias, is_debater=is_debater)

        self.capture_token_logprobs = capture_token_logprobs
        self.require_token_logprobs = require_token_logprobs
        if require_token_logprobs and not capture_token_logprobs:
            raise ValueError("require_token_logprobs=True requires capture_token_logprobs=True")

        # Pins are sent on EVERY request: OpenRouter otherwise spreads one slug
        # across backends with differing quantizations, destroying
        # reproducibility.
        self.provider_order = list(provider_order) if provider_order else None
        self.quantizations = list(quantizations) if quantizations else None
        self.allow_fallbacks = allow_fallbacks
        self.data_collection = data_collection

        # OpenRouter's top-level reasoning toggle, sent as extra_body
        # {"reasoning": {"enabled": ...}}. None = don't send (provider
        # default). enable_thinking=False is how a hybrid-thinking model
        # (e.g. qwen/qwen3.5-9b on parasail) is run as a no-think judge:
        # verified 2026-07-31 that default requests emit reasoning while
        # {"enabled": false} yields reasoning_tokens == 0.
        self.enable_thinking = enable_thinking

        resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get your API key from https://openrouter.ai/keys"
            )
        if not endpoint:
            raise ValueError("OpenRouterModel requires an OpenRouter model slug (e.g. 'openai/gpt-oss-120b')")

        resolved_base_url = base_url or os.getenv("OPENROUTER_BASE_URL") or self.DEFAULT_BASE_URL
        # 10 min per call: past that it is a provider stall we cannot recover
        # from, and a timeout lets the run move on instead of blocking a round.
        self.client = openai.OpenAI(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://github.com/ethanelasky/debate"),
                "X-Title": os.getenv("OPENROUTER_APP_TITLE", "debate"),
            },
        )
        self.endpoint = endpoint
        self.reasoning_effort = reasoning_effort
        self.logger = logging.getLogger(__name__)

    def _is_reasoning_model(self) -> bool:
        return False

    def _uses_responses_api(self) -> bool:
        return False

    def _supports_decision_logprobs(self) -> bool:
        # gpt-oss-{20b,120b} and several other OpenRouter backends return
        # EMPTY content when logprobs=true, silently killing the decision call.
        # Opt out by default; the caller's JSON-parse fallback recovers the
        # verdict from plain content.
        return False

    # -- gpt-oss Harmony channels -----------------------------------------

    def _extract_message_text_from_choice(self, choice: Any) -> str:
        text = super()._extract_message_text_from_choice(choice)
        if text:
            final, _ = _split_harmony(text)
            if final:
                return final
        message = getattr(choice, "message", None)
        for attr in ("reasoning", "reasoning_content"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value.strip():
                final, _ = _split_harmony(value)
                return final or _HARMONY_ANY_MARKER_RE.sub("", value).strip()
        return ""

    def _extract_thinking_from_choice(self, choice: Any) -> Optional[str]:
        message = getattr(choice, "message", None)
        for attr in ("reasoning", "reasoning_content"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value.strip():
                _, analysis = _split_harmony(value)
                return analysis or value
        content = getattr(message, "content", None)
        if isinstance(content, str) and "<|channel|>analysis" in content:
            return _split_harmony(content)[1]
        return None

    # -- request -----------------------------------------------------------

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
        wants_logprobs = speech_structure != SpeechStructure.OPEN_ENDED and self._supports_decision_logprobs()
        request: dict[str, Any] = {
            "model": self.endpoint,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "logprobs": wants_logprobs,
            "top_logprobs": 5 if wants_logprobs else None,
        }
        provider_opts: dict[str, Any] = {}
        if self.provider_order:
            provider_opts["order"] = list(self.provider_order)
        if self.quantizations:
            provider_opts["quantizations"] = list(self.quantizations)
        if self.allow_fallbacks is not None:
            provider_opts["allow_fallbacks"] = self.allow_fallbacks
        elif provider_opts:
            # Pins default to no-fallback so a run cannot silently migrate to a
            # different backend or quantization.
            provider_opts["allow_fallbacks"] = False
        if self.data_collection:
            provider_opts["data_collection"] = self.data_collection
        if self.capture_token_logprobs:
            # Judge-logit capture wants logprobs on ALL calls, including
            # OPEN_ENDED. require_parameters keeps OpenRouter from routing to a
            # provider that silently drops the param.
            request["logprobs"] = True
            provider_opts["require_parameters"] = True
        if provider_opts:
            # The SDK rejects unknown top-level kwargs; provider options must
            # ride in extra_body.
            request["extra_body"] = {"provider": provider_opts}
        # The parent forwards reasoning_effort only on the Responses-API
        # branch. Without the effort override a Harmony model runs at DEFAULT
        # effort and can burn the whole max_tokens budget in the analysis
        # channel, returning an empty final channel (no verdict). OpenRouter
        # INFERS "enabled" from "effort", so the two keys must never ship
        # together: {"effort": ..., "enabled": false} is self-contradictory and
        # the provider is free to honor either half.
        reasoning_opts: dict[str, Any] = {}
        if self.enable_thinking is False:
            reasoning_opts["enabled"] = False
        elif self.reasoning_effort:
            reasoning_opts["effort"] = self.reasoning_effort
        elif self.enable_thinking:
            reasoning_opts["enabled"] = True
        if reasoning_opts:
            request.setdefault("extra_body", {})["reasoning"] = reasoning_opts
        if temperature is not None:
            request["temperature"] = temperature
        if top_p is not None:
            request["top_p"] = top_p
        if num_return_sequences > 1:
            request["n"] = num_return_sequences

        completion = self.client.chat.completions.create(**request)
        self._tag_provenance(completion)
        if not self.capture_token_logprobs:
            return completion

        choices = list(getattr(completion, "choices", None) or [])
        if not choices or any(self._extract_message_text_from_choice(c) for c in choices):
            return completion

        # Every choice came back empty under logprobs=true — the documented
        # gpt-oss failure. The verdict is worth more than the capture, so retry
        # once without logprobs unless the run demands capture.
        if self.require_token_logprobs:
            raise RequiredTokenLogprobsError(
                f"OpenRouter model '{self.endpoint}' returned empty content with logprobs=true; "
                "require_token_logprobs is set, so refusing to retry without them"
            )
        self.logger.warning(
            "OpenRouter model '%s' returned empty content with logprobs=true; retrying ONCE "
            "without logprobs. Token-logprob capture is SACRIFICED for this call and the "
            "judge-logit reward falls back to the elicited JSON confidence.",
            self.endpoint,
        )
        retry = dict(request)
        retry["logprobs"] = False
        retry["top_logprobs"] = None
        # Keep the pins and the reasoning-effort override (dropping either
        # reintroduces the failure this retry exists to escape), but drop
        # require_parameters: it only guaranteed logprob support, which we just
        # gave up, and keeping it would restrict routing to the pool that failed.
        extra = dict(retry.get("extra_body") or {})
        provider = {k: v for k, v in (extra.get("provider") or {}).items() if k != "require_parameters"}
        if provider:
            extra["provider"] = provider
        else:
            extra.pop("provider", None)
        if extra:
            retry["extra_body"] = extra
        else:
            retry.pop("extra_body", None)
        completion = self.client.chat.completions.create(**retry)
        self._tag_provenance(completion)
        for choice in getattr(completion, "choices", None) or []:
            try:
                choice._debate_logprobs_sacrificed = True
            except Exception:
                pass
        return completion

    @staticmethod
    def _tag_provenance(completion: Any) -> None:
        """Stamp completion-level provenance onto each choice: the parent builds
        one ModelResponse per CHOICE and never sees the completion object, so
        the choice is the only carrier that reaches _build_response."""
        provider = getattr(completion, "provider", None)
        if not isinstance(provider, str) or not provider:
            extra = getattr(completion, "model_extra", None)
            provider = extra.get("provider") if isinstance(extra, dict) else None
        generation_id = getattr(completion, "id", None)
        for choice in getattr(completion, "choices", None) or []:
            try:
                choice._debate_served_provider = provider if isinstance(provider, str) and provider else None
                choice._debate_generation_id = generation_id if isinstance(generation_id, str) and generation_id else None
            except Exception:
                pass

    # -- capture seam ------------------------------------------------------

    def _build_response(self, *, choice: Any, **kwargs) -> ModelResponse:
        """Every chat choice funnels through here, including OPEN_ENDED ones,
        so this is where per-choice logprobs meet the ModelResponse."""
        response = super()._build_response(choice=choice, **kwargs)
        if choice is None:
            return response
        response.served_provider = getattr(choice, "_debate_served_provider", None)
        response.generation_id = getattr(choice, "_debate_generation_id", None)
        if not self.capture_token_logprobs:
            return response

        entries = getattr(getattr(choice, "logprobs", None), "content", None)
        token_texts: list[str] = []
        logprobs: list[float] = []
        parsed_ok = True
        for entry in entries or []:
            token = getattr(entry, "token", None)
            logprob = getattr(entry, "logprob", None)
            if not isinstance(token, str) or logprob is None:
                parsed_ok = False
                break
            token_texts.append(token)
            logprobs.append(float(logprob))
        if entries and parsed_ok:
            response.response_token_texts = token_texts
            response.response_logprobs = logprobs
            return response

        if getattr(choice, "_debate_logprobs_sacrificed", False):
            # The empty-content guard already logged the sacrifice loudly.
            return response
        message = (
            f"OpenRouter model '{self.endpoint}' was asked to capture token logprobs but the "
            "response carries no usable logprobs.content; the judge-logit continuous reward "
            "silently degrades to the elicited JSON confidence for this call. Set "
            "require_token_logprobs to fail hard instead, or pin a provider that honors "
            "logprobs via provider_order."
        )
        if self.require_token_logprobs:
            raise RequiredTokenLogprobsError(message)
        self.logger.warning(message)
        return response

    def copy(self, is_debater: Optional[bool] = None, **kwargs) -> "OpenRouterModel":
        return OpenRouterModel(
            alias=kwargs.get("alias", self.alias),
            is_debater=is_debater if is_debater is not None else self.is_debater,
            endpoint=self.endpoint,
            base_url=str(self.client.base_url),
            api_key=self.client.api_key,
            reasoning_effort=self.reasoning_effort,
            # Judges are produced via copy(); dropping these would silently
            # kill capture for every real judge.
            capture_token_logprobs=self.capture_token_logprobs,
            require_token_logprobs=self.require_token_logprobs,
            provider_order=self.provider_order,
            quantizations=self.quantizations,
            allow_fallbacks=self.allow_fallbacks,
            data_collection=self.data_collection,
            # Judges are produced via copy(); dropping this would silently
            # turn a no-think judge back into a thinking one.
            enable_thinking=self.enable_thinking,
        )
