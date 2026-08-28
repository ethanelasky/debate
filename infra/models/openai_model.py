"""OpenAI Chat Completions / Responses wrapper — the de-facto base class for
every OpenAI-compatible provider (Fireworks, OpenRouter, DashScope, local
vLLM/LM Studio servers).

Salvaged from ~/ai-debate/ai_infra/models/openai_model.py. Kept: client
construction, the request core, decision-logprob capture, reasoning-model
detection, server-side n>1, response extraction. Dropped: the sampling
capability/validation machinery (wrappers now just place the sampling kwargs
they support and ignore the rest), tool-call plumbing, token-fidelity
enforcement, BoN/probe hooks.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from typing import Any, Optional

import openai

from infra.models.base import (
    DEFAULT_DEBATER_A_NAME,
    DEFAULT_DEBATER_B_NAME,
    Model,
    ModelInput,
    ModelResponse,
    RoleType,
    SpeechStructure,
)
from infra.models.provider_gate import call_with_retry, run_batched_predict

DEFAULT_MAX_NEW_TOKENS = 10000
HARD_CAP_MAX_NEW_TOKENS = 127500
STRUCTURED_OUTPUT_SCHEMA_NAME = "verdict"
TRANSIENT_RESPONSES_ERROR_CODES = frozenset(
    {"server_error", "rate_limit_exceeded", "vector_store_timeout"}
)


def _safe_response_detail(value: Any) -> Any:
    """Small, credential-free projection of Responses error/detail objects."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: value[key] for key in ("code", "message", "reason") if key in value}
    detail = {
        key: getattr(value, key)
        for key in ("code", "message", "reason")
        if getattr(value, key, None) is not None
    }
    return detail or str(value)


class ResponsesResultAPIError(Exception):
    """A syntactically successful Responses call produced no usable result."""

    def __init__(self, provenance: dict[str, Any], generation_id: Optional[str] = None):
        self.raw_response = json.dumps(provenance, sort_keys=True)
        self.generation_id = generation_id
        super().__init__(self.raw_response)


class IncompleteResponseAPIError(ResponsesResultAPIError):
    """Responses API returned status='incomplete' without usable output.

    Deliberately not an openai.APIError: that constructor's signature varies
    across SDK versions, and this is an internal control-flow signal.
    """

    def __init__(self, endpoint: str, details: Any, generation_id: Optional[str] = None):
        safe_details = _safe_response_detail(details)
        self.reason = (
            safe_details.get("reason") if isinstance(safe_details, dict) else None
        )
        super().__init__(
            {
                "type": "responses_incomplete",
                "endpoint": endpoint,
                "status": "incomplete",
                "incomplete_details": safe_details,
            },
            generation_id,
        )


class ResponsesStatusAPIError(ResponsesResultAPIError):
    """Responses returned a nonterminal/failed status instead of completed."""

    def __init__(
        self, endpoint: str, status: Any, error: Any, generation_id: Optional[str] = None
    ):
        safe_error = _safe_response_detail(error)
        self.status = status
        self.error_code = (
            safe_error.get("code") if isinstance(safe_error, dict) else None
        )
        super().__init__(
            {
                "type": "responses_status",
                "endpoint": endpoint,
                "status": status,
                "error": safe_error,
            },
            generation_id,
        )


class ResponsesRefusalAPIError(ResponsesResultAPIError):
    """Responses returned explicit refusal content instead of judge output."""

    def __init__(self, endpoint: str, refusal: str, generation_id: Optional[str] = None):
        super().__init__(
            {
                "type": "responses_refusal",
                "endpoint": endpoint,
                "status": "completed",
                "refusal": refusal,
            },
            generation_id,
        )


class OpenAIModel(Model):
    # Gate registry key; concurrency/timeout/retry budgets live in
    # infra/models/provider_gate.py. Subclasses override per provider.
    PROVIDER = "openai"
    DEFAULT_MODEL_ENDPOINT = "gpt-4-0125-preview"

    # Retry spec for provider_gate.call_with_retry. Subclasses override.
    _RETRY_EXCEPTION: Any = Exception

    def __init__(
        self,
        alias: str,
        is_debater: bool = True,
        endpoint: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(alias=alias, is_debater=is_debater)
        openai.organization = os.getenv("OPENAI_ORGANIZATION")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client_kwargs: dict[str, Any] = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key or os.getenv("OPENAI_API_KEY"):
            client_kwargs["api_key"] = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(**client_kwargs)
        self.endpoint = endpoint or type(self).DEFAULT_MODEL_ENDPOINT
        self.reasoning_effort = reasoning_effort
        self.logger = logging.getLogger(__name__)

    # -- batch driver ------------------------------------------------------

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens: int = HARD_CAP_MAX_NEW_TOKENS,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> list[ModelResponse]:
        """Flat list of len(inputs) * num_return_sequences; inputs[i]'s samples
        occupy [i*n, (i+1)*n)."""
        max_new_tokens = min(max_new_tokens, HARD_CAP_MAX_NEW_TOKENS)

        def call_one(idx: int) -> list[ModelResponse]:
            return self.predict_single_input_n(
                model_input_list=inputs[idx],
                max_new_tokens=max_new_tokens,
                speech_structure=speech_structure,
                num_return_sequences=num_return_sequences,
                **kwargs,
            )

        nested = run_batched_predict(
            provider=self.PROVIDER,
            num_items=len(inputs),
            call_one=call_one,
            logger=self.logger,
            deadline_result_factory=lambda: [ModelResponse(failed=True) for _ in range(num_return_sequences)],
        )
        return [response for batch in nested for response in batch]

    def supports_server_side_sampling(self) -> bool:
        """Chat completions honors n>1; the Responses API has no equivalent."""
        return not self._uses_responses_api()

    def predict_single_input(
        self,
        model_input_list: list[ModelInput],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        **kwargs,
    ) -> ModelResponse:
        responses = self.predict_single_input_n(
            model_input_list=model_input_list,
            max_new_tokens=max_new_tokens,
            speech_structure=speech_structure,
            num_return_sequences=1,
            **kwargs,
        )
        return responses[0] if responses else ModelResponse(failed=True)

    def predict_single_input_n(
        self,
        model_input_list: list[ModelInput],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> list[ModelResponse]:
        messages = self.generate_llm_input_from_model_inputs(model_input_list)
        max_new_tokens = min(max_new_tokens, HARD_CAP_MAX_NEW_TOKENS)
        prompt_text = "\n".join(model_input.content for model_input in model_input_list)

        if num_return_sequences > 1 and self._uses_responses_api():
            raise ValueError(
                f"model '{self.endpoint}' uses the Responses API, which has no "
                f"num_return_sequences>1 (got {num_return_sequences})"
            )
        if self.reasoning_effort is not None:
            kwargs.setdefault("reasoning_effort", self.reasoning_effort)

        try:
            completion = call_with_retry(
                self.PROVIDER,
                lambda: self.call_openai(
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    speech_structure=speech_structure,
                    num_return_sequences=num_return_sequences,
                    **kwargs,
                ),
                exception=self._RETRY_EXCEPTION,
                giveup=self._retry_giveup,
            )
        except ResponsesResultAPIError as e:
            # Incomplete/refusal results give up immediately; failed statuses
            # arrive here only after exhausting the provider retry budget.
            # Degrade only this batch item so DebateRound invalidates it rather
            # than scoring it as a wrong verdict, while siblings continue.
            self.logger.warning("%s", e)
            return [
                ModelResponse(
                    prompt=prompt_text,
                    raw_response=e.raw_response,
                    failed=True,
                    stop_reason=(
                        "length"
                        if isinstance(e, IncompleteResponseAPIError)
                        and e.reason == "max_output_tokens"
                        else None
                    ),
                    served_provider=self.PROVIDER,
                    generation_id=e.generation_id,
                )
                for _ in range(num_return_sequences)
            ]
        except (ValueError, NotImplementedError):
            # Config errors (n>1 on an API that can't do it) are programmer
            # mistakes, not transient failures.
            raise
        except Exception as e:
            self.logger.warning(
                "%s calling '%s': %s", type(e).__name__, self.endpoint, e
            )
            detail = f"{type(e).__name__}: {e}"
            return [
                ModelResponse(
                    prompt=prompt_text,
                    raw_response=detail,
                    failed=True,
                    served_provider=self.PROVIDER,
                )
                for _ in range(num_return_sequences)
            ]

        responses: list[ModelResponse] = []
        if self._uses_responses_api():
            responses.append(
                self._build_response(
                    speech=self._extract_responses_text(completion),
                    thinking=self._extract_thinking(completion),
                    prompt_text=prompt_text,
                    speech_structure=speech_structure,
                    choice=None,
                )
            )
        else:
            choices = list(getattr(completion, "choices", None) or [])
            if not choices:
                self.logger.warning("no choices returned by '%s'", self.endpoint)
                return [ModelResponse(failed=True) for _ in range(num_return_sequences)]
            for choice in choices:
                responses.append(
                    self._build_response(
                        speech=self._extract_message_text_from_choice(choice),
                        thinking=self._extract_thinking_from_choice(choice),
                        prompt_text=prompt_text,
                        speech_structure=speech_structure,
                        choice=choice,
                    )
                )

        # Pad if the backend returned fewer choices than requested so callers
        # always see exactly num_return_sequences entries.
        while len(responses) < num_return_sequences:
            responses.append(ModelResponse(prompt=prompt_text, failed=True))
        return responses[:num_return_sequences]

    # -- request -----------------------------------------------------------

    def call_openai(
        self,
        messages: list[dict[str, Any]],
        speech_structure: SpeechStructure,
        max_new_tokens: int,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> Any:
        response_format = kwargs.pop("response_format", None)
        json_schema = kwargs.pop("json_schema", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        if kwargs:
            self.logger.debug("ignoring unsupported call kwargs: %s", sorted(kwargs))

        if self._uses_responses_api():
            request: dict[str, Any] = {
                "model": self.endpoint,
                "input": messages,
                "max_output_tokens": max_new_tokens,
            }
            if self._is_reasoning_model():
                request["reasoning"] = {"effort": reasoning_effort or "medium"}
            else:
                if temperature is not None:
                    request["temperature"] = temperature
                if top_p is not None:
                    request["top_p"] = top_p
            if json_schema is not None and self.supports_json_schema():
                # Responses Structured Outputs use text.format, not the
                # Chat Completions response_format envelope.
                request["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": STRUCTURED_OUTPUT_SCHEMA_NAME,
                        "schema": json_schema,
                        "strict": True,
                    }
                }
            response = self.client.responses.create(**request)
            status = getattr(response, "status", None)
            generation_id = getattr(response, "id", None)
            if status == "incomplete":
                raise IncompleteResponseAPIError(
                    self.endpoint,
                    getattr(response, "incomplete_details", None),
                    generation_id,
                )
            refusal = self._extract_responses_refusal(response)
            if refusal is not None:
                raise ResponsesRefusalAPIError(self.endpoint, refusal, generation_id)
            if status != "completed":
                raise ResponsesStatusAPIError(
                    self.endpoint,
                    status,
                    getattr(response, "error", None),
                    generation_id,
                )
            return response

        wants_logprobs = speech_structure != SpeechStructure.OPEN_ENDED and self._supports_decision_logprobs()
        request = {
            "model": self.endpoint,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "logprobs": wants_logprobs,
            "top_logprobs": 5 if wants_logprobs else None,
        }
        if num_return_sequences > 1:
            request["n"] = num_return_sequences
        if temperature is not None:
            request["temperature"] = temperature
        if top_p is not None:
            request["top_p"] = top_p
        if (
            json_schema is not None
            and response_format is None
            and self.supports_json_schema()
        ):
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": STRUCTURED_OUTPUT_SCHEMA_NAME,
                    "schema": json_schema,
                    "strict": True,
                },
            }
        if response_format:
            request["response_format"] = response_format
        try:
            return self.client.chat.completions.create(**request)
        except openai.BadRequestError as err:
            if "max_tokens" not in str(err) or "max_completion_tokens" not in str(err):
                raise
            request.pop("max_tokens", None)
            request["max_completion_tokens"] = max_new_tokens
            return self.client.chat.completions.create(**request)

    @staticmethod
    def _retry_giveup(e: Exception) -> bool:
        # Cap hits, refusals, and 400s are deterministic for an unchanged
        # request. A failed/nonterminal Responses status can be transient and
        # consumes the normal provider retry budget.
        if isinstance(
            e,
            (IncompleteResponseAPIError, ResponsesRefusalAPIError, openai.BadRequestError),
        ):
            return True
        if isinstance(e, ResponsesStatusAPIError):
            if e.status in ("queued", "in_progress"):
                return False
            return not (
                e.status == "failed"
                and e.error_code in TRANSIENT_RESPONSES_ERROR_CODES
            )
        if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
            return False
        if isinstance(e, openai.RateLimitError):
            return False
        if isinstance(e, openai.APIStatusError):
            return not (500 <= e.status_code < 600)
        return True

    # -- response assembly -------------------------------------------------

    def _build_response(
        self,
        *,
        speech: str,
        thinking: Optional[str],
        prompt_text: str,
        speech_structure: SpeechStructure,
        choice: Any,
    ) -> ModelResponse:
        response = ModelResponse(
            speech=speech,
            raw_response=speech,
            prompt=prompt_text,
            thinking=thinking,
        )
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason in ("stop", "length"):
            response.stop_reason = finish_reason

        if speech_structure != SpeechStructure.DECISION or choice is None:
            return response
        if not self._supports_decision_logprobs():
            return response
        try:
            a_odds, b_odds = self._decision_odds(choice)
        except ValueError as err:
            self.logger.warning("no usable decision logprobs from '%s': %s", self.endpoint, err)
            return response
        response.decision = (
            DEFAULT_DEBATER_A_NAME
            if a_odds > b_odds
            else (DEFAULT_DEBATER_B_NAME if (b_odds > a_odds or random.random() > 0.5) else DEFAULT_DEBATER_A_NAME)
        )
        response.probabilistic_decision = {
            DEFAULT_DEBATER_A_NAME: a_odds,
            DEFAULT_DEBATER_B_NAME: b_odds,
        }
        return response

    @staticmethod
    def _decision_odds(choice: Any) -> tuple[float, float]:
        """Renormalized P(_A) / P(_B) from the top-logprobs at the position
        where the model emitted the debater-name suffix."""
        suffixes = ["_A", "_B"]
        logprob_data = getattr(choice, "logprobs", None)
        if not logprob_data or not getattr(logprob_data, "content", None):
            raise ValueError("no logprob content returned")
        for entry in logprob_data.content:
            if entry.token not in suffixes:
                continue
            scores = {suffix: 0.0 for suffix in suffixes}
            for option in entry.top_logprobs:
                if option.token in suffixes:
                    scores[option.token] = math.exp(float(option.logprob))
            total = sum(scores.values())
            if total == 0:
                raise ValueError("top logprob scores sum to zero")
            return scores["_A"] / total, scores["_B"] / total
        raise ValueError("decision suffix not found in logprobs")

    def _extract_message_text_from_choice(self, choice: Any) -> str:
        message = getattr(choice, "message", None)
        if message is None:
            self.logger.warning("unexpected chat choice payload; returning empty string")
            return ""
        return self._extract_text(getattr(message, "content", "")).strip()

    def _extract_thinking_from_choice(self, choice: Any) -> Optional[str]:
        """No-op by default; subclasses that surface reasoning override it."""
        return None

    def _extract_thinking(self, completion: Any) -> Optional[str]:
        return None

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Flatten the varied SDK content payloads (str, content-part lists,
        typed objects) into plain text."""
        if not content:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(OpenAIModel._extract_text(item) for item in content)
        if isinstance(content, dict):
            for key in ("text", "content", "value", "output_text"):
                text = OpenAIModel._extract_text(content.get(key))
                if text:
                    return text
            return ""
        for attr in ("text", "content", "value", "output_text"):
            text = OpenAIModel._extract_text(getattr(content, attr, None))
            if text:
                return text
        return ""

    def _extract_responses_text(self, response: Any) -> str:
        """Assistant text out of a Responses API result."""
        text = self._extract_text(getattr(response, "output_text", None)).strip()
        if text:
            return text
        chunks: list[str] = []
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) == "function_call":
                continue
            chunks.append(self._extract_text(getattr(item, "content", None) or item))
        text = "".join(chunks).strip()
        if not text:
            self.logger.warning("failed to extract text from '%s' response", self.endpoint)
        return text

    @staticmethod
    def _extract_responses_refusal(response: Any) -> Optional[str]:
        """Return explicit Responses refusal content, including SDK objects."""

        def field(value: Any, name: str) -> Any:
            return value.get(name) if isinstance(value, dict) else getattr(value, name, None)

        for item in getattr(response, "output", None) or []:
            parts = field(item, "content") or []
            # Be liberal about a provider returning the refusal as a top-level
            # output item even though the OpenAI schema nests it in a message.
            for part in [item, *parts]:
                if field(part, "type") != "refusal":
                    continue
                refusal = field(part, "refusal")
                return refusal if isinstance(refusal, str) else str(refusal or "refused")
        return None

    # -- endpoint capability probes ---------------------------------------

    @staticmethod
    def _endpoint_is_reasoning_model(endpoint: Optional[str]) -> bool:
        name = (endpoint or "").lower()
        return (name.startswith("o") and len(name) > 1 and name[1].isdigit()) or "gpt-5" in name

    @classmethod
    def _endpoint_uses_responses_api(cls, endpoint: Optional[str]) -> bool:
        if cls._endpoint_is_reasoning_model(endpoint):
            return True
        name = (endpoint or "").lower()
        return any(keyword in name for keyword in ("gpt-4o", "gpt-4.1"))

    def _is_reasoning_model(self) -> bool:
        return self._endpoint_is_reasoning_model(self.endpoint)

    def _uses_responses_api(self) -> bool:
        """o-series / gpt-4o / gpt-4.1 are served only over Responses, which
        takes max_output_tokens and exposes no logprobs."""
        return self._endpoint_uses_responses_api(self.endpoint)

    def _supports_decision_logprobs(self) -> bool:
        return not self._uses_responses_api()

    @classmethod
    def _endpoint_supports_json_schema(cls, endpoint: Optional[str]) -> bool:
        # Structured Outputs started at gpt-4o-2024-08-06 (and the mini
        # 2024-07-18 snapshot). Stable aliases and subsequent model families
        # support it; earlier snapshots and gpt-4-turbo do not.
        name = (endpoint or "").lower()

        def dated_snapshot(prefix: str, minimum: str) -> bool:
            snapshot = name.removeprefix(prefix)
            is_date = (
                len(snapshot) == 10
                and snapshot[:4].isdigit()
                and snapshot[4] == "-"
                and snapshot[5:7].isdigit()
                and snapshot[7] == "-"
                and snapshot[8:].isdigit()
            )
            return is_date and snapshot >= minimum

        if name == "gpt-4o-mini":
            return True
        if name.startswith("gpt-4o-mini-"):
            return dated_snapshot("gpt-4o-mini-", "2024-07-18")
        if name == "gpt-4o":
            return True
        if name.startswith("gpt-4o-"):
            return dated_snapshot("gpt-4o-", "2024-08-06")

        def family_or_dated_snapshot(family: str) -> bool:
            if name == family:
                return True
            if not name.startswith(f"{family}-"):
                return False
            snapshot = name.removeprefix(f"{family}-")
            return (
                len(snapshot) == 10
                and snapshot[:4].isdigit()
                and snapshot[4] == "-"
                and snapshot[5:7].isdigit()
                and snapshot[7] == "-"
                and snapshot[8:].isdigit()
            )

        if any(
            family_or_dated_snapshot(family)
            for family in ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano")
        ):
            return True
        if name.startswith("gpt-5"):
            return True
        # Explicit allowlist: several Responses-capable o-series variants do
        # not support Structured Outputs (notably o1-mini and
        # o3-deep-research).
        return any(
            family_or_dated_snapshot(family)
            for family in ("o1", "o1-pro", "o3", "o3-mini", "o3-pro", "o4-mini")
        )

    def supports_json_schema(self) -> bool:
        """Whether this adapter consumes DebateRound's ``json_schema`` kwarg.

        The direct OpenAI implementation handles supported Responses
        endpoints. Some OpenAI-compatible subclasses replace ``call_openai``
        and do not forward the schema. LocalModel is the exception: its
        factored request builder consumes the same kwarg for vLLM.
        """
        if callable(getattr(self, "build_request_kwargs", None)):
            # LocalModel's vLLM request builder grammar-constrains json_schema.
            return True
        return (
            type(self).call_openai is OpenAIModel.call_openai
            and self._endpoint_supports_json_schema(self.endpoint)
        )

    # -- misc --------------------------------------------------------------

    @classmethod
    def generate_llm_input_from_model_inputs(
        cls, input_list: list[ModelInput], extra_suffix: str = ""
    ) -> list[dict[str, str]]:
        """ModelInputs -> OpenAI messages. All system messages are merged into
        one leading system message."""
        messages = [
            {"role": RoleType.USER.api_name, "content": item}
            if isinstance(item, str)
            else {"role": item.role.api_name, "content": item.content}
            for item in input_list
        ]
        if extra_suffix:
            messages.append({"role": RoleType.ASSISTANT.api_name, "content": extra_suffix})

        system_role = RoleType.SYSTEM.api_name
        system_content = "\n".join(m["content"] for m in messages if m["role"] == system_role)
        rest = [m for m in messages if m["role"] != system_role]
        return ([{"role": system_role, "content": system_content}] if system_content else []) + rest

    def copy(self, is_debater: Optional[bool] = None, **kwargs: Any) -> "OpenAIModel":
        return type(self)(
            alias=kwargs.get("alias", self.alias),
            is_debater=is_debater if is_debater is not None else self.is_debater,
            endpoint=self.endpoint,
            reasoning_effort=self.reasoning_effort,
        )
