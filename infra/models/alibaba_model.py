"""Alibaba DashScope (Qwen) via its OpenAI-compatible endpoint.

Streams by default: ``reasoning_content`` only arrives on stream deltas, and
that is the only way to capture Qwen thinking into ModelResponse.thinking.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from types import SimpleNamespace
from typing import Any, Optional

import httpx
import openai

from infra.models.base import SpeechStructure
from infra.models.openai_model import IncompleteResponseAPIError, OpenAIModel

# Hard wall-clock cap for a whole streaming call. DashScope sends keepalive
# chunks that prevent httpx read timeouts from ever firing, so total elapsed
# time is the only usable limit. 600s fits hard problems with 32k thinking
# budgets while leaving room for retries inside a round-creation window.
_STREAM_WALL_CLOCK_TIMEOUT = 600

# Repetition-loop detector. qwen3.5-35b sometimes enters infinite-thinking
# loops where the same N-char block repeats forever (cycle length unknown).
# Every _REPEAT_CHECK_EVERY chunks, scan candidate cycle lengths over a rolling
# tail and check whether the last 3*c chars are three equal c-char blocks.
# Three blocks (not two) makes false positives on legit prose ~zero.
_REPEAT_TAIL_CHARS = 240
_REPEAT_MIN_CYCLE = 8
_REPEAT_MAX_CYCLE = 60
_REPEAT_CHECK_EVERY = 50
_REPEAT_WARMUP_CHARS = 200

# BadRequestError messages that mean retrying can never help.
_DETERMINISTIC_400_PATTERNS = ("input length", "context length", "token limit")


class _StreamWallClockTimeout(Exception):
    """Streaming call exceeded its wall-clock budget, or looped."""


def _detect_repeat_cycle(tail: str) -> Optional[int]:
    for c in range(_REPEAT_MIN_CYCLE, min(_REPEAT_MAX_CYCLE, len(tail) // 3) + 1):
        block = tail[-c:]
        if tail[-2 * c : -c] == block and tail[-3 * c : -2 * c] == block:
            return c
    return None


def _is_deterministic_bad_request(e: Exception) -> bool:
    if "BadRequestError" not in type(e).__name__:
        return False
    message = str(e).lower()
    return any(pattern in message for pattern in _DETERMINISTIC_400_PATTERNS)


class AlibabaModel(OpenAIModel):
    # Gate key; budgets in provider_gate (alibaba: max_tries=6, factor=5,
    # min_interval_s=0.1 — the global ~10 req/s DashScope rate limit lives
    # there and is acquired before every attempt).
    PROVIDER = "alibaba"

    BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL_ENDPOINT = "qwen-plus"

    # Per-family max_tokens caps advertised by DashScope intl. Order matters:
    # longer prefixes must come first.
    MAX_TOKENS_BY_FAMILY = {
        "qwen3.6-flash": 65536,
        "qwen3.5": 65536,
        "qwen3.6": 16384,
        "qwen3-32b": 16384,
    }
    MAX_TOKENS_DEFAULT = 8192

    # Commercial endpoints supporting explicit context cache (cache_control
    # blocks). Open-weight snapshots silently ignore the field and get no
    # implicit-cache billing either, so their payloads stay unchanged. Cache
    # reads bill at 10% of input price, writes at 125%.
    EXPLICIT_CACHE_ENDPOINT_PREFIXES = (
        "qwen3.7-max",
        "qwen3.6-max",
        "qwen3-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "qwen-plus",
        "qwen3.6-flash",
        "qwen3.5-flash",
        "qwen-flash",
        "qwen3-coder-plus",
        "qwen3-coder-flash",
        "qwen3-vl-plus",
        "qwen3-vl-flash",
        "deepseek-v3.2",
    )

    def __init__(
        self,
        alias: str,
        is_debater: bool = True,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ):
        from infra.models.base import Model

        Model.__init__(self, alias=alias, is_debater=is_debater)

        resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not resolved_api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")

        self.client = openai.OpenAI(
            base_url=base_url or self.BASE_URL,
            api_key=resolved_api_key,
            # read = max gap between stream chunks, not total call length.
            timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
        )
        self.endpoint = endpoint or self.DEFAULT_MODEL_ENDPOINT
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.logger = logging.getLogger(__name__)
        self._cache_marking_logged = False

    def _is_reasoning_model(self) -> bool:
        return False

    def _uses_responses_api(self) -> bool:
        return False

    def _supports_decision_logprobs(self) -> bool:
        return False

    def supports_server_side_sampling(self) -> bool:
        """DashScope rejects n>1 whenever thinking is on."""
        return not self.enable_thinking

    @property
    def _max_tokens(self) -> int:
        for prefix, limit in self.MAX_TOKENS_BY_FAMILY.items():
            if self.endpoint.startswith(prefix):
                return limit
        return self.MAX_TOKENS_DEFAULT

    @staticmethod
    def _retry_giveup(e: Exception) -> bool:
        return (
            isinstance(e, (IncompleteResponseAPIError, _StreamWallClockTimeout))
            or _is_deterministic_bad_request(e)
            or not any(
                name in type(e).__name__
                for name in (
                    "APIError",
                    "APIConnectionError",
                    "RateLimitError",
                    "BadRequestError",
                    "InternalServerError",
                    "APITimeoutError",
                    "Timeout",
                )
            )
        )

    # -- explicit context cache -------------------------------------------

    def _with_cache_control(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Mark the first system message as an explicit-cache breakpoint.

        The debate system message carries the heavyweight stable prefix (story
        + question + positions) re-sent verbatim across turns, candidates, and
        rollouts. Everything through the marked block is cached server-side
        (~5 min TTL, min 1024 tokens). Returns new objects; no mutation.
        """
        if not self.endpoint.startswith(self.EXPLICIT_CACHE_ENDPOINT_PREFIXES):
            return messages
        out: list[dict[str, Any]] = []
        marked = False
        for message in messages:
            content = message.get("content")
            # Claim the FIRST system message even if its content is already
            # block-shaped; falling through would mark a later one instead.
            if not marked and message.get("role") == "system":
                marked = True
                if isinstance(content, str):
                    message = {
                        **message,
                        "content": [
                            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                        ],
                    }
                    if not self._cache_marking_logged:
                        self._cache_marking_logged = True
                        self.logger.info(
                            "[%s] explicit context-cache marking active (%d chars)",
                            self.endpoint,
                            len(content),
                        )
            out.append(message)
        return out

    @staticmethod
    def _prompt_chars(messages: list[dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                total += sum(len(block.get("text") or "") for block in content)
        return total

    # -- request -----------------------------------------------------------

    def call_openai(
        self,
        messages: list[dict[str, Any]],
        speech_structure: SpeechStructure,
        max_new_tokens: int,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> Any:
        messages = self._with_cache_control(messages)
        if num_return_sequences > 1:
            if self.enable_thinking:
                raise ValueError(
                    "DashScope rejects num_return_sequences>1 when enable_thinking=True. "
                    "Disable thinking on this model, or duplicate the input and use n=1."
                )
            return self._call_batch_n(
                messages=messages,
                speech_structure=speech_structure,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                **kwargs,
            )

        effective_max = min(max_new_tokens, self._max_tokens)
        request: dict[str, Any] = {
            "model": self.endpoint,
            "messages": messages,
            "max_tokens": effective_max,
            "logprobs": speech_structure != SpeechStructure.OPEN_ENDED,
            "top_logprobs": 5 if speech_structure != SpeechStructure.OPEN_ENDED else None,
            "stream": True,
            # include_usage: the final chunk carries usage, which is how we
            # confirm cached_tokens > 0 (i.e. that the cache is billing).
            "stream_options": {"include_usage": True},
            "extra_body": {"enable_thinking": self.enable_thinking},
        }
        for name in ("temperature", "top_p"):
            value = kwargs.get(name)
            if value is not None:
                request[name] = value

        call_id = f"{self.endpoint}:{id(messages) % 10000}"
        self.logger.info(
            "[%s] START streaming: max_tokens=%d, prompt_chars=%d",
            call_id,
            effective_max,
            self._prompt_chars(messages),
        )
        call_start = time.monotonic()
        watchdog = None
        chunk_count = 0
        try:
            stream = self.client.chat.completions.create(**request)

            # The in-loop elapsed check only catches stalls BETWEEN chunks; a
            # stream that yields zero chunks would block next(stream) forever.
            # Closing the underlying socket from a timer thread makes the
            # blocking read return.
            fired = {"hit": False}

            def _close_after_timeout() -> None:
                fired["hit"] = True
                self.logger.error("[%s] watchdog firing after %ds", call_id, _STREAM_WALL_CLOCK_TIMEOUT)
                try:
                    stream.close()
                except Exception as e:
                    self.logger.error("[%s] watchdog stream.close() raised: %r", call_id, e)

            watchdog = threading.Timer(_STREAM_WALL_CLOCK_TIMEOUT, _close_after_timeout)
            watchdog.daemon = True
            watchdog.start()

            content: list[str] = []
            reasoning: list[str] = []
            logprobs_data = None
            finish_reason = None
            final_usage = None
            reasoning_tail = ""
            reasoning_chars = 0

            for chunk in stream:
                if fired["hit"]:
                    break
                chunk_count += 1
                elapsed = time.monotonic() - call_start
                if elapsed > _STREAM_WALL_CLOCK_TIMEOUT:
                    stream.close()
                    raise _StreamWallClockTimeout(
                        f"stream exceeded {_STREAM_WALL_CLOCK_TIMEOUT}s after {chunk_count} chunks"
                    )
                if getattr(chunk, "usage", None):
                    final_usage = chunk.usage
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if getattr(delta, "reasoning_content", None):
                        reasoning.append(delta.reasoning_content)
                        reasoning_tail = (reasoning_tail + delta.reasoning_content)[-_REPEAT_TAIL_CHARS:]
                        reasoning_chars += len(delta.reasoning_content)
                    if getattr(delta, "content", None):
                        content.append(delta.content)
                    if getattr(choice, "logprobs", None):
                        logprobs_data = choice.logprobs
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason
                if reasoning_chars >= _REPEAT_WARMUP_CHARS and chunk_count % _REPEAT_CHECK_EVERY == 0:
                    period = _detect_repeat_cycle(reasoning_tail)
                    if period is not None:
                        self.logger.error(
                            "[%s] repetition loop: cycle=%d chars, thinking=%d chars, %d chunks",
                            call_id,
                            period,
                            reasoning_chars,
                            chunk_count,
                        )
                        stream.close()
                        raise _StreamWallClockTimeout(
                            f"repetition loop (cycle={period}) after {chunk_count} chunks"
                        )

            watchdog.cancel()
            if fired["hit"]:
                raise _StreamWallClockTimeout(
                    f"stream exceeded {_STREAM_WALL_CLOCK_TIMEOUT}s (watchdog); "
                    f"{chunk_count} chunks received"
                )

            content_chars = sum(len(s) for s in content)
            details = getattr(final_usage, "prompt_tokens_details", None)
            self.logger.info(
                "[%s] DONE in %.1fs: %d chunks, thinking=%d chars, content=%d chars, "
                "prompt_tokens=%s, cached_tokens=%s",
                call_id,
                time.monotonic() - call_start,
                chunk_count,
                reasoning_chars,
                content_chars,
                getattr(final_usage, "prompt_tokens", None),
                getattr(details, "cached_tokens", None) if details else None,
            )
            if content_chars == 0:
                self.logger.warning(
                    "[%s] empty content (thinking=%d chars); possible token budget exhaustion "
                    "at max_tokens=%d",
                    call_id,
                    reasoning_chars,
                    effective_max,
                )

            message = SimpleNamespace(
                content="".join(content),
                thinking="".join(reasoning) or None,
            )
            choice = SimpleNamespace(message=message, logprobs=logprobs_data, finish_reason=finish_reason)
            return SimpleNamespace(choices=[choice])

        except Exception as err:
            if watchdog is not None:
                watchdog.cancel()
            self.logger.warning(
                "[%s] FAILED after %.1fs (%d chunks): %s: %s",
                call_id,
                time.monotonic() - call_start,
                chunk_count,
                type(err).__name__,
                str(err)[:200],
            )
            raise

    def _call_batch_n(
        self,
        messages: list[dict[str, Any]],
        speech_structure: SpeechStructure,
        max_new_tokens: int,
        num_return_sequences: int,
        **kwargs,
    ) -> Any:
        """Non-streaming n>1. DashScope caps n at 4 and rejects the request
        outright when thinking is enabled; the caller enforces both."""
        effective_max = min(max_new_tokens, self._max_tokens)
        request: dict[str, Any] = {
            "model": self.endpoint,
            "messages": messages,
            "max_tokens": effective_max,
            "n": num_return_sequences,
            "logprobs": speech_structure != SpeechStructure.OPEN_ENDED,
            "top_logprobs": 5 if speech_structure != SpeechStructure.OPEN_ENDED else None,
            "stream": False,
            "extra_body": {"enable_thinking": False},
        }
        for name in ("temperature", "top_p"):
            value = kwargs.get(name)
            if value is not None:
                request[name] = value

        call_id = f"{self.endpoint}:{id(messages) % 10000}"
        t0 = time.monotonic()
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as err:
            self.logger.warning(
                "[%s] FAILED after %.1fs (batch-n=%d): %s: %s",
                call_id,
                time.monotonic() - t0,
                num_return_sequences,
                type(err).__name__,
                str(err)[:200],
            )
            raise
        self.logger.info(
            "[%s] DONE in %.1fs: batch-n=%d returned %d choices",
            call_id,
            time.monotonic() - t0,
            num_return_sequences,
            len(getattr(response, "choices", None) or []),
        )
        return response

    # -- response ----------------------------------------------------------

    def _extract_thinking_from_choice(self, choice: Any) -> Optional[str]:
        """Handles both the SimpleNamespace built by the streaming path
        (message.thinking) and a real non-streaming DashScope choice
        (message.reasoning_content)."""
        message = getattr(choice, "message", None)
        return getattr(message, "thinking", None) or getattr(message, "reasoning_content", None) or None

    def _extract_thinking(self, completion: Any) -> Optional[str]:
        choices = getattr(completion, "choices", None) or []
        return self._extract_thinking_from_choice(choices[0]) if choices else None

    def copy(self, is_debater: Optional[bool] = None, **kwargs) -> "AlibabaModel":
        return AlibabaModel(
            alias=kwargs.get("alias", self.alias),
            is_debater=is_debater if is_debater is not None else self.is_debater,
            endpoint=self.endpoint,
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
        )
