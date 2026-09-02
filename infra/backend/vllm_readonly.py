"""Read-only vLLM completions backend for exact token-prefix continuation.

This backend exists for frozen local policies that still need ``Policy``'s
two-phase token-budget enforcement.  It can sample, but every training method
fails explicitly: callers cannot accidentally optimize the frozen service.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from typing import Any

from infra.backend.base import (
    Backend,
    Datum,
    LossSpec,
    OptimParams,
    Sample,
    SamplingParams,
)


class VLLMCompletionsBackend(Backend):
    """Inference-only backend over vLLM's raw ``/v1/completions`` API.

    ``Policy`` renders the chat template locally, then its budget-forced
    sampler calls this backend once for the think phase and once for the
    visible phase.  Passing token IDs rather than text preserves the exact
    extended prefix across phases.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        tokenizer_path: str,
        workers: int = 16,
        tokenizer: Any = None,
        post_fn: Any = None,
        retry_attempts: int = 4,
        sleep_fn: Any = time.sleep,
    ):
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer = tokenizer
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.workers = workers
        self._post_override = post_fn
        self.retry_attempts = retry_attempts
        self._sleep = sleep_fn

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.retry_attempts <= 0:
            raise ValueError("retry_attempts must be positive")
        endpoint = f"{self.base_url}/completions"
        for attempt in range(self.retry_attempts):
            try:
                if self._post_override is not None:
                    return self._post_override(body)
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=7200) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:600]
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt + 1 == self.retry_attempts:
                    suffix = " after retries exhausted" if retryable else ""
                    raise RuntimeError(
                        f"HTTP {exc.code} from {endpoint}{suffix}: {detail}"
                    ) from None
            except urllib.error.URLError as exc:
                if attempt + 1 == self.retry_attempts:
                    raise RuntimeError(
                        f"transport retries exhausted for {endpoint}: {exc.reason}"
                    ) from None
            self._sleep(min(8.0, 2.0**attempt))
        raise AssertionError("unreachable retry loop")

    def _sample_one(
        self, prompt: list[int], params: SamplingParams, n: int
    ) -> list[Sample]:
        if params.max_tokens is None:
            raise ValueError("vLLM completions sampling requires max_tokens")
        body: dict[str, Any] = {
            "model": self.model,
            # Exact token-prefix continuation is the point of this backend.
            # Never decode/re-encode a prefix: tokenizers are not injective.
            "prompt": prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "n": n,
            "return_token_ids": True,
            # The budget-forced sampler distinguishes a naturally sampled
            # close from a cap hit by inspecting the returned token suffix.
            "include_stop_str_in_output": True,
        }
        if params.stop:
            body["stop"] = params.stop
        if params.min_tokens:
            body["min_tokens"] = int(params.min_tokens)
        response = self._post(body)
        choices = sorted(
            response.get("choices") or [], key=lambda choice: choice.get("index", 0)
        )
        if len(choices) != n:
            raise RuntimeError(f"vLLM returned {len(choices)} choices, expected {n}")
        out: list[Sample] = []
        for choice in choices:
            prompt_tokens = choice.get("prompt_token_ids")
            if prompt_tokens != prompt:
                raise RuntimeError(
                    "vLLM returned prompt_token_ids that differ from the exact requested prefix"
                )
            tokens = choice.get("token_ids")
            if not isinstance(tokens, list) or not all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in tokens
            ):
                raise RuntimeError("vLLM choice is missing valid token_ids")
            tokens = list(tokens)
            text = choice.get("text")
            if not isinstance(text, str):
                text = self.tokenizer.decode(tokens)
            out.append(
                Sample(
                    tokens=tokens,
                    # No optimizer consumes these; aligned placeholders keep
                    # the Sample fidelity contract intact.
                    logprobs=[0.0] * len(tokens),
                    text=text,
                    stop_reason=(
                        "length"
                        if choice.get("finish_reason") == "length"
                        else "stop"
                    ),
                )
            )
        return out

    def sample(
        self, prompts: list[list[int]], params: SamplingParams, n: int = 1
    ) -> list[list[Sample]]:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.workers
        ) as executor:
            futures = [
                executor.submit(self._sample_one, prompt, params, n)
                for prompt in prompts
            ]
            return [future.result() for future in futures]

    def sync_sampler(self) -> None:
        return None

    def forward(self, data: list[Datum]) -> list[list[float]]:
        raise NotImplementedError("read-only vLLM backend cannot run training forward")

    def forward_backward(
        self, data: list[Datum], loss: LossSpec
    ) -> dict[str, float]:
        raise NotImplementedError("read-only vLLM backend cannot train")

    def optim_step(self, params: OptimParams) -> dict[str, float]:
        raise NotImplementedError("read-only vLLM backend cannot optimize")

    def save(self, name: str) -> str:
        raise NotImplementedError("read-only vLLM backend has no checkpoint")

    def load(self, path: str) -> None:
        raise NotImplementedError("read-only vLLM backend is fixed at construction")
