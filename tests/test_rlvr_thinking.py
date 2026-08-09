"""RLVR's enable_thinking toggle: it must reach the Policy, and it must refuse
to be a no-op.

A chat template that never references the kwarg swallows it silently, so the
launch-time probe is the only thing standing between a config that says
"no-think" and an arm that trains with thinking on.
"""

import pytest

from infra.run_rlvr import _check_thinking_toggle
from infra.run_rlvr import validate_experiment as validate_rlvr


class HybridTokenizer:
    """Qwen3.x-shaped: False closes the think block immediately, True opens it."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kw):
        return [1, 2, 3] + ([4, 5] if kw.get("enable_thinking", True) else [4, 6, 5])


class IgnoringTokenizer:
    """OLMo-shaped: no thinking mode, template ignores the kwarg."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kw):
        return [1, 2, 3]


class BatchEncodingTokenizer(HybridTokenizer):
    """Some wrappers return a BatchEncoding of nested id lists, not a flat list."""

    def apply_chat_template(self, messages, **kw):
        return {"input_ids": [super().apply_chat_template(messages, **kw)]}


@pytest.mark.parametrize("value", [True, False])
def test_hybrid_template_passes(value):
    _check_thinking_toggle(HybridTokenizer(), value)
    _check_thinking_toggle(BatchEncodingTokenizer(), value)


@pytest.mark.parametrize("value", [True, False])
def test_template_ignoring_the_kwarg_raises(value):
    with pytest.raises(ValueError) as exc:
        _check_thinking_toggle(IgnoringTokenizer(), value)
    assert "silently dropped" in str(exc.value)
    assert f"enable_thinking: {str(value).lower()}" in str(exc.value)


def test_enable_thinking_is_an_accepted_experiment_key():
    validate_rlvr(
        {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "enable_thinking": False,
            "max_completion_tokens": 4096,
            "dataset": {"type": "math"},
            "training": {"steps": 2, "lr": 1e-5},
        }
    )


class _AlwaysThinkTokenizer:
    """Olmo Think line: no toggle exists, the generation prompt always opens
    <think>. Identical renders + <think> present must pass when the config
    asked for thinking."""

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=True, **kw):
        text = "<|user|>probe<|assistant|><think>"
        return [1, 2, 3] if tokenize else text


class _NoThinkTokenizer:
    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=True, **kw):
        return [1, 2, 3] if tokenize else "<|user|>probe<|assistant|>"


def test_always_thinking_template_passes_when_thinking_requested():
    from infra.run_rlvr import _check_thinking_toggle

    _check_thinking_toggle(_AlwaysThinkTokenizer(), True)  # no raise


def test_toggle_less_non_think_template_still_rejected():
    import pytest

    from infra.run_rlvr import _check_thinking_toggle

    with pytest.raises(ValueError, match="renders identically"):
        _check_thinking_toggle(_NoThinkTokenizer(), True)
