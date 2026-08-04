"""OpenRouterModel request-shape tests (no network).

Every call goes through a fake ``client.chat.completions.create`` that
captures the request body (the approach tests/test_mb_solo_monitor.py uses).
Covers the enable_thinking wiring: OpenRouter's thinking toggle rides in
``extra_body["reasoning"]["enabled"]`` and must share ONE reasoning dict with
``reasoning_effort`` ("effort"), survive the factory's OPENROUTER branch, and
survive ``copy()`` (judges are produced via copy()).
"""

from __future__ import annotations

from types import SimpleNamespace

from infra.models.base import ModelSettings, SpeechStructure
from infra.models.factory import instantiate_model
from infra.models.openrouter_model import OpenRouterModel

SLUG = "qwen/qwen3.5-9b"


def _model(**kwargs) -> tuple[OpenRouterModel, list[dict]]:
    model = OpenRouterModel(alias="judge", is_debater=False, endpoint=SLUG, api_key="test-key", **kwargs)
    requests: list[dict] = []

    def create(**request):
        requests.append(request)
        return SimpleNamespace(
            id="gen-1",
            provider="parasail",
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )

    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return model, requests


def _call(model: OpenRouterModel) -> None:
    model.call_openai([{"role": "user", "content": "hi"}], SpeechStructure.OPEN_ENDED, max_new_tokens=16)


def test_enable_thinking_false_sent_as_reasoning_enabled():
    model, requests = _model(enable_thinking=False)
    _call(model)
    assert requests[0]["extra_body"]["reasoning"] == {"enabled": False}


def test_enable_thinking_none_sends_no_reasoning_block():
    model, requests = _model()
    _call(model)
    assert "extra_body" not in requests[0]


def test_effort_and_enable_thinking_share_one_reasoning_dict():
    model, requests = _model(reasoning_effort="low", enable_thinking=False)
    _call(model)
    assert requests[0]["extra_body"]["reasoning"] == {"effort": "low", "enabled": False}


def test_factory_passes_enable_thinking(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    settings = ModelSettings(
        model_type="openrouter", alias="judge", model_file_path=SLUG, enable_thinking=False
    )
    model = instantiate_model(settings, is_debater=False, binding="eval")
    assert isinstance(model, OpenRouterModel)
    assert model.enable_thinking is False
    # Unset stays None (provider default) rather than degrading to False.
    settings_default = ModelSettings(model_type="openrouter", alias="judge", model_file_path=SLUG)
    assert instantiate_model(settings_default, is_debater=False, binding="eval").enable_thinking is None


def test_copy_preserves_enable_thinking():
    model = OpenRouterModel(alias="judge", endpoint=SLUG, api_key="test-key", enable_thinking=False)
    clone = model.copy(is_debater=False)
    assert clone.enable_thinking is False
