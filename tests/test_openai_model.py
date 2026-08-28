"""OpenAIModel request-shape tests (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from infra.models.base import ModelInput, RoleType, SpeechStructure
from infra.models.local_model import LocalModel
from infra.models.openai_model import (
    IncompleteResponseAPIError,
    OpenAIModel,
    ResponsesStatusAPIError,
)
from infra.models.provider_gate import get_gate


VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["Debater_A", "Debater_B", "Tie"]},
        "confidence": {"type": "number"},
    },
    "required": ["winner", "confidence"],
    "additionalProperties": False,
}


@pytest.fixture
def openai_three_try_gate():
    gate = get_gate()
    original = gate.entry("openai").config
    gate.update("openai", max_tries=3, backoff_factor=0, backoff_max_value=0)
    try:
        yield
    finally:
        gate.configure("openai", original)


def _responses_model(*, response, reasoning_effort: str = "low") -> tuple[OpenAIModel, list[dict]]:
    model = OpenAIModel(
        alias="judge",
        is_debater=False,
        endpoint="gpt-5.6-luna",
        api_key="test-key",
        reasoning_effort=reasoning_effort,
    )
    requests: list[dict] = []

    def create(**request):
        requests.append(request)
        return response

    model.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return model, requests


def _transient_sdk_error(kind: str) -> Exception:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    if kind == "timeout":
        return openai.APITimeoutError(request)
    if kind == "connection":
        return openai.APIConnectionError(message="connection lost", request=request)
    status = 429 if kind == "rate_limit" else 500
    response = httpx.Response(
        status,
        request=request,
        json={"error": {"message": f"{kind} failure"}},
    )
    error_type = openai.RateLimitError if kind == "rate_limit" else openai.InternalServerError
    return error_type(f"{kind} failure", response=response, body=response.json())


@pytest.mark.parametrize("max_new_tokens", [512, 1536])
def test_responses_json_schema_uses_text_format_and_low_reasoning(max_new_tokens):
    model, requests = _responses_model(
        response=SimpleNamespace(status="completed", output_text='{"winner":"Debater_A","confidence":1}')
    )

    result = model.predict_single_input(
        [ModelInput(role=RoleType.USER, content="Judge this debate")],
        max_new_tokens=max_new_tokens,
        json_schema=VERDICT_SCHEMA,
    )

    assert result.failed is False
    assert requests == [
        {
            "model": "gpt-5.6-luna",
            "input": [{"role": "user", "content": "Judge this debate"}],
            "max_output_tokens": max_new_tokens,
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "verdict",
                    "schema": VERDICT_SCHEMA,
                    "strict": True,
                }
            },
        }
    ]


def test_chat_completions_preserves_explicit_response_format():
    model = OpenAIModel(alias="judge", endpoint="gpt-4-turbo", api_key="test-key")
    requests: list[dict] = []
    completion = SimpleNamespace(choices=[])

    def create(**request):
        requests.append(request)
        return completion

    model.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    response_format = {"type": "json_object"}

    assert (
        model.call_openai(
            [{"role": "user", "content": "hi"}],
            SpeechStructure.OPEN_ENDED,
            max_new_tokens=16,
            response_format=response_format,
            json_schema=VERDICT_SCHEMA,
        )
        is completion
    )
    assert requests[0]["response_format"] is response_format


def test_old_chat_endpoint_does_not_send_unsupported_json_schema():
    model = OpenAIModel(alias="judge", endpoint="gpt-4-turbo", api_key="test-key")
    requests: list[dict] = []

    def create(**request):
        requests.append(request)
        return SimpleNamespace(choices=[])

    model.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    model.call_openai(
        [{"role": "user", "content": "hi"}],
        SpeechStructure.OPEN_ENDED,
        max_new_tokens=16,
        json_schema=VERDICT_SCHEMA,
    )

    assert "response_format" not in requests[0]


@pytest.mark.parametrize(
    "endpoint",
    [
        "gpt-4o-2024-05-13",
        "o1-mini",
        "o1-mini-2024-09-12",
        "o3-deep-research",
        "o3-deep-research-2025-06-26",
    ],
)
def test_unsupported_responses_endpoint_does_not_send_json_schema(endpoint):
    model = OpenAIModel(
        alias="judge", endpoint=endpoint, api_key="test-key"
    )
    requests: list[dict] = []

    def create(**request):
        requests.append(request)
        return SimpleNamespace(status="completed", output_text="plain text")

    model.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = model.predict_single_input(
        [ModelInput(role=RoleType.USER, content="hi")],
        max_new_tokens=16,
        json_schema=VERDICT_SCHEMA,
    )

    assert result.failed is False
    assert "text" not in requests[0]


@pytest.mark.parametrize(
    "endpoint",
    ["o1-pro", "o1-pro-2025-03-19", "o3-pro", "o3-pro-2025-06-10"],
)
def test_supported_pro_responses_endpoint_sends_json_schema(endpoint):
    model = OpenAIModel(alias="judge", endpoint=endpoint, api_key="test-key")
    requests: list[dict] = []

    def create(**request):
        requests.append(request)
        return SimpleNamespace(status="completed", output_text="{}")

    model.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = model.predict_single_input(
        [ModelInput(role=RoleType.USER, content="hi")],
        max_new_tokens=16,
        json_schema=VERDICT_SCHEMA,
    )

    assert result.failed is False
    assert requests[0]["text"]["format"]["schema"] == VERDICT_SCHEMA


def test_call_openai_incomplete_raises_without_retrying():
    incomplete = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    model, requests = _responses_model(response=incomplete)

    with pytest.raises(IncompleteResponseAPIError, match="max_output_tokens"):
        model.call_openai(
            [{"role": "user", "content": "Judge this debate"}],
            SpeechStructure.OPEN_ENDED,
            max_new_tokens=512,
            json_schema=VERDICT_SCHEMA,
        )

    assert len(requests) == 1


def test_batched_predict_contains_incomplete_to_failed_item():
    model = OpenAIModel(
        alias="judge",
        is_debater=False,
        endpoint="gpt-5.6-luna",
        api_key="test-key",
        reasoning_effort="low",
    )
    attempts: list[str] = []

    def create(**request):
        content = request["input"][0]["content"]
        attempts.append(content)
        if content == "hit cap":
            return SimpleNamespace(
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            )
        return SimpleNamespace(
            status="completed",
            output_text='{"winner":"Debater_B","confidence":0.9}',
        )

    model.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    results = model.predict(
        [
            [ModelInput(role=RoleType.USER, content="hit cap")],
            [ModelInput(role=RoleType.USER, content="complete")],
        ],
        max_new_tokens=512,
        json_schema=VERDICT_SCHEMA,
    )

    assert len(results) == 2
    assert results[0].failed is True
    assert results[0].stop_reason == "length"
    assert json.loads(results[0].raw_response or "")["incomplete_details"] == {
        "reason": "max_output_tokens"
    }
    assert results[1].failed is False
    assert results[1].speech == '{"winner":"Debater_B","confidence":0.9}'
    assert sorted(attempts) == ["complete", "hit cap"]


def test_content_filter_incomplete_is_terminal_but_not_length(openai_three_try_gate):
    model, requests = _responses_model(
        response=SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="content_filter"),
        )
    )

    result = model.predict_single_input(
        [ModelInput(role=RoleType.USER, content="Judge this debate")],
        max_new_tokens=512,
        json_schema=VERDICT_SCHEMA,
    )

    assert len(requests) == 1
    assert result.failed is True
    assert result.stop_reason is None
    assert json.loads(result.raw_response or "")["incomplete_details"] == {
        "reason": "content_filter"
    }


@pytest.mark.parametrize(
    ("status", "error_code", "expected_attempts"),
    [
        ("failed", "server_error", 3),
        ("failed", "rate_limit_exceeded", 3),
        ("failed", "vector_store_timeout", 3),
        ("queued", None, 3),
        ("in_progress", None, 3),
        ("cancelled", None, 1),
    ],
)
def test_responses_noncompleted_status_retries_then_fails(
    status, error_code, expected_attempts, openai_three_try_gate
):
    error = (
        SimpleNamespace(code=error_code, message="temporary failure")
        if error_code is not None
        else None
    )
    model, requests = _responses_model(
        response=SimpleNamespace(id=f"resp_{status}", status=status, error=error, output=[])
    )

    result = model.predict_single_input(
        [ModelInput(role=RoleType.USER, content="Judge this debate")],
        max_new_tokens=512,
        json_schema=VERDICT_SCHEMA,
    )

    assert len(requests) == expected_attempts
    assert result.failed is True
    assert result.stop_reason is None
    assert result.served_provider == "openai"
    assert result.generation_id == f"resp_{status}"
    expected_raw = {
        "endpoint": "gpt-5.6-luna",
        "error": (
            {"code": error_code, "message": "temporary failure"}
            if error_code is not None
            else None
        ),
        "status": status,
        "type": "responses_status",
    }
    assert json.loads(result.raw_response or "") == expected_raw


@pytest.mark.parametrize(
    ("status", "error_code"),
    [
        ("failed", "server_error"),
        ("failed", "rate_limit_exceeded"),
        ("failed", "vector_store_timeout"),
        ("queued", None),
        ("in_progress", None),
    ],
)
def test_transient_responses_status_can_recover_on_retry(
    status, error_code, openai_three_try_gate
):
    model = OpenAIModel(
        alias="judge",
        is_debater=False,
        endpoint="gpt-5.6-luna",
        api_key="test-key",
        reasoning_effort="low",
    )
    responses = [
        SimpleNamespace(
            status=status,
            error=(
                SimpleNamespace(code=error_code, message="temporary failure")
                if error_code is not None
                else None
            ),
            output=[],
        ),
        SimpleNamespace(status="completed", output_text='{"winner":"Tie","confidence":0.5}'),
    ]

    def create(**_request):
        return responses.pop(0)

    model.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = model.predict_single_input(
        [ModelInput(role=RoleType.USER, content="Judge this debate")],
        max_new_tokens=512,
        json_schema=VERDICT_SCHEMA,
    )

    assert result.failed is False
    assert result.speech == '{"winner":"Tie","confidence":0.5}'
    assert responses == []


@pytest.mark.parametrize(
    "error_code",
    [
        "invalid_prompt",
        "data_residency_mismatch",
        "bio_policy",
        "invalid_image",
        "image_content_policy_violation",
        "future_unknown_code",
        None,
    ],
)
def test_deterministic_or_unknown_failed_status_is_terminal(
    error_code, openai_three_try_gate
):
    error = (
        SimpleNamespace(code=error_code, message="request cannot be processed")
        if error_code is not None
        else None
    )
    model, requests = _responses_model(
        response=SimpleNamespace(
            id="resp_failed", status="failed", error=error, output=[]
        )
    )

    result = model.predict_single_input(
        [ModelInput(role=RoleType.USER, content="Judge this debate")],
        max_new_tokens=512,
        json_schema=VERDICT_SCHEMA,
    )

    assert len(requests) == 1
    assert result.failed is True
    raw = json.loads(result.raw_response or "")
    assert raw["status"] == "failed"
    assert raw["error"] == (
        {"code": error_code, "message": "request cannot be processed"}
        if error_code is not None
        else None
    )


def test_responses_status_error_retains_status_and_error_code_attrs():
    model, _requests = _responses_model(
        response=SimpleNamespace(
            id="resp_invalid",
            status="failed",
            error=SimpleNamespace(code="invalid_prompt", message="invalid prompt"),
            output=[],
        )
    )

    with pytest.raises(ResponsesStatusAPIError) as raised:
        model.call_openai(
            [{"role": "user", "content": "Judge this debate"}],
            SpeechStructure.OPEN_ENDED,
            max_new_tokens=512,
        )

    assert raised.value.status == "failed"
    assert raised.value.error_code == "invalid_prompt"


def test_responses_refusal_is_terminal_failed_result(openai_three_try_gate):
    refusal = SimpleNamespace(type="refusal", refusal="I cannot judge this request.")
    response = SimpleNamespace(
        status="completed",
        output_text="",
        output=[SimpleNamespace(type="message", content=[refusal])],
    )
    model, requests = _responses_model(response=response)

    result = model.predict_single_input(
        [ModelInput(role=RoleType.USER, content="Judge this debate")],
        max_new_tokens=512,
        json_schema=VERDICT_SCHEMA,
    )

    assert len(requests) == 1
    assert result.failed is True
    assert json.loads(result.raw_response or "") == {
        "endpoint": "gpt-5.6-luna",
        "refusal": "I cannot judge this request.",
        "status": "completed",
        "type": "responses_refusal",
    }


def test_bad_request_stops_after_max_tokens_fallback(openai_three_try_gate):
    model = OpenAIModel(alias="judge", endpoint="gpt-4-turbo", api_key="test-key")
    requests: list[dict] = []

    def bad_request(message: str) -> openai.BadRequestError:
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(400, request=request, json={"error": {"message": message}})
        return openai.BadRequestError(message, response=response, body=response.json())

    def create(**request):
        requests.append(request)
        if len(requests) == 1:
            raise bad_request("max_tokens is unsupported; use max_completion_tokens")
        raise bad_request("schema is invalid")

    model.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result = model.predict_single_input(
        [ModelInput(role=RoleType.USER, content="Judge this debate")],
        max_new_tokens=512,
    )

    assert result.failed is True
    assert len(requests) == 2
    assert "max_tokens" in requests[0]
    assert "max_completion_tokens" in requests[1]
    assert "BadRequestError" in (result.raw_response or "")


@pytest.mark.parametrize(
    "kind", ["timeout", "connection", "rate_limit", "internal_server"]
)
def test_transient_sdk_errors_retry_and_recover(kind, openai_three_try_gate):
    model = OpenAIModel(
        alias="judge", endpoint="gpt-5.6-luna", api_key="test-key"
    )
    calls = 0

    def create(**_request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _transient_sdk_error(kind)
        return SimpleNamespace(status="completed", output_text="recovered")

    model.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = model.predict_single_input(
        [ModelInput(role=RoleType.USER, content="hi")], max_new_tokens=16
    )

    assert calls == 2
    assert result.failed is False
    assert result.speech == "recovered"


@pytest.mark.parametrize(
    "kind", ["timeout", "connection", "rate_limit", "internal_server"]
)
def test_transient_sdk_errors_stop_at_retry_budget(kind, openai_three_try_gate):
    model = OpenAIModel(
        alias="judge", endpoint="gpt-5.6-luna", api_key="test-key"
    )
    calls = 0

    def create(**_request):
        nonlocal calls
        calls += 1
        raise _transient_sdk_error(kind)

    model.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = model.predict_single_input(
        [ModelInput(role=RoleType.USER, content="hi")], max_new_tokens=16
    )

    assert calls == 3
    assert result.failed is True
    assert type(_transient_sdk_error(kind)).__name__ in (result.raw_response or "")


def test_sdk_mock_transport_serializes_schema_and_detects_typed_refusal(
    openai_three_try_gate,
):
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "error": None,
                "incomplete_details": None,
                "model": "gpt-5.6-luna",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "refusal", "refusal": "I cannot judge this request."}
                        ],
                    }
                ],
                "parallel_tool_calls": True,
                "tools": [],
            },
        )

    model = OpenAIModel(
        alias="judge",
        is_debater=False,
        endpoint="gpt-5.6-luna",
        api_key="test-key",
        reasoning_effort="low",
    )
    model.client = openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        result = model.predict_single_input(
            [ModelInput(role=RoleType.USER, content="Judge this debate")],
            max_new_tokens=512,
            json_schema=VERDICT_SCHEMA,
        )
    finally:
        model.client.close()

    assert result.failed is True
    assert len(requests) == 1
    assert result.served_provider == "openai"
    assert result.generation_id == "resp_test"
    assert requests[0]["text"]["format"] == {
        "type": "json_schema",
        "name": "verdict",
        "schema": VERDICT_SCHEMA,
        "strict": True,
    }
    assert json.loads(result.raw_response or "")["type"] == "responses_refusal"


def test_json_schema_capability_tracks_request_adapter():
    direct = OpenAIModel(alias="judge", endpoint="gpt-5.6-luna", api_key="test-key")
    old_chat = OpenAIModel(alias="judge", endpoint="gpt-4-turbo", api_key="test-key")
    old_4o = OpenAIModel(alias="judge", endpoint="gpt-4o-2024-05-13", api_key="test-key")
    structured_4o = OpenAIModel(
        alias="judge", endpoint="gpt-4o-2024-08-06", api_key="test-key"
    )
    local = LocalModel(alias="judge", endpoint="qwen3.5-4b")

    class SchemaDroppingAdapter(OpenAIModel):
        def call_openai(self, *args, **kwargs):
            raise AssertionError("not called")

    dropping = SchemaDroppingAdapter(alias="judge", endpoint="custom", api_key="test-key")

    assert direct.supports_json_schema() is True
    assert old_chat.supports_json_schema() is False
    assert old_4o.supports_json_schema() is False
    assert structured_4o.supports_json_schema() is True
    assert local.supports_json_schema() is True
    assert dropping.supports_json_schema() is False


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("gpt-5.6-luna", True),
        ("gpt-5", True),
        ("gpt-4.1", True),
        ("gpt-4.1-mini-2025-04-14", True),
        ("gpt-4o", True),
        ("gpt-4o-2024-08-06", True),
        ("o1", True),
        ("o1-2024-12-17", True),
        ("o3", True),
        ("o3-2025-04-16", True),
        ("gpt-4-turbo", False),
        ("gpt-4o-2024-05-13", False),
        ("o1-mini", False),
        ("o1-mini-2024-09-12", False),
        ("o1-pro", True),
        ("o1-pro-2025-03-19", True),
        ("o3-deep-research", False),
        ("o3-deep-research-2025-06-26", False),
        ("o3-pro", True),
        ("o3-pro-2025-06-10", True),
    ],
)
def test_json_schema_capability_table(endpoint, expected):
    model = OpenAIModel(alias="judge", endpoint=endpoint, api_key="test-key")
    assert model.supports_json_schema() is expected
