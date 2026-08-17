"""Contract tests for the direct Piston case-execution adapter."""

from __future__ import annotations

import http.client
import json
import urllib.error

import pytest

from infra.envs.tasks import piston
from infra.envs.tasks.base import GraderInfrastructureError

BASE_URL = "http://127.0.0.1:2000"
RUNTIME = "3.12.0"


def _piston_response(
    *,
    stdout="3\n",
    stderr="",
    code=0,
    piston_signal=None,
    status=None,
):
    return {
        "language": "python",
        "version": RUNTIME,
        "run": {
            "stdout": stdout,
            "stderr": stderr,
            "output": stdout + stderr,
            "code": code,
            "signal": piston_signal,
            "message": None,
            "status": status,
            "cpu_time": 2.0,
            "wall_time": 3.0,
            "memory": 4096,
        },
    }


class _Response:
    def __init__(
        self,
        payload=None,
        *,
        raw=None,
        status_code=200,
        content_type="application/json; charset=utf-8",
    ):
        assert (payload is None) != (raw is None)
        self.raw = raw if raw is not None else json.dumps(payload).encode("utf-8")
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self.read_limits = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def getcode(self):
        return self.status_code

    def read(self, limit):
        self.read_limits.append(limit)
        return self.raw[:limit]


def _install_urlopen(monkeypatch, events):
    calls = []
    remaining_events = list(events)

    def urlopen(request, *, timeout):
        calls.append((request, timeout))
        if not remaining_events:
            raise AssertionError("unexpected extra Piston request")
        event = remaining_events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event

    monkeypatch.setattr(piston._DIRECT_OPENER, "open", urlopen)
    return calls


def _run(**overrides):
    kwargs = {
        "base_url": BASE_URL,
        "runtime_version": RUNTIME,
        "solution_code": "a, b = map(int, input().split())\nprint(a + b)",
        "test_input": "1 2\n",
        "remaining_seconds": 2.0,
    }
    kwargs.update(overrides)
    return piston.run_python_case(**kwargs)


def test_success_request_contains_only_candidate_material_and_deadline(monkeypatch):
    response = _Response(_piston_response())
    calls = _install_urlopen(monkeypatch, [response])
    moments = iter([100.0, 100.25])
    monkeypatch.setattr(piston.time, "perf_counter", lambda: next(moments))

    result = _run()

    assert result == {
        "returncode": 120,
        "timed_out": False,
        "output_limited": False,
        "stdout": "3\n",
        "stderr": "",
    }
    request, transport_timeout = calls[0]
    assert request.full_url == BASE_URL + "/api/v2/execute"
    assert request.method == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Accept") == "application/json"
    assert transport_timeout == pytest.approx(3.75)
    body = json.loads(request.data)
    assert body == {
        "language": "python",
        "version": RUNTIME,
        "files": [
            {
                "name": "solution.py",
                "content": "a, b = map(int, input().split())\nprint(a + b)",
                "encoding": "utf8",
            }
        ],
        "stdin": "1 2\n",
        "run_timeout": 1750,
        "run_cpu_time": 1750,
    }
    assert not any("expected" in key.lower() for key in body)
    assert response.read_limits == [piston._MAX_RESPONSE_BYTES + 1]


def test_piston_transport_ignores_process_proxy_configuration(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.test:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:8080")
    monkeypatch.setenv("NO_PROXY", "")
    calls = _install_urlopen(monkeypatch, [_Response(_piston_response())])

    assert _run()["returncode"] == 120
    assert len(calls) == 1
    proxy_handlers = [
        handler
        for handler in piston._DIRECT_OPENER.handlers
        if isinstance(handler, piston.urllib.request.ProxyHandler)
    ]
    # build_opener omits an explicitly empty ProxyHandler from its registered
    # handler list. The important invariant is that no environment-populated
    # proxy handler is present.
    assert proxy_handlers == []


def test_request_timeout_is_clamped_to_the_protocol_ceiling(monkeypatch):
    calls = _install_urlopen(monkeypatch, [_Response(_piston_response())])

    assert _run(remaining_seconds=91.0)["returncode"] == 120
    body = json.loads(calls[0][0].data)
    assert body["run_timeout"] == piston.MAX_RUN_TIMEOUT_MILLISECONDS
    assert body["run_cpu_time"] == piston.MAX_RUN_TIMEOUT_MILLISECONDS
    assert calls[0][1] == pytest.approx(92.0)


@pytest.mark.parametrize(
    ("status", "code", "piston_signal", "returncode", "timed_out", "limited"),
    [
        (None, 7, None, 1, False, False),
        ("RE", 1, None, 1, False, False),
        ("SG", None, "SIGSEGV", 1, False, False),
        ("TO", None, "SIGKILL", 1, True, False),
        ("OL", None, "SIGKILL", 1, False, True),
        ("EL", None, "SIGKILL", 1, False, True),
    ],
)
def test_candidate_outcomes_stay_ordinary_case_results(
    monkeypatch,
    status,
    code,
    piston_signal,
    returncode,
    timed_out,
    limited,
):
    _install_urlopen(
        monkeypatch,
        [
            _Response(
                _piston_response(
                    stdout="partial",
                    stderr="candidate stderr",
                    code=code,
                    piston_signal=piston_signal,
                    status=status,
                )
            )
        ],
    )

    result = _run()

    assert result == {
        "returncode": returncode,
        "timed_out": timed_out,
        "output_limited": limited,
        "stdout": "partial",
        "stderr": "candidate stderr",
    }


def test_direct_zero_exit_is_indistinguishable_from_normal_return(monkeypatch):
    """This records the known direct-source parity difference explicitly."""
    calls = _install_urlopen(monkeypatch, [_Response(_piston_response())])

    assert _run(solution_code="import os; os._exit(0)")["returncode"] == 120
    sent = json.loads(calls[0][0].data)
    assert sent["files"][0]["content"] == "import os; os._exit(0)"


def test_failure_status_with_zero_code_does_not_map_to_normal_sentinel(monkeypatch):
    calls = _install_urlopen(
        monkeypatch,
        [_Response(_piston_response(code=0, piston_signal=None, status="RE"))],
    )
    assert _run()["returncode"] == 1
    assert len(calls) == 1


def test_candidate_exit_120_cannot_collide_with_normal_sentinel(monkeypatch):
    _install_urlopen(
        monkeypatch,
        [_Response(_piston_response(code=120, piston_signal=None, status="RE"))],
    )

    result = _run(solution_code="raise SystemExit(120)")

    assert result["returncode"] != piston._NORMAL_RETURN_CODE
    assert result["returncode"] == 1


@pytest.mark.parametrize("status", ["XX", "UNKNOWN"])
def test_server_and_unsupported_statuses_are_infrastructure_failures(
    monkeypatch, status
):
    calls = _install_urlopen(
        monkeypatch,
        [
            _Response(_piston_response(code=None, piston_signal=None, status=status)),
            _Response(_piston_response()),
        ],
    )
    with pytest.raises(GraderInfrastructureError):
        _run()
    assert len(calls) == 1, "protocol failures must not be retried"


def test_http_400_is_fatal_without_retry(monkeypatch):
    error = urllib.error.HTTPError(BASE_URL, 400, "bad request", {}, None)
    calls = _install_urlopen(monkeypatch, [error, _Response(_piston_response())])

    with pytest.raises(GraderInfrastructureError, match="HTTP 400"):
        _run()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "first_failure",
    [
        urllib.error.HTTPError(BASE_URL, 503, "unavailable", {}, None),
        urllib.error.URLError("connection reset"),
        http.client.IncompleteRead(b"partial response"),
    ],
)
def test_5xx_and_transport_failure_are_retried_with_reduced_budget(
    monkeypatch, first_failure
):
    calls = _install_urlopen(
        monkeypatch, [first_failure, _Response(_piston_response())]
    )
    moments = iter([10.0, 10.0, 10.25])
    monkeypatch.setattr(piston.time, "perf_counter", lambda: next(moments))

    assert _run(remaining_seconds=1.0)["returncode"] == 120

    assert len(calls) == 2
    assert [json.loads(request.data)["run_timeout"] for request, _ in calls] == [
        1000,
        750,
    ]
    assert [timeout for _, timeout in calls] == pytest.approx([3.0, 2.75])


def test_transport_exhaustion_is_fatal_after_two_attempts(monkeypatch):
    calls = _install_urlopen(
        monkeypatch,
        [urllib.error.URLError("down"), TimeoutError("still down")],
    )

    with pytest.raises(GraderInfrastructureError, match="after 2 attempts"):
        _run()
    assert len(calls) == 2


def test_retry_cannot_run_past_remaining_solution_deadline(monkeypatch):
    calls = _install_urlopen(monkeypatch, [urllib.error.URLError("down")])
    moments = iter([20.0, 20.0, 21.1])
    monkeypatch.setattr(piston.time, "perf_counter", lambda: next(moments))

    with pytest.raises(GraderInfrastructureError, match="remaining solution deadline"):
        _run(remaining_seconds=1.0)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: [], "response schema"),
        (lambda payload: {**payload, "extra": True}, "response schema"),
        (
            lambda payload: {**payload, "version": "3.11.0"},
            "runtime identity",
        ),
        (
            lambda payload: {
                **payload,
                "run": {
                    key: value
                    for key, value in payload["run"].items()
                    if key != "status"
                },
            },
            "run schema",
        ),
        (
            lambda payload: {
                **payload,
                "run": {**payload["run"], "stdout": ["not", "text"]},
            },
            "process output",
        ),
        (
            lambda payload: {
                **payload,
                "run": {**payload["run"], "code": True},
            },
            "exit code",
        ),
        (
            lambda payload: {
                **payload,
                "run": {**payload["run"], "code": None, "signal": None},
            },
            "exit code/signal pair",
        ),
    ],
)
def test_malformed_response_schema_is_fatal_without_retry(monkeypatch, mutate, message):
    calls = _install_urlopen(
        monkeypatch,
        [_Response(mutate(_piston_response())), _Response(_piston_response())],
    )

    with pytest.raises(GraderInfrastructureError, match=message):
        _run()
    assert len(calls) == 1


@pytest.mark.parametrize("raw", [b"not json", b'"unterminated'])
def test_malformed_json_is_fatal_without_retry(monkeypatch, raw):
    calls = _install_urlopen(
        monkeypatch,
        [_Response(raw=raw), _Response(_piston_response())],
    )
    with pytest.raises(GraderInfrastructureError, match="malformed JSON"):
        _run()
    assert len(calls) == 1


def test_non_json_response_is_fatal(monkeypatch):
    calls = _install_urlopen(
        monkeypatch,
        [_Response(_piston_response(), content_type="text/html")],
    )
    with pytest.raises(GraderInfrastructureError, match="non-JSON"):
        _run()
    assert len(calls) == 1


def test_response_body_is_read_with_a_hard_cap(monkeypatch):
    monkeypatch.setattr(piston, "_MAX_RESPONSE_BYTES", 32)
    response = _Response(raw=b"x" * 33)
    _install_urlopen(monkeypatch, [response])

    with pytest.raises(GraderInfrastructureError, match="response limit"):
        _run()
    assert response.read_limits == [33]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"base_url": "file:///tmp/piston"}, "base URL"),
        ({"base_url": "http://user:secret@judge:2000"}, "base URL"),
        ({"base_url": "http://judge.internal:2000"}, "only on loopback"),
        ({"base_url": "http://127.0.0.1:2000/piston"}, "base URL"),
        ({"base_url": "http://[::1"}, "base URL"),
        ({"base_url": "http://127.0.0.1:99999"}, "base URL"),
        ({"runtime_version": "3.x"}, "exact semantic version"),
        ({"runtime_version": "*"}, "exact semantic version"),
        ({"remaining_seconds": 0}, "positive finite"),
        ({"remaining_seconds": float("inf")}, "positive finite"),
        ({"test_input": b"bytes"}, "must be strings"),
    ],
)
def test_invalid_trusted_inputs_fail_before_network(monkeypatch, overrides, message):
    def forbidden_urlopen(*args, **kwargs):
        raise AssertionError("network must not be called")

    monkeypatch.setattr(piston._DIRECT_OPENER, "open", forbidden_urlopen)
    with pytest.raises(GraderInfrastructureError, match=message):
        _run(**overrides)
