"""Focused coverage for the optional CodeContests Piston execution path."""

from __future__ import annotations

import json

import pytest

import infra.envs.tasks.codecontests as codecontests_module
from infra.envs.tasks.base import GraderInfrastructureError
from infra.envs.tasks.codecontests import (
    CodeContestsEnv,
    CodeContestsFamily,
    run_stdin_tests,
)

PISTON_URL = "https://piston.example.test"
PYTHON_VERSION = "3.10.0"
NORMAL_RETURN_CODE = 120
ECHO_SOLUTION = "print(input().strip())"

ROWS = [
    {
        "name": "echo-one",
        "problem": "Echo one line.",
        "rlvr_inputs": ["reward-one"],
        "rlvr_outputs": ["reward-one"],
        "truth_inputs": ["truth-one"],
        "truth_outputs": ["truth-one"],
    },
    {
        "name": "echo-two",
        "problem": "Echo one line again.",
        "rlvr_inputs": ["reward-two"],
        "rlvr_outputs": ["reward-two"],
        "truth_inputs": ["truth-two"],
        "truth_outputs": ["truth-two"],
    },
]


def _write_rows(path) -> str:
    path.write_text("\n".join(json.dumps(row) for row in ROWS) + "\n")
    return str(path)


def _case_result(stdout: str, *, timed_out: bool = False) -> dict[str, object]:
    return {
        "returncode": NORMAL_RETURN_CODE,
        "timed_out": timed_out,
        "output_limited": False,
        "stdout": stdout,
        "stderr": "",
    }


def _piston_case_result(
    stdout: str,
    *,
    timed_out: bool = False,
    candidate_time_seconds: float = 0.1,
) -> dict[str, object]:
    result = _case_result(stdout, timed_out=timed_out)
    result["candidate_time_seconds"] = candidate_time_seconds
    return result


def _piston_kwargs() -> dict[str, str]:
    return {
        "verifier": "piston",
        "piston_url": PISTON_URL,
        "piston_python_version": PYTHON_VERSION,
    }


def test_default_backend_still_calls_only_the_local_case_runner(monkeypatch):
    local_calls = []

    def fake_local(**kwargs):
        local_calls.append(kwargs)
        return _case_result("answer")

    def unexpected_piston(**kwargs):
        pytest.fail(f"default local verifier called Piston with {kwargs!r}")

    monkeypatch.setattr(codecontests_module, "_run_candidate_case", fake_local)
    monkeypatch.setattr(
        codecontests_module.piston, "run_python_case", unexpected_piston
    )

    result = run_stdin_tests("print('answer')", ["input"], ["answer"], timeout=5)

    assert result["passed"] is True
    assert len(local_calls) == 1
    assert local_calls[0]["solution_code"] == "print('answer')"
    assert local_calls[0]["test_input"] == "input"


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"verifier": "docker"}, "'local' or 'piston'"),
        (
            {"verifier": "local", "piston_url": PISTON_URL},
            "does not accept Piston-only settings",
        ),
        (
            {"verifier": "local", "piston_python_version": PYTHON_VERSION},
            "does not accept Piston-only settings",
        ),
        ({"verifier": "piston"}, "nonempty piston_url"),
        (
            {"verifier": "piston", "piston_url": "   "},
            "nonempty piston_url",
        ),
        (
            {"verifier": "piston", "piston_url": PISTON_URL},
            "exact semantic piston_python_version",
        ),
        (
            {
                "verifier": "piston",
                "piston_url": PISTON_URL,
                "piston_python_version": "*",
            },
            "wildcards and version ranges",
        ),
        (
            {
                "verifier": "piston",
                "piston_url": "http://judge.internal:2000",
                "piston_python_version": PYTHON_VERSION,
            },
            "only on loopback",
        ),
        (
            {
                "verifier": "piston",
                "piston_url": "https://piston.example.test/base-path",
                "piston_python_version": PYTHON_VERSION,
            },
            "base URL is invalid",
        ),
        (
            {
                "verifier": "piston",
                "piston_url": PISTON_URL,
                "piston_python_version": ">=3.10",
            },
            "wildcards and version ranges",
        ),
    ],
)
def test_dataset_config_rejects_invalid_verifier_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        CodeContestsFamily().source({"path": "unused.jsonl", **settings})


def test_piston_receives_only_source_stdin_and_remaining_budget(monkeypatch):
    calls = []
    expected_output = "EXPECTED-OUTPUT-MUST-STAY-IN-SUPERVISOR"

    def fake_piston(**kwargs):
        calls.append(kwargs)
        return _piston_case_result(expected_output)

    def unexpected_local(**kwargs):
        pytest.fail(f"selected Piston verifier called local runner with {kwargs!r}")

    monkeypatch.setattr(codecontests_module.piston, "run_python_case", fake_piston)
    monkeypatch.setattr(codecontests_module, "_run_candidate_case", unexpected_local)

    result = run_stdin_tests(
        ECHO_SOLUTION,
        ["candidate-stdin"],
        [expected_output],
        timeout=5,
        **_piston_kwargs(),
    )

    assert result["passed"] is True
    assert len(calls) == 1
    assert set(calls[0]) == {
        "base_url",
        "runtime_version",
        "solution_code",
        "test_input",
        "remaining_seconds",
    }
    assert calls[0]["base_url"] == PISTON_URL
    assert calls[0]["runtime_version"] == PYTHON_VERSION
    assert calls[0]["solution_code"] == ECHO_SOLUTION
    assert calls[0]["test_input"] == "candidate-stdin"
    assert 0 < calls[0]["remaining_seconds"] <= 5
    assert expected_output not in calls[0].values()


def test_piston_runtime_owns_syntax_validation(monkeypatch):
    calls = []

    def fake_piston(**kwargs):
        calls.append(kwargs)
        return {
            "returncode": 1,
            "timed_out": False,
            "output_limited": False,
            "stdout": "",
            "stderr": "SyntaxError: invalid syntax",
            "candidate_time_seconds": 0.1,
        }

    monkeypatch.setattr(codecontests_module.piston, "run_python_case", fake_piston)

    result = run_stdin_tests(
        "def (:", ["input"], ["output"], timeout=5, **_piston_kwargs()
    )

    assert len(calls) == 1
    assert calls[0]["solution_code"] == "def (:"
    assert result["status"] == "failed"
    assert result["passed"] is False
    assert "SyntaxError" in result["first_failure"]["stderr"]


def test_reward_and_family_grade_use_configured_piston(tmp_path, monkeypatch):
    path = _write_rows(tmp_path / "rows.jsonl")
    calls = []

    def fake_piston(**kwargs):
        calls.append(kwargs)
        return _piston_case_result(kwargs["test_input"])

    monkeypatch.setattr(codecontests_module.piston, "run_python_case", fake_piston)
    family = CodeContestsFamily()
    env = family.source(
        {
            "path": path,
            "test_path": path,
            "timeout_seconds": 5,
            **_piston_kwargs(),
        }
    )

    task = env.tasks(1, split="test")[0]
    task.meta.update(
        {
            "gdm_inputs": ["gdm-eval"],
            "gdm_outputs": ["gdm-eval"],
            "cco_inputs": ["cco-eval"],
            "cco_outputs": ["cco-eval"],
        }
    )
    reward, info = env.reward(task, f"```python\n{ECHO_SOLUTION}\n```")
    grade = family.grade(
        {"truth_inputs": ["family-truth"], "truth_outputs": ["family-truth"]},
        ECHO_SOLUTION,
    )

    assert isinstance(env, CodeContestsEnv)
    assert env.grade_workers == codecontests_module._MAX_CONCURRENT_PISTON_VERIFIERS
    assert family.grade_workers == codecontests_module._MAX_CONCURRENT_PISTON_VERIFIERS
    assert reward == pytest.approx(1.1)
    assert info["correct_relaxed"] == 1.0
    assert info["cco_correct"] == 1.0
    assert grade is True
    assert [call["test_input"] for call in calls] == [
        "cco-eval",
        "gdm-eval",
        "family-truth",
    ]
    assert all(call["base_url"] == PISTON_URL for call in calls)
    assert all(call["runtime_version"] == PYTHON_VERSION for call in calls)


def test_piston_cases_ignore_transport_time_and_share_candidate_budget(monkeypatch):
    now = [100.0]
    remaining_values = []

    monkeypatch.setattr(codecontests_module.time, "perf_counter", lambda: now[0])

    def fake_piston(**kwargs):
        remaining_values.append(kwargs["remaining_seconds"])
        # The synthetic five-second HTTP round trip is trusted transport, not
        # candidate execution. Only the reported 0.5 seconds consumes budget.
        now[0] += 5.0
        return _piston_case_result("ok", candidate_time_seconds=0.5)

    monkeypatch.setattr(codecontests_module.piston, "run_python_case", fake_piston)

    result = run_stdin_tests(
        ECHO_SOLUTION,
        ["one", "two", "three"],
        ["ok", "ok", "ok"],
        timeout=1,
        **_piston_kwargs(),
    )

    assert remaining_values == pytest.approx([1.0, 0.5])
    assert result["status"] == "timeout"
    assert result["timeout"] is True
    assert result["tests_passed"] == 2
    assert result["tests_total"] == 3
    assert result["first_failure"]["test_idx"] == 2


def test_piston_candidate_time_not_transport_time_can_leave_budget(monkeypatch):
    now = [100.0]
    remaining_values = []

    monkeypatch.setattr(codecontests_module.time, "perf_counter", lambda: now[0])

    def fake_piston(**kwargs):
        remaining_values.append(kwargs["remaining_seconds"])
        now[0] += 20.0
        return _piston_case_result("ok", candidate_time_seconds=0.25)

    monkeypatch.setattr(codecontests_module.piston, "run_python_case", fake_piston)

    result = run_stdin_tests(
        ECHO_SOLUTION,
        ["one", "two", "three"],
        ["ok", "ok", "ok"],
        timeout=1,
        **_piston_kwargs(),
    )

    assert remaining_values == pytest.approx([1.0, 0.75, 0.5])
    assert result["status"] == "passed"
    assert result["timeout"] is False


@pytest.mark.parametrize("candidate_time", [None, True, -1, float("nan")])
def test_piston_candidate_time_contract_is_fail_closed(monkeypatch, candidate_time):
    def fake_piston(**kwargs):
        result = _case_result("ok")
        if candidate_time is not None:
            result["candidate_time_seconds"] = candidate_time
        return result

    monkeypatch.setattr(codecontests_module.piston, "run_python_case", fake_piston)

    with pytest.raises(GraderInfrastructureError, match="case schema|candidate time"):
        run_stdin_tests(
            ECHO_SOLUTION,
            ["one"],
            ["ok"],
            timeout=1,
            **_piston_kwargs(),
        )


def test_piston_non_timeout_cannot_overspend_final_case(monkeypatch):
    monkeypatch.setattr(
        codecontests_module.piston,
        "run_python_case",
        lambda **kwargs: _piston_case_result(
            "ok", candidate_time_seconds=1.5
        ),
    )

    with pytest.raises(GraderInfrastructureError, match="beyond.*candidate budget"):
        run_stdin_tests(
            ECHO_SOLUTION,
            ["one"],
            ["ok"],
            timeout=1,
            **_piston_kwargs(),
        )


def test_piston_non_timeout_cannot_overspend_cumulative_budget(monkeypatch):
    remaining_values = []
    candidate_times = iter([0.6, 0.5])

    def fake_piston(**kwargs):
        remaining_values.append(kwargs["remaining_seconds"])
        return _piston_case_result(
            "ok", candidate_time_seconds=next(candidate_times)
        )

    monkeypatch.setattr(codecontests_module.piston, "run_python_case", fake_piston)

    with pytest.raises(GraderInfrastructureError, match="beyond.*candidate budget"):
        run_stdin_tests(
            ECHO_SOLUTION,
            ["one", "two"],
            ["ok", "ok"],
            timeout=1,
            **_piston_kwargs(),
        )

    assert remaining_values == pytest.approx([1.0, 0.4])


def test_piston_semaphore_is_held_for_the_entire_solution(monkeypatch):
    events = []

    class RecordingSemaphore:
        held = False

        def acquire(self):
            assert not self.held
            self.held = True
            events.append("acquire")

        def release(self):
            assert self.held
            self.held = False
            events.append("release")

    semaphore = RecordingSemaphore()
    monkeypatch.setattr(
        codecontests_module, "_piston_verifier_semaphore", semaphore
    )

    def fake_piston(**kwargs):
        assert semaphore.held
        events.append(f"run:{kwargs['test_input']}")
        return _piston_case_result("ok")

    monkeypatch.setattr(codecontests_module.piston, "run_python_case", fake_piston)

    result = run_stdin_tests(
        ECHO_SOLUTION,
        ["one", "two"],
        ["ok", "ok"],
        timeout=5,
        **_piston_kwargs(),
    )

    assert result["passed"] is True
    assert events == ["acquire", "run:one", "run:two", "release"]


def test_piston_infrastructure_errors_propagate(monkeypatch):
    error = GraderInfrastructureError("piston transport unavailable")

    def fail_piston(**kwargs):
        raise error

    monkeypatch.setattr(codecontests_module.piston, "run_python_case", fail_piston)

    with pytest.raises(GraderInfrastructureError) as exc_info:
        run_stdin_tests(
            ECHO_SOLUTION,
            ["input"],
            ["output"],
            timeout=5,
            **_piston_kwargs(),
        )
    assert exc_info.value is error

    family = CodeContestsFamily(
        timeout_seconds=5,
        verifier="piston",
        piston_url=PISTON_URL,
        piston_python_version=PYTHON_VERSION,
    )
    with pytest.raises(GraderInfrastructureError) as family_exc_info:
        family.grade(
            {"truth_inputs": ["input"], "truth_outputs": ["output"]},
            ECHO_SOLUTION,
        )
    assert family_exc_info.value is error


def test_protocol_identity_tracks_backend_protocol_version_but_not_endpoint(tmp_path):
    path = _write_rows(tmp_path / "rows.jsonl")
    base_config = {"path": path, "test_path": path, "eval_subset_size": 2}

    local_family = CodeContestsFamily()
    local_family.source(base_config)
    local_identity = local_family.protocol_identity()

    first_piston_family = CodeContestsFamily()
    first_piston_family.source({**base_config, **_piston_kwargs()})
    first_piston_identity = first_piston_family.protocol_identity()

    other_endpoint_family = CodeContestsFamily()
    other_endpoint_family.source(
        {
            **base_config,
            **_piston_kwargs(),
            "piston_url": "https://other-piston.example.test",
        }
    )
    other_endpoint_identity = other_endpoint_family.protocol_identity()

    other_version_family = CodeContestsFamily()
    other_version_family.source(
        {
            **base_config,
            **_piston_kwargs(),
            "piston_python_version": "3.11.9",
        }
    )
    other_version_identity = other_version_family.protocol_identity()

    assert local_identity["verifier"] == "local"
    assert local_identity["piston_protocol"] == "none"
    assert local_identity["piston_python_version"] == "none"
    assert first_piston_identity["verifier"] == "piston"
    assert first_piston_identity["piston_protocol"] == "codecontests-piston-v2"
    assert first_piston_identity["piston_python_version"] == PYTHON_VERSION
    assert "piston_url" not in first_piston_identity
    assert PISTON_URL not in first_piston_identity.values()
    assert other_endpoint_identity == first_piston_identity
    assert other_version_identity != first_piston_identity
    assert other_version_identity["piston_python_version"] == "3.11.9"
