from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from infra.envs.tasks.base import GraderInfrastructureError
from scripts import smoke_piston_verifier as smoke


def _result(
    status: str,
    *,
    passed: bool,
    timed_out: bool,
    tests_passed: int,
    tests_total: int,
) -> dict[str, Any]:
    return {
        "status": status,
        "passed": passed,
        "timeout": timed_out,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
    }


class FakeProductionSupervisor:
    def __init__(self, *, fail_followers: bool = False) -> None:
        self.lock = threading.Lock()
        self.active_timeouts = 0
        self.maximum_active_timeouts = 0
        self.follower_active_counts: list[int] = []
        self.calls: list[dict[str, Any]] = []
        self.timeout_calls = 0
        self.fail_followers = fail_followers

    def __call__(
        self,
        solution_code: str,
        inputs: list[str],
        outputs: list[str],
        timeout: int,
        **settings: str,
    ) -> dict[str, Any]:
        with self.lock:
            self.calls.append(
                {
                    "solution_code": solution_code,
                    "inputs": inputs,
                    "outputs": outputs,
                    "timeout": timeout,
                    **settings,
                }
            )

        if solution_code == smoke._TIMEOUT_SOURCE:
            with self.lock:
                self.timeout_calls += 1
                timeout_call = self.timeout_calls
                self.active_timeouts += 1
                self.maximum_active_timeouts = max(
                    self.maximum_active_timeouts, self.active_timeouts
                )
            if timeout_call == 1:
                # The standalone timeout phase precedes saturation.
                time.sleep(0.01)
            else:
                time.sleep(0.05)
            with self.lock:
                self.active_timeouts -= 1
            return _result(
                "timeout",
                passed=False,
                timed_out=True,
                tests_passed=0,
                tests_total=1,
            )

        if solution_code == smoke._FOLLOWER_SOURCE:
            with self.lock:
                self.follower_active_counts.append(self.active_timeouts)
            if self.fail_followers:
                raise GraderInfrastructureError("synthetic Piston outage")

        if solution_code in {smoke._EXIT_1_SOURCE, smoke._EXIT_120_SOURCE}:
            return _result(
                "failed",
                passed=False,
                timed_out=False,
                tests_passed=0,
                tests_total=1,
            )
        if solution_code == smoke._OUTPUT_FLOOD_SOURCE:
            return _result(
                "candidate_error",
                passed=False,
                timed_out=False,
                tests_passed=0,
                tests_total=1,
            )
        return _result(
            "passed",
            passed=True,
            timed_out=False,
            tests_passed=len(inputs),
            tests_total=len(inputs),
        )


def test_preflight_exercises_every_contract_and_starts_leaders_first(capsys):
    verifier = FakeProductionSupervisor()

    smoke.run_preflight(
        url="http://127.0.0.1:2000",
        runtime="3.12.0",
        verifier=verifier,
    )

    assert verifier.maximum_active_timeouts == 4
    assert len(verifier.follower_active_counts) == 28
    saturation_leaders = [
        call
        for call in verifier.calls
        if call["solution_code"] == smoke._TIMEOUT_SOURCE
    ][1:]
    followers = [
        call
        for call in verifier.calls
        if call["solution_code"] == smoke._FOLLOWER_SOURCE
    ]
    assert len(saturation_leaders) == 4
    assert all(call["timeout"] == 3 for call in saturation_leaders)
    assert len(followers) == 28
    assert all(call["verifier"] == "piston" for call in verifier.calls)
    assert all(
        call["piston_url"] == "http://127.0.0.1:2000"
        for call in verifier.calls
    )
    assert all(
        call["piston_python_version"] == "3.12.0"
        for call in verifier.calls
    )
    assert verifier.calls[0]["inputs"] == [
        "",
        "unterminated",
        "line\n",
        "line\r\n",
    ]
    assert len(verifier.calls[5]["inputs"][0]) == 600 * 1024
    assert verifier.calls[6]["solution_code"] == smoke._EARLY_EXIT_SOURCE
    assert len(verifier.calls[6]["inputs"][0]) == 600 * 1024
    assert "[7/7] 4-slot saturation" in capsys.readouterr().out


def test_preflight_saturates_the_effective_two_slot_concurrency(monkeypatch, capsys):
    verifier = FakeProductionSupervisor()
    monkeypatch.setattr(
        smoke.codecontests, "_MAX_CONCURRENT_PISTON_VERIFIERS", 2
    )

    smoke.run_preflight(
        url="http://127.0.0.1:2000",
        verifier=verifier,
    )

    assert verifier.maximum_active_timeouts == 2
    assert len(verifier.follower_active_counts) == 28
    saturation_leaders = [
        call
        for call in verifier.calls
        if call["solution_code"] == smoke._TIMEOUT_SOURCE
    ][1:]
    followers = [
        call
        for call in verifier.calls
        if call["solution_code"] == smoke._FOLLOWER_SOURCE
    ]
    assert len(saturation_leaders) == 2
    assert all(call["timeout"] == 3 for call in saturation_leaders)
    assert len(followers) == 28
    assert "[7/7] 2-slot saturation" in capsys.readouterr().out


def test_saturation_does_not_drain_queued_followers_after_infrastructure_error():
    verifier = FakeProductionSupervisor(fail_followers=True)

    with pytest.raises(GraderInfrastructureError, match="synthetic Piston outage"):
        smoke.run_preflight(
            url="http://127.0.0.1:2000",
            verifier=verifier,
        )

    assert 1 <= len(verifier.follower_active_counts) <= 4


@pytest.mark.parametrize("effective", [0, -1, 5, True, "4"])
def test_preflight_rejects_invalid_effective_concurrency(monkeypatch, effective):
    verifier = FakeProductionSupervisor()
    monkeypatch.setattr(
        smoke.codecontests,
        "_MAX_CONCURRENT_PISTON_VERIFIERS",
        effective,
    )

    with pytest.raises(smoke.PreflightFailure, match="must be an integer from 1 through 4"):
        smoke.run_preflight(
            url="http://127.0.0.1:2000",
            verifier=verifier,
        )

    assert verifier.calls == []


def test_preflight_fails_on_a_verdict_mismatch():
    def always_wrong(*args, **kwargs):
        return _result(
            "failed",
            passed=False,
            timed_out=False,
            tests_passed=0,
            tests_total=4,
        )

    with pytest.raises(smoke.PreflightFailure, match="byte-exact stdin mismatch"):
        smoke.run_preflight(
            url="http://127.0.0.1:2000",
            verifier=always_wrong,
        )
