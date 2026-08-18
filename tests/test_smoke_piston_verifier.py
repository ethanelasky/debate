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
        self.local = threading.local()

    def perf_counter(self) -> float:
        return getattr(self.local, "now", 0.0)

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
                # Keep these concurrency tests fast while presenting the
                # three-second duration that a correct production timeout has.
                self.local.now = 3.0
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


class DeterministicSaturationSupervisor:
    """Return valid verdicts while exposing synthetic per-thread durations."""

    def __init__(self, leader_durations: list[float]) -> None:
        self.lock = threading.Lock()
        self.local = threading.local()
        self.leader_durations = iter(leader_durations)
        self.leader_calls = 0
        self.follower_calls = 0

    def perf_counter(self) -> float:
        return getattr(self.local, "now", 0.0)

    def __call__(
        self,
        solution_code: str,
        inputs: list[str],
        outputs: list[str],
        timeout: int,
        **settings: str,
    ) -> dict[str, Any]:
        if solution_code == smoke._TIMEOUT_SOURCE:
            with self.lock:
                duration = next(self.leader_durations)
                self.leader_calls += 1
            self.local.now = duration
            return _result(
                "timeout",
                passed=False,
                timed_out=True,
                tests_passed=0,
                tests_total=1,
            )

        assert solution_code == smoke._FOLLOWER_SOURCE
        with self.lock:
            self.follower_calls += 1
        return _result(
            "passed",
            passed=True,
            timed_out=False,
            tests_passed=1,
            tests_total=1,
        )


def test_saturation_timing_accepts_four_simultaneous_slots(monkeypatch):
    verifier = DeterministicSaturationSupervisor([3.0, 3.0, 3.0, 3.0])
    monkeypatch.setattr(smoke.time, "perf_counter", verifier.perf_counter)

    smoke._check_saturation(
        verifier=verifier,
        settings=smoke._settings("http://127.0.0.1:2000", "3.12.0"),
        concurrency=4,
    )

    assert verifier.leader_calls == 4
    assert verifier.follower_calls == 28


def test_saturation_timing_rejects_immediate_fake_timeouts(monkeypatch):
    verifier = DeterministicSaturationSupervisor([0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(smoke.time, "perf_counter", verifier.perf_counter)

    with pytest.raises(
        smoke.PreflightFailure,
        match=r"returned after 0\.000s, before the 2\.5s minimum",
    ):
        smoke._check_saturation(
            verifier=verifier,
            settings=smoke._settings("http://127.0.0.1:2000", "3.12.0"),
            concurrency=4,
        )

    assert verifier.leader_calls == 4


def test_saturation_timing_rejects_hidden_two_slot_queue_with_long_http_grace(
    monkeypatch,
):
    # The longer transport grace lets both three-second waves return valid
    # timeout verdicts; independent leader timing must still reject the queue.
    assert smoke.codecontests.piston._RESPONSE_GRACE_SECONDS == 10.0
    verifier = DeterministicSaturationSupervisor([3.0, 3.0, 6.0, 6.0])
    monkeypatch.setattr(smoke.time, "perf_counter", verifier.perf_counter)

    with pytest.raises(
        smoke.PreflightFailure,
        match=r"took 6\.000s, reaching the 5\.0s hidden two-wave boundary",
    ):
        smoke._check_saturation(
            verifier=verifier,
            settings=smoke._settings("http://127.0.0.1:2000", "3.12.0"),
            concurrency=4,
        )

    assert verifier.leader_calls == 4


def test_preflight_exercises_every_contract_and_starts_leaders_first(
    monkeypatch, capsys
):
    verifier = FakeProductionSupervisor()
    monkeypatch.setattr(smoke.time, "perf_counter", verifier.perf_counter)

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
    monkeypatch.setattr(smoke.time, "perf_counter", verifier.perf_counter)
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


def test_saturation_does_not_drain_queued_followers_after_infrastructure_error(
    monkeypatch,
):
    verifier = FakeProductionSupervisor(fail_followers=True)
    monkeypatch.setattr(smoke.time, "perf_counter", verifier.perf_counter)

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
