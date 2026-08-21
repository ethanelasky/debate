from __future__ import annotations

from types import SimpleNamespace

import pytest

from infra.backend import verl as verl_mod
from infra.backend.verl import VerlBackend, VerlBackendConfig


def _record(rank: int, *, success: bool = True, after=(0, 0)) -> dict:
    return {
        "rank": rank,
        "success": success,
        "before": (16, 32),
        "after": after,
    }


class _CheckpointManager:
    def __init__(self):
        self.update_calls: list[int] = []

    def update_weights(self, global_step: int) -> None:
        self.update_calls.append(global_step)


def _backend(worker_group, *, n_gpus: int) -> VerlBackend:
    backend = VerlBackend.__new__(VerlBackend)
    backend.config = VerlBackendConfig(model_path="test-model", n_gpus=n_gpus)
    backend.wg = worker_group
    backend.checkpoint_manager = _CheckpointManager()
    backend._global_step = 7
    backend._rollout_awake = False
    return backend


class _ExecutingAllRanksWorkerGroup:
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.current_rank: int | None = None
        self.calls: list[tuple[str, object]] = []

    def execute_all_sync(self, method_name, func):
        self.calls.append((method_name, func))
        results = []
        for rank in range(self.world_size):
            self.current_rank = rank
            results.append(func())
        return results


class _ScriptedWorkerGroup:
    def __init__(self, world_size: int, results):
        self.world_size = world_size
        self.results = results
        self.calls: list[tuple[str, object]] = []

    def execute_all_sync(self, method_name, func):
        self.calls.append((method_name, func))
        return self.results


def test_two_training_ranks_both_release_before_rollout_wakes(monkeypatch):
    wg = _ExecutingAllRanksWorkerGroup(world_size=2)
    released: list[int] = []

    def fake_release():
        assert wg.current_rank is not None
        released.append(wg.current_rank)
        return _record(wg.current_rank)

    monkeypatch.setattr(verl_mod, "_release_training_cache", fake_release)
    backend = _backend(wg, n_gpus=2)

    backend.sync_sampler()

    assert released == [0, 1]
    assert wg.calls == [("execute_func_rank_zero", fake_release)]
    assert backend.checkpoint_manager.update_calls == [7]
    assert backend._rollout_awake is True


def test_rank_one_release_failure_blocks_rollout_wake():
    wg = _ScriptedWorkerGroup(2, [_record(0), _record(1, success=False)])
    backend = _backend(wg, n_gpus=2)

    with pytest.raises(RuntimeError, match="failed on rank 1"):
        backend.sync_sampler()

    assert backend.checkpoint_manager.update_calls == []
    assert backend._rollout_awake is False


@pytest.mark.parametrize(
    "results", [[_record(0)], [_record(0), _record(1), _record(2)]]
)
def test_release_cardinality_mismatch_blocks_rollout_wake(results):
    wg = _ScriptedWorkerGroup(2, results)
    backend = _backend(wg, n_gpus=2)

    with pytest.raises(RuntimeError, match="cardinality mismatch"):
        backend.sync_sampler()

    assert backend.checkpoint_manager.update_calls == []
    assert backend._rollout_awake is False


def test_nonfinite_release_telemetry_blocks_rollout_wake():
    wg = _ScriptedWorkerGroup(2, [_record(0), _record(1, after=(0, float("nan")))])
    backend = _backend(wg, n_gpus=2)

    with pytest.raises(RuntimeError, match="nonfinite or negative"):
        backend.sync_sampler()

    assert backend.checkpoint_manager.update_calls == []
    assert backend._rollout_awake is False


def test_one_training_rank_uses_same_all_rank_release_path(monkeypatch):
    wg = _ExecutingAllRanksWorkerGroup(world_size=1)

    def fake_release():
        assert wg.current_rank == 0
        return _record(0)

    monkeypatch.setattr(verl_mod, "_release_training_cache", fake_release)
    backend = _backend(wg, n_gpus=1)

    backend.sync_sampler()

    assert wg.calls == [("execute_func_rank_zero", fake_release)]
    assert backend.checkpoint_manager.update_calls == [7]
    assert backend._rollout_awake is True


def test_missing_all_rank_worker_group_api_fails_closed_before_wake():
    backend = _backend(SimpleNamespace(world_size=2), n_gpus=2)

    with pytest.raises(RuntimeError, match="does not support execute_all_sync"):
        backend.sync_sampler()

    assert backend.checkpoint_manager.update_calls == []
    assert backend._rollout_awake is False
