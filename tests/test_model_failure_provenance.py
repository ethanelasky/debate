"""Failed frozen-model generations retain safe, actionable provenance."""

from infra.envs.debate.protocol import CompiledSlot, Kind, Slot
from infra.envs.debate.round import DebateRound, DebateState, SlotResult
from infra.models.base import ModelResponse


def test_failed_model_response_is_retained_in_state_meta():
    round_ = object.__new__(DebateRound)
    state = DebateState(bindings={})
    step = CompiledSlot(
        index=5,
        turn=2,
        speaker="judge",
        seq=0,
        slot=Slot(name="verdict", kind=Kind.DECISION),
    )
    response = ModelResponse(
        speech="PRIVATE MODEL OUTPUT",
        prompt="sk-secret-must-not-be-copied",
        raw_response='{"type":"responses_incomplete","status":"incomplete"}',
        failed=True,
        stop_reason="length",
        served_provider="openai",
        generation_id="resp_123",
    )

    round_._ingest(
        state,
        step,
        SlotResult(
            text="",
            response=response,
            failed=True,
            fail_reason="model_failed",
        ),
    )

    assert state.failed == "judge/verdict: model_failed"
    assert state.meta["model_failure"] == {
        "slot": "judge/verdict@2",
        "fail_reason": "model_failed",
        "stop_reason": "length",
        "raw_response": '{"type":"responses_incomplete","status":"incomplete"}',
        "served_provider": "openai",
        "generation_id": "resp_123",
    }
    assert "sk-secret" not in repr(state.meta["model_failure"])
    assert "PRIVATE MODEL OUTPUT" not in repr(state.meta["model_failure"])


def test_non_model_failure_does_not_invent_model_provenance():
    round_ = object.__new__(DebateRound)
    state = DebateState(bindings={})
    step = CompiledSlot(
        index=0,
        turn=0,
        speaker="alice",
        seq=0,
        slot=Slot(name="proposal", kind=Kind.SOLUTION),
    )

    round_._ingest(
        state,
        step,
        SlotResult(text="", failed=True, fail_reason="fidelity"),
    )

    assert state.failed == "alice/proposal: fidelity"
    assert "model_failure" not in state.meta
