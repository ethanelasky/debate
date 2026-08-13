"""Batch-level rollout coverage counters, including trajectory-free batches."""

from __future__ import annotations

import pytest

from infra.backend.base import Sample, SamplingParams
from infra.envs.base import SingleTurnEnv, SlotLimits, Task, budget_forced_sample
from infra.envs.debate.protocol import Protocol
from infra.envs.debate.round import (
    DebateRound,
    DebateState,
    GenRequest,
    PolicySeat,
    SlotResult,
)
from infra.envs.planned import PlannedEnv


def _sample(text: str) -> Sample:
    tokens = list(range(1, len(text) + 1))
    return Sample(
        tokens=tokens,
        logprobs=[-0.1] * len(tokens),
        text=text,
        stop_reason="stop",
        prompt_tokens=[99],
    )


class _ScriptedPolicy:
    def __init__(self, *stages: list[list[str]]):
        self.stages = list(stages)

    def predict(self, convos, n=1, limits=None):
        stage = self.stages.pop(0)
        return [[_sample(text) for text in row] for row in stage]


class _ExplodingPolicy:
    def predict(self, convos, n=1, limits=None):
        raise RuntimeError("generation exploded")


class _AnswerExplodingPolicy:
    def __init__(self, plan_stage):
        self.plan_stage = plan_stage
        self.calls = 0

    def predict(self, convos, n=1, limits=None):
        self.calls += 1
        if self.calls == 1:
            return [[_sample(text) for text in row] for row in self.plan_stage]
        raise RuntimeError("answer generation exploded")


class _ShapePolicy:
    """Minimal Policy duck type returning pre-built backend result shapes."""

    def __init__(self, *stages):
        self.stages = list(stages)

    def predict(self, convos, n=1, limits=None):
        return self.stages.pop(0)


def _nested_samples(rows):
    return [[_sample(text) for text in row] for row in rows]


def _requests(n):
    return [
        GenRequest(
            messages=[{"role": "user", "content": f"answer {i}"}],
            limits=SlotLimits(max_total_tokens=8),
        )
        for i in range(n)
    ]


class _Prompts:
    messages = [{"role": "user", "content": "answer"}]

    def supplied_templates(self):
        return {"ANSWER_GEN_USER": "answer", "PLAN_USER": "plan <PROBLEM>"}


class _Source(SingleTurnEnv):
    prompts = _Prompts()

    def tasks(self, n, split="train"):
        return [
            Task(messages=[{"role": "user", "content": "answer"}], meta={"question": f"q{i}"})
            for i in range(n)
        ]

    def reward(self, task, text):
        return 1.0, {}


def test_single_turn_coverage_normal_partial_and_all_drop():
    env = _Source()
    tasks = env.tasks(2)

    groups = env.rollout(tasks, _ScriptedPolicy([["a", "b"], ["c", "d"]]), group_size=2)
    assert [len(group) for group in groups] == [2, 2]
    assert env.last_rollout_info == {
        "tasks_requested": 2,
        "samples_attempted": 4,
        "samples_kept": 4,
        "samples_dropped_fidelity": 0,
    }

    groups = env.rollout(tasks, _ScriptedPolicy([["a", ""], ["c", "d"]]), group_size=2)
    assert [len(group) for group in groups] == [1, 2]
    assert env.last_rollout_info == {
        "tasks_requested": 2,
        "samples_attempted": 4,
        "samples_kept": 3,
        "samples_dropped_fidelity": 1,
    }

    groups = env.rollout(tasks, _ScriptedPolicy([["", ""], ["", ""]]), group_size=2)
    assert groups == [[], []]
    assert env.last_rollout_info == {
        "tasks_requested": 2,
        "samples_attempted": 4,
        "samples_kept": 0,
        "samples_dropped_fidelity": 4,
    }


def test_single_turn_empty_rollout_has_zero_counters():
    env = _Source()
    assert env.rollout([], _ScriptedPolicy([]), group_size=2) == []
    assert env.last_rollout_info == {
        "tasks_requested": 0,
        "samples_attempted": 0,
        "samples_kept": 0,
        "samples_dropped_fidelity": 0,
    }


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        ([['a', 'b']], "returned 1 result groups for 2 requests"),
        ([['a', 'b'], ['c', 'd'], ['e', 'f']], "returned 3 result groups for 2 requests"),
        ([[], ['c', 'd']], "request 0 returned 0 samples; expected exactly 2"),
        ([['a', 'b', 'x'], ['c', 'd']], "request 0 returned 3 samples; expected exactly 2"),
    ],
)
def test_single_turn_rejects_malformed_predict_shapes(stage, message):
    env = _Source()
    with pytest.raises(RuntimeError, match=message):
        env.rollout(env.tasks(2), _ScriptedPolicy(stage), group_size=2)
    assert env.last_rollout_records == []
    assert env.last_rollout_info == {
        "tasks_requested": 2,
        "samples_attempted": 0,
        "samples_kept": 0,
        "samples_dropped_fidelity": 0,
    }


def test_single_turn_generation_failure_clears_previous_observations():
    env = _Source()
    env.rollout(env.tasks(1), _ScriptedPolicy([["a"]]), group_size=1)
    assert env.last_rollout_records

    with pytest.raises(RuntimeError, match="generation exploded"):
        env.rollout(env.tasks(2), _ExplodingPolicy(), group_size=1)
    assert env.last_rollout_records == []
    assert env.last_rollout_info == {
        "tasks_requested": 2,
        "samples_attempted": 0,
        "samples_kept": 0,
        "samples_dropped_fidelity": 0,
    }


def test_planned_coverage_normal_and_partial_drop():
    env = PlannedEnv(_Source(), plan_max_tokens=8)
    tasks = env.tasks(1)

    groups = env.rollout(tasks, _ScriptedPolicy([["p1", "p2"]], [["a"], ["b"]]), 2)
    assert [len(group) for group in groups] == [2]
    assert env.last_rollout_info == {
        "tasks_requested": 1,
        "plan_samples_attempted": 2,
        "plan_samples_kept": 2,
        "plan_samples_dropped_fidelity": 0,
        "answer_requests": 2,
        "answer_samples_attempted": 2,
        "answer_samples_kept": 2,
        "answer_samples_dropped_fidelity": 0,
        "samples_dropped_fidelity": 0,
    }

    groups = env.rollout(
        tasks,
        _ScriptedPolicy([["p1", "", "p3"]], [["a"], [""]]),
        group_size=3,
    )
    assert [len(group) for group in groups] == [1]
    assert env.last_rollout_info == {
        "tasks_requested": 1,
        "plan_samples_attempted": 3,
        "plan_samples_kept": 2,
        "plan_samples_dropped_fidelity": 1,
        "answer_requests": 2,
        "answer_samples_attempted": 2,
        "answer_samples_kept": 1,
        "answer_samples_dropped_fidelity": 1,
        "samples_dropped_fidelity": 2,
    }


def test_planned_all_plan_drops_preserve_zero_answer_counters():
    env = PlannedEnv(_Source(), plan_max_tokens=8)
    groups = env.rollout(env.tasks(1), _ScriptedPolicy([["", ""]]), group_size=2)

    assert groups == [[]]
    assert env.last_rollout_info == {
        "tasks_requested": 1,
        "plan_samples_attempted": 2,
        "plan_samples_kept": 0,
        "plan_samples_dropped_fidelity": 2,
        "answer_requests": 0,
        "answer_samples_attempted": 0,
        "answer_samples_kept": 0,
        "answer_samples_dropped_fidelity": 0,
        "samples_dropped_fidelity": 2,
    }


def test_planned_empty_rollout_has_zero_counters():
    env = PlannedEnv(_Source(), plan_max_tokens=8)
    assert env.rollout([], _ScriptedPolicy([]), group_size=2) == []
    assert env.last_rollout_info == {
        "tasks_requested": 0,
        "plan_samples_attempted": 0,
        "plan_samples_kept": 0,
        "plan_samples_dropped_fidelity": 0,
        "answer_requests": 0,
        "answer_samples_attempted": 0,
        "answer_samples_kept": 0,
        "answer_samples_dropped_fidelity": 0,
        "samples_dropped_fidelity": 0,
    }


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        ([['p1', 'p2']], "returned 1 result groups for 2 requests"),
        (
            [['p1', 'p2'], ['p3', 'p4'], ['p5', 'p6']],
            "returned 3 result groups for 2 requests",
        ),
        ([[], ['p3', 'p4']], "request 0 returned 0 samples; expected exactly 2"),
        (
            [['p1', 'p2', 'extra'], ['p3', 'p4']],
            "request 0 returned 3 samples; expected exactly 2",
        ),
    ],
)
def test_planned_rejects_malformed_plan_shapes(stage, message):
    env = PlannedEnv(_Source(), plan_max_tokens=8)
    with pytest.raises(RuntimeError, match=message):
        env.rollout(env.tasks(2), _ScriptedPolicy(stage), group_size=2)
    assert env.last_rollout_records == []
    assert env.last_rollout_info["plan_samples_attempted"] == 0
    assert env.last_rollout_info["answer_requests"] == 0


@pytest.mark.parametrize(
    ("answer_stage", "message"),
    [
        ([], "returned 0 result groups for 1 requests"),
        ([['a'], ['extra']], "returned 2 result groups for 1 requests"),
        ([[]], "request 0 returned 0 samples; expected exactly 1"),
        ([['a', 'extra']], "request 0 returned 2 samples; expected exactly 1"),
    ],
)
def test_planned_rejects_malformed_answer_shapes(answer_stage, message):
    env = PlannedEnv(_Source(), plan_max_tokens=8)
    with pytest.raises(RuntimeError, match=message):
        env.rollout(env.tasks(1), _ScriptedPolicy([["p1"]], answer_stage), group_size=1)
    assert env.last_rollout_records == []
    assert env.last_rollout_info["plan_samples_attempted"] == 1
    assert env.last_rollout_info["plan_samples_kept"] == 1
    assert env.last_rollout_info["answer_requests"] == 1
    assert env.last_rollout_info["answer_samples_attempted"] == 0


def test_planned_generation_failures_do_not_leave_stale_observations():
    env = PlannedEnv(_Source(), plan_max_tokens=8)
    env.rollout(env.tasks(1), _ScriptedPolicy([["p1"]], [["a"]]), group_size=1)
    assert env.last_rollout_records

    with pytest.raises(RuntimeError, match="generation exploded"):
        env.rollout(env.tasks(2), _ExplodingPolicy(), group_size=1)
    assert env.last_rollout_records == []
    assert env.last_rollout_info["tasks_requested"] == 2
    assert env.last_rollout_info["plan_samples_attempted"] == 0
    assert env.last_rollout_info["answer_requests"] == 0

    with pytest.raises(RuntimeError, match="answer generation exploded"):
        env.rollout(env.tasks(1), _AnswerExplodingPolicy([["p1"]]), group_size=1)
    assert env.last_rollout_records == []
    assert env.last_rollout_info["tasks_requested"] == 1
    assert env.last_rollout_info["plan_samples_attempted"] == 1
    assert env.last_rollout_info["plan_samples_kept"] == 1
    assert env.last_rollout_info["answer_requests"] == 1
    assert env.last_rollout_info["answer_samples_attempted"] == 0


def test_planned_all_answer_drops_preserve_counters_without_trajectories():
    env = PlannedEnv(_Source(), plan_max_tokens=8)
    groups = env.rollout(
        env.tasks(1), _ScriptedPolicy([["p1", "p2"]], [[""], [""]]), group_size=2
    )

    assert groups == [[]]
    assert env.last_rollout_info == {
        "tasks_requested": 1,
        "plan_samples_attempted": 2,
        "plan_samples_kept": 2,
        "plan_samples_dropped_fidelity": 0,
        "answer_requests": 2,
        "answer_samples_attempted": 2,
        "answer_samples_kept": 0,
        "answer_samples_dropped_fidelity": 2,
        "samples_dropped_fidelity": 2,
    }


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([['a']], "returned 1 result groups for 2 requests"),
        ([['a'], ['b'], ['extra']], "returned 3 result groups for 2 requests"),
        ([[], ['b']], "request 0 returned 0 samples; expected exactly 1"),
        ([['a', 'extra'], ['b']], "request 0 returned 2 samples; expected exactly 1"),
    ],
)
def test_trained_debate_seat_rejects_malformed_predict_shapes(rows, message):
    seat = PolicySeat(_ShapePolicy(_nested_samples(rows)))
    with pytest.raises(RuntimeError, match=message):
        seat.generate(_requests(2))


class _RoundPrompts:
    def system(self, speaker, bindings):
        return "system"

    def instruction(self, slot_name, speaker, bindings):
        return "answer"

    def preamble_messages(self, speaker, bindings):
        return []

    def attributed(self, author_name, slot_name, text, **kwargs):
        return text


_RETRY_PROTOCOL = Protocol.parse(
    {"turns": [{"alice": [{"name": "answer", "kind": "solution"}]}]}
)


def _retry_states(n=2):
    return [DebateState(bindings={"alice": {"NAME": "Alice"}}) for _ in range(n)]


def _retry_round(seat):
    return DebateRound(
        _RETRY_PROTOCOL,
        {"alice": seat},
        _RoundPrompts(),
        answer_extractor=lambda text: (
            text if text == "valid" else None,
            text == "valid",
        ),
        solution_retries=1,
        solution_retry_feedback=lambda text, attempt: f"retry {attempt}",
    )


@pytest.mark.parametrize(
    ("retry_rows", "message"),
    [
        ([['valid']], "returned 1 result groups for 2 requests"),
        (
            [['valid'], ['valid'], ['extra']],
            "returned 3 result groups for 2 requests",
        ),
        ([[], ['valid']], "request 0 returned 0 samples; expected exactly 1"),
        (
            [['valid', 'extra'], ['valid']],
            "request 0 returned 2 samples; expected exactly 1",
        ),
    ],
)
def test_trained_debate_retry_rejects_malformed_predict_shapes(retry_rows, message):
    policy = _ShapePolicy(
        _nested_samples([['bad'], ['bad']]),
        _nested_samples(retry_rows),
    )
    with pytest.raises(RuntimeError, match=message):
        _retry_round(PolicySeat(policy)).run(_retry_states())


class _FlatRetrySeat:
    def __init__(self, retry_results):
        self.stages = [
            [SlotResult(text="bad"), SlotResult(text="bad")],
            retry_results,
        ]

    def generate(self, requests):
        return self.stages.pop(0)


@pytest.mark.parametrize("retry_results", [[SlotResult(text="valid")], [SlotResult(text="valid")] * 3])
def test_debate_round_rejects_retry_result_count_before_zip(retry_results):
    with pytest.raises(RuntimeError, match=r"retry generation returned [13] results for 2 requests"):
        _retry_round(_FlatRetrySeat(retry_results)).run(_retry_states())


class _CharTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


_CHAR_TOKENIZER = _CharTokenizer()


def _phase_sample(text="thought", stop_reason="length"):
    tokens = _CHAR_TOKENIZER.encode(text)
    return Sample(
        tokens=tokens,
        logprobs=[-0.1] * len(tokens),
        text=text,
        stop_reason=stop_reason,
    )


def _run_two_phase(*backend_stages):
    stages = list(backend_stages)

    def sample_fn(prompts, params, n=1):
        return stages.pop(0)

    return budget_forced_sample(
        sample_fn,
        _CHAR_TOKENIZER,
        [_CHAR_TOKENIZER.encode("<think>")] * 2,
        SamplingParams(max_tokens=40),
        SlotLimits(max_think_tokens=10, max_visible_tokens=10, max_total_tokens=40),
    )


@pytest.mark.parametrize(
    ("phase1", "message"),
    [
        ([[_phase_sample()]], "returned 1 result groups for 2 requests"),
        ([[_phase_sample()]] * 3, "returned 3 result groups for 2 requests"),
        ([[], [_phase_sample()]], "request 0 returned 0 samples; expected exactly 1"),
        (
            [[_phase_sample(), _phase_sample()], [_phase_sample()]],
            "request 0 returned 2 samples; expected exactly 1",
        ),
    ],
)
def test_two_phase_rejects_malformed_phase1_shapes(phase1, message):
    with pytest.raises(RuntimeError, match=message):
        _run_two_phase(phase1)


@pytest.mark.parametrize(
    ("phase2", "message"),
    [
        ([[_phase_sample("a", "stop")]], "returned 1 result groups for 2 requests"),
        (
            [[_phase_sample("a", "stop")]] * 3,
            "returned 3 result groups for 2 requests",
        ),
        (
            [[], [_phase_sample("b", "stop")]],
            "request 0 returned 0 samples; expected exactly 1",
        ),
        (
            [
                [_phase_sample("a", "stop"), _phase_sample("extra", "stop")],
                [_phase_sample("b", "stop")],
            ],
            "request 0 returned 2 samples; expected exactly 1",
        ),
    ],
)
def test_two_phase_rejects_malformed_phase2_shapes(phase2, message):
    phase1 = [[_phase_sample()], [_phase_sample()]]
    with pytest.raises(RuntimeError, match=message):
        _run_two_phase(phase1, phase2)
