"""Plan-then-answer rollouts (infra/envs/planned.py) and the `plan` task-prompt
key: the RLVR twin of the debate protocols' pre-proposal plan slot.

Covers the regression that motivated it: alice's turn-0 private slot rendered
with no problem statement anywhere in context (the proposer has no pre_debate
preamble, and the generic scratchpad cue carries no <TOPIC>), so the model
literally asked to be given the problem. The plan slot's cue is now the task
family's `plan` template, spliced like <ANSWER_GEN_USER> — both arms render
the same bytes, and both carry the problem."""

from __future__ import annotations

import pytest
import yaml

from infra.backend.base import Sample
from infra.envs.base import SingleTurnEnv, Task
from infra.envs.planned import PlannedEnv
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file

PROBLEM = "What is $1 + 1$?"


# ------------------------------------------------------------ fakes


class FakePolicy:
    """Duck-types Policy.predict: echoes a canned text per call, recording the
    conversations it was asked to continue."""

    def __init__(self, texts):
        self.texts = list(texts)  # one per predict() call
        self.calls = []           # (convos, n, limits)

    def predict(self, convos, n=1, limits=None):
        self.calls.append((convos, n, limits))
        text = self.texts.pop(0)
        return [
            [
                Sample(
                    tokens=[1, 2, 3],
                    logprobs=[0.0, 0.0, 0.0],
                    text=text,
                    stop_reason="stop",
                    prompt_tokens=[7, 8],
                )
                for _ in range(n)
            ]
            for _ in convos
        ]


class FakeMathSource(SingleTurnEnv):
    prompts = load_generation_prompts(resolve_prompt_file(None, "math.yaml"))

    def tasks(self, n, split="train"):
        return [
            Task(
                messages=self.prompts.render({"PROBLEM": PROBLEM}),
                meta={"question": PROBLEM, "gt": 2.0},
            )
            for _ in range(n)
        ]

    def reward(self, task, text):
        return (1.0 if "2" in text else 0.0), {"correct": float("2" in text)}


# ------------------------------------------------------- loader: `plan` key


def test_math_yaml_supplies_plan_template():
    p = load_generation_prompts(resolve_prompt_file(None, "math.yaml"))
    plan = p.supplied_templates()["PLAN_USER"]
    assert "<PROBLEM>" in plan
    assert "PRIVATE scratchpad" in plan
    # not part of the solo context: plain single-turn arms are unchanged
    assert all(plan != m["content"] for m in p.messages)


def test_nocot_plan_is_neutral():
    plan = load_generation_prompts(
        resolve_prompt_file(None, "math_nocot.yaml")
    ).supplied_templates()["PLAN_USER"]
    assert "<PROBLEM>" in plan
    assert "compute" not in plan.lower()  # no reasoning coaching in the nocot arm


def test_plan_without_problem_placeholder_rejected(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "messages": [{"role": "user", "content": "<PROBLEM>"}],
                "plan": "no placeholder here",
            }
        )
    )
    with pytest.raises(ValueError, match="<PROBLEM>"):
        load_generation_prompts(path)


def test_plan_as_message_name_rejected(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "messages": [
                    {"role": "user", "content": "<PROBLEM>", "name": "PLAN_USER"}
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="top-level `plan` key"):
        load_generation_prompts(path)


# --------------------------------------------------------- PlannedEnv rollout


def test_two_turn_rollout_shape():
    env = PlannedEnv(FakeMathSource(), plan_max_tokens=1000)
    policy = FakePolicy(texts=["my plan", "\\boxed{2}"])
    tasks = env.tasks(1)
    groups = env.rollout(tasks, policy, group_size=1)

    plan_call, answer_call = policy.calls
    plan_convo = plan_call[0][0]
    # turn 1: the task's system message + the plan cue — the problem IS there
    assert [m["role"] for m in plan_convo] == ["system", "user"]
    assert PROBLEM in plan_convo[1]["content"]
    assert plan_call[2].max_total_tokens == 1000
    # turn 2: turn 1 + the plan as an assistant turn + the task's own
    # eliciting message, verbatim (the single-sourced ANSWER_GEN_USER)
    answer_convo = answer_call[0][0]
    assert answer_convo[:2] == plan_convo
    assert answer_convo[2] == {"role": "assistant", "content": "my plan"}
    assert answer_convo[3] == tasks[0].messages[1]

    (traj,) = groups[0]
    assert len(traj.datums) == 2  # plan turn + answer turn, both trained
    assert traj.reward == 1.0
    assert env.last_rollout_records[0]["completion"] == "\\boxed{2}"
    assert env.last_rollout_records[0]["messages"] == answer_convo


def test_inner_without_plan_template_rejected(tmp_path):
    path = tmp_path / "noplan.yaml"
    path.write_text(yaml.safe_dump({"messages": [{"role": "user", "content": "<PROBLEM>"}]}))

    class NoPlanSource(FakeMathSource):
        prompts = load_generation_prompts(path)

    with pytest.raises(ValueError, match="plan"):
        PlannedEnv(NoPlanSource(), plan_max_tokens=1000)


# ------------------------------------- cross-arm byte identity + the regression


def _debate_env(protocol_turns):
    from infra.envs.debate.env import DebateEnv, DebateEnvConfig
    from infra.envs.debate.judge import JudgeConfig
    from infra.envs.debate.protocol import Protocol

    return DebateEnv(
        DebateEnvConfig(
            protocol=Protocol.parse({"turns": protocol_turns}),
            prompt_file="infra/prompts/debate/hendrycks_math.yaml",
            prompt_entry="math_proposer_critic",
            trained_speakers=["alice"],
            frozen_models={"bob": object(), "judge": object()},
            judge=JudgeConfig(),
            fresh_positions=True,
        ),
        FakeMathSource(),
        __import__("infra.envs.tasks.math", fromlist=["MathFamily"]).MathFamily(),
    )


PLAN_PROTOCOL = [
    {"alice": [{"name": "plan", "visibility": "private"},
               {"name": "proposal", "kind": "solution"}]},
    {"bob": [{"name": "critique"}]},
    {"judge": [{"name": "deliberation", "visibility": "private"},
               {"name": "verdict", "kind": "decision"}]},
]


def test_debate_plan_cue_equals_rlvr_plan_turn():
    """The invariant the splice exists for, plan side: the rendered debate plan
    cue and the RLVR plan turn's user message are the same bytes."""
    debate = _debate_env(PLAN_PROTOCOL)
    rendered = debate.prompts.instruction("plan", "alice", {"TOPIC": PROBLEM})
    rlvr = PlannedEnv(FakeMathSource(), plan_max_tokens=1000)
    assert rendered == rlvr._plan_cue(rlvr.tasks(1)[0])


def test_alice_plan_slot_context_contains_the_problem():
    """The original bug: alice's turn-0 private slot had no problem statement
    anywhere in its context (no proposer preamble + a problem-free generic
    scratchpad cue), so she asked to be given the problem."""
    from infra.envs.debate.round import render_context

    debate = _debate_env(PLAN_PROTOCOL)
    state = debate._build_state(debate.tasks(1)[0], flipped=False)
    slot0 = debate.protocol.compile()[0]
    assert slot0.slot.name == "plan" and slot0.speaker == "alice"
    messages = render_context(state, slot0, debate.prompts)
    assert any(PROBLEM in m["content"] for m in messages)
