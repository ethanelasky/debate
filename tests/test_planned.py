"""Plan-then-answer rollouts (infra/envs/planned.py) and the `plan` task-prompt
key: the RLVR twin of the debate protocols' pre-proposal plan slot.

What these pin, and why:

- Behavior through the REAL Policy/backend seam (not a fake that mirrors the
  wrapper): the plan cap actually reaches the sampler, the answer turn's
  prompt actually contains the plan, datums carry the right prompt boundaries,
  fidelity drops don't derail the batch, reward comes from the answer alone.
- The SHIPPED CONFIGS, entry by entry: every seat's first-slot context
  contains the problem. This is the regression that motivated the plan slot —
  alice's turn-0 private scratchpad rendered with no problem anywhere in
  context, and no test failed, because tests only exercised toy protocols.
  Renaming the config slot back to `scratchpad` fails these.
- Cross-arm byte identity per config: the debate plan cue and the RLVR plan
  turn's user message are the same bytes, and the config's plan_tokens equals
  the plan slot's cap (the eval arm samples plans under the trained budget).
"""

from __future__ import annotations

import warnings

import pytest
import yaml

from test_single_turn_env import ScriptedBackend

from infra.backend.base import SamplingParams
from infra.config import load_experiment
from infra.envs.base import Policy, SingleTurnEnv, Task
from infra.envs.planned import PlannedEnv
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file

PROBLEM = "What is $1 + 1$?"


# ------------------------------------------------------------ scaffolding


class RecordingBackend(ScriptedBackend):
    """ScriptedBackend that records what each sample() call was asked for."""

    def __init__(self, script):
        super().__init__(script)
        self.calls = []  # (decoded prompts, params.max_tokens, n)

    def sample(self, prompts, params, n=1):
        self.calls.append(([self.tokenizer.decode(p) for p in prompts], params.max_tokens, n))
        return super().sample(prompts, params, n)


def _tiny_prompts(tmp_path):
    """A minimal task-prompt config, kept short so FakeTokenizer's 512-char
    chat-template window holds the whole context and prompts decode verbatim."""
    path = tmp_path / "tiny.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "messages": [
                    {"role": "system", "content": "SYS"},
                    {"role": "user", "name": "ANSWER_GEN_USER", "content": "Solve: <PROBLEM>"},
                ],
                "plan": "Plan privately: <PROBLEM>",
            }
        )
    )
    return load_generation_prompts(path)


class TinySource(SingleTurnEnv):
    """reward = 1.0 iff the ANSWER contains GOOD (never reads the plan)."""

    def __init__(self, prompts):
        self.prompts = prompts

    def tasks(self, n, split="train"):
        return [
            Task(
                messages=self.prompts.render({"PROBLEM": f"q{i}"}),
                meta={"question": f"q{i}"},
            )
            for i in range(n)
        ]

    def reward(self, task, text):
        return (1.0 if "GOOD" in text else 0.0), {"correct": float("GOOD" in text)}


def _run(tmp_path, script, *, n_tasks=1, group_size=1, plan_max=1000, ceiling=2000):
    env = PlannedEnv(TinySource(_tiny_prompts(tmp_path)), plan_max_tokens=plan_max)
    backend = RecordingBackend(script)
    policy = Policy(backend, SamplingParams(max_tokens=ceiling))
    groups = env.rollout(env.tasks(n_tasks), policy, group_size)
    return env, backend, groups


# ------------------------------------------------------- loader: `plan` key


def test_plan_is_not_part_of_the_solo_context(tmp_path):
    """The plain single-turn arm must be byte-unchanged by the plan key."""
    p = _tiny_prompts(tmp_path)
    assert "PLAN_USER" in p.supplied_templates()
    rendered = p.render({"PROBLEM": "x"})
    assert all("Plan privately" not in m["content"] for m in rendered)


def test_plan_without_problem_placeholder_rejected(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(
        yaml.safe_dump(
            {"messages": [{"role": "user", "content": "<PROBLEM>"}], "plan": "no tag"}
        )
    )
    with pytest.raises(ValueError, match="<PROBLEM>"):
        load_generation_prompts(path)


def test_plan_as_message_name_rejected(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(
        yaml.safe_dump(
            {"messages": [{"role": "user", "content": "<PROBLEM>", "name": "PLAN_USER"}]}
        )
    )
    with pytest.raises(ValueError, match="top-level `plan` key"):
        load_generation_prompts(path)


def test_inner_without_plan_template_rejected(tmp_path):
    path = tmp_path / "noplan.yaml"
    path.write_text(yaml.safe_dump({"messages": [{"role": "user", "content": "<PROBLEM>"}]}))
    with pytest.raises(ValueError, match="plan"):
        PlannedEnv(TinySource(load_generation_prompts(path)), plan_max_tokens=1000)


# ---------------------------------------- rollout behavior through real Policy


def test_plan_cap_reaches_the_sampler(tmp_path):
    _, backend, _ = _run(tmp_path, ["thoughts", "GOOD"])
    (_, plan_max, plan_n), (_, ans_max, _) = backend.calls
    assert plan_max == 1000  # SlotLimits cap, not the 2000 ceiling
    assert ans_max == 2000   # answer turn runs under the policy ceiling


def test_answer_prompt_carries_plan_cue_plan_and_elicit(tmp_path):
    _, backend, _ = _run(tmp_path, ["thoughts", "GOOD"])
    (plan_prompts, _, _), (ans_prompts, _, _) = backend.calls
    # turn 1: system + plan cue with the problem; the eliciting message is NOT
    # there (the answer must not be producible from the plan turn)
    assert "[system] SYS" in plan_prompts[0]
    assert "Plan privately: q0" in plan_prompts[0]
    assert "Solve: q0" not in plan_prompts[0]
    # turn 2: turn 1 + the plan as an assistant turn + ANSWER_GEN_USER verbatim
    assert plan_prompts[0] in ans_prompts[0]
    assert "[assistant] thoughts" in ans_prompts[0]
    assert ans_prompts[0].index("[assistant] thoughts") < ans_prompts[0].index("Solve: q0")


def test_datums_have_per_turn_prompt_boundaries(tmp_path):
    _, backend, groups = _run(tmp_path, ["thoughts", "GOOD"])
    (traj,) = groups[0]
    plan_datum, ans_datum = traj.datums
    tok = backend.tokenizer
    assert tok.decode(plan_datum.tokens[plan_datum.prompt_len :]) == "thoughts"
    assert tok.decode(ans_datum.tokens[ans_datum.prompt_len :]) == "GOOD"
    # the answer datum's prompt region includes the plan it conditioned on
    assert "thoughts" in tok.decode(ans_datum.tokens[: ans_datum.prompt_len])
    assert len(ans_datum.advantages) == len(ans_datum.tokens) - ans_datum.prompt_len


def test_reward_reads_the_answer_not_the_plan(tmp_path):
    # GOOD appears only in the plan: the answer is what gets scored
    _, _, groups = _run(tmp_path, ["GOOD plan", "bad answer"])
    assert groups[0][0].reward == 0.0
    assert groups[0][0].info["correct"] == 0.0


def test_groups_and_per_sample_plans(tmp_path):
    # 2 tasks x group 2: each answer continues ITS OWN plan, groups stay per task
    script = ["pA", "pB", "pC", "pD", "GOOD", "GOOD", "GOOD", "GOOD"]
    _, backend, groups = _run(tmp_path, script, n_tasks=2, group_size=2)
    (_, _, plan_n) = backend.calls[0]
    assert plan_n == 2
    ans_prompts = backend.calls[1][0]
    assert [p.split("[assistant] ")[1].split("\n")[0] for p in ans_prompts] == ["pA", "pB", "pC", "pD"]
    assert ["q0" in p for p in ans_prompts] == [True, True, False, False]
    assert [len(g) for g in groups] == [2, 2]


def test_plan_fidelity_failure_drops_the_chain(tmp_path):
    # "" plan = zero tokens = fidelity failure: no answer sampled for it
    _, backend, groups = _run(tmp_path, ["", "ok plan", "GOOD"], group_size=2)
    assert len(backend.calls[1][0]) == 1  # one answer convo, not two
    assert len(groups[0]) == 1
    assert groups[0][0].info["samples_dropped_fidelity"] == 1.0


def test_answer_fidelity_failure_drops_the_trajectory(tmp_path):
    _, _, groups = _run(tmp_path, ["ok plan", "good plan", "", "GOOD"], group_size=2)
    assert len(groups[0]) == 1
    assert groups[0][0].reward == 1.0


# --------------------------------------------- the shipped configs, entry by entry

MATH_PC_ENTRIES = [
    "math_pc_olmo_l5",
    "math_pc_olmo_l5_g8",
    "math_pc_olmo_l5_g8_nocot",
    "math_pc_olmo_smoke",
]


def _math_source(exp):
    prompts = load_generation_prompts(
        resolve_prompt_file((exp.get("dataset") or {}).get("prompt_file"), "math.yaml")
    )
    src = TinySource(prompts)
    src.tasks = lambda n, split="train": [
        Task(messages=prompts.render({"PROBLEM": PROBLEM}), meta={"question": PROBLEM})
        for _ in range(n)
    ]
    return src


def _debate_env(exp):
    from infra.envs.debate.env import DebateEnv, DebateEnvConfig
    from infra.envs.debate.judge import JudgeConfig
    from infra.envs.debate.protocol import Protocol
    from infra.envs.tasks.math import MathFamily

    protocol = Protocol.parse(exp["protocol"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DebateEnv(
            DebateEnvConfig(
                protocol=protocol,
                prompt_file=exp["prompt_config"]["file_path"],
                prompt_entry=exp["prompt_config"]["entry"],
                trained_speakers=[],
                frozen_models={s: object() for s in protocol.speakers},
                judge=JudgeConfig(),
                fresh_positions=exp.get("fresh_positions", True),
            ),
            _math_source(exp),
            MathFamily(),
        )


@pytest.fixture(params=MATH_PC_ENTRIES)
def exp(request):
    return load_experiment("configs/math_pc_olmo.yaml", request.param)


def test_every_seat_first_slot_context_contains_the_problem(exp):
    """THE regression: alice's first slot is generated before anything is
    public, so unless its own cue/preamble carries the problem, she is asked
    to plan a debate she cannot see. Checked for every seat, on the protocol
    each config actually ships."""
    from infra.envs.debate.round import SlotRecord, render_context

    env = _debate_env(exp)
    state = env._build_state(env.tasks(1)[0], flipped=False)
    slots = env.protocol.compile()
    first_by_speaker = {}
    for cs in slots:
        first_by_speaker.setdefault(cs.speaker, cs)
    for speaker, cs in first_by_speaker.items():
        state.records = [
            SlotRecord(slot=prior, text=f"<{prior.speaker}/{prior.slot.name}>")
            for prior in slots
            if prior.index < cs.index
        ]
        messages = render_context(state, cs, env.prompts)
        assert any(PROBLEM in m["content"] for m in messages), (
            f"{exp}: no problem statement anywhere in {speaker}'s first-slot "
            f"({cs.slot.name}) context"
        )


def test_debate_plan_cue_equals_rlvr_plan_turn(exp):
    """Byte identity across arms, per config: the rendered debate plan cue is
    the RLVR plan turn's user message, for the prompt_file the config uses."""
    env = _debate_env(exp)
    slot0 = env.protocol.compile()[0]
    assert (slot0.speaker, slot0.slot.name) == ("alice", "plan")
    rendered = env.prompts.instruction("plan", "alice", {"TOPIC": PROBLEM})
    rlvr = PlannedEnv(_math_source(exp), plan_max_tokens=1000)
    assert rendered == rlvr._plan_cue(rlvr.tasks(1)[0])


def test_plan_tokens_matches_the_plan_slot_cap(exp):
    """run_debate wraps the RLVR eval with exp['plan_tokens']; if it drifts
    from the protocol's plan-slot cap, eval plans sample under a different
    budget than trained plans."""
    from infra.envs.debate.protocol import Protocol

    slot0 = Protocol.parse(exp["protocol"]).compile()[0]
    assert exp.get("plan_tokens") == slot0.slot.max_total_tokens
