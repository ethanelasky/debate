"""End-to-end DebateEnv rollout with scripted seats (no network, no GPUs).

Alice (trained, via a fake Backend) proposes and defends; Bob (frozen, scripted)
critiques; a scripted judge emits valid competitive JSON. Checks trajectories,
rewards, datums, groups, and blindness-sensitive context rendering.
"""

import threading
import time

import pytest

from dataclasses import replace as _replace

from infra.backend.base import Backend, Sample, SamplingParams
from infra.envs.base import Policy, Task
from infra.envs.debate.env import DebateEnv, DebateEnvConfig
from infra.envs.debate.judge import JudgeConfig
from infra.envs.debate.rewards import ScoringConfig
from infra.envs.debate.protocol import Protocol
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file
from infra.envs.tasks.base import AnswerParse
from infra.envs.tasks.math import MathFamily
from infra.models.base import Model, ModelResponse

import yaml

PROTOCOL = Protocol.parse(
    yaml.safe_load(
        """
turns:
  - alice: [{name: proposal, kind: solution}]
  - bob:   [{name: critique}]
  - alice: [{name: alice_rebuttal}]
  - judge: [{name: verdict, kind: decision}]
"""
    )
)

PROMPT_FILE = "infra/prompts/debate/hendrycks_math.yaml"


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kw):
        text = "\n".join(f"[{m['role']}] {m['content']}" for m in messages)
        return [ord(c) % 4096 for c in text][-512:] or [1]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 4096 for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class ScriptedBackend(Backend):
    """Returns scripted texts per call in order; tokens are codepoints so the
    decode-roundtrip fidelity property holds."""

    def __init__(self, script):
        self.tokenizer = FakeTokenizer()
        self.script = list(script)
        self.contexts: list[str] = []  # decoded prompts seen, for blindness asserts

    def sync_sampler(self):
        pass

    def sample(self, prompts, params, n=1):
        out = []
        for p in prompts:
            self.contexts.append(self.tokenizer.decode(p))
            text = self.script.pop(0)
            toks = self.tokenizer.encode(text)
            out.append(
                [Sample(tokens=toks, logprobs=[-0.2] * len(toks), text=text, stop_reason="stop")]
                * n
            )
        return out

    def forward(self, data):
        raise NotImplementedError

    def forward_backward(self, data, loss):
        raise NotImplementedError

    def optim_step(self, params):
        raise NotImplementedError

    def save(self, name):
        return name

    def load(self, path):
        pass


class ScriptedModel(Model):
    def __init__(self, alias, script):
        super().__init__(alias=alias)
        self.script = list(script)

    def predict(self, inputs, max_new_tokens=1000, num_return_sequences=1, **kw):
        out = []
        for _ in inputs:
            text = self.script.pop(0)
            out.append(ModelResponse(speech=text, raw_response=text, stop_reason="stop"))
        return out


class TaskSource:
    # Real task envs expose their answer-generation prompts; DebateEnv splices
    # them into slots that ask for them (the math pack's proposal slot does).
    prompts = load_generation_prompts(resolve_prompt_file(None, "math.yaml"))

    def tasks(self, n, split="train"):
        return [
            Task(
                messages=[{"role": "user", "content": f"What is {i}+{i}?"}],
                meta={"question": f"What is {i}+{i}?", "gt": float(2 * i)},
            )
            for i in range(1, n + 1)
        ]


def make_env(trained, judge_script, scoring=None, task_source=None, family=None):
    return DebateEnv(
        DebateEnvConfig(
            protocol=PROTOCOL,
            prompt_file=PROMPT_FILE,
            prompt_entry="math_proposer_critic",
            trained_speakers=trained,
            frozen_models={
                "bob": ScriptedModel("bob", ["Your step 2 is wrong."] * 64),
                "judge": ScriptedModel("judge", judge_script),
            },
            judge=JudgeConfig(),
            scoring=scoring or ScoringConfig(),
            fresh_positions=True,
        ),
        task_source if task_source is not None else TaskSource(),
        family if family is not None else MathFamily(),
    )


GOOD_VERDICT = '{"winner": "Debater_A", "confidence": 0.8}'
BAD_VERDICT = '{"winner": "Debater_B", "confidence": 0.7}'


def test_full_rollout_trained_alice():
    # slot-batched order: BOTH proposals first, then both defenses
    backend = ScriptedBackend(
        ["I compute. \\boxed{2}", "I compute. \\boxed{4}", "My defense stands.", "Still right."]
    )
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_env(["alice"], [GOOD_VERDICT, BAD_VERDICT])
    groups = env.rollout(TaskSource().tasks(2), policy, group_size=1)

    assert len(groups) == 2
    (t1,), (t2,) = groups
    assert t1.reward == 1.0 and t2.reward == -1.0        # ladder solo_win / solo_loss
    assert len(t1.datums) == 2                           # proposal + defense
    assert t1.info["solution_correct"] == 1.0
    assert t1.info["judge_conf_json"] == 0.8
    assert all(d.prompt_len > 0 for d in t1.datums)

    # blindness/context checks on what alice's trained backend actually saw:
    proposal_ctx, defense_ctx = backend.contexts[0], backend.contexts[2]
    assert "step 2 is wrong" not in proposal_ctx         # bob's critique is later
    assert "step 2 is wrong" in defense_ctx              # defense sees the critique
    assert "\\boxed{2}" in defense_ctx                   # own earlier speech present


def test_preamble_carries_the_problem_to_critic_and_judge_not_the_system_card():
    """Old-repo layout (Ethan, 2026-08-03): the problem reaches the critic via
    pre_debate and the judge via pre_debate_judge — a preamble on their FIRST
    user message — not via the debater system cards, which are now pure role
    material. The proposer gets NO preamble: its opening cue is the RLVR
    answer-generation message and must stay byte-identical to it."""
    backend = ScriptedBackend(["Thinking. \\boxed{7}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_env(["alice"], [GOOD_VERDICT])
    problem = "What is 1+1?"

    seen: dict[str, list[list]] = {"bob": [], "judge": []}
    for name in ("bob", "judge"):
        model = env.config.frozen_models[name]
        model.predict = (
            lambda inputs, _n=name, _orig=model.predict, **kw: (
                seen[_n].append([{"role": m.role.name.lower(), "content": m.content} for m in inputs[0]]),
                _orig(inputs, **kw),
            )[1]
        )
    env.rollout(TaskSource().tasks(1), policy, group_size=1)

    # no debater system card states the problem any more
    for speaker in ("alice", "bob", "judge"):
        assert problem not in env.prompts.system(speaker, env._build_state(
            TaskSource().tasks(1)[0], flipped=False
        ).bindings[speaker])

    for name in ("bob", "judge"):
        msgs = seen[name][0]
        first_user = next(m["content"] for m in msgs if m["role"] == "user")
        assert "The debate concerns the following problem:" in first_user, name
        assert problem in first_user, name
        assert first_user.index("The debate concerns the following problem:") == 0, name

    # the proposer's first user message is the answer-generation cue, unprefixed
    proposal_ctx = backend.contexts[0]
    assert "The debate concerns the following problem:" not in proposal_ctx


def test_fresh_position_binds_into_critique_and_verdict():
    backend = ScriptedBackend(["Thinking. \\boxed{7}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_env(["alice"], [GOOD_VERDICT])
    bob = env.config.frozen_models["bob"]
    seen = []
    orig = bob.predict

    def spy(inputs, **kw):
        seen.append("\n".join(mi.content for mi in inputs[0]))
        return orig(inputs, **kw)

    bob.predict = spy
    env.rollout(TaskSource().tasks(1), policy, group_size=1)
    assert "7" in seen[0]  # alice's extracted answer bound into bob's prompt


class NoQuestionTaskSource(TaskSource):
    """Violates the TaskFamily contract: meta lacks 'question'. Inherits
    TaskSource.prompts so construction gets past the template splice and the
    rollout reaches the meta check this test is actually about."""

    def tasks(self, n, split="train"):
        return [
            Task(messages=[{"role": "user", "content": "What is 2+2?"}], meta={"gt": 4.0})
            for _ in range(n)
        ]


def test_missing_question_meta_raises_contract_error():
    backend = ScriptedBackend(["\\boxed{2}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    src = NoQuestionTaskSource()
    env = make_env(["alice"], [GOOD_VERDICT], task_source=src)
    with pytest.raises(ValueError, match=r"NoQuestionTaskSource.*question.*TaskFamily"):
        env.rollout(src.tasks(1), policy, group_size=1)


class CountingFamily(MathFamily):
    """grade() counts calls (and sleeps, so accidental serial re-grading would
    at least be visible in runtime); always correct."""

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def grade(self, meta, solution):
        with self._lock:
            self.calls += 1
        time.sleep(0.05)
        return True


def test_grading_deduped_within_group():
    # group_size=2, identical scripted proposals -> same (task, solution) key
    # twice; family.grade must run exactly once, both trajs get the metric.
    backend = ScriptedBackend(["\\boxed{2}", "\\boxed{2}", "Defense.", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    family = CountingFamily()
    env = make_env(["alice"], [GOOD_VERDICT, GOOD_VERDICT], family=family)
    groups = env.rollout(TaskSource().tasks(1), policy, group_size=2)
    assert family.calls == 1
    trajs = [t for g in groups for t in g]
    assert len(trajs) == 2
    assert all(t.info["solution_correct"] == 1.0 for t in trajs)
    assert env.last_rollout_info["grade_errors"] == 0
    assert env.last_rollout_info["grader_requests"] == 1
    assert env.last_rollout_info["grader_error_rate"] == 0.0


def test_grade_exception_recorded_not_propagated():
    class ExplodingFamily(MathFamily):
        def grade(self, meta, solution):
            raise RuntimeError("verifier fell over")

    backend = ScriptedBackend(["\\boxed{2}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_env(["alice"], [GOOD_VERDICT], family=ExplodingFamily())
    groups = env.rollout(TaskSource().tasks(1), policy, group_size=1)
    (t,) = groups[0]
    assert "solution_correct" not in t.info  # graded as None, not a crash
    assert env.last_rollout_info["grade_errors"] == 1
    assert env.last_rollout_info["grader_requests"] == 1
    assert env.last_rollout_info["grader_error_rate"] == 1.0


def test_grading_goes_through_grade_batch():
    """The seam a learned verifier overrides: when a family provides its own
    grade_batch, DebateEnv must use it — per-pair grade() is never called."""

    class BatchOnlyFamily(MathFamily):
        def grade(self, meta, solution):
            raise AssertionError("per-pair grade() bypasses the grade_batch seam")

        def grade_batch(self, items):
            self.last_grade_errors = 0
            return [True for _ in items]

    backend = ScriptedBackend(["\\boxed{2}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_env(["alice"], [GOOD_VERDICT], family=BatchOnlyFamily())
    groups = env.rollout(TaskSource().tasks(1), policy, group_size=1)
    (t,) = groups[0]
    assert t.info["solution_correct"] == 1.0
    assert env.last_rollout_info["grade_errors"] == 0


@pytest.mark.parametrize("results", [[], [True, False]])
def test_grade_batch_cardinality_must_match_unique_requests(results):
    class MalformedBatchFamily(MathFamily):
        def grade_batch(self, items):
            self.last_grade_errors = 0
            return results

    backend = ScriptedBackend(["\\boxed{2}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_env(["alice"], [GOOD_VERDICT], family=MalformedBatchFamily())
    with pytest.raises(
        RuntimeError,
        match=r"MalformedBatchFamily\.grade_batch returned (0|2) results for 1 unique grading requests",
    ):
        env.rollout(TaskSource().tasks(1), policy, group_size=1)


def test_unscoreable_verdict_drops_debate():
    backend = ScriptedBackend(["\\boxed{2}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    # judge emits garbage for all attempts -> retries exhausted -> state failed
    env = make_env(["alice"], ["not json at all"] * 8)
    groups = env.rollout(TaskSource().tasks(1), policy, group_size=1)
    assert groups == [[]]


def test_solution_census_precedes_failure_filtering():
    """Produced answers survive both their own parse failure and a later
    judge failure in the rollout-level denominator."""
    # Proposals are generated as one batch. State 1 fails immediately because
    # no relaxed answer can be extracted; state 2 reaches its judge, which
    # then fails. The two surviving defenses are the next backend batch.
    backend = ScriptedBackend(
        ["\\boxed{2}", "not an answer", "the answer is 6", "Defense.", "Defense."]
    )
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_env(["alice"], [GOOD_VERDICT, "not json"])
    env.config.verdict_retries = 0
    env.rollout(TaskSource().tasks(3), policy, group_size=1)

    info = env.last_rollout_info
    assert info["expected_solution_slots"] == 3.0
    assert info["produced_solution_slots"] == 3.0
    assert info["answer_format_valid_count"] == 1.0
    assert info["extracted_solution_slots"] == 2.0
    # Includes state 2's extracted answer despite its later judge failure.
    assert info["gradeable_solution_slots"] == 2.0
    assert info["answer_format_valid_rate"] == pytest.approx(1 / 3)
    assert info["solution_production_rate"] == 1.0
    assert info["extracted_solution_rate"] == pytest.approx(2 / 3)
    assert info["gradeable_solution_rate"] == 1.0
    assert info["grader_requests"] == 2.0
    assert info["grader_error_rate"] == 0.0
    assert info["debates_failed"] == 2


def test_solution_census_missing_generation_and_all_fail():
    # An empty Sample fails fidelity before DebateRound can append a record.
    backend = ScriptedBackend([""])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_env(["alice"], [])
    groups = env.rollout(TaskSource().tasks(1), policy, group_size=1)

    assert groups == [[]]
    expected = {
        "debates": 1,
        "debates_failed": 1,
        "debates_unscoreable": 0,
        "fail_reasons": {"alice/proposal": 1},
        "grade_errors": 0,
        "grader_requests": 0.0,
        "expected_solution_slots": 1.0,
        "produced_solution_slots": 0.0,
        "answer_format_valid_count": 0.0,
        "extracted_solution_slots": 0.0,
        "gradeable_solution_slots": 0.0,
        "answer_format_valid_rate": 0.0,
        "solution_production_rate": 0.0,
        "extracted_solution_rate": 0.0,
        "gradeable_solution_rate": 0.0,
        "grader_error_rate": 0.0,
    }
    assert {key: env.last_rollout_info[key] for key in expected} == expected


def test_continuous_scoring_uses_json_confidence():
    backend = ScriptedBackend(["\\boxed{2}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_env(
        ["alice"],
        [GOOD_VERDICT],
        scoring=ScoringConfig(scoring="continuous", confidence_source="json"),
    )
    groups = env.rollout(TaskSource().tasks(1), policy, group_size=1)
    assert groups[0][0].reward == pytest.approx(2 * 0.8 - 1)  # winner margin


def test_docent_export(tmp_path):
    backend = ScriptedBackend(["\\boxed{2}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_env(["alice"], [GOOD_VERDICT])
    env.rollout(TaskSource().tasks(1), policy, group_size=1)

    from infra.envs.debate.docent_export import agent_runs, export_jsonl

    runs = agent_runs(env)
    assert len(runs) == 1
    run = runs[0]
    names = [t.name for t in run.transcripts]
    assert names == ["omniscient", "view:alice", "view:bob", "view:judge"]
    omni = run.transcripts[0]
    assert len(omni.messages) == 1 + 4  # system + proposal/critique/defense/verdict
    assert run.metadata["verdict"]["winner"] == "Debater_A"
    assert run.metadata["verdict"]["confidence"]["Debater_A"]["json"] == 0.8
    assert omni.messages[1].metadata["answer_format_valid"] is True
    # judge view: rendered context ends with its own verdict as assistant
    judge_view = run.transcripts[3]
    assert judge_view.messages[0].role == "system"

    def _text(m):
        return m.text if hasattr(m, "text") else str(getattr(m, "content", ""))

    # message 1 is the judge's pre_debate_judge preamble, its own user message;
    # the attributed transcript starts in the message after it
    assert "The debate concerns the following problem:" in _text(judge_view.messages[1])
    assert "Debater_A said:" in _text(judge_view.messages[2])
    path = export_jsonl(runs, str(tmp_path / "debates.jsonl"))
    import json as _json

    line = _json.loads(open(path).read().splitlines()[0])
    assert line["metadata"]["verdict"]["ok"] is True


def test_self_play_both_seats_harvested():
    backend = ScriptedBackend(
        ["\\boxed{2}", "Wrong step!", "Defense."]  # alice proposal, bob critique, alice defense
    )
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = DebateEnv(
        DebateEnvConfig(
            protocol=PROTOCOL,
            prompt_file=PROMPT_FILE,
            prompt_entry="math_proposer_critic",
            trained_speakers=["alice", "bob"],
            frozen_models={"judge": ScriptedModel("judge", [GOOD_VERDICT])},
            fresh_positions=True,
        ),
        TaskSource(),
        MathFamily(),
    )
    groups = env.rollout(TaskSource().tasks(1), policy, group_size=1)
    # seat-split groups: proposer and critic rewards are anti-correlated and
    # must never share a GRPO baseline
    assert len(groups) == 2
    trajs = [t for g in groups for t in g]
    assert len(trajs) == 2                                # both seats harvested
    assert {t.info["seat"] for t in trajs} == {"alice", "bob"}
    rewards = sorted(t.reward for t in trajs)
    assert rewards == [-1.0, 1.0]                         # zero-sum competitive
    assert {len(t.datums) for t in trajs} == {1, 2}       # bob 1 slot, alice 2


def test_grpo_pack_per_datum_rewards():
    from infra.backend.base import Datum
    from infra.envs.base import Trajectory
    from infra.rl.datums import grpo_pack

    def mk(traj_rewards, datum_rewards=None):
        d = lambda: Datum(tokens=[1, 2, 3], prompt_len=2, sampler_logprobs=[-0.1], advantages=[0.0])
        return Trajectory(datums=[d(), d()], reward=traj_rewards, datum_rewards=datum_rewards)

    # uniform rewards: identical to per-trajectory normalization (both datums
    # of a trajectory get the SAME advantage)
    datums, _ = grpo_pack([[mk(1.0), mk(0.0)]], drop_zero_advantage=False)
    advs = [d.advantages[0] for d in datums]
    assert advs[0] == advs[1] and advs[2] == advs[3] and advs[0] == -advs[2]

    # slot bonus on datum 0 only: position 0 group has variance, position 1 constant
    datums, stats = grpo_pack(
        [[mk(0.0, [0.1, 0.0]), mk(0.0, [0.0, 0.0])]], drop_zero_advantage=True
    )
    assert stats["pack/n_datums_dropped_zero_advantage"] == 2.0  # both position-1 datums
    assert len(datums) == 2 and datums[0].advantages[0] == -datums[1].advantages[0] != 0.0


def test_format_reward_targets_solution_datum():
    from infra.envs.debate.rewards import ScoringConfig

    backend = ScriptedBackend(["\\boxed{2}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_env(
        ["alice"],
        [GOOD_VERDICT],
        scoring=ScoringConfig(
            shaping=[
                {
                    "kind": "format_reward",
                    "coeff": 0.25,
                    "slots": ["proposal"],
                    "flag": "answer_format_valid",
                }
            ]
        ),
    )
    groups = env.rollout(TaskSource().tasks(1), policy, group_size=1)
    (t,) = groups[0]
    # proposal datum gets +0.25 (boxed present); defense datum only the base +1
    assert t.datum_rewards == [1.25, 1.0]
    assert t.reward == pytest.approx(1.125)  # mean, for logging


def test_overshoot_shaping_rejects_a_slot_with_no_think_cap():
    """A slot with no max_think_tokens can never be force-closed, so pricing
    overshoot on it buys a constant 0.0 — the same value-shaped void that made
    dataset.think_overshoot_penalty read as "0.0 on the debate arm" when it was
    in fact inert there. Refuse it at construction instead of paying it."""
    from infra.envs.debate.rewards import ScoringConfig

    with pytest.raises(ValueError, match="max_think_tokens"):
        make_env(
            ["alice"],
            [GOOD_VERDICT],
            scoring=ScoringConfig(
                shaping=[
                    {
                        "kind": "think_overshoot_penalty",
                        "coeff": 0.1,
                        "slots": ["proposal"],
                    }
                ]
            ),
        )


def test_format_shaping_and_census_share_the_extraction_parse():
    class SingleParseFamily(MathFamily):
        def __init__(self):
            self.parse_calls = 0

        def parse_answers(self, text):
            self.parse_calls += 1
            if self.parse_calls > 1:
                raise AssertionError("solution answer was parsed more than once")
            return AnswerParse(strict=2.0, relaxed=2.0)

    family = SingleParseFamily()
    backend = ScriptedBackend(["any answer", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_env(
        ["alice"],
        [GOOD_VERDICT],
        family=family,
        scoring=ScoringConfig(
            shaping=[
                {
                    "kind": "format_reward",
                    "coeff": 0.25,
                    "slots": ["proposal"],
                    "flag": "answer_format_valid",
                }
            ]
        ),
    )

    groups = env.rollout(TaskSource().tasks(1), policy, group_size=1)

    assert family.parse_calls == 1
    assert groups[0][0].datum_rewards == [1.25, 1.0]
    assert env.last_rollout_info["answer_format_valid_count"] == 1.0


def make_solo_env(protocol=PROTOCOL, fresh_positions=True):
    return DebateEnv(
        DebateEnvConfig(
            protocol=protocol,
            prompt_file=PROMPT_FILE,
            prompt_entry="math_proposer_critic",
            trained_speakers=["alice"],
            frozen_models={
                "bob": ScriptedModel("bob", ["Your step 2 is wrong."] * 64),
                "judge": ScriptedModel("judge", [GOOD_VERDICT] * 8),
            },
            fresh_positions=fresh_positions,
            first_speech_non_debate_aware=True,
        ),
        TaskSource(),
        MathFamily(),
    )


def test_non_debate_aware_first_speech():
    backend = ScriptedBackend(["I compute. \\boxed{2}", "My defense stands."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_solo_env()
    bob = env.config.frozen_models["bob"]
    seen = []
    orig = bob.predict

    def spy(inputs, **kw):
        seen.append("\n".join(mi.content for mi in inputs[0]))
        return orig(inputs, **kw)

    bob.predict = spy
    groups = env.rollout(TaskSource().tasks(1), policy, group_size=1)

    # what the trained backend actually saw for the proposal (FakeTokenizer
    # keeps only the prompt tail, so read the full contexts off the state)
    assert "What is 1+1?" in backend.contexts[0]
    from infra.envs.debate.round import render_context

    state = env.last_states[0]
    slots = PROTOCOL.compile()
    render = lambda cs: "\n".join(m["content"] for m in render_context(state, cs, env.prompts))
    proposal_ctx, defense_ctx = render(slots[0]), render(slots[2])
    assert proposal_ctx == "What is 1+1?"          # the task's own messages, verbatim
    assert "PROPOSER" not in proposal_ctx          # no debate system card
    assert "Debater_B" not in proposal_ctx         # opponent never mentioned
    # the defense is debate-aware again, and renders the solo cue as the thing
    # alice's own proposal answered
    assert "PROPOSER" in defense_ctx
    assert "Debater_B" in defense_ctx
    assert "step 2 is wrong" in defense_ctx
    assert "What is 1+1?" in defense_ctx
    assert "\\boxed{2}" in defense_ctx

    assert "\\boxed{2}" in seen[0]                 # critic still sees the speech
    assert "2" in seen[0]                          # position binding intact

    (t,) = groups[0]
    assert len(t.datums) == 2 and all(d.prompt_len > 0 for d in t.datums)


def test_non_debate_aware_requires_fresh_positions():
    with pytest.raises(ValueError, match="fresh_positions"):
        make_solo_env(fresh_positions=False)


def test_non_debate_aware_requires_leading_solution_slot():
    proto = Protocol.parse(
        yaml.safe_load(
            """
turns:
  - bob:   [{name: critique}]
  - alice: [{name: proposal, kind: solution}]
  - judge: [{name: verdict, kind: decision}]
"""
        )
    )
    with pytest.raises(ValueError, match="debater solution slot"):
        make_solo_env(protocol=proto)


def test_non_debate_aware_requires_public_proposal():
    # a private/ephemeral opening would silently hide the speech from the
    # critic and judge (row 10: the solo speech is a PUBLIC record)
    proto = Protocol.parse(
        yaml.safe_load(
            """
turns:
  - alice: [{name: proposal, kind: solution, visibility: private}]
  - bob:   [{name: critique}]
  - judge: [{name: verdict, kind: decision}]
"""
        )
    )
    with pytest.raises(ValueError, match="public"):
        make_solo_env(protocol=proto)


def test_non_debate_aware_requires_user_message():
    class NoUserTaskSource:
        def tasks(self, n, split="train"):
            return [
                Task(messages=[{"role": "system", "content": "solve"}], meta={"question": "q", "gt": 1.0})
                for _ in range(n)
            ]

    backend = ScriptedBackend(["\\boxed{2}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    env = make_solo_env()
    with pytest.raises(ValueError, match="user message"):
        env.rollout(NoUserTaskSource().tasks(1), policy, group_size=1)


def test_length_normalize_balances_seats():
    from infra.backend.base import Datum
    from infra.envs.base import Trajectory
    from infra.rl.datums import grpo_pack

    def datum(n):
        return Datum(
            tokens=list(range(n + 2)), prompt_len=2,
            sampler_logprobs=[-0.1] * n, advantages=[0.0] * n,
        )

    # alice: 2 datums of 10 tokens; bob: 1 datum of 2 tokens; zero-sum rewards
    alice = [Trajectory(datums=[datum(10), datum(10)], reward=r) for r in (1.0, 0.0)]
    bob = [Trajectory(datums=[datum(2)], reward=r) for r in (0.0, 1.0)]

    def total_mass(groups, mode):
        datums, _ = grpo_pack(groups, length_normalize=mode, drop_zero_advantage=False)
        return sum(sum(abs(a) for a in d.advantages) for d in datums)

    # token-sum semantics: alice's mass is 10x bob's (20 tokens x 2 datums vs 2)
    raw_alice = total_mass([alice], "none")
    raw_bob = total_mass([bob], "none")
    assert raw_alice == pytest.approx(10 * raw_bob)

    # trajectory mode: equal mass per trajectory regardless of slots/lengths
    norm_alice = total_mass([alice], "trajectory")
    norm_bob = total_mass([bob], "trajectory")
    assert norm_alice == pytest.approx(norm_bob)

    # datum mode: mass proportional to datum COUNT only
    assert total_mass([alice], "datum") == pytest.approx(2 * total_mass([bob], "datum"))

    # count mode: no token scaling, equal per-trajectory datum weighting
    datums, _ = grpo_pack([alice], length_normalize="count", drop_zero_advantage=False)
    assert abs(datums[0].advantages[0]) == pytest.approx(abs(datums[0].advantages[0]))
    assert total_mass([bob], "count") == pytest.approx(raw_bob)  # 1 datum: /1


def test_empty_speech_census_counts_a_swallowed_think_block():
    """A trained seat whose pre-opened <think> never closes hands the round an
    empty speech (round._split_think fails closed). The rollout census is what
    reports it: a debate the judge decided on a seat that said nothing."""
    backend = ScriptedBackend(
        ["reasoning</think>I compute. \\boxed{2}", "I still need to check the algebra"]
    )
    policy = Policy(backend, SamplingParams(max_tokens=128), {"enable_thinking": True})
    env = make_env(["alice"], [GOOD_VERDICT])
    env.rollout(TaskSource().tasks(1), policy, group_size=1)

    info = env.last_rollout_info
    assert info["trained_speeches"] == 2.0          # proposal + alice_rebuttal
    assert info["empty_speeches"] == 1.0            # the reply was all think
    assert info["empty_speeches_by_slot"] == {"alice_rebuttal": 1.0}
    # The proposal closed its block, so its answer still reaches the grader.
    assert info["extracted_solution_slots"] == 1.0


BUDGETED_PROTOCOL = Protocol.parse(
    yaml.safe_load(
        """
turns:
  - alice: [{name: proposal, kind: solution, max_total_tokens: 128}]
  - bob:   [{name: critique, max_total_tokens: 64}]
  - alice: [{name: alice_rebuttal, max_total_tokens: 64}]
  - judge: [{name: verdict, kind: decision, max_total_tokens: 64}]
"""
    )
)


class CapHittingBackend(ScriptedBackend):
    """Scripted, but reports the named speeches as cut at their cap."""

    def __init__(self, script, truncated_indices):
        super().__init__(script)
        self._truncated = set(truncated_indices)
        self._served = 0

    def sample(self, prompts, params, n=1):
        out = super().sample(prompts, params, n)
        for group in out:
            hit = self._served in self._truncated
            self._served += 1
            if hit:
                for i, s in enumerate(group):
                    group[i] = _replace(s, stop_reason="length")
        return out


def _budgeted_env(backend_script, truncated_indices, coeff=0.1):
    from infra.envs.debate.rewards import ScoringConfig

    backend = CapHittingBackend(backend_script, truncated_indices)
    env = DebateEnv(
        DebateEnvConfig(
            protocol=BUDGETED_PROTOCOL,
            prompt_file=PROMPT_FILE,
            prompt_entry="math_proposer_critic",
            trained_speakers=["alice"],
            frozen_models={
                "bob": ScriptedModel("bob", ["Your step 2 is wrong."] * 64),
                "judge": ScriptedModel("judge", [GOOD_VERDICT] * 64),
            },
            judge=JudgeConfig(),
            scoring=ScoringConfig(
                shaping=[
                    {
                        "kind": "speech_overshoot_penalty",
                        "coeff": coeff,
                        "slots": ["critique", "alice_rebuttal"],
                    }
                ]
            ),
            fresh_positions=True,
        ),
        TaskSource(),
        MathFamily(),
    )
    return env, backend


def test_speech_overshoot_penalty_prices_the_slot_that_ran_out_of_budget():
    """A speech cut at its cap reaches the judge mid-sentence. The hard cap
    truncates it; this term is what tells the policy to fit."""
    env, backend = _budgeted_env(["\\boxed{2}", "My reply runs long"], truncated_indices=[1])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    groups = env.rollout(TaskSource().tasks(1), policy, group_size=1)
    (t,) = groups[0]
    proposal, reply = t.datum_rewards
    assert proposal == 1.0            # untruncated, and the term skips the solution slot
    assert reply == pytest.approx(0.9)  # 1.0 - 0.1, paid once on the slot that overran


def test_speech_overshoot_penalty_is_silent_when_every_speech_fits():
    env, backend = _budgeted_env(["\\boxed{2}", "Short reply."], truncated_indices=[])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    groups = env.rollout(TaskSource().tasks(1), policy, group_size=1)
    (t,) = groups[0]
    assert t.datum_rewards == [1.0, 1.0]


def test_speech_overshoot_shaping_rejects_a_slot_with_no_budget():
    """The twin of the think-cap refusal: with neither max_total_tokens nor
    max_visible_tokens the slot can never run out, so the term would buy a
    constant 0.0."""
    from infra.envs.debate.rewards import ScoringConfig

    with pytest.raises(ValueError, match="max_total_tokens"):
        make_env(
            ["alice"],
            [GOOD_VERDICT],
            scoring=ScoringConfig(
                shaping=[{"kind": "speech_overshoot_penalty", "coeff": 0.1, "slots": ["proposal"]}]
            ),
        )


# ------------------------------------------- shortening an inherited protocol


def test_null_slots_delete_a_speaker_and_an_emptied_turn_drops():
    """`null` is how a child SHORTENS a protocol, and it is the only way.

    Config lists deep-merge BY INDEX and the merge walks only the override's
    entries, so declaring fewer turns leaves the parent's extra ones in place:
    a 3-turn override of this 5-turn parent keeps both rebuttals and unions the
    judge into the alice_rebuttal turn. `null` already means "clear the
    inherited key" elsewhere in these configs (max_think_tokens on the reply
    slots of math_pc_l5), so it means the same here.
    """
    proto = Protocol.parse(
        {
            "turns": [
                {"alice": [{"name": "proposal", "kind": "solution", "max_total_tokens": 64}]},
                {"bob": [{"name": "critique", "max_total_tokens": 32}]},
                {"alice": None},
                {"bob": None},
                {"judge": [{"name": "verdict", "kind": "decision", "max_total_tokens": 16}]},
            ]
        }
    )
    assert [cs.slot.name for cs in proto.compile()] == ["proposal", "critique", "verdict"]
    # The judge turn keeps its own identity rather than being merged into the
    # turn a deleted speaker vacated.
    assert [cs.speaker for cs in proto.compile()] == ["alice", "bob", "judge"]


def test_a_turn_keeps_its_other_speakers_when_one_is_deleted():
    proto = Protocol.parse(
        {
            "turns": [
                {"alice": [{"name": "proposal", "kind": "solution", "max_total_tokens": 64}]},
                {"bob": [{"name": "critique", "max_total_tokens": 32}], "alice": None},
                {"judge": [{"name": "verdict", "kind": "decision", "max_total_tokens": 16}]},
            ]
        }
    )
    assert [cs.slot.name for cs in proto.compile()] == ["proposal", "critique", "verdict"]


def test_an_empty_slot_list_is_still_an_error():
    """`[]` merges into a populated list as a NO-OP rather than clearing it, so
    an author who writes it is expecting a deletion that will not happen."""
    with pytest.raises(ValueError, match="non-empty list"):
        Protocol.parse({"turns": [{"alice": []}]})
    with pytest.raises(ValueError, match="non-empty mapping"):
        Protocol.parse({"turns": [{}]})
