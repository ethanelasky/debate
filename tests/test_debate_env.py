"""End-to-end DebateEnv rollout with scripted seats (no network, no GPUs).

Alice (trained, via a fake Backend) proposes and defends; Bob (frozen, scripted)
critiques; a scripted judge emits valid competitive JSON. Checks trajectories,
rewards, datums, groups, and blindness-sensitive context rendering.
"""

import pytest

from infra.backend.base import Backend, Sample, SamplingParams
from infra.envs.base import Policy, Task
from infra.envs.debate.env import DebateEnv, DebateEnvConfig
from infra.envs.debate.judge import JudgeConfig
from infra.envs.debate.rewards import ScoringConfig
from infra.envs.debate.topology import Topology
from infra.models.base import Model, ModelResponse

import yaml

TOPOLOGY = Topology.parse(
    yaml.safe_load(
        """
turns:
  - alice: [{name: proposal, kind: solution}]
  - bob:   [{name: critique}]
  - alice: [{name: defense}]
  - judge: [{name: verdict, kind: decision}]
"""
    )
)

PROMPT_FILE = "infra/envs/debate/prompt_configs/hendrycks_math.yaml"


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
    def tasks(self, n, split="train"):
        return [
            Task(
                messages=[{"role": "user", "content": f"What is {i}+{i}?"}],
                meta={"question": f"What is {i}+{i}?", "gt": float(2 * i)},
            )
            for i in range(1, n + 1)
        ]


def extract_number(text):
    import re

    m = re.search(r"\\boxed\{([-\d.]+)\}", text)
    return float(m.group(1)) if m else None


def make_env(trained, judge_script, scoring=None):
    return DebateEnv(
        DebateEnvConfig(
            topology=TOPOLOGY,
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
        TaskSource(),
        extract_number,
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


def test_unscoreable_verdict_drops_debate():
    backend = ScriptedBackend(["\\boxed{2}", "Defense."])
    policy = Policy(backend, SamplingParams(max_tokens=128))
    # judge emits garbage for all attempts -> retries exhausted -> state failed
    env = make_env(["alice"], ["not json at all"] * 8)
    groups = env.rollout(TaskSource().tasks(1), policy, group_size=1)
    assert groups == [[]]


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
    # judge view: rendered context ends with its own verdict as assistant
    judge_view = run.transcripts[3]
    assert judge_view.messages[0].role == "system"
    assert "Debater_A said:" in "".join(
        m.text if hasattr(m, "text") else str(getattr(m, "content", "")) for m in judge_view.messages[1:2]
    )
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
            topology=TOPOLOGY,
            prompt_file=PROMPT_FILE,
            prompt_entry="math_proposer_critic",
            trained_speakers=["alice", "bob"],
            frozen_models={"judge": ScriptedModel("judge", [GOOD_VERDICT])},
            fresh_positions=True,
        ),
        TaskSource(),
        extract_number,
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
            shaping=[{"kind": "format_reward", "coeff": 0.25, "slots": ["proposal"], "flag": "strict_boxed"}]
        ),
    )
    groups = env.rollout(TaskSource().tasks(1), policy, group_size=1)
    (t,) = groups[0]
    # proposal datum gets +0.25 (boxed present); defense datum only the base +1
    assert t.datum_rewards == [1.25, 1.0]
    assert t.reward == pytest.approx(1.125)  # mean, for logging
