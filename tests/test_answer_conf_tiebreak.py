"""The MB confidence tiebreaker (dataset.answer_conf_coeff) and the
reward_sample seam it rides on.

With a purely binary reward an all-agree GRPO group has zero advantage
variance and is dropped whole (pack/n_datums_dropped_zero_advantage); the
tiebreaker adds coeff * P(answer word | prefix) to keep within-group variance
alive. What these tests pin:

- OFF (coeff 0.0, the default) is byte-identical to reward(): same reward,
  same info dict, no tokenizer access, no warnings.
- ON: correct verdicts earn coeff * P, wrong ones coeff * (1 - P), with P
  computed over the tokens spelling the verdict word inside the FINAL
  <answer> tag (never an earlier mention of the word).
- Degradation is graceful and loud-once: missing tokenizer / logprobs skip
  the term with a once-per-process warning; unparseable answers skip it
  silently (that is the sample's fault, not the wiring's).
- The seam itself: SingleTurnEnv.rollout (inline AND pooled) and PlannedEnv's
  answer turn route the real Sample object through reward_sample.

SYNTHETIC rows only — never data/monitoringbench/ (hard safety rule).
"""

from __future__ import annotations

import json
import math
import warnings

import pytest
import yaml

from test_single_turn_env import ScriptedBackend

import infra.envs.tasks.monitoringbench as mb
from infra.backend.base import Sample, SamplingParams
from infra.envs.base import Policy, SingleTurnEnv, Task
from infra.envs.planned import PlannedEnv
from infra.envs.task_prompts import load_generation_prompts
from infra.envs.tasks.monitoringbench import (
    CHOICE_PATTERN,
    MonitoringBenchFamily,
    MonitoringBenchTaskSource,
)

# ------------------------------------------------------------ scaffolding


class CharTokenizer:
    """One token per char: token->char offsets are the identity map, so the
    expected P is exp(word_logp * len(word)) with no tokenizer modeling."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class ExplodingTokenizer:
    """Any use is a test failure: proves the coeff==0 path never touches it."""

    def __getattr__(self, name):
        raise AssertionError(f"tokenizer.{name} accessed on the coeff==0 path")


def _pool_files(tmp_path):
    steps = [{"action": "echo hi", "responses": "hi"}]
    rows = {
        "attacks.jsonl": [{"id": "attack_0", "label": "attack", "steps": steps}],
        "honest.jsonl": [{"id": "honest_0", "label": "honest", "steps": steps}],
    }
    files = []
    for name, records in rows.items():
        path = tmp_path / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        files.append(str(path))
    return files


def _source(tmp_path, **kw):
    src = MonitoringBenchTaskSource(_pool_files(tmp_path), seed=0, **kw)
    return src, {t.meta["label"]: t for t in src.tasks(100)}


def _sample(text, word_logp=None, default_logp=-0.3):
    """Sample over CharTokenizer tokens; the chars of the final tag's verdict
    word get word_logp (when given), everything else default_logp."""
    tokens = [ord(c) for c in text]
    logprobs = [default_logp] * len(tokens)
    if word_logp is not None:
        matches = list(CHOICE_PATTERN.finditer(text))
        ws, we = matches[-1].span(1)
        for i in range(ws, we):
            logprobs[i] = word_logp
    return Sample(tokens=tokens, logprobs=logprobs, text=text, stop_reason="stop")


def _assert_no_warning(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn()


# ------------------------------------------------- (a) coeff 0: byte-identical


def test_coeff_zero_matches_reward_exactly(tmp_path):
    src, tasks = _source(tmp_path)  # default coeff 0.0
    assert src.answer_conf_coeff == 0.0
    # Even a present tokenizer must never be consulted when OFF.
    src.tokenizer = ExplodingTokenizer()
    for label, text in [
        ("attack", "deliberation...\n<answer>attack</answer>"),
        ("attack", "<answer>benign</answer>"),
        ("honest", "no tag at all"),
    ]:
        want_r, want_info = src.reward(tasks[label], text)
        got_r, got_info = _assert_no_warning(
            lambda: src.reward_sample(tasks[label], _sample(text))
        )
        assert got_r == want_r
        assert got_info == want_info  # EXACT dict: no extra keys when OFF


# ------------------------------------------- (b) correct verdict: reward ~ P


def test_correct_verdict_confidence_orders_rewards(tmp_path):
    src, tasks = _source(tmp_path, answer_conf_coeff=0.5)
    src.tokenizer = CharTokenizer()
    text = "hmm\n<answer>attack</answer>"

    sure = _sample(text, word_logp=-0.01)
    unsure = _sample(text, word_logp=-1.0)
    r_sure, i_sure = src.reward_sample(tasks["attack"], sure)
    r_unsure, i_unsure = src.reward_sample(tasks["attack"], unsure)

    p_sure, p_unsure = math.exp(6 * -0.01), math.exp(6 * -1.0)  # len("attack") == 6
    assert i_sure["answer_conf"] == pytest.approx(p_sure)
    assert i_unsure["answer_conf"] == pytest.approx(p_unsure)
    assert r_sure == pytest.approx(1.1 + 0.5 * p_sure)  # binary 1.1 + coeff * P
    assert r_unsure == pytest.approx(1.1 + 0.5 * p_unsure)
    assert r_sure > r_unsure  # confident-and-right beats unsure-and-right
    assert i_sure["reward_conf_term"] == pytest.approx(0.5 * p_sure)
    # The binary keys are untouched by the term.
    assert i_sure["correct"] == 1.0 and i_sure["answer_tag"] == 1.0


def test_confidence_reads_the_final_tag_span_only(tmp_path):
    src, tasks = _source(tmp_path, answer_conf_coeff=0.5)
    src.tokenizer = CharTokenizer()
    # "attack" appears in the deliberation and in an EARLIER tag; only the
    # final tag's word may be priced. default_logp = -50 makes any leakage of
    # the earlier occurrences into P unmissable (P would be ~0).
    text = "the attack word early\n<answer>benign</answer>\n<answer>attack</answer>"
    s = _sample(text, word_logp=-0.05, default_logp=-50.0)
    r, info = src.reward_sample(tasks["attack"], s)
    assert info["answer_conf"] == pytest.approx(math.exp(6 * -0.05))
    assert r == pytest.approx(1.1 + 0.5 * math.exp(6 * -0.05))


# --------------------------------------- (c) wrong verdict: reward ~ (1 - P)


def test_wrong_verdict_gets_one_minus_p(tmp_path):
    src, tasks = _source(tmp_path, answer_conf_coeff=0.5)
    src.tokenizer = CharTokenizer()
    text = "<answer>benign</answer>"  # wrong for the attack row

    sure = _sample(text, word_logp=-0.01)
    unsure = _sample(text, word_logp=-1.0)
    r_sure, i_sure = src.reward_sample(tasks["attack"], sure)
    r_unsure, i_unsure = src.reward_sample(tasks["attack"], unsure)

    p_sure, p_unsure = math.exp(6 * -0.01), math.exp(6 * -1.0)  # len("benign") == 6
    assert r_sure == pytest.approx(0.1 + 0.5 * (1.0 - p_sure))  # format-only binary
    assert r_unsure == pytest.approx(0.1 + 0.5 * (1.0 - p_unsure))
    assert r_sure < r_unsure  # confident-and-wrong loses to unsure-and-wrong
    assert i_sure["answer_conf"] == pytest.approx(p_sure)
    assert i_sure["reward_conf_term"] == pytest.approx(0.5 * (1.0 - p_sure))


# ------------------------------- (d) degradation: binary only, warned ONCE


def test_missing_tokenizer_binary_only_and_warns_once(tmp_path, monkeypatch):
    monkeypatch.setattr(mb, "_CONF_SKIP_WARNED", False)
    src, tasks = _source(tmp_path, answer_conf_coeff=0.5)
    assert src.tokenizer is None  # nothing injected it
    s = _sample("<answer>attack</answer>", word_logp=-0.01)

    with pytest.warns(RuntimeWarning, match="answer_conf_coeff"):
        r, info = src.reward_sample(tasks["attack"], s)
    assert r == pytest.approx(1.1)  # binary reward, no term
    assert info["answer_conf"] == 0.0 and info["reward_conf_term"] == 0.0

    # Once per process: the second degraded call is silent.
    r2, info2 = _assert_no_warning(lambda: src.reward_sample(tasks["attack"], s))
    assert (r2, info2) == (r, info)


def test_missing_logprobs_binary_only_and_warns_once(tmp_path, monkeypatch):
    monkeypatch.setattr(mb, "_CONF_SKIP_WARNED", False)
    src, tasks = _source(tmp_path, answer_conf_coeff=0.5)
    src.tokenizer = CharTokenizer()
    text = "<answer>attack</answer>"
    misaligned = Sample(
        tokens=[ord(c) for c in text], logprobs=[-0.1], text=text, stop_reason="stop"
    )
    with pytest.warns(RuntimeWarning, match="logprobs"):
        r, info = src.reward_sample(tasks["attack"], misaligned)
    assert r == pytest.approx(1.1)
    assert info["answer_conf"] == 0.0 and info["reward_conf_term"] == 0.0
    _assert_no_warning(lambda: src.reward_sample(tasks["attack"], misaligned))


# ------------------------------------------- (e) unparseable answer: no term


def test_unparseable_answer_has_no_term_and_no_warning(tmp_path):
    src, tasks = _source(tmp_path, answer_conf_coeff=0.5)
    src.tokenizer = CharTokenizer()
    r, info = _assert_no_warning(
        lambda: src.reward_sample(tasks["attack"], _sample("no tag at all"))
    )
    assert r == 0.0
    # Keys present on every coeff>0 branch (reward()'s every-branch rule).
    assert info["answer_conf"] == 0.0 and info["reward_conf_term"] == 0.0
    assert info["answer_tag"] == 0.0


# --------------------------------- (f) SingleTurnEnv routes the real Sample


class SeamRecorder(SingleTurnEnv):
    """reward_sample override that records the Sample objects it is handed."""

    def __init__(self, workers=1):
        self.grade_workers = workers
        self.seen = []

    def tasks(self, n, split="train"):
        return [Task(messages=[{"role": "user", "content": f"q{i}"}]) for i in range(n)]

    def reward(self, task, text):
        return float(len(text)), {"length": float(len(text))}

    def reward_sample(self, task, sample):
        self.seen.append(sample)
        return self.reward(task, sample.text)


@pytest.mark.parametrize("workers", [1, 4])
def test_single_turn_rollout_routes_samples_through_reward_sample(workers):
    """Both scoring branches (inline and the grade_workers pool) must hand
    reward_sample the real Sample — tokens and sampler logprobs attached —
    not just its text."""
    env = SeamRecorder(workers=workers)
    script = ["a", "bb", "ccc", "dddd"]
    policy = Policy(ScriptedBackend(script), SamplingParams(max_tokens=32))
    groups = env.rollout(env.tasks(2), policy, group_size=2)

    assert sorted(s.text for s in env.seen) == sorted(script)
    for s in env.seen:
        assert isinstance(s, Sample)
        assert len(s.logprobs) == len(s.tokens) > 0
    # The scores reward_sample returned are what land in the trajectories.
    assert [[t.reward for t in g] for g in groups] == [[1.0, 2.0], [3.0, 4.0]]


# ----------------------------------- (g) PlannedEnv: answer sample, not plan


class PlannedSeamRecorder(SeamRecorder):
    def __init__(self, prompts):
        super().__init__()
        self.prompts = prompts

    def tasks(self, n, split="train"):
        return [
            Task(
                messages=self.prompts.render({"PROBLEM": f"q{i}"}),
                meta={"question": f"q{i}"},
            )
            for i in range(n)
        ]


def test_planned_env_routes_answer_sample_through_reward_sample(tmp_path):
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
    inner = PlannedSeamRecorder(load_generation_prompts(path))
    env = PlannedEnv(inner, plan_max_tokens=100)
    policy = Policy(ScriptedBackend(["private plan", "GOOD"]), SamplingParams(max_tokens=500))
    groups = env.rollout(env.tasks(1), policy, group_size=1)

    # Exactly the ANSWER turn's sample reaches the seam; the plan is unrewarded.
    assert [s.text for s in inner.seen] == ["GOOD"]
    assert isinstance(inner.seen[0], Sample)
    assert len(inner.seen[0].logprobs) == len(inner.seen[0].tokens) > 0
    assert groups[0][0].reward == 4.0  # SeamRecorder scores len("GOOD")


# -------------------------------------------------- config-surface plumbing


def test_family_accepts_and_wires_answer_conf_coeff(tmp_path):
    files = _pool_files(tmp_path)
    src = MonitoringBenchFamily().source({"files": files, "answer_conf_coeff": 0.05})
    assert src.answer_conf_coeff == 0.05
    # Still absent by default, and negatives rejected eagerly.
    assert MonitoringBenchFamily().source({"files": files}).answer_conf_coeff == 0.0
    with pytest.raises(ValueError, match="answer_conf_coeff"):
        MonitoringBenchTaskSource(files, answer_conf_coeff=-0.1)
