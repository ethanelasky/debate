"""SingleTurnEnv: the RLVR rollout shape shared by every task-source env.

MathEnv and CodeContestsEnv had byte-for-byte equivalent rollout() methods
differing only in whether reward() ran in a thread pool. These pin the
behaviors both of them relied on, so the shared implementation cannot drift:
group shape, fidelity drops, and pooled == inline results.
"""

import threading

import pytest

from infra.backend.base import Backend, Sample, SamplingParams
from infra.envs.base import Policy, SingleTurnEnv, Task


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
    """One scripted text per (prompt, sample) in flat order. A text of "" is a
    fidelity failure: zero tokens, which is what fidelity_ok() rejects."""

    def __init__(self, script):
        self.tokenizer = FakeTokenizer()
        self.script = list(script)

    def sync_sampler(self):
        pass

    def sample(self, prompts, params, n=1):
        out = []
        for _ in prompts:
            samples = []
            for _ in range(n):
                text = self.script.pop(0)
                toks = self.tokenizer.encode(text)
                samples.append(
                    Sample(tokens=toks, logprobs=[-0.2] * len(toks), text=text, stop_reason="stop")
                )
            out.append(samples)
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
        raise NotImplementedError


class ScoreByLength(SingleTurnEnv):
    """reward = len(text); info carries the text back for order assertions."""

    def __init__(self, workers=1):
        self.grade_workers = workers
        self.calls = []

    def tasks(self, n, split="train"):
        return [Task(messages=[{"role": "user", "content": f"q{i}"}]) for i in range(n)]

    def reward(self, task, text):
        self.calls.append(text)
        return float(len(text)), {"length": float(len(text))}


def _policy(script):
    return Policy(ScriptedBackend(script), SamplingParams(max_tokens=32))


def test_one_group_per_task_in_order():
    env = ScoreByLength()
    tasks = env.tasks(2)
    groups = env.rollout(tasks, _policy(["a", "bb", "ccc", "dddd"]), group_size=2)

    assert len(groups) == 2
    assert [[t.reward for t in g] for g in groups] == [[1.0, 2.0], [3.0, 4.0]]
    assert all(len(t.datums) == 1 for g in groups for t in g)


def test_info_reaches_the_trajectory():
    env = ScoreByLength()
    groups = env.rollout(env.tasks(1), _policy(["abc"]), group_size=1)
    assert groups[0][0].info["length"] == 3.0


def test_fidelity_failures_are_dropped_and_counted():
    env = ScoreByLength()
    # "" -> zero tokens -> fidelity_ok() False
    groups = env.rollout(env.tasks(2), _policy(["a", "", "ccc", "dddd"]), group_size=2)

    assert [len(g) for g in groups] == [1, 2]
    assert groups[0][0].info["samples_dropped_fidelity"] == 1.0
    assert env.calls == ["a", "ccc", "dddd"]  # never graded


def test_drop_counter_lost_when_first_group_empty():
    """The pre-existing quirk, pinned so a refactor cannot change it silently:
    the counter rides on groups[0][0], so an empty first group discards it."""
    env = ScoreByLength()
    groups = env.rollout(env.tasks(2), _policy(["", "", "ccc", "dddd"]), group_size=2)

    assert [len(g) for g in groups] == [0, 2]
    assert all("samples_dropped_fidelity" not in t.info for g in groups for t in g)


def test_datums_carry_prompt_tokens():
    env = ScoreByLength()
    groups = env.rollout(env.tasks(1), _policy(["abc"]), group_size=1)
    datum = groups[0][0].datums[0]
    assert datum.prompt_len > 0
    assert len(datum.tokens) == datum.prompt_len + 3


@pytest.mark.parametrize("workers", [1, 4])
def test_pooled_and_inline_agree(workers):
    """grade_workers is a latency knob only: the pool must not reorder results
    or change which sample lands in which group."""
    env = ScoreByLength(workers=workers)
    script = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"]
    groups = env.rollout(env.tasks(3), _policy(script), group_size=2)

    assert [[t.reward for t in g] for g in groups] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert [[t.info["length"] for t in g] for g in groups] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


class ExplodingGrader(ScoreByLength):
    def reward(self, task, text):
        if text == "boom":
            raise RuntimeError("verifier crashed")
        return super().reward(task, text)


@pytest.mark.parametrize("workers", [1, 4])
def test_reward_exceptions_propagate(workers):
    """A crashing grader must fail the rollout, pooled or not — never be
    swallowed into a silent zero-reward trajectory."""
    env = ExplodingGrader(workers=workers)
    with pytest.raises(RuntimeError, match="verifier crashed"):
        env.rollout(env.tasks(2), _policy(["a", "boom", "ccc", "dddd"]), group_size=2)


def test_pooled_reward_failure_cancels_work_that_has_not_started():
    """A failed verifier invalidates the batch; queued candidates must not run."""
    release_second = threading.Event()
    second_started = threading.Event()

    class FailFastEnv(ScoreByLength):
        def reward(self, task, text):
            self.calls.append(text)
            if text == "boom":
                assert second_started.wait(timeout=1)
                raise RuntimeError("verifier crashed")
            if text == "running":
                second_started.set()
                release_second.wait(timeout=0.2)
            elif text.startswith("queued-"):
                # If a worker wins the small race between publishing the
                # exception and the caller cancelling futures, keep it from
                # draining the remaining queue before cancellation lands.
                release_second.wait(timeout=0.2)
            return float(len(text)), {"length": float(len(text))}

    env = FailFastEnv(workers=2)
    try:
        with pytest.raises(RuntimeError, match="verifier crashed"):
            env.rollout(
                env.tasks(10),
                _policy(["boom", "running", *[f"queued-{i}" for i in range(8)]]),
                group_size=1,
            )
    finally:
        release_second.set()

    assert "boom" in env.calls
    # The two workers plus at most one dequeue race may start; the remaining
    # submissions are cancelled rather than executing a doomed evaluation.
    assert len(env.calls) <= 3
