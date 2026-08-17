"""SingleTurnEnv: the RLVR rollout shape shared by every task-source env.

MathEnv and CodeContestsEnv had byte-for-byte equivalent rollout() methods
differing only in whether reward() ran in a thread pool. These pin the
behaviors both of them relied on, so the shared implementation cannot drift:
group shape, fidelity drops, and pooled == inline results.
"""

import concurrent.futures
import threading

import pytest

from infra.backend.base import Backend, Sample, SamplingParams
from infra.envs import base as env_base
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


class LengthStopBackend(ScriptedBackend):
    """Texts ending in "!" report stop_reason="length" (hit the token cap)."""

    def sample(self, prompts, params, n=1):
        out = super().sample(prompts, params, n)
        for samples in out:
            for s in samples:
                if s.text.endswith("!"):
                    s.stop_reason = "length"
        return out


def test_truncated_flag_reaches_info_and_aggregate():
    from infra.train import _aggregate

    env = ScoreByLength()
    policy = Policy(LengthStopBackend(["abc!", "ab"]), SamplingParams(max_tokens=32))
    groups = env.rollout(env.tasks(1), policy, group_size=2)
    assert [t.info["truncated"] for t in groups[0]] == [1.0, 0.0]
    assert _aggregate(groups[0], "train")["train/truncated"] == 0.5


def test_aggregate_reward_std_is_population_std():
    from infra.train import _aggregate

    env = ScoreByLength()
    groups = env.rollout(env.tasks(1), _policy(["a", "ccc"]), group_size=2)
    out = _aggregate(groups[0], "train")
    assert out["train/reward_mean"] == 2.0
    assert out["train/reward_std"] == 1.0
    same = env.rollout(env.tasks(1), _policy(["dd", "ee"]), group_size=2)
    assert _aggregate(same[0], "train")["train/reward_std"] == 0.0


def test_fidelity_failures_are_dropped_and_counted():
    env = ScoreByLength()
    # "" -> zero tokens -> fidelity_ok() False
    groups = env.rollout(env.tasks(2), _policy(["a", "", "ccc", "dddd"]), group_size=2)

    assert [len(g) for g in groups] == [1, 2]
    assert env.last_rollout_info["samples_dropped_fidelity"] == 1
    assert env.calls == ["a", "ccc", "dddd"]  # never graded


def test_drop_counter_survives_when_first_group_empty():
    env = ScoreByLength()
    groups = env.rollout(env.tasks(2), _policy(["", "", "ccc", "dddd"]), group_size=2)

    assert [len(g) for g in groups] == [0, 2]
    assert env.last_rollout_info["samples_dropped_fidelity"] == 2
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


def test_pooled_reward_failure_aborts_queued_calls_and_preserves_original_error(
    monkeypatch,
):
    """A later-position failure is observed while the first result blocks.

    The patched event lets the already-running call leave only after the pool
    has recorded the fatal error.  Any queued wrapper that a freed worker
    picks up must then decline to enter reward().
    """
    abort = threading.Event()
    running_started = threading.Event()
    all_submitted = threading.Event()
    failure = RuntimeError("fatal verifier failure")

    real_executor = concurrent.futures.ThreadPoolExecutor

    class DrainQueuedExecutor(real_executor):
        """Test pool that drains queued wrappers even when asked to cancel."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.submissions = 0

        def submit(self, fn, /, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            self.submissions += 1
            if self.submissions == 4:
                all_submitted.set()
            return future

        def shutdown(self, wait=True, *, cancel_futures=False):
            return super().shutdown(wait=wait, cancel_futures=False)

    class CoordinatedFailure(ScoreByLength):
        def __init__(self):
            super().__init__(workers=2)
            self.entered = []

        def reward(self, task, text):
            self.entered.append(text)
            if text == "running":
                running_started.set()
                if not abort.wait(timeout=2):
                    raise TimeoutError("pool never published its abort")
                return 1.0, {"length": 1.0}
            if text == "fatal":
                if not running_started.wait(timeout=2):
                    raise TimeoutError("the first worker never started")
                if not all_submitted.wait(timeout=2):
                    raise TimeoutError("queued work was not submitted")
                raise failure
            raise AssertionError(f"queued reward entered after fatal error: {text}")

    # _fail_fast_thread_map creates exactly one cooperative event per batch.
    monkeypatch.setattr(env_base, "Event", lambda: abort)
    monkeypatch.setattr(
        env_base.concurrent.futures,
        "ThreadPoolExecutor",
        DrainQueuedExecutor,
    )
    # Disable both cancellation paths: queued wrappers must actually execute,
    # proving their abort check (rather than Future.cancel) blocks reward().
    monkeypatch.setattr(env_base.concurrent.futures.Future, "cancel", lambda self: False)
    env = CoordinatedFailure()

    with pytest.raises(RuntimeError, match="fatal verifier failure") as caught:
        env.rollout(
            env.tasks(1),
            _policy(["running", "fatal", "queued-a", "queued-b"]),
            group_size=4,
        )

    assert caught.value is failure
    assert set(env.entered) == {"running", "fatal"}


def test_pooled_reward_preserves_submission_order_after_out_of_order_completion():
    second_finished = threading.Event()

    class ReverseCompletion(ScoreByLength):
        def __init__(self):
            super().__init__(workers=2)
            self.completion_order = []

        def reward(self, task, text):
            if text == "first":
                if not second_finished.wait(timeout=2):
                    raise TimeoutError("second reward never completed")
            else:
                second_finished.set()
            self.completion_order.append(text)
            return float(len(text)), {"length": float(len(text))}

    env = ReverseCompletion()
    groups = env.rollout(env.tasks(1), _policy(["first", "second"]), group_size=2)

    assert env.completion_order == ["second", "first"]
    assert [trajectory.reward for trajectory in groups[0]] == [5.0, 6.0]
