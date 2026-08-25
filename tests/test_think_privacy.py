"""The think channel is PRIVATE: no form of a completion's reasoning may
enter the public transcript text. The pre-opened-template case (completion
carries only '</think>') leaked 27% of transcript views in the live 32B
think debate arm (2026-08-12) — _split_think's pair regex never matched."""

from types import SimpleNamespace

from infra.backend.base import Sample
from infra.envs.debate.round import _split_think


def _stub_sample(text: str) -> Sample:
    tokens = list(range(1, len(text) + 1))
    return Sample(
        tokens=tokens,
        logprobs=[-0.1] * len(tokens),
        text=text,
        stop_reason="length",
        prompt_tokens=[99],
    )


def test_paired_tags_split():
    think, text = _split_think("<think>secret reasoning</think>The answer is 5.")
    assert think == "secret reasoning"
    assert text == "The answer is 5."


def test_preopened_template_close_only_split():
    # Chat template pre-opens <think>: completion has ONLY the close.
    think, text = _split_think("secret reasoning about the problem</think>The answer is 5.")
    assert think == "secret reasoning about the problem"
    assert text == "The answer is 5."
    assert "secret" not in text and "</think>" not in text


def test_no_think_passthrough():
    think, text = _split_think("Just an answer: 5.")
    assert think is None and text == "Just an answer: 5."


def test_multiline_preopened():
    raw = "line one\nline two\n</think>\n\nFinal: \\boxed{7}"
    think, text = _split_think(raw)
    assert think == "line one\nline two"
    assert text == "Final: \\boxed{7}"


def test_unterminated_preopened_think_is_never_published():
    """The budget ran out before the model closed its pre-opened <think>, so
    every token is reasoning. It must not become the public speech: that is
    the live failure where 7/12 critiques, 6/12 defenses and 8/12 rebuttals
    reached the judge as raw scratchpad."""
    raw = "The user wants me to argue as the Critic. Let me solve it myself first"
    think, text = _split_think(raw, thinking_opened=True)
    assert text == ""
    assert think == raw
    assert "argue as the Critic" not in text


def test_unterminated_think_stays_speech_for_a_non_thinking_seat():
    """A seat whose template never opens <think> emits plain speech with no
    tags. Swallowing THAT would silence every frozen API debater."""
    think, text = _split_think("Just an answer: 5.", thinking_opened=False)
    assert think is None and text == "Just an answer: 5."


def test_thinking_opened_does_not_change_a_closed_block():
    think, text = _split_think("reasoning</think>The answer is 5.", thinking_opened=True)
    assert think == "reasoning" and text == "The answer is 5."


def test_empty_completion_reports_no_thinking():
    think, text = _split_think("   ", thinking_opened=True)
    assert think is None and text == ""


def test_trained_seat_publishes_nothing_when_its_think_never_closes():
    """The wiring, not just the helper: a seat whose template opened <think>
    and whose slot has no think cap (so no budget forcing, no regions) must
    hand the round an EMPTY speech with the reasoning kept private."""
    from infra.envs.base import SlotLimits
    from infra.envs.debate.round import GenRequest, PolicySeat

    raw = "I need to argue the answer is wrong. Let me redo the algebra first"

    class _ThinkingPolicy:
        chat_template_kwargs = {"enable_thinking": True}

        def predict(self, convos, n=1, limits=None):
            return [[_stub_sample(raw)] for _ in convos]

    req = [GenRequest(messages=[{"role": "user", "content": "critique"}],
                      limits=SlotLimits(max_total_tokens=768))]
    result = PolicySeat(_ThinkingPolicy()).generate(req)[0]
    assert result.text == ""
    assert result.thinking == raw


def test_trained_seat_keeps_speech_when_the_seat_does_not_think():
    from infra.envs.base import SlotLimits
    from infra.envs.debate.round import GenRequest, PolicySeat

    class _PlainPolicy:
        chat_template_kwargs = {"enable_thinking": False}

        def predict(self, convos, n=1, limits=None):
            return [[_stub_sample("Your step 3 is wrong.")] for _ in convos]

    req = [GenRequest(messages=[{"role": "user", "content": "critique"}],
                      limits=SlotLimits(max_total_tokens=768))]
    result = PolicySeat(_PlainPolicy()).generate(req)[0]
    assert result.text == "Your step 3 is wrong."
    assert result.thinking is None


def test_empty_speeches_are_counted_per_slot():
    """An empty speech is the safe outcome, not a harmless one: the judge
    rules on a debate where a seat said nothing. The census is what makes a
    missing think budget visible without reading transcripts."""
    from infra.envs.debate.env import DebateEnv
    from infra.envs.debate.protocol import Protocol
    from infra.envs.debate.round import DebateState, SlotRecord

    protocol = Protocol.parse(
        {"turns": [{"alice": [{"name": "proposal", "kind": "solution", "max_total_tokens": 8}]},
                   {"bob": [{"name": "critique", "max_total_tokens": 8}]}]}
    )
    compiled = {cs.slot.name: cs for cs in protocol.compile()}
    state = DebateState(bindings={})
    state.records = [
        SlotRecord(slot=compiled["proposal"], text="The answer is 5."),
        SlotRecord(slot=compiled["critique"], text=""),
        SlotRecord(slot=compiled["critique"], text="   "),
    ]
    env = DebateEnv.__new__(DebateEnv)
    env.config = SimpleNamespace(trained_speakers=["alice", "bob"])

    census = env._speech_census([state])
    assert census["trained_speeches"] == 3.0
    assert census["empty_speeches"] == 2.0
    assert census["empty_speeches_by_slot"] == {"critique": 2.0}


def test_shipped_qwen_debate_arm_caps_every_trained_slot():
    """The arm that runs on the pod: each trained slot leaves room to speak
    after its think phase closes, and the proposal is the RLVR arm's own
    generation budget."""
    from pathlib import Path

    from infra.config import load_experiment

    configs = Path(__file__).resolve().parents[1] / "configs"
    exp = load_experiment(configs / "math_pc_debate.yaml", "mathl5_qwen35_pc_debate_cispo_verl")
    rlvr = load_experiment(configs / "math_qwen35.yaml", "mathl5_qwen35_cispo")

    slots = {s["name"]: s for turn in exp["protocol"]["turns"] for sl in turn.values() for s in sl}
    for name in ("proposal", "critique", "defense", "rebuttal"):
        s = slots[name]
        assert s["max_think_tokens"] is not None, name
        assert s["max_think_tokens"] < s["max_total_tokens"], name
    assert slots["proposal"]["max_think_tokens"] == rlvr["think_tokens"]
    assert (
        slots["proposal"]["max_total_tokens"]
        == rlvr["think_tokens"] + rlvr["max_completion_tokens"]
    )
