"""The think channel is PRIVATE: no form of a completion's reasoning may
enter the public transcript text. The pre-opened-template case (completion
carries only '</think>') leaked 27% of transcript views in the live 32B
think debate arm (2026-08-12) — _split_think's pair regex never matched."""

from infra.envs.debate.round import _split_think


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
