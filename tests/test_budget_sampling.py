"""Budget-forced sampling: phase structure, injection, regions, masks."""

from infra.backend.base import SamplingParams
from infra.envs.base import SlotLimits, budget_forced_sample, datum_from_sample

from infra.backend.base import Sample


class CharTok:
    """Tokens are codepoints; decode/encode are exact inverses."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


TOK = CharTok()
CLOSE = "</think>\n\n"


def scripted_backend(script):
    """script: list of texts (or (text, stop_reason)) returned per call, per prompt."""
    calls = []

    def sample_fn(prompts, params, n=1):
        calls.append({"prompts": prompts, "params": params, "n": n})
        outs = script.pop(0)
        result = []
        for i in range(len(prompts)):
            group = []
            for j in range(n):
                spec = outs[i * n + j] if isinstance(outs, list) else outs
                text, reason = spec if isinstance(spec, tuple) else (spec, "stop")
                toks = TOK.encode(text)
                group.append(
                    Sample(tokens=toks, logprobs=[-0.1] * len(toks), text=text, stop_reason=reason)
                )
            result.append(group)
        return result

    return sample_fn, calls


def run(script, limits, n=1, prompt="<think>"):
    fn, calls = scripted_backend(script)
    prompts = [TOK.encode(prompt)]
    out = budget_forced_sample(fn, TOK, prompts, SamplingParams(max_tokens=500), limits, n=n)
    return out[0], calls


def test_forced_injection_masks_close():
    # phase 1 hits the cap mid-think (no </think>); close must be injected
    [s], calls = run(
        [[("reasoning without close", "length")], [("the answer", "stop")]],
        SlotLimits(max_think_tokens=30),
    )
    assert calls[0]["params"].stop == ["</think>"]
    assert calls[0]["params"].max_tokens == 30
    decoded = TOK.decode(s.tokens)
    assert CLOSE in decoded and decoded.endswith("the answer")
    kinds = [r.kind for r in s.regions]
    assert kinds == ["think", "forced_close", "visible"]
    fc = next(r for r in s.regions if r.kind == "forced_close")
    assert all(lp == 0.0 for lp in s.logprobs[fc.start : fc.end])
    assert s.stop_reason == "stop"
    s.prompt_tokens = TOK.encode("<think>")
    d = datum_from_sample(s)
    assert d.mask is not None
    assert all(m == 0.0 for m in d.mask[fc.start : fc.end])
    assert all(m == 1.0 for m in d.mask[: fc.start]) and all(m == 1.0 for m in d.mask[fc.end :])


def test_natural_close_no_injection():
    [s], _ = run(
        [[("short thought</think>", "stop")], [("answer text", "stop")]],
        SlotLimits(max_think_tokens=100),
    )
    kinds = [r.kind for r in s.regions]
    assert kinds == ["think", "visible"]  # no forced_close
    assert all(lp == -0.1 for lp in s.logprobs)  # all sampled
    s.prompt_tokens = []
    assert datum_from_sample(s).mask is None


def test_died_in_think():
    [s], calls = run(
        [[("infinite pondering" + chr(0), "stop")]],
        SlotLimits(max_think_tokens=100),
    )
    assert len(calls) == 1  # no phase 2
    assert [r.kind for r in s.regions] == ["think"]
    assert s.stop_reason == "stop"


def test_never_opens_think():
    [s], calls = run(
        [[("just an answer, no thinking", "stop")]],
        SlotLimits(max_think_tokens=50),
        prompt="plain prompt",  # template did not open a think block
    )
    assert len(calls) == 1
    assert [r.kind for r in s.regions] == ["visible"]


def test_visible_cap_binds_phase2():
    [s], calls = run(
        [[("t</think>", "stop")], [("x" * 400, "length")]],
        SlotLimits(max_visible_tokens=40),
    )
    assert calls[1]["params"].max_tokens == 40
    assert s.stop_reason == "length"


def test_n_gt_1_phase_per_sample():
    samples, calls = run(
        [
            [("a</think>", "stop"), ("b no close", "length")],
            # max_visible binds both continuations to one cap -> ONE phase-2 call
            [("ans1", "stop"), ("ans2", "stop")],
        ],
        SlotLimits(max_think_tokens=60, max_visible_tokens=40),
        n=2,
    )
    assert len(samples) == 2 and len(calls) == 2
    assert calls[1]["n"] == 1 and len(calls[1]["prompts"]) == 2
    k0 = [r.kind for r in samples[0].regions]
    k1 = [r.kind for r in samples[1].regions]
    assert k0 == ["think", "visible"] and k1 == ["think", "forced_close", "visible"]


class SpecialTok(CharTok):
    """Same exact-inverse codec, with a special-token vocabulary."""

    all_special_tokens = ["<|im_end|>", "<think>", "</think>"]


def run_special(script, limits, prompt="<think>"):
    fn, calls = scripted_backend(script)
    out = budget_forced_sample(
        fn, SpecialTok(), [TOK.encode(prompt)], SamplingParams(max_tokens=500), limits
    )
    return out[0][0], calls


def test_rebuilt_text_strips_special_tokens_after_forced_close():
    """budget_forced_sample rebuilds Sample.text over the joined phases. A raw
    decode here puts the EOS string back into text the debate round splices
    into the next speaker's context, where the template re-tokenizes it into a
    turn boundary mid-message."""
    s, _ = run_special(
        [[("reasoning without close", "length")], [("the answer<|im_end|>", "stop")]],
        SlotLimits(max_think_tokens=30),
    )
    assert "<|im_end|>" not in s.text
    assert s.text.endswith("the answer")
    # the think markers are the split contract; they must survive
    assert "</think>" in s.text


def test_rebuilt_text_strips_special_tokens_on_the_all_visible_path():
    s, _ = run_special(
        [[("no thinking here<|im_end|>", "stop")]],
        SlotLimits(max_think_tokens=50),
        prompt="plain prompt",
    )
    assert s.text == "no thinking here"


def test_rebuilt_text_strips_special_tokens_when_it_dies_in_think():
    s, _ = run_special(
        [[("still reasoning<|im_end|>" + chr(0), "stop")]],
        SlotLimits(max_think_tokens=30),
    )
    assert [r.kind for r in s.regions] == ["think"]
    assert "<|im_end|>" not in s.text
    assert "still reasoning" in s.text
