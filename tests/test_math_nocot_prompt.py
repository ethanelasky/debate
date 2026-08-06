"""math_nocot.yaml: the neutral (no-CoT-prompt) variant for the CoT-discovery
experiments. Contract: derived from math.yaml by deletion only — no CoT
elicitation ("step-by-step", "rigorous"), no CoT suppression ("only",
"do not"), boxed-answer requirement retained, and the debate proposal slot
renders it byte-identically to the RLVR user message (same single-source
splice as math.yaml)."""

from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file

NOCOT = "infra/prompts/tasks/math_nocot.yaml"


def _prompts():
    return load_generation_prompts(resolve_prompt_file(NOCOT, "unused"))


def _system(p) -> str:
    return next(m["content"] for m in p.messages if m["role"] == "system")


def _user(p) -> str:
    return p.supplied_templates()["ANSWER_GEN_USER"]


def test_no_cot_elicitation_or_suppression():
    p = _prompts()
    text = (_system(p) + "\n" + _user(p)).lower()
    for banned in ("step-by-step", "step by step", "rigorous", "reason"):
        assert banned not in text  # does not ask for CoT
    for banned in ("only", "do not", "no prose"):
        assert banned not in text  # does not forbid CoT either


def test_boxed_requirement_and_placeholder_retained():
    p = _prompts()
    assert "EXACTLY one \\boxed{...} on the last line" in _user(p)
    assert "<PROBLEM>" in _user(p)


def test_lines_are_a_subset_of_math_yaml():
    """Deletion-only contract: every non-empty line is a math.yaml line, or a
    leading fragment of one (deletion of a trailing clause within a line)."""
    base = load_generation_prompts(resolve_prompt_file("infra/prompts/tasks/math.yaml", "unused"))
    base_lines = (_system(base) + "\n" + _user(base)).splitlines()
    p = _prompts()
    for line in (_system(p) + "\n" + _user(p)).splitlines():
        if line.strip():
            assert any(
                b == line or b.startswith(line.rstrip()) for b in base_lines
            ), f"authored (not deleted) line: {line!r}"


def test_debate_proposal_splices_nocot_bytes():
    """The debate pack's proposal slot IS the nocot user message (with <PROBLEM>
    rebound to <TOPIC>) — the same parity contract math.yaml has."""
    from infra.envs.debate.env import _splice
    from infra.envs.debate.prompts import load_prompt_library
    from infra.envs.debate.protocol import Protocol

    import yaml

    proto = Protocol.parse(
        yaml.safe_load(
            "turns: [{alice: [{name: proposal, kind: solution}]}, {bob: [{name: critique}]},"
            " {alice: [{name: defense}]}, {judge: [{name: verdict, kind: decision}]}]"
        )
    )
    lib = load_prompt_library(
        "infra/prompts/debate/hendrycks_math.yaml", "math_proposer_critic", proto
    )
    p = _prompts()
    subs = {k: v.replace("<PROBLEM>", "<TOPIC>") for k, v in p.supplied_templates().items()}
    lib.slots = {n: _splice(t, subs) for n, t in lib.slots.items()}
    rendered = str(lib.slots["proposal"])
    assert _user(p).replace("<PROBLEM>", "<TOPIC>") in rendered
