import pytest

from infra.envs.debate.prompts import (
    PromptLibrary,
    RenderedPrompts,
    load_prompt_library,
    render,
    slot_template,
    validate_prompts,
)
from infra.envs.debate.topology import Topology
from infra.envs.task_prompts import load_generation_prompts, resolve_prompt_file


def task_prompts(name: str):
    """The task family's answer-generation config — the single source for
    format wording the debate yamls no longer restate."""
    return load_generation_prompts(resolve_prompt_file(None, name))


MATH_YAML = "infra/envs/debate/prompt_configs/hendrycks_math.yaml"
CODECONTESTS_YAML = "infra/envs/debate/prompt_configs/codecontests.yaml"

# DESIGN-debate-env.md §1, proposer_critic_2round.
PC_TOPOLOGY = {
    "turns": [
        {"alice": [{"name": "proposal", "kind": "solution"}]},
        {"bob": [{"name": "critique"}]},
        {"alice": [{"name": "defense"}]},
        {"bob": [{"name": "rebuttal"}]},
        {"judge": [{"name": "deliberation", "visibility": "private"},
                   {"name": "verdict", "kind": "decision"}]},
    ]
}

BINDINGS = {
    "NAME": "Alice",
    "OPPONENT_NAME": "Bob",
    "TOPIC": "What is 2+2?",
    "POSITION": "4",
    "OPPONENT_POSITION": "5",
}


def splice_task_prompts(lib, name):
    """What DebateEnv does before rendering: splice the task family's
    answer_gen_user into slots that ask for it by <ANSWER_GEN_USER>, with
    <PROBLEM> rebound to this layer's <TOPIC>. Tests that render a pack's slots
    need the same treatment, since the packs no longer carry that wording."""
    from infra.envs.debate.env import _splice

    subs = {
        k: v.replace("<PROBLEM>", "<TOPIC>")
        for k, v in task_prompts(name).supplied_templates().items()
    }
    lib.slots = {
        n: ({s: _splice(t, subs) for s, t in e.items()} if isinstance(e, dict) else _splice(e, subs))
        for n, e in lib.slots.items()
    }
    return lib


@pytest.fixture
def lib():
    return splice_task_prompts(
        load_prompt_library(MATH_YAML, "math_proposer_critic"), "math.yaml"
    )


@pytest.fixture
def topology():
    return Topology.parse(PC_TOPOLOGY)


# ------------------------------------------------------------------ loading


def test_extends_merges_vars_over_parent(lib, tmp_path):
    assert set(lib.system) == {"alice", "bob", "judge"}
    assert set(lib.slots) == {"scratchpad", "proposal", "critique", "defense", "rebuttal", "deliberation", "verdict"}

    path = tmp_path / "p.yaml"
    path.write_text(
        "_base:\n"
        "  vars: {A: '1', B: '2'}\n"
        "  system: {alice: 'base <A>'}\n"
        "  slots: {s: 'x'}\n"
        "child:\n"
        "  _extends: _base\n"
        "  vars: {B: 'two'}\n"
        "  slots: {t: 'y'}\n"
    )
    child = load_prompt_library(path, "child")
    assert child.vars == {"A": "1", "B": "two"}
    assert child.system == {"alice": "base <A>"}
    assert child.slots == {"s": "x", "t": "y"}


def test_unknown_entry_raises(lib):
    with pytest.raises(KeyError, match="math_proposer_critic"):
        load_prompt_library(MATH_YAML, "nope")


# ---------------------------------------------------------- slot_template


def test_string_template_serves_any_speaker(lib):
    assert slot_template(lib, "proposal", "alice") is slot_template(lib, "proposal", "zebra")


def test_map_template_is_per_speaker():
    lib = PromptLibrary(slots={"closing": {"alice": "case FOR", "bob": "case AGAINST"}})
    assert slot_template(lib, "closing", "alice") == "case FOR"
    assert slot_template(lib, "closing", "bob") == "case AGAINST"
    with pytest.raises(KeyError) as e:
        slot_template(lib, "closing", "judge")
    assert "closing" in str(e.value) and "judge" in str(e.value)


def test_missing_slot_named_in_message(lib):
    with pytest.raises(KeyError, match="ghost"):
        slot_template(lib, "ghost", "alice")


# ------------------------------------------------------------------ render


def test_vars_applied_and_bindings_override():
    out = render("<TOPIC> | <NOTE>", {"TOPIC": "t"}, {"NOTE": "from vars"})
    assert out == "t | from vars"
    assert render("<A>", {"A": "binding"}, {"A": "var"}) == "binding"


def test_unbound_uppercase_placeholder_raises():
    with pytest.raises(ValueError, match="MISSING"):
        render("hello <MISSING>", {"OTHER": "x"})


def test_lowercase_tags_pass_through(lib):
    out = render(slot_template(lib, "proposal", "alice"), {"TOPIC": "2+2"}, lib.vars)
    assert "<problem>2+2</problem>" in out
    assert "<TOPIC>" not in out
    crit = render(slot_template(lib, "critique", "bob"), {"OPPONENT_POSITION": "4"}, lib.vars)
    assert "boxed answer is 4" in crit


def test_substituted_text_is_not_rescanned():
    assert render("<A>", {"A": "<NOT_A_PLACEHOLDER>"}) == "<NOT_A_PLACEHOLDER>"


# --------------------------------------------------------- validate_prompts


def test_validate_passes_for_math_pc(lib, topology):
    validate_prompts(lib, topology, fresh_positions=True)
    validate_prompts(lib, topology, fresh_positions=False)


def test_validate_passes_for_codecontests_pc(topology):
    cc = load_prompt_library(CODECONTESTS_YAML, "codecontests_proposer_critic")
    assert set(cc.system) == {"alice", "bob", "judge"}
    assert set(cc.slots) == set(load_prompt_library(MATH_YAML, "math_proposer_critic").slots)
    # The proposal slot is single-sourced from the task family's
    # answer-generation config (the RLVR arm sends the same composition) and is
    # spliced in by DebateEnv, so the debate yaml must NOT restate the wording.
    assert slot_template(cc, "proposal", "alice").strip() == "<ANSWER_GEN_USER>"
    assert "```python" in task_prompts("codecontests.yaml").answer_gen_user
    validate_prompts(cc, topology, fresh_positions=True)
    validate_prompts(cc, topology, fresh_positions=False)


def test_codecontests_pc_is_forced_disagree_with_old_pack_methodology():
    """Competitive forced-disagree format (Ethan, 2026-08-03), carrying the old
    code-review pack's two-phase tracing methodology but none of its
    collaborative stance machinery."""
    cc = load_prompt_library(CODECONTESTS_YAML, "codecontests_proposer_critic")
    critique = slot_template(cc, "critique", "bob")
    assert "You are FORCED to disagree" in critique
    assert "PHASE 1 — PLAN YOUR TEST INPUTS" in critique
    assert "PHASE 2 — TRACE EACH INPUT" in critique
    assert "CRITICAL — you are FORCED to disagree" in cc.system["bob"]
    assert "NEVER concede that the Proposer is right" in cc.system["bob"]
    # no collaborative machinery anywhere in the pack
    whole = "\n".join(list(cc.system.values()) + [slot_template(cc, s, "judge") for s in cc.slots])
    for banned in ("AGREE", "DISAGREE", "COLLABORATIVE", "both can win", '"stance"'):
        assert banned not in whole, banned
    # correctness-only: TLE/complexity arguments are out of scope for everyone
    for speaker in ("alice", "bob"):
        assert "Do NOT argue about time complexity" in cc.system[speaker]
    assert "Ignore efficiency, style, and performance" in cc.system["judge"]
    assert "Time complexity, efficiency, or performance arguments" in cc.system["judge"]


def test_codecontests_verdict_slot_matches_competitive_parser():
    """Single winner-JSON verdict — the competitive default, same schema text
    as the math pack, so no judge_config override is needed."""
    from infra.envs.debate.judge import JudgeConfig, SeatVerdict, parse_verdict, verdict_from_slot

    cc = load_prompt_library(CODECONTESTS_YAML, "codecontests_proposer_critic")
    math = load_prompt_library(MATH_YAML, "math_proposer_critic")
    seats = ["Debater_A", "Debater_B"]
    tmpl = render(
        slot_template(cc, "verdict", "judge"),
        {"NAME": seats[0], "OPPONENT_NAME": seats[1]},
        cc.vars,
    )
    schema_line = '{"winner": "Debater_A" | "Debater_B", "confidence": 0.50-1.00}'
    assert schema_line in tmpl
    # schema text is character-identical to math's, and the alternation line is
    # not valid JSON, so nothing parses off the example
    assert schema_line in render(
        slot_template(math, "verdict", "judge"),
        {"NAME": seats[0], "OPPONENT_NAME": seats[1]},
        math.vars,
    )
    # the Tie example in the template is itself valid JSON (inherited from
    # math), so a judge echoing the template must be read from its LAST object
    assert parse_verdict(tmpl + '\n{"winner": "Debater_A", "confidence": 0.9}', "competitive", seats) == {
        "winner": "Debater_A",
        "confidence": 0.9,
    }
    filled = 'The Proposer is right.\n{"winner": "Debater_A", "confidence": 0.80}'
    v = verdict_from_slot(filled, None, None, JudgeConfig(), seats)  # competitive default
    assert v.ok and v.winner == seats[0]
    assert v.seats == {seats[0]: SeatVerdict.CORRECT, seats[1]: SeatVerdict.INCORRECT}
    # the Tie rule the pack documents
    tie = verdict_from_slot('{"winner": "Tie", "confidence": 0.5}', None, None, JudgeConfig(), seats)
    assert tie.ok and tie.winner is None
    # the old collaborative parser cannot read it — the pack is no longer collaborative
    assert parse_verdict(filled, "collaborative", seats) is None


def test_validate_fails_on_missing_slot_template(lib, topology):
    lib.slots.pop("rebuttal")
    with pytest.raises(ValueError, match="rebuttal"):
        validate_prompts(lib, topology)


def test_validate_fails_on_missing_system_prompt(lib, topology):
    lib.system.pop("judge")
    with pytest.raises(ValueError, match="judge"):
        validate_prompts(lib, topology)


def test_validate_fails_bindability_when_position_in_solution_slot(lib, topology):
    lib.slots["proposal"] = lib.slots["proposal"] + "\nYour answer is <POSITION>."
    with pytest.raises(ValueError, match="POSITION"):
        validate_prompts(lib, topology, fresh_positions=True)
    validate_prompts(lib, topology, fresh_positions=False)  # assigned mode: fine


def test_validate_fails_when_critic_needs_position_before_any_solution(lib):
    topo = Topology.parse({"turns": [{"bob": [{"name": "critique"}]}]})
    lib.system = {"bob": lib.system["bob"]}
    with pytest.raises(ValueError, match="OPPONENT_POSITION"):
        validate_prompts(lib, topo, fresh_positions=True)


def test_bindability_ignores_placeholders_covered_by_vars(topology):
    lib = PromptLibrary(
        system={s: "sys" for s in topology.speakers},
        slots={
            "proposal": "<POSITION>",
            "critique": "c",
            "defense": "d",
            "rebuttal": "r",
            "deliberation": "del",
            "verdict": "v",
        },
        vars={"POSITION": "static"},
    )
    validate_prompts(lib, topology, fresh_positions=True)


# ------------------------------------------------------------ RenderedPrompts


def test_rendered_prompts_protocol(lib):
    rp = RenderedPrompts(lib)
    assert rp.system("alice", BINDINGS).startswith("<role>You are Alice, the PROPOSER")
    assert "<problem>What is 2+2?</problem>" in rp.instruction("proposal", "alice", BINDINGS)
    # judge reads the proposer's answer via OPPONENT_POSITION — the key the
    # round loop actually rebinds for non-solvers when a solution lands
    assert "boxed answer: 5" in rp.instruction("deliberation", "judge", BINDINGS)
    verdict = rp.instruction("verdict", "judge", BINDINGS)
    assert '{"winner": "Alice" | "Bob", "confidence": 0.50-1.00}' in verdict


def test_attributed_format_golden(lib):
    assert RenderedPrompts(lib).attributed("Bob", "critique", "you erred") == "Bob said:\nyou erred"


def test_rendered_prompts_missing_system_speaker(lib):
    with pytest.raises(KeyError, match="carol"):
        RenderedPrompts(lib).system("carol", BINDINGS)


def test_block_lists_join_and_extend_by_index(tmp_path):
    import yaml as _yaml

    from infra.envs.debate.prompts import load_prompt_library

    p = tmp_path / "p.yaml"
    p.write_text(
        _yaml.safe_dump(
            {
                "_base": {
                    "system": {"alice": ["identity block", "rules block", "format block"]},
                    "slots": {"speech": ["do the thing"]},
                },
                "child": {
                    "_extends": "_base",
                    # override ONLY block 1; keep 0 and 2; append block 3
                    "system": {"alice": ["identity block", "CHILD RULES", "format block", "extra"]},
                },
            }
        )
    )
    base = load_prompt_library(p, "_base")
    assert base.system["alice"] == "identity block\n\nrules block\n\nformat block"
    child = load_prompt_library(p, "child")
    assert child.system["alice"] == "identity block\n\nCHILD RULES\n\nformat block\n\nextra"
    assert child.slots["speech"] == "do the thing"  # inherited untouched


def test_preview_shows_block_placement():
    import yaml as _yaml

    from infra.envs.debate.prompts import load_prompt_library, preview
    from infra.envs.debate.topology import Topology

    lib = splice_task_prompts(
        load_prompt_library(
            "infra/envs/debate/prompt_configs/hendrycks_math.yaml", "math_proposer_critic"
        ),
        "math.yaml",
    )
    topo = Topology.parse(
        _yaml.safe_load(
            "turns: [{alice: [{name: proposal, kind: solution}]}, {bob: [{name: critique}]},"
            " {alice: [{name: defense}]}, {judge: [{name: verdict, kind: decision}]}]"
        )
    )
    out = preview(lib, topo)
    assert "<alice/proposal output>" in out          # stubbed prior slots
    assert "<TOPIC>" in out                          # runtime placeholders kept visible
    assert "EXACTLY one \\boxed{...}" in out         # spliced task wording renders for real
    assert "== judge — context for its final slot (verdict)" in out
