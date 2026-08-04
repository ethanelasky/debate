import pytest

from infra.envs.debate.prompts import (
    PromptLibrary,
    RenderedPrompts,
    load_prompt_library,
    render,
    slot_template,
    validate_prompts,
)
from infra.envs.debate.protocol import Protocol
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


def load_pack(path: str, entry: str, proto=None):
    """Packs are role-conditioned, so loading one needs a protocol."""
    return load_prompt_library(path, entry, proto or Protocol.parse(PC_TOPOLOGY))


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
        load_pack(MATH_YAML, "math_proposer_critic"), "math.yaml"
    )


@pytest.fixture
def protocol():
    return Protocol.parse(PC_TOPOLOGY)


# ------------------------------------------------------------------ loading


MINIMAL_ENTRY = (
    "_base:\n"
    "  vars: {A: '1', B: '2'}\n"
    "  overall_system: 'shared <A>'\n"
    "  slot_stages: {proposal: pre_opening_speech_proposer}\n"
    "  pre_opening_speech_proposer: 'x'\n"
    "child:\n"
    "  _extends: _base\n"
    "  vars: {B: 'two'}\n"
    "  debater_system_proposer: 'proposer card'\n"
)


def test_extends_merges_vars_over_parent(lib, tmp_path):
    assert set(lib.system) == {"alice", "bob", "judge"}
    assert set(lib.slots) == {"scratchpad", "proposal", "critique", "defense", "rebuttal", "deliberation", "verdict"}

    path = tmp_path / "p.yaml"
    path.write_text(MINIMAL_ENTRY)
    proto = Protocol.parse({"turns": [{"alice": [{"name": "proposal", "kind": "solution"}]}]})
    child = load_prompt_library(path, "child", proto)
    assert child.vars == {"A": "1", "B": "two"}
    assert child.system == {"alice": "shared <A>\n\nproposer card"}
    assert child.slots == {"proposal": "x"}


def test_unknown_entry_raises(lib):
    with pytest.raises(KeyError, match="math_proposer_critic"):
        load_pack(MATH_YAML, "nope")


def test_load_without_protocol_refuses(tmp_path):
    """Stages are role-conditioned; there is nothing to compose without the
    protocol that says which seat proposes, critiques and judges."""
    path = tmp_path / "p.yaml"
    path.write_text(MINIMAL_ENTRY)
    with pytest.raises(ValueError, match="needs the protocol"):
        load_prompt_library(path, "child")


def test_unrendered_stage_key_rejected(tmp_path):
    """A typo'd role stage, or a cue stage slot_stages stopped naming, is dead
    prompt text — which is exactly how the old repo's pre_debate went dark."""
    path = tmp_path / "p.yaml"
    path.write_text(MINIMAL_ENTRY + "  pre_debate_typo: 'oops'\n")
    proto = Protocol.parse({"turns": [{"alice": [{"name": "proposal", "kind": "solution"}]}]})
    with pytest.raises(ValueError, match="pre_debate_typo"):
        load_prompt_library(path, "child", proto)


def test_slot_stages_naming_undefined_stage_rejected(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(
        "solo:\n"
        "  overall_system: 'sys'\n"
        "  slot_stages: {proposal: nope}\n"
    )
    proto = Protocol.parse({"turns": [{"alice": [{"name": "proposal", "kind": "solution"}]}]})
    with pytest.raises(ValueError, match="nope"):
        load_prompt_library(path, "solo", proto)


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


def test_validate_passes_for_math_pc(lib, protocol):
    validate_prompts(lib, protocol, fresh_positions=True)
    validate_prompts(lib, protocol, fresh_positions=False)


def test_validate_passes_for_codecontests_pc(protocol):
    cc = load_pack(CODECONTESTS_YAML, "codecontests_proposer_critic")
    assert set(cc.system) == {"alice", "bob", "judge"}
    assert set(cc.slots) == set(load_pack(MATH_YAML, "math_proposer_critic").slots)
    # The proposal slot is single-sourced from the task family's
    # answer-generation config (the RLVR arm sends the same composition) and is
    # spliced in by DebateEnv, so the debate yaml must NOT restate the wording.
    assert slot_template(cc, "proposal", "alice").strip() == "<ANSWER_GEN_USER>"
    assert "```python" in task_prompts("codecontests.yaml").answer_gen_user
    validate_prompts(cc, protocol, fresh_positions=True)
    validate_prompts(cc, protocol, fresh_positions=False)


def test_codecontests_pc_is_stance_framed_with_old_pack_methodology():
    """Competitive format (Ethan, 2026-08-03) carrying the old code-review
    pack's two-phase tracing methodology but none of its collaborative
    stance-choice machinery. The critic's disagreement is framed as its ROLE,
    not as compulsion (Ethan, 2026-08-04)."""
    cc = load_pack(CODECONTESTS_YAML, "codecontests_proposer_critic")
    critique = slot_template(cc, "critique", "bob")
    assert "Your role is to prove the program INCORRECT" in critique
    assert "PHASE 1 — PLAN YOUR TEST INPUTS" in critique
    assert "PHASE 2 — TRACE EACH INPUT" in critique
    assert "Your role is to prove the Proposer wrong" in cc.system["bob"]
    # the strategy line survives the reframing: consistent with the stance
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

    cc = load_pack(CODECONTESTS_YAML, "codecontests_proposer_critic")
    math = load_pack(MATH_YAML, "math_proposer_critic")
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


def test_validate_fails_on_missing_slot_template(lib, protocol):
    lib.slots.pop("rebuttal")
    with pytest.raises(ValueError, match="rebuttal"):
        validate_prompts(lib, protocol)


def test_validate_fails_on_missing_system_prompt(lib, protocol):
    lib.system.pop("judge")
    with pytest.raises(ValueError, match="judge"):
        validate_prompts(lib, protocol)


def test_judge_never_told_a_position_was_assigned(protocol):
    """Frank, 2026-08-04: the judge must see a natural debate. Nothing it
    receives may say a side ARGUES BECAUSE IT WAS TOLD TO, and nothing may tell
    it to discount a participant. The debaters' own cards still assign their
    positions — that is private to them, and asserted here so the split does not
    silently collapse in the other direction.

    Scoped to the judge's ACTUAL surface, not the whole pack: system card,
    preamble, and every slot cue the judge speaks. A shared stage would land
    here too, which is the regression this exists to catch (the role framing
    lived in `overall_system`, delivered to all three seats, until this rule)."""
    for path, entry, subject in (
        (MATH_YAML, "math_proposer_critic", "answer"),
        (CODECONTESTS_YAML, "codecontests_proposer_critic", "program"),
    ):
        pack = load_pack(path, entry)
        # prompts are hard-wrapped prose, so compare on normalized whitespace
        flat = lambda s: " ".join(s.split())  # noqa: E731
        judge_slots = [
            cs.slot.name for cs in protocol.compile() if cs.speaker == "judge"
        ]
        seen_by_judge = flat(
            "\n".join(
                [pack.system["judge"], pack.preamble.get("judge", "")]
                + [slot_template(pack, name, "judge") for name in judge_slots]
            )
        )
        for banned in (
            "FORCED",
            "forced to disagree",
            "regardless of what you privately believe",
            "role is to argue",
            "is required to argue",
            "weigh their claims skeptically",
            "(Critic)",
            "(Proposer)",
            "the Critic",
            "the Proposer",
        ):
            assert banned not in seen_by_judge, f"{entry}: judge sees {banned!r}"
        # ...while the debaters' cards keep their OWN assigned positions
        assert "Your role is to prove the Proposer wrong" in flat(pack.system["bob"])
        assert "NEVER concede that the Proposer is right" in flat(pack.system["bob"])
        assert "<proposer-role>" in pack.system["alice"]
        # A seat is told its own position, never that the OTHER seat's is
        # locked. `debater_system` is shared by both debaters, so a stance
        # claim parked there reaches alice — the same leak overall_system had,
        # one level down. Alice may still learn objections are coming (its own
        # <proposer-role> says so); it may not learn they are compulsory.
        for banned in (
            "the Critic argues that",
            "Critic's role is to",
            "FORCED",
            "forced to disagree",
            "NEVER concede that the Proposer is right",
        ):
            assert banned not in flat(pack.system["alice"]), f"{entry}: alice sees {banned!r}"
        # the symmetric skepticism that replaced the one-sided <critic-bias>
        assert "Do NOT simply trust either participant's claims" in flat(pack.system["judge"])
        assert subject in flat(pack.system["judge"])


def test_validate_fails_on_empty_composed_system_card(lib, protocol):
    """Cards are COMPOSED from stages, so 'defines no stage for this role'
    surfaces as an empty string rather than a missing key."""
    lib.system["bob"] = ""
    with pytest.raises(ValueError, match="empty system prompt for speaker 'bob'"):
        validate_prompts(lib, protocol)


def test_validate_fails_bindability_when_position_in_solution_slot(lib, protocol):
    lib.slots["proposal"] = lib.slots["proposal"] + "\nYour answer is <POSITION>."
    with pytest.raises(ValueError, match="POSITION"):
        validate_prompts(lib, protocol, fresh_positions=True)
    validate_prompts(lib, protocol, fresh_positions=False)  # assigned mode: fine


def test_validate_fails_when_critic_needs_position_before_any_solution(lib):
    proto = Protocol.parse({"turns": [{"bob": [{"name": "critique"}]}]})
    lib.system = {"bob": lib.system["bob"]}
    with pytest.raises(ValueError, match="OPPONENT_POSITION"):
        validate_prompts(lib, proto, fresh_positions=True)


def test_bindability_ignores_placeholders_covered_by_vars(protocol):
    lib = PromptLibrary(
        system={s: "sys" for s in protocol.speakers},
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
    validate_prompts(lib, protocol, fresh_positions=True)


# ------------------------------------------------------------ RenderedPrompts


def test_rendered_prompts_protocol(lib):
    rp = RenderedPrompts(lib)
    # composed card: overall_system, then the role card
    card = rp.system("alice", BINDINGS)
    assert card.startswith("<role>You are a participant in a proposer-critic debate")
    assert "<role>You are Alice, the PROPOSER.</role>" in card
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
                    "overall_system": ["identity block", "rules block", "format block"],
                    "slot_stages": {"speech": "pre_speech_proposer"},
                    "pre_speech_proposer": ["do the thing"],
                },
                "child": {
                    "_extends": "_base",
                    # override ONLY block 1; keep 0 and 2; append block 3
                    "overall_system": ["identity block", "CHILD RULES", "format block", "extra"],
                },
            }
        )
    )
    proto = Protocol.parse({"turns": [{"alice": [{"name": "speech"}]}]})
    base = load_prompt_library(p, "_base", proto)
    assert base.system["alice"] == "identity block\n\nrules block\n\nformat block"
    child = load_prompt_library(p, "child", proto)
    assert child.system["alice"] == "identity block\n\nCHILD RULES\n\nformat block\n\nextra"
    assert child.slots["speech"] == "do the thing"  # inherited untouched


def test_preview_shows_block_placement():
    import yaml as _yaml

    from infra.envs.debate.prompts import load_prompt_library, preview
    from infra.envs.debate.protocol import Protocol

    proto = Protocol.parse(
        _yaml.safe_load(
            "turns: [{alice: [{name: proposal, kind: solution}]}, {bob: [{name: critique}]},"
            " {alice: [{name: defense}]}, {judge: [{name: verdict, kind: decision}]}]"
        )
    )
    lib = splice_task_prompts(
        load_prompt_library(
            "infra/envs/debate/prompt_configs/hendrycks_math.yaml", "math_proposer_critic", proto
        ),
        "math.yaml",
    )
    out = preview(lib, proto)
    assert "<alice/proposal output>" in out          # stubbed prior slots
    assert "<TOPIC>" in out                          # runtime placeholders kept visible
    assert "EXACTLY one \\boxed{...}" in out         # spliced task wording renders for real
    assert "== judge — context for its final slot (verdict)" in out
