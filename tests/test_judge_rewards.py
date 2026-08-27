"""Pins for the judge decision-token scan, the ladder, and shaping."""

from __future__ import annotations

import math

import pytest

from infra.backend.base import Sample
from infra.envs.debate.judge import (
    Confidence,
    JudgeConfig,
    LogitStatus,
    SeatVerdict,
    Verdict,
    decision_index,
    decision_point,
    match_seat,
    parse_verdict,
    scan_decision_logit,
    verdict_from_slot,
)
from infra.envs.debate.rewards import (
    BudgetPenalty,
    LadderValues,
    LengthPenalty,
    RoundTokenReport,
    ScoringConfig,
    SlotTokenCounts,
    SpeechOvershootPenalty,
    build_shaping,
    final_reward,
    ladder,
    score,
    validate_scoring,
)
from infra.models.base import ModelResponse, ModelSettings

SEATS = ["Debater_A", "Debater_B"]

# Qwen3-8B's tokenization of the winner value, with the JSON scaffolding
# around it as single pieces.
PREFIX = '{"winner": "'
SUFFIX = '", "confidence": 0.9}'
WINNER_PIECES = [PREFIX, "De", "b", "ater", " A", SUFFIX]


def texts_response(pieces, logprobs, stop_reason="stop"):
    return ModelResponse(
        speech="".join(pieces),
        response_token_texts=list(pieces),
        response_logprobs=list(logprobs),
        stop_reason=stop_reason,
    )


def ids_sample(pieces, logprobs, stop_reason="stop"):
    """Trained-seat channel: ids plus a decode_fn that maps id i -> pieces[i]."""
    ids = list(range(len(pieces)))
    sample = Sample(tokens=ids, logprobs=list(logprobs), text="".join(pieces), stop_reason=stop_reason)
    return sample, lambda toks: "".join(pieces[t] for t in toks)


def lp_for(p, n_other=0.0):
    return math.log(p) - n_other


# 1 -------------------------------------------------------------- strict crossing


def test_strict_crossing_selects_the_divergent_token():
    text = "".join(WINNER_PIECES)
    value_start = text.index('"', text.index("winner") + 8) + 1
    d = decision_point("Debater A", value_start, SEATS + ["Tie"])
    assert decision_index(WINNER_PIECES, d) == 4  # ' A', not 'ater'


def test_strict_crossing_when_divergence_lands_on_a_token_boundary():
    # 'De','b','ater' end exactly at char 7 of the value; a '>=' comparison
    # would return 'ater' (index 2) instead of the decision token.
    assert decision_index(["De", "b", "ater", " A"], 7) == 3


# 2 -------------------------------------------------------- divergence point


def test_divergence_point_with_no_shared_prefix_scores_first_value_token():
    seats = ["Alice", "Bob"]
    pieces = [PREFIX, "Al", "ice", SUFFIX]
    logprobs = [0.0, math.log(0.5), math.log(0.8), 0.0]
    parsed = parse_verdict("".join(pieces), "competitive", seats)
    out = scan_decision_logit(texts_response(pieces, logprobs), None, "competitive", seats, parsed)
    # d == value_start: both value tokens overlap and their logprobs sum.
    assert out["Alice"][1] is LogitStatus.SCORED
    assert out["Alice"][0] == pytest.approx(0.4)
    assert out["Bob"][0] == pytest.approx(0.6)


def test_multi_token_divergent_suffix_sums_to_value_end():
    seats = ["Debater_Alpha", "Debater_Beta"]
    pieces = [PREFIX, "De", "b", "ater", " Al", "pha", SUFFIX]
    logprobs = [0.0, 0.0, 0.0, 0.0, math.log(0.9), math.log(0.5), 0.0]
    parsed = parse_verdict("".join(pieces), "competitive", seats)
    out = scan_decision_logit(texts_response(pieces, logprobs), None, "competitive", seats, parsed)
    # Every token overlapping [divergence, value_end) contributes; the JSON
    # suffix token does not.
    assert out["Debater_Alpha"][0] == pytest.approx(0.45)


# 3 ------------------------------------------------------------- both channels


def test_texts_channel_normalizes_bpe_surfaces():
    pieces = [PREFIX, "De", "b", "ater", "ĠA", SUFFIX]  # Ġ == leading space
    logprobs = [0.0] * 4 + [math.log(0.7), 0.0]
    parsed = parse_verdict('{"winner": "Debater A", "confidence": 0.9}', "competitive", SEATS)
    out = scan_decision_logit(texts_response(pieces, logprobs), None, "competitive", SEATS, parsed)
    assert out["Debater_A"][0] == pytest.approx(0.7)


def test_ids_channel_matches_texts_channel():
    logprobs = [0.0] * 4 + [math.log(0.7), 0.0]
    sample, decode = ids_sample(WINNER_PIECES, logprobs)
    parsed = parse_verdict("".join(WINNER_PIECES), "competitive", SEATS)
    out = scan_decision_logit(sample, decode, "competitive", SEATS, parsed)
    assert out["Debater_A"][0] == pytest.approx(0.7)
    assert out["Debater_B"][0] == pytest.approx(0.3)


@pytest.mark.parametrize("channel", ["texts", "ids"])
def test_truncation_and_misalignment_gate_both_channels(channel):
    logprobs = [0.0] * 6
    parsed = parse_verdict("".join(WINNER_PIECES), "competitive", SEATS)
    if channel == "texts":
        truncated = (texts_response(WINNER_PIECES, logprobs, stop_reason="length"), None)
        misaligned = (texts_response(WINNER_PIECES, logprobs[:-1]), None)
    else:
        s, decode = ids_sample(WINNER_PIECES, logprobs, stop_reason="length")
        truncated = (s, decode)
        s2, decode2 = ids_sample(WINNER_PIECES, logprobs[:-1])
        misaligned = (s2, decode2)
    out = scan_decision_logit(*truncated[:1], truncated[1], "competitive", SEATS, parsed)
    assert all(status is LogitStatus.TRUNCATED for _, status in out.values())
    out = scan_decision_logit(*misaligned[:1], misaligned[1], "competitive", SEATS, parsed)
    assert all(status is LogitStatus.MISALIGNED for _, status in out.values())


def test_no_pieces_without_decode_fn():
    sample, _ = ids_sample(WINNER_PIECES, [0.0] * 6)
    parsed = parse_verdict("".join(WINNER_PIECES), "competitive", SEATS)
    out = scan_decision_logit(sample, None, "competitive", SEATS, parsed)
    assert all(status is LogitStatus.NO_PIECES for _, status in out.values())


def test_join_reparse_disagreement_is_misaligned():
    pieces = [PREFIX, "De", "b", "ater", " A", SUFFIX]
    parsed = parse_verdict('{"winner": "Debater_B", "confidence": 0.9}', "competitive", SEATS)
    out = scan_decision_logit(texts_response(pieces, [0.0] * 6), None, "competitive", SEATS, parsed)
    assert all(status is LogitStatus.MISALIGNED for _, status in out.values())


# 4 --------------------------------------------------------------- no overwrite


def _verdict_with_both_channels():
    logprobs = [0.0] * 4 + [math.log(0.62), 0.0]
    text = "".join(WINNER_PIECES)
    return verdict_from_slot(text, texts_response(WINNER_PIECES, logprobs), None, JudgeConfig(), SEATS)


def test_json_and_logit_confidences_both_retained():
    v = _verdict_with_both_channels()
    assert v.ok and v.winner == "Debater_A"
    assert v.logit_pieces_source == "texts"
    assert v.confidence["Debater_A"].json == pytest.approx(0.9)
    assert v.confidence["Debater_A"].logit == pytest.approx(0.62)
    assert v.confidence["Debater_B"].json == pytest.approx(0.1)
    assert v.confidence["Debater_B"].logit == pytest.approx(0.38)

    json_scored = score(v, "competitive", ScoringConfig(scoring="continuous", confidence_source="json"))
    logit_scored = score(v, "competitive", ScoringConfig(scoring="continuous", confidence_source="logit"))
    assert json_scored["Debater_A"].value == pytest.approx(0.8)
    assert json_scored["Debater_A"].source == "json"
    assert logit_scored["Debater_A"].value == pytest.approx(0.24)
    assert logit_scored["Debater_A"].source == "logit"


# 5 ------------------------------------------------------------ provenance


def _competitive_verdict(json_conf=0.9, logit=None, status=LogitStatus.NO_PIECES):
    jc = json_conf
    return Verdict(
        schema="competitive",
        seats={"Debater_A": SeatVerdict.CORRECT, "Debater_B": SeatVerdict.INCORRECT},
        confidence={
            "Debater_A": Confidence(jc, logit, "elicited" if jc is not None else "absent", status),
            "Debater_B": Confidence(
                None if jc is None else 1 - jc,
                None if logit is None else 1 - logit,
                "elicited" if jc is not None else "absent",
                status,
            ),
        },
        winner="Debater_A",
        ok=True,
        raw="",
    )


def test_fallbacks_keep_the_table_value_and_are_labelled():
    v = _competitive_verdict(logit=None, status=LogitStatus.TRUNCATED)
    out = score(v, "competitive", ScoringConfig(scoring="continuous", confidence_source="logit"))
    assert out["Debater_A"].value == 1.0 and out["Debater_A"].source == "binary_fallback"
    assert out["Debater_B"].value == -1.0 and out["Debater_B"].source == "binary_fallback"

    absent = _competitive_verdict(json_conf=None)
    out = score(absent, "competitive", ScoringConfig(scoring="continuous", confidence_source="json"))
    assert out["Debater_A"].source == "binary_fallback"

    # No cross-source fallback: json present, logit absent, logit selected.
    v2 = _competitive_verdict(json_conf=0.9, logit=None, status=LogitStatus.NO_PIECES)
    out = score(v2, "competitive", ScoringConfig(scoring="continuous", confidence_source="logit"))
    assert out["Debater_A"].source == "binary_fallback" and out["Debater_A"].value == 1.0


# 6 ---------------------------------------------------------------- ladders


def _verdict(seats: dict, ok=True, schema="competitive", winner=None):
    return Verdict(
        schema=schema,
        seats=seats,
        confidence={s: Confidence(None, None, "absent", LogitStatus.NO_PIECES) for s in seats},
        winner=winner,
        ok=ok,
        raw="",
    )


C, I, T, U = SeatVerdict.CORRECT, SeatVerdict.INCORRECT, SeatVerdict.TIE, SeatVerdict.UNKNOWN


@pytest.mark.parametrize(
    "rulings,expect,cell,scoreable",
    [
        ({"Debater_A": C, "Debater_B": I}, (1.0, -1.0), "a_win", True),
        ({"Debater_A": I, "Debater_B": C}, (-1.0, 1.0), "b_win", True),
        ({"Debater_A": I, "Debater_B": I}, (-1.0, -1.0), "double_loss", True),
        ({"Debater_A": T, "Debater_B": T}, (0.0, 0.0), "tie", True),
        ({"Debater_A": C, "Debater_B": C}, (0.0, 0.0), "double_win", True),
        ({"Debater_A": U, "Debater_B": C}, (0.0, 0.0), "unknown", False),
    ],
)
def test_ladder_cells(rulings, expect, cell, scoreable):
    out = ladder(_verdict(rulings), "competitive")
    assert (out["Debater_A"].value, out["Debater_B"].value) == expect
    assert out["Debater_A"].cell == cell and out["Debater_A"].scoreable is scoreable


def test_ladder_parse_failure_is_unscoreable():
    out = ladder(_verdict({s: U for s in SEATS}, ok=False), "competitive")
    assert all(r.value == 0.0 and not r.scoreable and r.cell == "failed" for r in out.values())


def test_collaborative_one_seat_tie_symmetrizes():
    v = verdict_from_slot(
        '{"Debater_A": {"verdict": "tie", "confidence": 0.8},'
        ' "Debater_B": {"verdict": "correct", "confidence": 0.9}}',
        None,
        None,
        JudgeConfig(schema_name="collaborative"),
        SEATS,
    )
    assert set(v.seats.values()) == {SeatVerdict.TIE}
    assert all(c.json is None and c.json_provenance == "tie" for c in v.confidence.values())
    out = ladder(v, "collaborative")
    assert all(r.cell == "tie" and r.scoreable for r in out.values())


def test_single_seat_ladder_rows():
    for ruling, value, cell, scoreable in [
        (C, 1.0, "win", True),
        (I, -1.0, "loss", True),
        (T, 0.0, "tie", True),
        (U, 0.0, "unknown", False),
    ]:
        out = ladder(_verdict({"Consultant": ruling}), "collaborative")
        assert out["Consultant"].value == value
        assert out["Consultant"].cell == cell
        assert out["Consultant"].scoreable is scoreable


def test_ladder_values_are_configurable():
    values = LadderValues(solo_win=2.0, solo_loss=-0.5, tie=0.25, double_loss=-3.0, double_win=0.75)
    out = ladder(_verdict({"Debater_A": C, "Debater_B": I}), "competitive", values)
    assert (out["Debater_A"].value, out["Debater_B"].value) == (2.0, -0.5)
    out = ladder(_verdict({"Debater_A": C, "Debater_B": C}), "collaborative", values)
    assert out["Debater_A"].value == 0.75


# 7 -------------------------------------------------------------- continuous


def test_continuous_competitive_margins_sum_to_zero():
    v = _competitive_verdict(json_conf=0.75)
    out = score(v, "competitive", ScoringConfig(scoring="continuous"))
    assert out["Debater_A"].value == pytest.approx(0.5)
    assert out["Debater_B"].value == pytest.approx(-0.5)
    assert out["Debater_A"].value + out["Debater_B"].value == pytest.approx(0.0)


def test_continuous_keeps_table_for_neither_and_tie():
    neither = verdict_from_slot('{"winner": "Neither", "confidence": 0.9}', None, None, JudgeConfig(), SEATS)
    assert neither.seats == {"Debater_A": SeatVerdict.INCORRECT, "Debater_B": SeatVerdict.INCORRECT}
    out = score(neither, "competitive", ScoringConfig(scoring="continuous"))
    assert (out["Debater_A"].value, out["Debater_B"].value) == (-1.0, -1.0)
    assert out["Debater_A"].cell == "double_loss" and out["Debater_A"].scoreable

    tie = verdict_from_slot('{"winner": "Tie", "confidence": 0.9}', None, None, JudgeConfig(), SEATS)
    out = score(tie, "competitive", ScoringConfig(scoring="continuous"))
    assert all(r.value == 0.0 and r.scoreable for r in out.values())


def test_continuous_collaborative_signs_each_seat():
    v = verdict_from_slot(
        '{"Debater_A": {"verdict": "correct", "confidence": 0.8},'
        ' "Debater_B": {"verdict": "incorrect", "confidence": 0.6}}',
        None,
        None,
        JudgeConfig(schema_name="collaborative"),
        SEATS,
    )
    out = score(v, "collaborative", ScoringConfig(scoring="continuous"))
    assert out["Debater_A"].value == pytest.approx(0.6)
    assert out["Debater_B"].value == pytest.approx(-0.2)


# 8 --------------------------------------------------------- validate_scoring


def _judge_settings(**kw):
    base = {"alias": "judge", "capture_token_logprobs": True}
    base.update(kw)
    return ModelSettings(**base)


def test_validate_scoring_requires_untouched_sampling_under_logit():
    logit = ScoringConfig(confidence_source="logit")
    ok = _judge_settings(sampling={"eval": {"temperature": 1.0, "top_p": 1.0}})
    validate_scoring(logit, ok)

    for bad in (
        {"sampling": {"eval": {"temperature": 0.7, "top_p": 1.0}}},
        {"sampling": {"eval": {"temperature": 1.0, "top_p": 0.9}}},
        {"sampling": {"eval": {"temperature": 1.0, "top_p": 1.0, "top_k": 20}}},
    ):
        with pytest.raises(ValueError):
            validate_scoring(logit, _judge_settings(**bad))

    with pytest.raises(ValueError):
        validate_scoring(logit, _judge_settings(capture_token_logprobs=False, sampling={"eval": {"temperature": 1.0}}))

    # json source does not constrain the judge's sampling at all.
    validate_scoring(ScoringConfig(confidence_source="json"), _judge_settings(capture_token_logprobs=False))


# 9 ------------------------------------------------------------ parse robustness


def test_verdict_after_prose():
    text = "A reasoned my case better.\n\n{\"winner\": \"Debater_A\", \"confidence\": 0.9}"
    assert parse_verdict(text, "competitive", SEATS)["winner"] == "Debater_A"


def test_verdict_in_json_fence():
    text = 'Here is my ruling:\n```json\n{"winner": "Debater_B", "confidence": 0.7}\n```\n'
    assert parse_verdict(text, "competitive", SEATS)["winner"] == "Debater_B"


def test_schema_example_then_real_verdict_takes_the_last():
    text = (
        'The format is {"winner": "Debater_A", "confidence": 0.85}.\n'
        'My verdict: {"winner": "Debater_B", "confidence": 0.6}'
    )
    assert parse_verdict(text, "competitive", SEATS)["winner"] == "Debater_B"


def test_unparseable_verdict_is_not_ok():
    v = verdict_from_slot("I decline to rule.", None, None, JudgeConfig(), SEATS)
    assert not v.ok and set(v.seats.values()) == {SeatVerdict.UNKNOWN}
    assert all(c.json is None and c.logit is None for c in v.confidence.values())
    assert all(not r.scoreable and r.value == 0.0 for r in ladder(v, "competitive").values())


def test_confidence_outside_range_is_clamped_with_warning():
    with pytest.warns(UserWarning):
        v = verdict_from_slot('{"winner": "Debater_A", "confidence": 1.4}', None, None, JudgeConfig(), SEATS)
    assert v.confidence["Debater_A"].json == 1.0


def test_collaborative_requires_every_seat():
    assert parse_verdict('{"Debater_A": {"verdict": "correct"}}', "collaborative", SEATS) is None


# 10 ------------------------------------------------------------ alias matcher


def test_alias_matcher_refuses_substrings_in_both_directions():
    assert match_seat("Debater_A", SEATS) == "Debater_A"
    assert match_seat("debater a", SEATS) == "Debater_A"  # separator normalization
    assert match_seat("not Debater_A", SEATS) is None  # seat name inside the value
    assert match_seat("B", SEATS) is None  # bare token inside a seat name
    assert match_seat("", SEATS) is None
    # A distinguishing multi-token suffix still resolves.
    assert match_seat("Debater_A", ["run3_Debater_A", "run3_Debater_B"]) == "run3_Debater_A"


def test_parse_rejects_a_winner_that_matches_no_seat():
    assert parse_verdict('{"winner": "not Debater_A"}', "competitive", SEATS) is None
    assert parse_verdict('{"winner": "B"}', "competitive", SEATS) is None


def test_decision_point_ignores_the_value_itself():
    assert decision_point("Debater A", 0, ["Debater_A", "Debater_B"]) == 8


# 11 ---------------------------------------------------------------- shaping


def _report():
    return RoundTokenReport(
        counts={
            ("Debater_A", "speech"): SlotTokenCounts(think=100, visible=200, total=300, cap_total=600),
            ("Debater_A", "closing"): SlotTokenCounts(think=10, visible=40, total=50, cap_total=100),
            ("Debater_B", "speech"): SlotTokenCounts(think=0, visible=100, total=100, cap_total=600),
        }
    )


def test_per_slot_penalty_hits_only_the_offending_slot():
    scored = score(_competitive_verdict(), "competitive")
    term = LengthPenalty(coeff=0.001, counts="total", slots=["speech"], per_slot=True)
    delta = term.apply(scored, _report())
    assert delta.per_seat == {}
    assert delta.per_slot == {("Debater_A", "speech"): -0.3, ("Debater_B", "speech"): -0.1}
    assert final_reward("Debater_A", "speech", scored, [delta]) == pytest.approx(0.7)
    assert final_reward("Debater_A", "closing", scored, [delta]) == pytest.approx(1.0)


def test_per_seat_penalty_broadcasts_across_slots():
    scored = score(_competitive_verdict(), "competitive")
    term = LengthPenalty(coeff=0.001, counts="think", per_slot=False)
    delta = term.apply(scored, _report())
    assert delta.per_slot == {}
    assert delta.per_seat["Debater_A"] == pytest.approx(-0.11)  # 100 + 10 think tokens
    assert final_reward("Debater_A", "closing", scored, [delta]) == pytest.approx(0.89)


def test_normalize_cap_divides_by_the_slot_cap():
    scored = score(_competitive_verdict(), "competitive")
    term = LengthPenalty(coeff=1.0, counts="total", slots=["speech"], normalize="cap")
    delta = term.apply(scored, _report())
    assert delta.per_slot[("Debater_A", "speech")] == pytest.approx(-0.5)  # 300 / 600


def test_shaping_never_applies_to_unscoreable_trajectories():
    scored = ladder(_verdict({s: SeatVerdict.UNKNOWN for s in SEATS}, ok=False), "competitive")
    delta = LengthPenalty(coeff=0.001).apply(scored, _report())
    assert delta.per_slot == {} and delta.per_seat == {}


def test_build_shaping_from_yaml_style_dicts():
    terms = build_shaping([{"kind": "length_penalty", "coeff": 0.0005, "counts": "visible", "per_slot": True}])
    assert len(terms) == 1 and isinstance(terms[0], LengthPenalty)
    assert terms[0].coeff == 0.0005 and terms[0].counts == "visible"
    with pytest.raises(ValueError):
        build_shaping([{"kind": "nope"}])


# 11b ------------------------------------------- Kenton word limit (§3.4)


def _words_report(a_speech: int, b_speech: int, a_closing: int = 0):
    """Same shape as _report(), but the quantity the word limit prices."""
    return RoundTokenReport(
        counts={
            ("Debater_A", "speech"): SlotTokenCounts(
                think=100, visible=200, total=300, cap_total=600, visible_words=a_speech
            ),
            ("Debater_A", "closing"): SlotTokenCounts(
                think=10, visible=40, total=50, cap_total=100, visible_words=a_closing
            ),
            ("Debater_B", "speech"): SlotTokenCounts(
                think=0, visible=100, total=100, cap_total=600, visible_words=b_speech
            ),
        }
    )


def test_word_limit_charges_only_the_excess():
    """"A soft additive penalty proportional to the excess" — so the charge is
    the distance past the limit, not the length and not a flat trip."""
    scored = score(_competitive_verdict(), "competitive")
    term = BudgetPenalty(coeff=0.002, limit=150, slots=["speech"])
    delta = term.apply(scored, _words_report(a_speech=200, b_speech=175))
    assert delta.per_slot[("Debater_A", "speech")] == pytest.approx(-0.10)  # 50 over
    assert delta.per_slot[("Debater_B", "speech")] == pytest.approx(-0.05)  # 25 over


def test_a_speech_inside_the_limit_pays_nothing_however_long():
    """No pressure below the limit: the term must not become a general
    shortness reward, which is the gradient that collapsed the RLVR arm."""
    scored = score(_competitive_verdict(), "competitive")
    term = BudgetPenalty(coeff=0.002, limit=150, slots=["speech"])
    delta = term.apply(scored, _words_report(a_speech=150, b_speech=1))
    assert delta.per_slot[("Debater_A", "speech")] == 0.0
    assert delta.per_slot[("Debater_B", "speech")] == 0.0


def test_word_limit_separates_speeches_the_flat_term_prices_identically():
    """The regression the proportional form exists to fix.

    Under a 150-word cue with a 320-token ceiling, a 200-word speech and a
    151-word speech both fit the cap, so ``truncated`` is False on both and the
    flat term charges them the SAME (nothing). Priced on the excess they differ
    by a factor of 50, which is the signal the cue was always asserting and the
    reward never paid.
    """
    scored = score(_competitive_verdict(), "competitive")
    flat = SpeechOvershootPenalty(coeff=0.1, slots=["speech"])
    prop = BudgetPenalty(coeff=0.002, limit=150, slots=["speech"])
    report = _words_report(a_speech=200, b_speech=151)

    flat_delta = flat.apply(scored, report)
    assert flat_delta.per_slot[("Debater_A", "speech")] == flat_delta.per_slot[
        ("Debater_B", "speech")
    ] == 0.0

    prop_delta = prop.apply(scored, report)
    assert prop_delta.per_slot[("Debater_A", "speech")] == pytest.approx(-0.10)
    assert prop_delta.per_slot[("Debater_B", "speech")] == pytest.approx(-0.002)


def test_flat_mode_charges_once_however_far_over():
    """The other half of the toggle. Same limit, same coeff, same speeches —
    only the shape differs, and under `flat` a 1-word overrun and a 50-word one
    cost exactly the same. That indifference is the property that makes a flat
    term invisible to a group-relative baseline."""
    scored = score(_competitive_verdict(), "competitive")
    term = BudgetPenalty(coeff=0.1, limit=150, mode="flat", slots=["speech"])
    delta = term.apply(scored, _words_report(a_speech=200, b_speech=151))
    assert delta.per_slot[("Debater_A", "speech")] == pytest.approx(-0.1)
    assert delta.per_slot[("Debater_B", "speech")] == pytest.approx(-0.1)


def test_flat_mode_still_free_inside_the_limit():
    scored = score(_competitive_verdict(), "competitive")
    term = BudgetPenalty(coeff=0.1, limit=150, mode="flat", slots=["speech"])
    delta = term.apply(scored, _words_report(a_speech=150, b_speech=0))
    assert delta.per_slot[("Debater_A", "speech")] == 0.0
    assert delta.per_slot[("Debater_B", "speech")] == 0.0


def test_an_unknown_mode_is_refused_rather_than_defaulted():
    """The two shapes differ by a factor of the excess, so a typo must not
    quietly pick one."""
    with pytest.raises(ValueError, match="mode"):
        BudgetPenalty(coeff=0.002, limit=150, mode="linear")
    with pytest.raises(ValueError, match="mode"):
        build_shaping([{"kind": "budget_penalty", "coeff": 0.002, "mode": "proprtional"}])


def test_word_limit_exempts_the_solution_turn_by_slot_selection():
    """Kenton exempts Alice's solution turn; here that is the `slots` list, so
    a term aimed at the reply slots must leave the graded slot untouched."""
    scored = score(_competitive_verdict(), "competitive")
    term = BudgetPenalty(coeff=0.002, limit=150, slots=["closing"])
    delta = term.apply(scored, _words_report(a_speech=999, b_speech=999, a_closing=200))
    assert ("Debater_A", "speech") not in delta.per_slot
    assert delta.per_slot == {("Debater_A", "closing"): pytest.approx(-0.10)}


def test_word_limit_builds_from_yaml_style_dict():
    terms = build_shaping(
        [{"kind": "budget_penalty", "coeff": 0.002, "limit": 150, "slots": ["critique"]}]
    )
    assert len(terms) == 1 and isinstance(terms[0], BudgetPenalty)
    assert terms[0].coeff == 0.002 and terms[0].limit == 150 and terms[0].slots == ["critique"]


def test_budget_penalty_can_price_solution_slot_tokens():
    """The answer-generation use. The RLVR path prices proposal length through
    SingleTurnEnv's soft_token_budget, which a debate never reaches — DebateEnv
    extends Env with its own rollout — so on this arm a proposal budget has to
    be a slot term or it does not exist."""
    scored = score(_competitive_verdict(), "competitive")
    report = RoundTokenReport(
        counts={
            ("Debater_A", "speech"): SlotTokenCounts(
                think=4000, visible=1024, total=5024, cap_total=5024, visible_words=200
            ),
            ("Debater_B", "speech"): SlotTokenCounts(
                think=2000, visible=200, total=2200, cap_total=5024, visible_words=40
            ),
        }
    )
    term = BudgetPenalty(coeff=0.0002, limit=4000, counts="total", slots=["speech"])
    delta = term.apply(scored, report)
    assert delta.per_slot[("Debater_A", "speech")] == pytest.approx(-0.2048)  # 1024 over
    assert delta.per_slot[("Debater_B", "speech")] == 0.0                     # inside


def test_words_and_tokens_are_not_interchangeable_on_the_same_slot():
    """Why `counts` exists rather than a token proxy for the word limit: the
    same slot is 200 words and 5024 tokens, and a budget of 150 means very
    different things depending on which one it reads."""
    scored = score(_competitive_verdict(), "competitive")
    report = RoundTokenReport(
        counts={
            ("Debater_A", "speech"): SlotTokenCounts(
                think=4000, visible=1024, total=5024, cap_total=5024, visible_words=200
            ),
        }
    )
    by_words = BudgetPenalty(coeff=1.0, limit=150, counts="words", slots=["speech"])
    by_visible = BudgetPenalty(coeff=1.0, limit=150, counts="visible", slots=["speech"])
    assert by_words.apply(scored, report).per_slot[("Debater_A", "speech")] == -50.0
    assert by_visible.apply(scored, report).per_slot[("Debater_A", "speech")] == -874.0


def test_an_unknown_counts_is_refused():
    with pytest.raises(ValueError, match="counts"):
        BudgetPenalty(coeff=0.002, limit=150, counts="tokens")


def test_build_shaping_accepts_the_merge_identity_key():
    """`name` addresses a term across _extends; it is not a term field."""
    terms = build_shaping(
        [{"name": "reply_word_budget", "kind": "budget_penalty", "coeff": 0.002,
          "limit": 150, "slots": ["critique"]}]
    )
    assert len(terms) == 1 and isinstance(terms[0], BudgetPenalty)
    assert terms[0].coeff == 0.002 and terms[0].slots == ["critique"]


def test_a_drop_that_matched_nothing_is_an_error():
    """A surviving `_drop` means the name did not exist in the parent, so the
    author expected a removal that did not happen -- exactly the silent no-op
    this mechanism replaced."""
    with pytest.raises(ValueError, match="_drop"):
        build_shaping([{"name": "typo_here", "_drop": True}])
