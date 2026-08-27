"""How `_extends` merges lists.

The repo uses lists for two incompatible things, and one merge rule for both
is what produced dead `coeff: 0.0` position-holders in arms, 25 config
comments warning about by-index merge, and a written rule in three files
saying never to put `protocol:` in them.
"""

from __future__ import annotations

import pytest

from infra.config import deep_merge


# --------------------------------------------------------- named: by name


def test_named_entries_merge_by_name_not_position():
    base = {"shaping": [{"name": "a", "coeff": 1}, {"name": "b", "coeff": 2}]}
    over = {"shaping": [{"name": "b", "coeff": 9}]}
    assert deep_merge(base, over)["shaping"] == [
        {"name": "a", "coeff": 1},
        {"name": "b", "coeff": 9},
    ]


def test_an_unknown_name_appends_rather_than_overwriting_position_zero():
    base = {"shaping": [{"name": "a", "coeff": 1}]}
    over = {"shaping": [{"name": "z", "coeff": 5}]}
    assert deep_merge(base, over)["shaping"] == [
        {"name": "a", "coeff": 1},
        {"name": "z", "coeff": 5},
    ]


def test_drop_removes_an_inherited_entry():
    """The capability by-index cannot express at all. Before this, a child that
    wanted a term gone had to leave it in place at coeff 0.0."""
    base = {"shaping": [{"name": "a", "coeff": 1}, {"name": "b", "coeff": 2}]}
    over = {"shaping": [{"name": "b", "_drop": True}]}
    assert deep_merge(base, over)["shaping"] == [{"name": "a", "coeff": 1}]


def test_a_named_entry_still_deep_merges_its_own_fields():
    base = {"shaping": [{"name": "a", "coeff": 1, "opts": {"x": 1, "y": 2}}]}
    over = {"shaping": [{"name": "a", "opts": {"y": 9}}]}
    assert deep_merge(base, over)["shaping"] == [
        {"name": "a", "coeff": 1, "opts": {"x": 1, "y": 9}}
    ]


def test_duplicate_names_fall_back_to_by_index():
    """A by-name merge would have to pick one of two entries sharing a name,
    and picking wrong silently changes a reward function. Falling back keeps
    the old, predictable behaviour instead."""
    base = {"xs": [{"name": "a", "coeff": 1}, {"name": "a", "coeff": 2}]}
    over = {"xs": [{"name": "a", "coeff": 9}]}
    assert deep_merge(base, over)["xs"] == [
        {"name": "a", "coeff": 9},
        {"name": "a", "coeff": 2},
    ]


# ------------------------------------------- scalar lists: still by index


def test_scalar_lists_stay_by_index_because_prompts_rely_on_it():
    """These look like values you would want replaced, and making them so broke
    the prompt packs: hendrycks_math_boxreq overrides block 0 of judge_system
    and needs the parent's evaluation-steps block to survive at index 1.

    Narrowing a scalar field is expressed by dropping the named entry that
    owns it and restating it, not by shortening the list in place.
    """
    base = {"judge_system": ["<role>...", "<evaluation-steps>...", "<proof>..."]}
    out = deep_merge(base, {"judge_system": ["<role>NEW"]})["judge_system"]
    assert out == ["<role>NEW", "<evaluation-steps>...", "<proof>..."]


# ------------------------------------------- positional records: by index


def test_unnamed_dicts_still_merge_by_index():
    """The case the original rule exists for: override one field of
    debaters[0] without redeclaring the rest of it."""
    base = {"debaters": [{"model": "a", "temp": 1.0}, {"model": "b", "temp": 1.0}]}
    over = {"debaters": [{"temp": 0.5}]}
    assert deep_merge(base, over)["debaters"] == [
        {"model": "a", "temp": 0.5},
        {"model": "b", "temp": 1.0},
    ]


def test_a_partially_named_list_stays_by_index():
    base = {"xs": [{"name": "a", "v": 1}, {"v": 2}]}
    over = {"xs": [{"v": 9}]}
    assert deep_merge(base, over)["xs"] == [{"name": "a", "v": 9}, {"v": 2}]


def test_protocol_turns_are_positional_and_keep_index_semantics():
    """Turns are speaker -> slots maps with no name, so they must not pick up
    by-name behaviour; math_pc_l5 overrides turn 0 alone and relies on it."""
    base = {"turns": [{"alice": [{"name": "proposal", "max_total_tokens": 3072}]},
                      {"bob": [{"name": "critique"}]}]}
    over = {"turns": [{"alice": [{"name": "proposal", "max_total_tokens": 5024}]}]}
    out = deep_merge(base, over)["turns"]
    assert out[0]["alice"][0]["max_total_tokens"] == 5024
    assert out[1] == {"bob": [{"name": "critique"}]}
