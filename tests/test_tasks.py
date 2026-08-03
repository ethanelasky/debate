"""Unit tests for the task-family layer (registry, math family, config guards).

Everything here is offline: MathFamily.source() is only ever called with an
invalid config, where reject_unknown_keys fires before any dataset loading.
"""

import pytest

from infra.envs.tasks import get_family
from infra.envs.tasks.base import reject_unknown_keys
from infra.envs.tasks.math import MathFamily, _parse_levels


def test_registry_lookup():
    assert isinstance(get_family("math"), MathFamily)


def test_registry_unknown_name_lists_known_families():
    with pytest.raises(ValueError) as exc:
        get_family("nope")
    assert "math" in str(exc.value)


@pytest.mark.parametrize(
    "meta, solution, expected",
    [
        ({"gt": 2.0}, 2.0, True),
        ({"gt": 2.0}, 3.0, False),
        ({"gt": 2.0}, None, None),          # unparseable slot
        ({}, 2.0, None),                    # no ground truth
        ({"gt": 2.0}, "not a number", None),
        ({"gt": 2.0}, 2.0 + 1e-9, True),    # inside the 1e-6 tolerance
    ],
)
def test_math_grade(meta, solution, expected):
    assert MathFamily().grade(meta, solution) is expected


def test_extractor_strict_vs_relaxed():
    strict = MathFamily().extractor(False)
    relaxed = MathFamily().extractor(True)
    assert strict("the answer is 7") is None
    assert relaxed("the answer is 7") == 7.0
    assert strict("\\boxed{42}") == 42.0
    assert relaxed("\\boxed{42}") == 42.0


def test_format_flags_strict_boxed():
    fam = MathFamily()
    assert fam.format_flags("\\boxed{3}")["strict_boxed"] == 1.0
    assert fam.format_flags("no box")["strict_boxed"] == 0.0


@pytest.mark.parametrize(
    "spec, expected",
    [(5, (5,)), ("3-4", (3, 4)), ("5", (5,)), ([3, 4], (3, 4))],
)
def test_parse_levels(spec, expected):
    assert _parse_levels(spec) == expected


def test_reject_unknown_keys():
    reject_unknown_keys({"levels": 5}, {"levels", "seed"}, "math")  # no raise
    with pytest.raises(ValueError) as exc:
        reject_unknown_keys({"levels": 5, "bogus_key": 1}, {"levels", "seed"}, "math")
    assert "bogus_key" in str(exc.value)


def test_math_source_rejects_unknown_key_before_loading():
    with pytest.raises(ValueError) as exc:
        MathFamily().source({"bogus_key": 1})
    assert "bogus_key" in str(exc.value)
