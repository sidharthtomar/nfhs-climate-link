"""Tests for outcome construction.

The headline test is test_suffix_trap: it encodes the specific failure mode
that motivated the label-based design, so a future refactor that reintroduces
positional mapping fails CI instead of shipping.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfhs_climate.config import PROTECTION_ITEMS  # noqa: E402
from nfhs_climate.data.nfhs import (  # noqa: E402
    build_outcomes,
    build_protection_flags,
    weighted_mean,
)


def make_frame(rows):
    """rows: list of dicts keyed by NFHS-5 protection variable name."""
    cols = list(PROTECTION_ITEMS["nfhs5"])
    df = pd.DataFrame([{c: r.get(c, 0) for c in cols} for r in rows])
    df["v005"] = 1_000_000
    return df


def test_exclusive_pad_user():
    df = make_frame([{"s260c": 1}])  # sanitary napkins only
    out = build_outcomes(build_protection_flags(df))
    assert out["any_hygienic_narrow"].iloc[0] == 1
    assert out["exclusive_hygienic_narrow"].iloc[0] == 1


def test_mixed_user_is_not_exclusive():
    df = make_frame([{"s260c": 1, "s260a": 1}])  # pads AND cloth
    out = build_outcomes(build_protection_flags(df))
    assert out["any_hygienic_narrow"].iloc[0] == 1
    assert out["exclusive_hygienic_narrow"].iloc[0] == 0


def test_locally_prepared_flips_with_definition():
    """The single choice worth ~28 points of prevalence."""
    df = make_frame([{"s260b": 1}])  # locally prepared napkins only
    out = build_outcomes(build_protection_flags(df))
    assert out["exclusive_hygienic_narrow"].iloc[0] == 0
    assert out["exclusive_hygienic_broad"].iloc[0] == 1


def test_suffix_trap():
    """s260e is MENSTRUAL CUP; s260f is NOTHING.

    In NFHS-4 the 'nothing' response sits at suffix (e). Any code that maps
    across rounds by letter turns the most deprived respondents into cup
    users -- inverting the outcome exactly where it matters most.
    """
    assert PROTECTION_ITEMS["nfhs5"]["s260e"] == "menstrual_cup"
    assert PROTECTION_ITEMS["nfhs5"]["s260f"] == "nothing"
    assert PROTECTION_ITEMS["nfhs4"]["s257e"] == "nothing"
    assert "s257e" not in PROTECTION_ITEMS["nfhs4"] or (
        PROTECTION_ITEMS["nfhs4"]["s257e"] != PROTECTION_ITEMS["nfhs5"]["s260e"]
    )

    uses_nothing = make_frame([{"s260f": 1}])
    out = build_outcomes(build_protection_flags(uses_nothing))
    assert out["any_hygienic_narrow"].iloc[0] == 0, "'nothing' must never be hygienic"

    uses_cup = make_frame([{"s260e": 1}])
    out = build_outcomes(build_protection_flags(uses_cup))
    assert out["exclusive_hygienic_narrow"].iloc[0] == 1


def test_no_response_is_missing_not_zero():
    """A woman with no positive response is unknown, not unhygienic."""
    df = make_frame([{}])
    out = build_outcomes(build_protection_flags(df))
    assert np.isnan(out["any_hygienic_narrow"].iloc[0])


def test_weighted_mean_respects_weights():
    s = pd.Series([1.0, 0.0])
    w = pd.Series([3.0, 1.0])
    assert weighted_mean(s, w) == pytest.approx(0.75)


def test_build_outcomes_requires_flags_first():
    with pytest.raises(RuntimeError):
        build_outcomes(make_frame([{"s260c": 1}]))
