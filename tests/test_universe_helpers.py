from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.universe import (
    MembershipWindow,
    _expand_active_pairs,
    _normalise_membership_frame,
    _normalise_price_frame,
    apply_membership_windows,
)


def test_normalise_price_frame_requires_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        _normalise_price_frame([1, 2, 3])


def test_normalise_price_frame_renames_and_coerces_types() -> None:
    prices = pd.DataFrame({"Date": ["2020-01-01"], "Symbol": [1], "value": [0.1]})

    normalised = _normalise_price_frame(prices)

    assert list(normalised.columns)[:2] == ["date", "symbol"]
    assert normalised["date"].iloc[0] == pd.Timestamp("2020-01-01")
    assert normalised["symbol"].dtype == object


def test_normalise_price_frame_validates_columns() -> None:
    prices = pd.DataFrame({"symbol": ["AAA"], "value": [1]})

    with pytest.raises(ValueError, match="date"):
        _normalise_price_frame(prices)

    with pytest.raises(ValueError, match="symbol"):
        _normalise_price_frame(pd.DataFrame({"date": ["2020-01-01"], "value": [1]}))


def test_normalise_membership_frame_from_mapping_and_invalid_dates() -> None:
    membership = {"AAA": (MembershipWindow(pd.Timestamp("2020-01-01"), None),)}

    frame = _normalise_membership_frame(membership)

    assert frame.columns.tolist() == ["symbol", "effective_date", "end_date"]
    assert frame.loc[0, "symbol"] == "AAA"
    assert frame.loc[0, "effective_date"] == pd.Timestamp("2020-01-01")

    bad_membership = {"AAA": ({"effective_date": None},)}
    with pytest.raises(ValueError, match="valid effective dates"):
        _normalise_membership_frame(bad_membership)


def test_normalise_membership_frame_missing_required_columns() -> None:
    missing_symbol = pd.DataFrame({"effective_date": ["2020-01-01"]})
    with pytest.raises(ValueError, match="symbol"):
        _normalise_membership_frame(missing_symbol)

    missing_effective = pd.DataFrame({"symbol": ["AAA"]})
    with pytest.raises(ValueError, match="effective_date"):
        _normalise_membership_frame(missing_effective)


def test_normalise_membership_frame_defaults_end_date_and_handles_empty() -> None:
    membership = pd.DataFrame({"symbol": ["AAA"], "effective_date": ["2020-01-01"]})

    frame = _normalise_membership_frame(membership)

    assert frame["end_date"].isna().all()

    empty = _normalise_membership_frame(pd.DataFrame())
    assert empty.empty


def test_expand_active_pairs_handles_empty_inputs() -> None:
    assert _expand_active_pairs(pd.DataFrame(), pd.Index([])).empty

    membership = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "effective_date": [pd.Timestamp("2020-02-01")],
            "end_date": [pd.Timestamp("2020-02-28")],
        }
    )
    assert _expand_active_pairs(membership, pd.Index(["2020-01-01"])).empty


def test_apply_membership_windows_returns_unmodified_for_empty_inputs() -> None:
    frame = pd.DataFrame()
    membership = {"AAA": (MembershipWindow(pd.Timestamp("2020-01-01"), None),)}
    assert apply_membership_windows(frame, membership).empty

    non_empty = pd.DataFrame({"AAA": [1]})
    assert apply_membership_windows(non_empty, {}).equals(non_empty)
