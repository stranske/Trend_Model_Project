from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trend_analysis.universe import (
    MembershipWindow,
    apply_membership_windows,
    build_membership_mask,
    gate_universe,
    load_universe_membership,
)


def test_load_universe_membership_requires_effective_date(tmp_path: Path) -> None:
    csv = tmp_path / "membership.csv"
    csv.write_text("fund,effective_date,end_date\nFundA,,\n", encoding="utf-8")

    with pytest.raises(ValueError, match="effective_date"):
        load_universe_membership(csv)


def test_load_universe_membership_requires_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_universe_membership(missing)


def test_load_universe_membership_validates_columns(tmp_path: Path) -> None:
    csv = tmp_path / "membership.csv"
    csv.write_text("fund,end_date\nFundA,2020-01-01\n", encoding="utf-8")

    with pytest.raises(ValueError, match="effective_date"):
        load_universe_membership(csv)


def test_load_universe_membership_rejects_blank_fund(tmp_path: Path) -> None:
    csv = tmp_path / "membership.csv"
    csv.write_text("fund,effective_date\n ,2020-01-01\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fund name"):
        load_universe_membership(csv)


def test_apply_membership_masks_adds_and_drops() -> None:
    idx = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    frame = pd.DataFrame(
        {
            "FundA": [0.1, 0.2, 0.3],
            "FundB": [0.4, 0.5, 0.6],
            "FundC": [0.7, 0.8, 0.9],
        },
        index=idx,
    )
    membership = {
        "FundA": (MembershipWindow(pd.Timestamp("2020-01-31"), None),),
        # FundB enters after the first month
        "FundB": (MembershipWindow(pd.Timestamp("2020-02-29"), None),),
        # FundC exits after February
        "FundC": (
            MembershipWindow(pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")),
        ),
    }

    masked = apply_membership_windows(frame, membership)

    assert pd.isna(masked.loc[pd.Timestamp("2020-01-31"), "FundB"])
    assert not pd.isna(masked.loc[pd.Timestamp("2020-02-29"), "FundB"])
    assert pd.isna(masked.loc[pd.Timestamp("2020-03-31"), "FundC"])
    assert masked.loc[pd.Timestamp("2020-02-29"), "FundC"] == pytest.approx(0.8)


def test_apply_membership_windows_coerces_index_and_ignores_unknown_funds() -> None:
    frame = pd.DataFrame({"AAA": [1, 2]}, index=["2020-01-01", "2020-01-02"])
    membership = {
        "AAA": (MembershipWindow(pd.Timestamp("2020-01-02"), None),),
        "ZZZ": (MembershipWindow(pd.Timestamp("2020-01-01"), None),),
    }

    masked = apply_membership_windows(frame, membership)

    assert isinstance(masked.index, pd.DatetimeIndex)
    assert pd.isna(masked.loc[pd.Timestamp("2020-01-01"), "AAA"])
    assert masked.loc[pd.Timestamp("2020-01-02"), "AAA"] == 2
    # Unknown fund is ignored and does not add a column
    assert "ZZZ" not in masked.columns


def test_gate_universe_enforces_membership_daily() -> None:
    prices = pd.DataFrame(
        {
            "Date": [
                "2020-01-31",
                "2020-01-31",
                "2020-02-29",
                "2020-02-29",
                "2020-03-31",
            ],
            "Symbol": ["AAA", "BBB", "AAA", "BBB", "BBB"],
            "return": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    membership = pd.DataFrame(
        {
            "fund": ["AAA", "BBB"],
            "effective_date": ["2020-01-01", "2020-02-01"],
            "end_date": [None, "2020-02-29"],
        }
    )

    gated = gate_universe(prices, membership, pd.Timestamp("2020-03-31"))

    assert list(gated["symbol"].unique()) == ["AAA", "BBB"]
    feb_rows = gated[gated["date"] == pd.Timestamp("2020-02-29")]
    assert set(feb_rows["symbol"]) == {"AAA", "BBB"}
    march_rows = gated[gated["date"] == pd.Timestamp("2020-03-31")]
    assert march_rows.empty
    jan_rows = gated[gated["date"] == pd.Timestamp("2020-01-31")]
    assert jan_rows["symbol"].tolist() == ["AAA"]


def test_gate_universe_validates_as_of_and_membership() -> None:
    prices = pd.DataFrame({"date": ["2020-01-01"], "symbol": ["AAA"], "value": [1]})

    with pytest.raises(ValueError, match="Unknown datetime string format"):
        gate_universe(prices, prices, "invalid-date")

    empty_membership = pd.DataFrame(columns=["symbol", "effective_date", "end_date"])
    empty = gate_universe(prices, empty_membership, pd.Timestamp("2020-01-01"))
    assert empty.empty

    future_membership = pd.DataFrame(
        {"symbol": ["AAA"], "effective_date": ["2020-02-01"], "end_date": [None]}
    )
    future = gate_universe(prices, future_membership, pd.Timestamp("2020-01-15"))
    assert future.empty


def test_gate_universe_rebalance_only_uses_single_date(tmp_path: Path) -> None:
    prices = pd.DataFrame(
        {
            "date": ["2020-02-29", "2020-02-29", "2020-03-31"],
            "symbol": ["AAA", "BBB", "BBB"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    membership_csv = tmp_path / "membership.csv"
    membership_csv.write_text(
        "symbol,effective_date,end_date\nAAA,2020-01-01,\nBBB,2020-03-01,\n",
        encoding="utf-8",
    )

    gated = gate_universe(
        prices,
        str(membership_csv),
        pd.Timestamp("2020-02-29"),
        rebalance_only=True,
    )

    assert list(gated["symbol"]) == ["AAA"]
    assert (gated["date"] == pd.Timestamp("2020-02-29")).all()


def test_gate_universe_matches_date_symbol_pairs() -> None:
    prices = pd.DataFrame(
        {
            "date": [
                "2020-01-31",
                "2020-02-29",
                "2020-02-29",
                "2020-03-31",
                "2020-01-31",
            ],
            "symbol": ["BBB", "AAA", "BBB", "AAA", "AAA"],
            "value": [5, 1, 2, 3, 4],
        }
    )
    membership = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "effective_date": ["2020-01-01", "2020-02-01"],
            "end_date": [None, "2020-02-29"],
        }
    )

    gated = gate_universe(prices, membership, pd.Timestamp("2020-03-31"))

    expected_pairs = [
        (pd.Timestamp("2020-01-31"), "AAA"),
        (pd.Timestamp("2020-02-29"), "AAA"),
        (pd.Timestamp("2020-02-29"), "BBB"),
        (pd.Timestamp("2020-03-31"), "AAA"),
    ]
    assert list(zip(gated["date"], gated["symbol"])) == expected_pairs


def test_gate_universe_drops_when_active_pairs_empty() -> None:
    prices = pd.DataFrame(
        {"date": ["2020-01-31"], "symbol": ["AAA"], "value": [1]},
    )
    membership = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "effective_date": ["2020-02-01"],
            "end_date": ["2020-02-29"],
        }
    )

    gated = gate_universe(prices, membership, pd.Timestamp("2020-01-31"))

    assert gated.empty


def test_build_membership_mask_marks_entries_and_exits() -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    membership = pd.DataFrame(
        {
            "fund": ["Alpha", "Beta", "Gamma"],
            "effective_date": ["2020-01-01", "2020-02-01", "2020-01-01"],
            "end_date": [None, None, "2020-02-29"],
        }
    )

    mask = build_membership_mask(dates, membership)

    assert not bool(mask.loc[pd.Timestamp("2020-01-31"), "Beta"])
    assert bool(mask.loc[pd.Timestamp("2020-02-29"), "Beta"])
    assert not bool(mask.loc[pd.Timestamp("2020-03-31"), "Gamma"])


def test_build_membership_mask_accepts_membership_table() -> None:
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    membership = {
        "AAA": (
            MembershipWindow(pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")),
        )
    }

    mask = build_membership_mask(dates, membership)

    assert bool(mask.loc[pd.Timestamp("2020-01-31"), "AAA"])
    assert bool(mask.loc[pd.Timestamp("2020-02-29"), "AAA"])
    assert mask.columns.tolist() == ["AAA"]


def test_build_membership_mask_handles_empty_inputs() -> None:
    empty_dates = build_membership_mask([], {})
    assert empty_dates.empty

    mask = build_membership_mask(["2020-01-01"], pd.DataFrame())
    assert mask.empty
