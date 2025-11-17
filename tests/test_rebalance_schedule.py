from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.schedules import apply_rebalance_schedule, get_rebalance_dates


def test_get_rebalance_dates_month_end_handles_missing_trading_days() -> None:
    index = pd.bdate_range("2023-01-02", periods=45)
    index = index[index < "2023-03-01"]
    index = index.drop(pd.Timestamp("2023-01-31"))
    index = index.drop(pd.Timestamp("2023-02-28"))

    calendar = get_rebalance_dates(index, "monthly")
    expected = pd.DatetimeIndex(
        [pd.Timestamp("2023-01-30"), pd.Timestamp("2023-02-27")],
        name="rebalance_date",
    )
    assert calendar.equals(expected)


def test_get_rebalance_dates_weekly_skips_holidays() -> None:
    index = pd.bdate_range("2023-01-02", periods=15)
    index = index.drop(pd.Timestamp("2023-01-06"))

    calendar = get_rebalance_dates(index, "weekly")
    expected = pd.DatetimeIndex(
        [
            pd.Timestamp("2023-01-05"),
            pd.Timestamp("2023-01-13"),
            pd.Timestamp("2023-01-20"),
        ],
        name="rebalance_date",
    )
    assert calendar.equals(expected)


def test_get_rebalance_dates_custom_schedule_intersection() -> None:
    index = pd.bdate_range("2023-01-02", periods=10)
    custom = ["2023-01-01", "2023-01-05", "2023-01-12"]

    calendar = get_rebalance_dates(index, custom)
    expected = pd.DatetimeIndex(
        [pd.Timestamp("2023-01-05"), pd.Timestamp("2023-01-12")],
        name="rebalance_date",
    )
    assert calendar.equals(expected)


def test_apply_rebalance_schedule_only_changes_on_calendar() -> None:
    index = pd.bdate_range("2023-01-02", periods=15)
    positions = pd.DataFrame(
        {
            "A": np.linspace(0.1, 1.5, len(index)),
            "B": np.linspace(-0.2, 0.8, len(index)),
        },
        index=index,
    )
    calendar = get_rebalance_dates(index, "weekly")

    applied = apply_rebalance_schedule(positions, calendar)

    np.testing.assert_allclose(applied.loc[calendar], positions.loc[calendar])

    diffs = applied.diff().abs().sum(axis=1).fillna(0.0)
    changed = diffs[diffs > 1e-12]
    assert changed.index.equals(calendar)

    mid_week = pd.Timestamp("2023-01-10")
    assert applied.loc[mid_week].equals(applied.loc[calendar[0]])


def test_apply_rebalance_schedule_preserves_initial_window() -> None:
    index = pd.bdate_range("2023-01-02", periods=5)
    weights = pd.Series([0.2, 0.25, 0.3, 0.35, 0.4], index=index, name="weights")
    calendar = get_rebalance_dates(index, "weekly")

    applied = apply_rebalance_schedule(weights, calendar)

    assert applied.iloc[0] == weights.iloc[0]
    # The second trading day occurs before the first rebalance date, so it
    # should still reflect the initial portfolio.
    assert applied.iloc[1] == weights.iloc[0]


def test_apply_rebalance_schedule_preserves_dtype_and_overlapping_calendar() -> None:
    index = pd.bdate_range("2023-01-02", periods=6)
    positions = pd.Series([1, 2, 3, 4, 5, 6], index=index, dtype="int64")
    calendar = pd.DatetimeIndex([index[2], index[5]])

    applied = apply_rebalance_schedule(positions, calendar)

    assert applied.dtype == positions.dtype
    assert applied.loc[index[3]] == positions.loc[index[2]]


def test_apply_rebalance_schedule_errors_when_calendar_missing() -> None:
    index = pd.bdate_range("2023-01-02", periods=3)
    positions = pd.Series([0.1, 0.2, 0.3], index=index)
    calendar = [index[-1] + pd.Timedelta(days=5)]

    with pytest.raises(ValueError, match="No rebalance dates overlap"):
        apply_rebalance_schedule(positions, calendar)
