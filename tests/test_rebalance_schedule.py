from __future__ import annotations

import numpy as np
import pandas as pd

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
