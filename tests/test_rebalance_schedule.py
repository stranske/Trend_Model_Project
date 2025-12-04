from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.schedules import (
    _coerce_datetime_index,
    _match_timezone,
    apply_rebalance_schedule,
    get_rebalance_dates,
    normalize_positions,
)


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


def test_get_rebalance_dates_validates_frequency_inputs() -> None:
    index = pd.bdate_range("2024-01-02", periods=5)

    with pytest.raises(ValueError, match="freq must be a non-empty string"):
        get_rebalance_dates(index, "  ")

    with pytest.raises(ValueError, match="Unsupported frequency alias: quarterlyy"):
        get_rebalance_dates(index, "quarterlyy")

    with pytest.raises(TypeError, match="freq must be a string frequency alias"):
        get_rebalance_dates(index, 123)  # type: ignore[arg-type]


def test_get_rebalance_dates_matches_timezone_and_deduplicates_custom() -> None:
    index = pd.date_range("2024-02-01", periods=4, freq="D", tz="UTC")
    custom = pd.DatetimeIndex(["2024-02-02", "2024-02-03", "2024-02-02"])

    calendar = get_rebalance_dates(index, custom)

    expected = pd.DatetimeIndex(
        [pd.Timestamp("2024-02-02", tz="UTC"), pd.Timestamp("2024-02-03", tz="UTC")],
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

    normalised = normalize_positions(positions)
    applied = apply_rebalance_schedule(positions, calendar)

    np.testing.assert_allclose(applied.loc[calendar], normalised.loc[calendar])

    diffs = applied.diff().abs().sum(axis=1).fillna(0.0)
    changed = diffs[diffs > 1e-12]
    assert changed.index.equals(calendar)

    mid_week = pd.Timestamp("2023-01-10")
    assert applied.loc[mid_week].equals(normalised.loc[calendar[0]])


def test_apply_rebalance_schedule_requires_datetime_index() -> None:
    positions = pd.Series([0.1, 0.2], index=["2023-01-01", "2023-01-02"])
    with pytest.raises(TypeError, match="positions index must be a DatetimeIndex"):
        apply_rebalance_schedule(positions, [pd.Timestamp("2023-01-02")])


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

    assert applied.dtype == np.dtype("float64")
    expected = normalize_positions(positions.to_frame("position"))["position"]
    assert applied.loc[index[3]] == expected.loc[index[2]]


def test_apply_rebalance_schedule_resorts_to_original_order() -> None:
    ascending = pd.bdate_range("2023-01-02", periods=4)
    scrambled = pd.DatetimeIndex(
        [ascending[2], ascending[0], ascending[3], ascending[1]]
    )
    positions = pd.Series([1.0, 2.0, 3.0, 4.0], index=scrambled, name="positions")

    calendar = get_rebalance_dates(scrambled, "weekly")
    applied = apply_rebalance_schedule(positions, calendar)

    assert applied.index.equals(scrambled)
    assert applied.loc[scrambled[1]] == applied.loc[scrambled[0]]


def test_apply_rebalance_schedule_errors_when_calendar_missing() -> None:
    index = pd.bdate_range("2023-01-02", periods=3)
    positions = pd.Series([0.1, 0.2, 0.3], index=index)
    calendar = [index[-1] + pd.Timedelta(days=5)]

    with pytest.raises(ValueError, match="No rebalance dates overlap"):
        apply_rebalance_schedule(positions, calendar)


def test_normalize_positions_clamps_and_masks() -> None:
    index = pd.to_datetime(["2023-01-31", "2023-02-28"])
    df = pd.DataFrame(
        {
            "AAA": [1.4, -1.6],
            "BBB": [np.nan, 0.4],
            "IGNORED": [0.5, -0.25],
        },
        index=index,
    )

    normalized = normalize_positions(df, eligible=["BBB", "AAA"])

    assert list(normalized.columns) == ["BBB", "AAA"]
    np.testing.assert_allclose(
        normalized.iloc[0].to_numpy(),
        np.array([0.0, 1.0]),
    )
    np.testing.assert_allclose(
        normalized.iloc[1].to_numpy(),
        np.array([0.4, -1.0]),
    )


def test_normalize_positions_requires_datetime_index() -> None:
    df = pd.DataFrame({"AAA": [0.1]}, index=["not-a-date"])
    with pytest.raises(TypeError, match="normalize_positions contract"):
        normalize_positions(df)


def test_normalize_positions_validates_inputs() -> None:
    positions = pd.DataFrame(
        {"A": [0.1, 0.2]}, index=pd.to_datetime(["2023-01-01", "2023-01-01"])
    )

    with pytest.raises(
        TypeError, match="positions must be provided as a pandas DataFrame"
    ):
        normalize_positions([1, 2])  # type: ignore[arg-type]


def test_normalize_positions_rejects_duplicate_columns_and_empty_eligible() -> None:
    df = pd.DataFrame([[0.1, 0.2]], columns=["AAA", "AAA"], index=pd.to_datetime(["2024-01-01"]))

    with pytest.raises(
        ValueError, match="positions columns must be unique per the normalize_positions contract"
    ):
        normalize_positions(df)

    with pytest.raises(
        ValueError, match="eligible must include at least one symbol per the normalize_positions contract"
    ):
        normalize_positions(df.loc[:, ~df.columns.duplicated()], eligible=[])


def test_normalize_positions_rejects_duplicate_index() -> None:
    duplicated_index = pd.to_datetime(["2024-02-01", "2024-02-01"])
    df = pd.DataFrame({"AAA": [0.1, 0.2]}, index=duplicated_index)

    with pytest.raises(
        ValueError, match="positions index must be unique per the normalize_positions contract"
    ):
        normalize_positions(df)


def test_match_timezone_aligns_naive_and_aware_indices() -> None:
    aware_template = pd.date_range("2024-03-01", periods=2, freq="D", tz="UTC")
    naive_index = pd.DatetimeIndex(["2024-03-01", "2024-03-02"], name="rebalance_date")

    localized = _match_timezone(naive_index, aware_template)
    assert str(localized.tz) == "UTC"
    assert localized.tz_convert(None).equals(naive_index)

    aware_index = pd.DatetimeIndex(["2024-03-01", "2024-03-02"], tz="UTC")
    dropped = _match_timezone(aware_index, pd.DatetimeIndex(["2024-03-01", "2024-03-02"]))
    assert dropped.tz is None
    assert dropped.equals(pd.DatetimeIndex(["2024-03-01", "2024-03-02"]))


def test_normalize_positions_two_asset_reproducible_weights() -> None:
    index = pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31"])
    df = pd.DataFrame(
        {
            "AAA": [0.75, 1.5, -2.0],
            "BBB": [np.nan, -0.8, 0.5],
        },
        index=index,
    )

    normalized = normalize_positions(df)

    expected_last = pd.Series({"AAA": -1.0, "BBB": 0.5})
    pd.testing.assert_series_equal(
        normalized.iloc[-1], expected_last, check_names=False
    )


def test_coerce_datetime_index_rejects_unusable_values() -> None:
    with pytest.raises(TypeError, match="prices must be convertible"):
        _coerce_datetime_index(pd.Index([object()]), name="prices")


def test_match_timezone_localises_when_template_tz_set() -> None:
    idx = pd.DatetimeIndex(["2024-01-01"])
    template = pd.DatetimeIndex(["2024-01-01"], tz="UTC")

    converted = _match_timezone(idx, template)

    assert str(converted.tz) == "UTC"


def test_apply_rebalance_schedule_reindexes_to_eligible_set() -> None:
    index = pd.bdate_range("2024-03-01", periods=4)
    positions = pd.DataFrame(
        {
            "AAA": [0.3, 0.4, 0.2, 0.1],
            "BBB": [0.7, 0.6, 0.8, 0.9],
        },
        index=index,
    )

    schedule = get_rebalance_dates(index, "weekly")
    applied = apply_rebalance_schedule(positions, schedule, eligible=["BBB", "CCC"])

    assert list(applied.columns) == ["BBB", "CCC"]
    assert (applied["CCC"] == 0.0).all()


def test_match_timezone_drops_timezone_when_template_naive() -> None:
    idx = pd.DatetimeIndex(["2024-01-01"], tz="UTC")
    template = pd.DatetimeIndex(["2024-01-01"])

    converted = _match_timezone(idx, template)

    assert converted.tz is None
