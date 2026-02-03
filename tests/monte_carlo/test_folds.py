from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.monte_carlo.folds import FoldGenerator


def test_explicit_mode_requires_fold_starts() -> None:
    index = pd.date_range("2020-01-31", periods=6, freq="ME")

    with pytest.raises(ValueError, match="fold_starts"):
        FoldGenerator(mode="explicit").generate(index)


def test_explicit_mode_dedupes_and_aligns_starts() -> None:
    index = pd.date_range("2020-01-31", periods=6, freq="ME")
    generator = FoldGenerator(
        mode="explicit",
        fold_starts=["2020-05-31", "2020-03-31", "2020-05-31"],
    )

    folds = generator.generate(index)

    assert [fold.fold_id for fold in folds] == [1, 2]
    assert [fold.forecast_start for fold in folds] == [
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2020-05-31"),
    ]
    assert folds[0].calibration_end == pd.Timestamp("2020-02-29")
    assert folds[0].calibration_start == pd.Timestamp("2020-01-31")
    assert folds[0].label == "2020-03"


def test_rolling_mode_builds_calibration_windows_and_alignment() -> None:
    index = pd.date_range("2020-01-31", periods=12, freq="ME")
    generator = FoldGenerator(
        mode="rolling",
        start="2020-03-15",
        end="2020-09-30",
        step_months=3,
        calibration_lookback_years=1.0,
    )

    folds = generator.generate(index)

    assert [fold.forecast_start for fold in folds] == [
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2020-06-30"),
        pd.Timestamp("2020-09-30"),
    ]
    assert folds[0].calibration_end == pd.Timestamp("2020-02-29")
    assert folds[0].calibration_start == pd.Timestamp("2020-01-31")
    assert folds[1].calibration_end == pd.Timestamp("2020-05-31")
    assert folds[1].calibration_start == pd.Timestamp("2020-01-31")


def test_rolling_mode_step_years_and_n_folds_limit() -> None:
    index = pd.date_range("2020-01-31", periods=18, freq="ME")
    generator = FoldGenerator(
        mode="rolling",
        start="2020-02-29",
        end="2021-12-31",
        step_years=0.5,
        n_folds=2,
    )

    folds = generator.generate(index)

    assert [fold.forecast_start for fold in folds] == [
        pd.Timestamp("2020-02-29"),
        pd.Timestamp("2020-08-31"),
    ]


def test_count_spaced_mode_applies_lookback_and_calibration_window() -> None:
    index = pd.date_range("2020-01-31", periods=24, freq="ME")
    generator = FoldGenerator(
        mode="count_spaced",
        start="2021-01-31",
        end="2021-12-31",
        n_folds=2,
        calibration_lookback_years=1.0,
    )

    folds = generator.generate(index)

    assert [fold.forecast_start for fold in folds] == [
        pd.Timestamp("2021-01-31"),
        pd.Timestamp("2021-12-31"),
    ]
    assert folds[0].calibration_end == pd.Timestamp("2020-12-31")
    assert folds[0].calibration_start == pd.Timestamp("2020-01-31")
    assert folds[1].calibration_end == pd.Timestamp("2021-11-30")
    assert folds[1].calibration_start == pd.Timestamp("2020-11-30")


def test_count_spaced_mode_shifts_start_for_lookback() -> None:
    index = pd.date_range("2020-01-31", periods=24, freq="ME")
    generator = FoldGenerator(
        mode="count_spaced",
        start="2020-01-31",
        end="2021-12-31",
        n_folds=2,
        calibration_lookback_years=1.5,
    )

    folds = generator.generate(index)

    assert [fold.forecast_start for fold in folds] == [
        pd.Timestamp("2021-07-31"),
        pd.Timestamp("2021-12-31"),
    ]
    assert folds[0].calibration_end == pd.Timestamp("2021-06-30")
    assert folds[0].calibration_start == pd.Timestamp("2020-01-31")


def test_count_spaced_mode_rejects_invalid_range() -> None:
    index = pd.date_range("2020-01-31", periods=6, freq="ME")
    generator = FoldGenerator(
        mode="count_spaced",
        start="2021-01-31",
        end="2020-12-31",
        n_folds=2,
    )

    with pytest.raises(ValueError, match="fold start must be before fold end"):
        generator.generate(index)
