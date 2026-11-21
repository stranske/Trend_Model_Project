from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trend_analysis.monte_carlo.folds import (
    FoldGenerator,
    _align_to_index,
    _previous_in_index,
)
from trend_analysis.monte_carlo.history import load_price_history
from trend_analysis.monte_carlo.models.bootstrap import StationaryBootstrapModel
from trend_analysis.monte_carlo.runner import MonteCarloRunner
from trend_analysis.monte_carlo.scenario import MonteCarloScenario
from trend_analysis.monte_carlo.strategy import StrategyVariant


def test_from_config_returns_none_when_disabled() -> None:
    generator = FoldGenerator.from_config({"enabled": False, "mode": "explicit"})

    assert generator is None


def test_generate_sorts_and_drops_timezone_from_index() -> None:
    index = pd.DatetimeIndex(
        [
            "2021-03-31",
            "2021-01-31",
            "2021-02-28",
        ]
    ).tz_localize("UTC")
    generator = FoldGenerator(mode="count_spaced", start="2021-02-15", n_folds=1)

    folds = generator.generate(index)

    assert folds[0].forecast_start == pd.Timestamp("2021-02-28")
    assert folds[0].calibration_end == pd.Timestamp("2021-01-31")
    assert folds[0].calibration_start == pd.Timestamp("2021-01-31")


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


def test_count_spaced_mode_single_fold_respects_lookback() -> None:
    index = pd.date_range("2020-01-31", periods=36, freq="ME")
    generator = FoldGenerator(
        mode="count_spaced",
        start="2020-01-31",
        end="2022-12-31",
        n_folds=1,
        calibration_lookback_years=1.0,
    )

    folds = generator.generate(index)

    assert len(folds) == 1
    assert folds[0].forecast_start == pd.Timestamp("2021-01-31")
    assert folds[0].calibration_end == pd.Timestamp("2020-12-31")
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


def test_align_to_index_exact_match() -> None:
    index = pd.date_range("2020-01-31", periods=3, freq="ME")

    assert _align_to_index(pd.Timestamp("2020-02-29"), index) == pd.Timestamp("2020-02-29")


def test_align_to_index_before_first_returns_first() -> None:
    index = pd.date_range("2020-01-31", periods=3, freq="ME")

    assert _align_to_index(pd.Timestamp("2019-12-15"), index) == pd.Timestamp("2020-01-31")


def test_align_to_index_between_returns_next() -> None:
    index = pd.date_range("2020-01-31", periods=4, freq="ME")

    assert _align_to_index(pd.Timestamp("2020-02-10"), index) == pd.Timestamp("2020-02-29")


def test_align_to_index_after_last_returns_last() -> None:
    index = pd.date_range("2020-01-31", periods=3, freq="ME")

    assert _align_to_index(pd.Timestamp("2020-06-30"), index) == pd.Timestamp("2020-03-31")


def test_align_to_index_with_duplicates_skips_to_next_unique() -> None:
    index = pd.DatetimeIndex(
        [
            "2020-01-31",
            "2020-02-29",
            "2020-02-29",
            "2020-03-31",
        ]
    )

    assert _align_to_index(pd.Timestamp("2020-03-01"), index) == pd.Timestamp("2020-03-31")


def test_align_to_index_with_duplicates_exact_match_returns_value() -> None:
    index = pd.DatetimeIndex(
        [
            "2020-01-31",
            "2020-02-29",
            "2020-02-29",
            "2020-03-31",
        ]
    )

    assert _align_to_index(pd.Timestamp("2020-02-29"), index) == pd.Timestamp("2020-02-29")


def test_align_to_index_empty_index_raises() -> None:
    index = pd.DatetimeIndex([])

    with pytest.raises(ValueError, match="history_index must contain at least one date"):
        _align_to_index(pd.Timestamp("2020-01-31"), index)


def test_previous_in_index_between_returns_previous() -> None:
    index = pd.date_range("2020-01-31", periods=4, freq="ME")

    assert _previous_in_index(pd.Timestamp("2020-02-10"), index) == pd.Timestamp("2020-01-31")


def test_previous_in_index_exact_value_returns_previous() -> None:
    index = pd.date_range("2020-01-31", periods=4, freq="ME")

    assert _previous_in_index(pd.Timestamp("2020-03-31"), index) == pd.Timestamp("2020-02-29")


def test_previous_in_index_with_duplicates_between_returns_duplicate() -> None:
    index = pd.DatetimeIndex(
        [
            "2020-01-31",
            "2020-02-29",
            "2020-02-29",
            "2020-03-31",
        ]
    )

    assert _previous_in_index(pd.Timestamp("2020-03-01"), index) == pd.Timestamp("2020-02-29")


def test_previous_in_index_after_last_returns_last() -> None:
    index = pd.date_range("2020-01-31", periods=3, freq="ME")

    assert _previous_in_index(pd.Timestamp("2020-12-31"), index) == pd.Timestamp("2020-03-31")


def test_previous_in_index_before_first_raises() -> None:
    index = pd.date_range("2020-01-31", periods=3, freq="ME")

    with pytest.raises(ValueError, match="fold_start must be after the earliest history date"):
        _previous_in_index(pd.Timestamp("2019-12-31"), index)


def test_previous_in_index_singleton_after_returns_only_value() -> None:
    index = pd.DatetimeIndex(["2020-01-31"])

    assert _previous_in_index(pd.Timestamp("2020-02-15"), index) == pd.Timestamp("2020-01-31")


def test_previous_in_index_at_start_raises() -> None:
    index = pd.date_range("2020-01-31", periods=3, freq="ME")

    with pytest.raises(ValueError, match="fold_start must be after the earliest history date"):
        _previous_in_index(pd.Timestamp("2020-01-31"), index)


def test_previous_in_index_empty_index_raises() -> None:
    index = pd.DatetimeIndex([])

    with pytest.raises(ValueError, match="history_index must contain at least one date"):
        _previous_in_index(pd.Timestamp("2020-01-31"), index)


def _runner_base_config(csv_path: Path) -> dict[str, Any]:
    return {
        "version": "0.1.0",
        "data": {
            "csv_path": str(csv_path),
            "date_column": "Date",
            "frequency": "M",
            "allow_risk_free_fallback": True,
        },
        "preprocessing": {},
        "vol_adjust": {"enabled": False, "target_vol": 0.1, "window": {"length": 3}},
        "sample_split": {"method": "ratio", "ratio": 0.6},
        "portfolio": {
            "selection_mode": "all",
            "weighting": {"name": "equal"},
            "cost_model": {"per_trade_bps": 0, "half_spread_bps": 0},
        },
        "benchmarks": {},
        "metrics": {"registry": ["annual_return", "volatility", "sharpe_ratio"]},
        "regime": {},
        "export": {},
        "run": {"monthly_cost": 0.0},
    }


def _runner_for_folds(csv_path: Path, folds_config: dict[str, Any]) -> MonteCarloRunner:
    scenario = MonteCarloScenario(
        name="fold_default_start_check",
        base_config="config/defaults.yml",
        monte_carlo={
            "mode": "two_layer",
            "n_paths": 2,
            "horizon_years": 0.5,
            "frequency": "M",
            "seed": 42,
            "jobs": 1,
        },
        strategy_set={"curated": [StrategyVariant(name="EqualWeight")]},
        return_model={"kind": "stationary_bootstrap", "params": {"block_size": 2}},
        folds=folds_config,
    )
    return MonteCarloRunner(scenario, base_config=_runner_base_config(csv_path))


def test_default_start_has_feasible_calibration_history() -> None:
    fixture_path = Path("tests/fixtures/regime_lookback_sensitive.csv")
    history = load_price_history(fixture_path)

    # Both implicit-date modes derive the earliest start with >= 2 calibration observations.
    rolling_cfg = {"enabled": True, "mode": "rolling", "step_months": 3}
    rolling_gen = FoldGenerator.from_config(rolling_cfg)
    assert rolling_gen is not None
    rolling_folds = rolling_gen.generate(history.index)
    assert [fold.forecast_start for fold in rolling_folds] == [
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2020-06-30"),
    ]
    assert rolling_folds[0].calibration_start == pd.Timestamp("2020-01-31")
    assert rolling_folds[0].calibration_end == pd.Timestamp("2020-02-29")
    rolling_window = history.loc[
        rolling_folds[0].calibration_start : rolling_folds[0].calibration_end
    ]
    assert len(rolling_window) == 2
    assert len(StationaryBootstrapModel().fit(rolling_window).historical_log_returns) == 1

    spaced_cfg = {"enabled": True, "mode": "count_spaced", "n_folds": 2}
    spaced_gen = FoldGenerator.from_config(spaced_cfg)
    assert spaced_gen is not None
    spaced_folds = spaced_gen.generate(history.index)
    assert [fold.forecast_start for fold in spaced_folds] == [
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2020-08-31"),
    ]
    assert spaced_folds[0].calibration_start == pd.Timestamp("2020-01-31")
    assert spaced_folds[0].calibration_end == pd.Timestamp("2020-02-29")
    spaced_window = history.loc[spaced_folds[0].calibration_start : spaced_folds[0].calibration_end]
    assert len(spaced_window) == 2
    assert len(StationaryBootstrapModel().fit(spaced_window).historical_log_returns) == 1

    # Supported end-to-end Monte Carlo runner entry point executes both implicit-date modes.
    for cfg in (rolling_cfg, spaced_cfg):
        results = _runner_for_folds(fixture_path, cfg).run(jobs=1)
        assert results.errors == []
        assert results.metadata["n_folds"] == 2
        assert results.metadata["folds"][0]["forecast_start"].startswith("2020-03-31")

    # Compatibility controls: supported explicit dates and calibration windows are retained.
    explicit_gen = FoldGenerator.from_config(
        {"enabled": True, "mode": "explicit", "fold_starts": ["2020-05-31"]}
    )
    assert explicit_gen is not None
    explicit_folds = explicit_gen.generate(history.index)
    assert [fold.forecast_start for fold in explicit_folds] == [pd.Timestamp("2020-05-31")]
    assert explicit_folds[0].calibration_start == pd.Timestamp("2020-01-31")
    assert explicit_folds[0].calibration_end == pd.Timestamp("2020-04-30")

    explicit_rolling_gen = FoldGenerator.from_config(
        {"enabled": True, "mode": "rolling", "start": "2020-02-29", "n_folds": 1}
    )
    assert explicit_rolling_gen is not None
    explicit_rolling_folds = explicit_rolling_gen.generate(history.index)
    assert [fold.forecast_start for fold in explicit_rolling_folds] == [pd.Timestamp("2020-02-29")]
    assert explicit_rolling_folds[0].calibration_start == pd.Timestamp("2020-01-31")
    assert explicit_rolling_folds[0].calibration_end == pd.Timestamp("2020-01-31")

    explicit_spaced_gen = FoldGenerator.from_config(
        {"enabled": True, "mode": "count_spaced", "start": "2020-04-30", "n_folds": 2}
    )
    assert explicit_spaced_gen is not None
    explicit_spaced_folds = explicit_spaced_gen.generate(history.index)
    assert [fold.forecast_start for fold in explicit_spaced_folds] == [
        pd.Timestamp("2020-04-30"),
        pd.Timestamp("2020-08-31"),
    ]

    # Explicitly impossible dates and bounds raise actionable insufficient-history errors
    # naming the missing history (including the legacy index.min() boundary 2020-01-31).
    impossible_cases: list[tuple[dict[str, Any], tuple[str, ...]]] = [
        (
            {"enabled": True, "mode": "explicit", "fold_starts": ["2020-01-31"]},
            ("insufficient calibration history", "2020-01-31"),
        ),
        (
            {"enabled": True, "mode": "rolling", "start": "2019-12-31"},
            ("insufficient calibration history", "2019-12-31", "2020-01-31"),
        ),
        (
            {"enabled": True, "mode": "count_spaced", "start": "2020-01-31", "n_folds": 2},
            ("insufficient calibration history", "2020-01-31"),
        ),
        (
            {"enabled": True, "mode": "explicit", "fold_starts": ["2021-01-31"]},
            ("insufficient calibration history", "2021-01-31", "2020-08-31"),
        ),
        (
            {"enabled": True, "mode": "rolling", "start": "2021-01-31"},
            ("insufficient calibration history", "2021-01-31", "2020-08-31"),
        ),
        (
            {"enabled": True, "mode": "rolling", "end": "2020-02-29"},
            ("insufficient calibration history", "2020-02-29", "2020-03-31"),
        ),
        (
            {"enabled": True, "mode": "count_spaced", "end": "2020-01-31", "n_folds": 1},
            ("insufficient calibration history", "2020-01-31", "2020-08-31"),
        ),
        (
            {
                "enabled": True,
                "mode": "count_spaced",
                "n_folds": 2,
                "calibration_lookback_years": 2.0,
            },
            ("insufficient calibration history", "2022-01-31", "2020-01-31 to 2020-08-31"),
        ),
    ]
    for impossible_cfg, expected_tokens in impossible_cases:
        gen = FoldGenerator.from_config(impossible_cfg)
        assert gen is not None
        with pytest.raises(ValueError) as exc_info:
            gen.generate(history.index)
        message = str(exc_info.value)
        for token in expected_tokens:
            assert token.lower() in message.lower(), f"Missing {token!r} in {message!r}"


def test_default_start_on_minimal_three_point_checked_in_series() -> None:
    sample_path = Path("data/raw/managers/sample_manager.csv")
    history = load_price_history(sample_path)

    for cfg in (
        {"enabled": True, "mode": "rolling"},
        {"enabled": True, "mode": "count_spaced", "n_folds": 1},
    ):
        gen = FoldGenerator.from_config(cfg)
        assert gen is not None
        folds = gen.generate(history.index)
        assert len(folds) == 1
        assert folds[0].forecast_start == pd.Timestamp("2023-03-31")
        assert folds[0].calibration_start == pd.Timestamp("2023-01-31")
        assert folds[0].calibration_end == pd.Timestamp("2023-02-28")
        window = history.loc[folds[0].calibration_start : folds[0].calibration_end]
        assert len(StationaryBootstrapModel().fit(window).historical_log_returns) == 1

    two_point_index = history.index[:2]
    for mode_cfg in (
        {"enabled": True, "mode": "rolling"},
        {"enabled": True, "mode": "count_spaced", "n_folds": 1},
    ):
        gen = FoldGenerator.from_config(mode_cfg)
        assert gen is not None
        with pytest.raises(
            ValueError,
            match=r"Insufficient calibration history.*requires at least 3.*2023-01-31 to 2023-02-28",
        ):
            gen.generate(two_point_index)


def test_rolling_default_start_respects_calibration_lookback_years() -> None:
    index = pd.date_range("2020-01-31", periods=24, freq="ME")
    generator = FoldGenerator(
        mode="rolling",
        step_months=6,
        calibration_lookback_years=1.0,
    )

    folds = generator.generate(index)

    assert [fold.forecast_start for fold in folds] == [
        pd.Timestamp("2021-01-31"),
        pd.Timestamp("2021-07-31"),
    ]
    assert folds[0].calibration_start == pd.Timestamp("2020-01-31")
    assert folds[0].calibration_end == pd.Timestamp("2020-12-31")
