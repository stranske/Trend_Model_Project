"""Regression tests for per-path analysis-window alignment in the MC runner.

These guard the demo Monte Carlo fixes: a simulated path is a short forward
projection, so the runner must evaluate it against windows derived from the
path itself rather than the historical windows baked into ``base_config``.
Without this, every in/out window is empty and the demo produces no NAV paths,
an empty summary, and a Sharpe "distribution" that is really the path index.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from trend_analysis.api import RunResult
from trend_analysis.monte_carlo.runner import MonteCarloRunner, _PathContext
from trend_analysis.monte_carlo.scenario import MonteCarloScenario


def _demo_like_config() -> dict[str, Any]:
    """A defaults.yml-shaped config that degenerates without window alignment.

    It carries a historical ``multi_period`` window, a fixed historical
    ``sample_split`` date, and 63-period rolling lookbacks for both
    ``vol_adjust`` and the trend ``signals`` -- the exact combination that
    leaves a short simulated path 100% cash with empty NAV paths.
    """
    return {
        "version": "1",
        "data": {
            "date_column": "Date",
            "frequency": "M",
            "allow_risk_free_fallback": True,
        },
        "preprocessing": {},
        "vol_adjust": {
            "enabled": True,
            "target_vol": 0.10,
            "floor_vol": 0.015,
            "window": {"length": 63, "decay": "ewma", "lambda": 0.94},
        },
        "sample_split": {"method": "date", "date": "2017-12-31"},
        "portfolio": {"selection_mode": "all", "weighting_scheme": "equal"},
        "benchmarks": {},
        "metrics": {"registry": ["annual_return", "volatility", "sharpe_ratio"]},
        "regime": {},
        "export": {},
        "run": {"monthly_cost": 0.0},
        "signals": {"kind": "tsmom", "window": 63, "lag": 1, "vol_adjust": False},
        "multi_period": {
            "frequency": "A",
            "in_sample_len": 3,
            "out_sample_len": 1,
            "start": "1990-01",
            "end": "2024-12",
        },
    }


def _price_history(n: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2018-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(7)
    rets = pd.DataFrame(
        rng.normal(0.01, 0.03, size=(n, 3)),
        index=dates,
        columns=["AssetA", "AssetB", "AssetC"],
    )
    return (1.0 + rets).cumprod() * 100.0


def _scenario(horizon_years: float = 2.0, n_paths: int = 4) -> MonteCarloScenario:
    return MonteCarloScenario(
        name="window_alignment_test",
        base_config="base.yml",
        monte_carlo={
            "mode": "two_layer",
            "n_paths": n_paths,
            "horizon_years": horizon_years,
            "frequency": "M",
            "seed": 11,
        },
        return_model={"kind": "stationary_bootstrap", "params": {"mean_block_len": 2}},
        enable_fold_runs=False,
    )


def _context(dates: pd.DatetimeIndex) -> _PathContext:
    returns = pd.DataFrame({"Date": dates, "AssetA": 0.01, "AssetB": 0.02})
    return _PathContext(
        path_id=0,
        prices=pd.DataFrame({"AssetA": [1.0], "AssetB": [1.0]}),
        returns=returns,
        score_frame=pd.DataFrame(),
        path_hash="hash",
        seed=1,
    )


def _runner(horizon_years: float = 2.0, n_paths: int = 4) -> MonteCarloRunner:
    return MonteCarloRunner(
        _scenario(horizon_years=horizon_years, n_paths=n_paths),
        base_config=_demo_like_config(),
        price_history=_price_history(),
    )


def test_align_path_windows_neutralises_historical_windows() -> None:
    runner = _runner(horizon_years=2.0)  # n_periods=24 -> max_window=12
    merged = dict(_demo_like_config())
    dates = pd.date_range("2025-01-31", periods=24, freq="ME")
    runner._align_path_windows(merged, _context(dates))

    # Single-period data-relative split replaces the fixed historical date.
    assert merged["sample_split"] == {"method": "ratio", "ratio": 0.7}
    # Annual 3-in/1-out cannot fit a 2-year path -> multi_period dropped.
    assert "multi_period" not in merged
    # Over-long rolling lookbacks capped at half the horizon.
    assert merged["vol_adjust"]["window"]["length"] == 12
    assert merged["signals"]["window"] == 12


def test_align_multi_period_keeps_fitting_schedule_rebased_to_path() -> None:
    runner = _runner(horizon_years=0.5)  # n_periods=6
    merged = {
        "multi_period": {
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 2,
            "start": "1990-01",
            "end": "1990-06",
        }
    }
    dates = pd.date_range("2025-01-31", periods=6, freq="ME")
    runner._align_multi_period(merged, _context(dates))

    assert "multi_period" in merged
    # Re-based onto the simulated path's span, not the original historical dates.
    assert merged["multi_period"]["start"] == "2025-01"
    assert merged["multi_period"]["end"] == "2025-06"
    assert merged["multi_period"]["in_sample_len"] == 2


def test_align_path_windows_caps_to_retained_multi_period_in_sample_span() -> None:
    runner = _runner(horizon_years=0.5)  # n_periods=6, half-horizon cap would be 3
    merged = {
        "sample_split": {"method": "date", "date": "2017-12-31"},
        "vol_adjust": {"enabled": True, "window": {"length": 63}},
        "signals": {"kind": "tsmom", "window": 63, "min_periods": 63},
        "multi_period": {
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 2,
            "start": "1990-01",
            "end": "1990-06",
        },
    }
    dates = pd.date_range("2025-01-31", periods=6, freq="ME")
    runner._align_path_windows(merged, _context(dates))

    assert merged["multi_period"]["start"] == "2025-01"
    assert merged["vol_adjust"]["window"]["length"] == 2
    assert merged["signals"]["window"] == 2
    assert merged["signals"]["min_periods"] == 2


def test_align_path_windows_clamps_fallback_cap_to_path_length() -> None:
    runner = _runner(horizon_years=2.0)
    merged = {
        "sample_split": {"method": "date", "date": "2017-12-31"},
        "vol_adjust": {"enabled": True, "window": {"length": 63}},
        "signals": {"kind": "tsmom", "window": 63, "min_periods": 63},
    }
    dates = pd.date_range("2025-01-31", periods=1, freq="ME")
    runner._align_path_windows(merged, _context(dates))

    assert merged["vol_adjust"]["window"]["length"] == 1
    assert merged["signals"]["window"] == 1
    assert merged["signals"]["min_periods"] == 1


def test_path_context_score_frame_uses_path_aligned_windows(
    monkeypatch: Any,
) -> None:
    runner = _runner(horizon_years=0.5)
    dates = pd.date_range("2025-01-31", periods=6, freq="ME")
    returns = pd.DataFrame(
        {
            "AssetA": [0.01, 0.02, -0.01, 0.03, 0.01, 0.02],
            "AssetB": [0.02, -0.01, 0.01, 0.02, 0.03, -0.01],
        },
        index=dates,
    )

    class _PathResult:
        prices = (1.0 + returns).cumprod() * 100.0
        log_returns = np.log1p(returns)

    captured: dict[str, Mapping[str, Any] | None] = {}

    def fake_score_frame(
        _returns: pd.DataFrame,
        config_data: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        captured["config"] = config_data
        return pd.DataFrame({"sharpe_ratio": [1.0]}, index=["AssetA"])

    monkeypatch.setattr(runner, "_compute_score_frame", fake_score_frame)

    context = runner._generate_path_context(
        path_id=0,
        seed=1,
        model=None,
        n_periods=6,
        path_result=_PathResult(),
        fold_id=None,
        fold_label=None,
    )

    score_config = captured["config"]
    assert isinstance(score_config, Mapping)
    assert score_config["sample_split"] == {"method": "ratio", "ratio": 0.7}
    assert "multi_period" not in score_config
    assert not context.score_frame.empty
    assert "sharpe_ratio" in context.score_frame.columns


def test_align_multi_period_dropped_without_context() -> None:
    runner = _runner()
    merged = {"multi_period": {"frequency": "A", "in_sample_len": 3, "out_sample_len": 1}}
    runner._align_multi_period(merged, None)
    assert "multi_period" not in merged


def test_fit_windows_leave_fitting_lookbacks_untouched() -> None:
    runner = _runner(horizon_years=2.0)  # max_window=12
    merged = {
        "vol_adjust": {"enabled": True, "window": {"length": 6}},
        "signals": {"kind": "tsmom", "window": 4},
    }
    runner._fit_vol_adjust_window(merged, 12)
    runner._fit_signal_window(merged, 12)
    assert merged["vol_adjust"]["window"]["length"] == 6
    assert merged["signals"]["window"] == 4


def test_extract_portfolio_metrics_prefers_out_user_stats() -> None:
    runner = _runner()

    class _Stats:
        def __init__(self, sharpe: float) -> None:
            self.cagr = 0.1
            self.vol = 0.05
            self.sharpe = sharpe
            self.is_avg_corr = None  # non-numeric fields are skipped

    per_fund = pd.DataFrame({"sharpe": [0.1, 0.2]}, index=["FundA", "FundB"])
    run_result = RunResult(
        metrics=per_fund,
        details={"out_user_stats": _Stats(1.5), "out_ew_stats": _Stats(1.2)},
        seed=0,
        environment={},
    )
    metrics, source = runner._extract_portfolio_metrics(run_result)
    assert source == "out_user_stats"
    assert metrics["sharpe"] == 1.5
    assert "is_avg_corr" not in metrics


def test_extract_portfolio_metrics_accepts_series_stats() -> None:
    runner = _runner()
    per_fund = pd.DataFrame({"sharpe": [0.1, 0.2]}, index=["FundA", "FundB"])
    run_result = RunResult(
        metrics=per_fund,
        details={
            "out_user_stats": pd.Series(
                {"sharpe": 1.5, "bad_inf": np.inf, "is_avg_corr": True}
            )
        },
        seed=0,
        environment={},
    )

    metrics, source = runner._extract_portfolio_metrics(run_result)

    assert source == "out_user_stats"
    assert metrics == {"sharpe": 1.5}


def test_extract_portfolio_metrics_falls_back_to_metrics_frame() -> None:
    runner = _runner()
    per_scheme = pd.DataFrame({"sharpe": [0.7]}, index=["equal_weight"])
    run_result = RunResult(metrics=per_scheme, details={}, seed=0, environment={})
    metrics, source = runner._extract_portfolio_metrics(run_result)
    assert source == "equal_weight"
    assert metrics["sharpe"] == 0.7


def test_runner_produces_real_sharpe_and_nav_paths_for_demo_like_config() -> None:
    """End-to-end guard for both reported symptoms on a demo-like config."""
    runner = _runner(horizon_years=2.0, n_paths=6)
    results = runner.run(jobs=1)

    rf = results.results_frame
    assert not rf.empty
    # Symptom 2: a real Sharpe column exists with finite values (not path ids).
    assert "sharpe" in rf.columns
    assert np.isfinite(rf["sharpe"].to_numpy()).any()
    # Metrics come from the portfolio, not an arbitrary fund row.
    assert set(rf["metric_source"].dropna()) <= {"out_user_stats", "out_ew_stats"}
    assert not results.summary_frame.empty

    # Symptom 1: NAV paths are populated over the simulated horizon and vary.
    nav_paths = results.metadata.get("nav_paths")
    assert isinstance(nav_paths, pd.DataFrame)
    assert not nav_paths.empty
    assert isinstance(nav_paths.index, pd.DatetimeIndex)
    final_row = nav_paths.iloc[-1].to_numpy()
    assert np.nanstd(final_row) > 0.0
