from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trend_analysis.monte_carlo.results import build_results_frame
from trend_analysis.monte_carlo.runner import MonteCarloRunner
from trend_analysis.monte_carlo.scenario import MonteCarloScenario
from trend_analysis.monte_carlo.strategy import StrategyVariant


def _price_history() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=48, freq="M")
    base = np.linspace(100.0, 140.0, len(dates))
    prices = pd.DataFrame(
        {
            "AssetA": base,
            "AssetB": base * 1.05,
        },
        index=dates,
    )
    return prices


def _base_config() -> dict[str, Any]:
    return {
        "version": "0.1.0",
        "data": {
            "date_column": "Date",
            "frequency": "M",
            "allow_risk_free_fallback": True,
        },
        "preprocessing": {},
        "vol_adjust": {"enabled": False, "target_vol": 0.1, "window": {"length": 3}},
        "sample_split": {"method": "ratio", "ratio": 0.6},
        "portfolio": {"selection_mode": "all", "weighting_scheme": "equal"},
        "benchmarks": {},
        "metrics": {"registry": ["annual_return", "volatility", "sharpe_ratio"]},
        "regime": {},
        "export": {},
        "run": {"monthly_cost": 0.0},
    }


def _scenario(mode: str) -> MonteCarloScenario:
    strategies = [
        StrategyVariant(name="StrategyA"),
        StrategyVariant(name="StrategyB"),
    ]
    return MonteCarloScenario(
        name="mc_test",
        base_config="config/defaults.yml",
        monte_carlo={
            "mode": mode,
            "n_paths": 5,
            "horizon_years": 1.0,
            "frequency": "M",
            "seed": 123,
            "jobs": 1,
        },
        strategy_set={"curated": strategies},
        return_model={"kind": "stationary_bootstrap", "params": {"block_size": 3}},
    )


def _sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["path_id", "strategy"]).reset_index(drop=True)


def test_runner_two_layer_small_scenario() -> None:
    scenario = _scenario("two_layer")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )
    updates: list[dict[str, Any]] = []

    def _callback(payload: dict[str, Any]) -> None:
        updates.append(payload)

    results = runner.run(progress_callback=_callback, jobs=1)

    assert len(results.results_frame) == 10
    assert results.summary_frame.shape[0] == 2
    assert len(updates) == 5
    assert updates[-1]["completed"] == 5
    assert updates[-1]["total"] == 5

    path_hashes = results.results_frame.groupby("path_id")["path_hash"].nunique()
    assert path_hashes.max() == 1


def test_runner_mixture_samples_strategy_per_path() -> None:
    scenario = _scenario("mixture")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )

    results = runner.run(jobs=1)

    assert len(results.results_frame) == 5
    assert results.results_frame["strategy"].nunique() > 1


def test_run_deterministic_with_fixed_seed() -> None:
    scenario = _scenario("two_layer")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )

    first = _sorted_frame(runner.run(jobs=1).results_frame)
    second = _sorted_frame(runner.run(jobs=1).results_frame)
    pd.testing.assert_frame_equal(first, second)


def test_run_two_layer_deterministic() -> None:
    scenario = _scenario("two_layer")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )
    model = runner._build_price_model()
    n_periods = runner._compute_n_periods()
    strategies = runner._resolve_strategies()
    path_seeds, _ = runner._build_seeds()

    evals1, _ = runner._run_two_layer(
        model=model,
        n_periods=n_periods,
        strategies=strategies,
        path_seeds=path_seeds,
        progress_callback=None,
        jobs=1,
    )
    evals2, _ = runner._run_two_layer(
        model=model,
        n_periods=n_periods,
        strategies=strategies,
        path_seeds=path_seeds,
        progress_callback=None,
        jobs=1,
    )

    frame1 = _sorted_frame(build_results_frame(evals1))
    frame2 = _sorted_frame(build_results_frame(evals2))
    pd.testing.assert_frame_equal(frame1, frame2)


def test_run_mixture_deterministic() -> None:
    scenario = _scenario("mixture")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )
    model = runner._build_price_model()
    n_periods = runner._compute_n_periods()
    strategies = runner._resolve_strategies()
    path_seeds, strategy_seeds = runner._build_seeds()

    evals1, _ = runner._run_mixture(
        model=model,
        n_periods=n_periods,
        strategies=strategies,
        path_seeds=path_seeds,
        strategy_seeds=strategy_seeds,
        progress_callback=None,
        jobs=1,
    )
    evals2, _ = runner._run_mixture(
        model=model,
        n_periods=n_periods,
        strategies=strategies,
        path_seeds=path_seeds,
        strategy_seeds=strategy_seeds,
        progress_callback=None,
        jobs=1,
    )

    frame1 = _sorted_frame(build_results_frame(evals1))
    frame2 = _sorted_frame(build_results_frame(evals2))
    pd.testing.assert_frame_equal(frame1, frame2)
