from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import trend_analysis.monte_carlo.runner as runner_module
from trend_analysis.api import RunResult
from trend_analysis.monte_carlo.costs import CostProcess
from trend_analysis.monte_carlo.runner import MonteCarloRunner
from trend_analysis.monte_carlo.scenario import MonteCarloScenario
from trend_analysis.monte_carlo.strategy import StrategyVariant


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


def _price_history() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    base = np.linspace(100.0, 120.0, len(dates))
    return pd.DataFrame({"AssetA": base, "AssetB": base * 1.05}, index=dates)


def test_cost_process_fixed_distribution_applies_slippage() -> None:
    config = {
        "default_regime": "calm",
        "regimes": {
            "calm": {
                "distribution": {"kind": "fixed", "value": 5.0},
                "slippage_multiplier": 1.5,
            }
        },
    }
    process = CostProcess.from_config(config)
    assert process is not None

    dates = pd.date_range("2021-01-31", periods=3, freq="ME")
    regimes = pd.Series(["calm", "unknown", "calm"], index=dates, dtype="string")
    turnover = pd.Series([0.1, 0.2, 0.3], index=dates)
    rng = np.random.default_rng(7)

    output = process.sample(regimes=regimes, turnover=turnover, index=None, rng=rng)

    assert output.cost_bps.tolist() == [5.0, 5.0, 5.0]
    expected = turnover * (5.0 / 10000.0) * 1.5
    pd.testing.assert_series_equal(output.transaction_costs, expected)


def test_cost_process_normal_distribution_reasonable_mean() -> None:
    config = {
        "default_regime": "base",
        "regimes": {"base": {"distribution": {"kind": "normal", "mean": 8.0, "std": 0.5}}},
    }
    process = CostProcess.from_config(config)
    assert process is not None

    rng = np.random.default_rng(123)
    regimes = pd.Series(["base"] * 5000)
    output = process.sample(regimes=regimes, turnover=0.1, index=None, rng=rng)

    mean = float(output.cost_bps.mean())
    assert 7.7 < mean < 8.3


def test_cost_process_lognormal_stress_higher_mean_and_variance() -> None:
    config = {
        "default_regime": "calm",
        "regimes": {
            "calm": {"distribution": {"kind": "lognormal", "mean": 1.0, "sigma": 0.1}},
            "stress": {"distribution": {"kind": "lognormal", "mean": 1.3, "sigma": 0.4}},
        },
    }
    process = CostProcess.from_config(config)
    assert process is not None

    rng = np.random.default_rng(42)
    regimes = pd.Series(["calm"] * 5000 + ["stress"] * 5000)
    output = process.sample(regimes=regimes, turnover=0.1, index=None, rng=rng)

    calm = output.cost_bps[regimes == "calm"]
    stress = output.cost_bps[regimes == "stress"]

    assert calm.mean() > 0
    assert stress.mean() > calm.mean()
    assert stress.var() > calm.var()


def test_runner_integration_records_costs(monkeypatch: Any) -> None:
    costs_cfg = {
        "default_regime": "calm",
        "regimes": {
            "calm": {"distribution": {"kind": "fixed", "value": 4.0}},
            "stress": {
                "distribution": {"kind": "fixed", "value": 8.0},
                "slippage_multiplier": 1.2,
            },
        },
    }
    scenario = MonteCarloScenario(
        name="mc_costs",
        base_config="config/defaults.yml",
        monte_carlo={
            "mode": "two_layer",
            "n_paths": 1,
            "horizon_years": 1.0,
            "frequency": "M",
            "seed": 11,
            "jobs": 1,
        },
        strategy_set={"curated": [StrategyVariant(name="StrategyA")]},
        return_model={"kind": "stationary_bootstrap", "params": {"block_size": 3}},
        costs=costs_cfg,
    )

    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )

    def fake_run_simulation(config: Any, returns: pd.DataFrame) -> RunResult:
        metrics = pd.DataFrame({"annual_return": [0.1]}, index=["user_weight"])
        out_index = pd.date_range("2021-01-31", periods=3, freq="ME")
        details = {
            "out_sample_scaled": pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=out_index),
            "regime_labels_out": pd.Series(
                ["calm", "stress", "calm"], index=out_index, dtype="string"
            ),
        }
        turnover = pd.Series([0.1, 0.2, 0.3], index=out_index, name="turnover")
        return RunResult(
            metrics=metrics,
            details=details,
            seed=0,
            environment={},
            turnover=turnover,
        )

    monkeypatch.setattr(runner_module, "run_simulation", fake_run_simulation)

    results = runner.run(jobs=1)

    assert "total_cost_drag" in results.results_frame.columns
    assert results.evaluations
    diag = results.evaluations[0].diagnostic
    assert diag is not None and "costs" in diag
    costs = diag["costs"]
    assert isinstance(costs["transaction_costs"], pd.Series)
    assert len(costs["transaction_costs"]) == 3
