from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import trend_analysis.monte_carlo.runner as runner_module
from trend_analysis.api import RunResult
from trend_analysis.monte_carlo.costs import CostProcess
from trend_analysis.monte_carlo.registry import load_scenario
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


def test_cost_process_canonical_regime_stochastic_trade_cost_bps_dist() -> None:
    config = {
        "kind": "regime_stochastic",
        "default_regime": "calm",
        "calm": {"trade_cost_bps": {"dist": "fixed", "value": 6.0}},
        "stress": {
            "trade_cost_bps": {"dist": "fixed", "value": 12.0},
            "slippage_multiplier": 1.25,
        },
    }
    process = CostProcess.from_config(config)
    assert process is not None

    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    regimes = pd.Series(["calm", "stress", "unknown"], index=idx, dtype="string")
    turnover = pd.Series([0.10, 0.20, 0.30], index=idx)
    out = process.sample(
        regimes=regimes, turnover=turnover, index=None, rng=np.random.default_rng(1)
    )

    assert out.cost_bps.tolist() == [6.0, 12.0, 6.0]
    assert out.slippage_multiplier.tolist() == [1.0, 1.25, 1.0]


def test_cost_process_accepts_legacy_numeric_shorthand() -> None:
    process = CostProcess.from_config({"regimes": {"calm": 5, "stress": 11.5}})
    assert process is not None

    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    out = process.sample(
        regimes=pd.Series(["calm", "stress"], index=idx, dtype="string"),
        turnover=pd.Series([0.1, 0.2], index=idx),
        index=None,
        rng=np.random.default_rng(2),
    )
    assert out.cost_bps.tolist() == [5.0, 11.5]


def test_runner_integration_records_costs(monkeypatch: Any) -> None:
    scenario = load_scenario("cost_regime_example")
    scenario.monte_carlo.n_paths = 10
    scenario.strategy_set = {"curated": [StrategyVariant(name="StrategyA")]}
    assert scenario.costs is not None

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
    transaction_costs = costs["transaction_costs"]
    assert isinstance(transaction_costs, pd.Series)
    assert len(transaction_costs) == 3
    cost_bps = pd.Series(costs["cost_bps"], index=transaction_costs.index, dtype=float)
    assert (cost_bps > 0.0).all()
    assert float(cost_bps.iloc[1]) > float(max(cost_bps.iloc[0], cost_bps.iloc[2]))
    assert costs["slippage_multiplier"].tolist() == [1.0, 1.8, 1.0]
    expected_costs = (
        pd.Series([0.1, 0.2, 0.3], index=transaction_costs.index, name="turnover")
        * (cost_bps / 10000.0)
        * pd.Series([1.0, 1.8, 1.0], index=transaction_costs.index)
    )
    expected_costs.name = transaction_costs.name
    pd.testing.assert_series_equal(transaction_costs, expected_costs)
    total_cost_drag = pd.to_numeric(results.results_frame["total_cost_drag"], errors="coerce")
    assert total_cost_drag.notna().all()
    assert bool((total_cost_drag.abs() > 0.0).all())


def test_cost_regime_scenario_loads_from_registry() -> None:
    scenario = load_scenario("cost_regime_example")
    assert isinstance(scenario, MonteCarloScenario)
    assert scenario.name == "cost_regime_example"
    assert scenario.costs is not None


def test_runner_injects_cash_series_when_risk_free_column_absent(monkeypatch: Any) -> None:
    scenario = load_scenario("cost_regime_example")
    scenario.monte_carlo.n_paths = 10
    scenario.strategy_set = {"curated": [StrategyVariant(name="StrategyA")]}
    scenario.costs = None
    captured_returns: list[pd.DataFrame] = []

    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )

    def fake_run_simulation(config: Any, returns: pd.DataFrame) -> RunResult:
        captured_returns.append(returns.copy())
        metrics = pd.DataFrame({"annual_return": [0.1]}, index=["user_weight"])
        return RunResult(metrics=metrics, details={}, seed=0, environment={})

    monkeypatch.setattr(runner_module, "run_simulation", fake_run_simulation)

    _ = runner.run(jobs=1)

    assert captured_returns
    for returns in captured_returns:
        assert "CASH" in returns.columns
        cash_series = returns["CASH"]
        assert pd.api.types.is_numeric_dtype(cash_series)
        assert np.isfinite(cash_series.to_numpy(dtype=float, copy=False)).all()


def test_inject_cash_warns_when_fallback_disabled(monkeypatch: Any, caplog: Any) -> None:
    """When allow_risk_free_fallback is False the runner should log a warning
    explaining why CASH injection was skipped."""
    scenario = load_scenario("cost_regime_example")
    scenario.monte_carlo.n_paths = 10
    scenario.strategy_set = {"curated": [StrategyVariant(name="StrategyA")]}
    scenario.costs = None
    # Remove the scenario-level data override so the base_config value wins.
    if isinstance(scenario.raw, dict):
        scenario.raw.pop("data", None)

    base_cfg = _base_config()
    base_cfg["data"]["allow_risk_free_fallback"] = False

    runner = MonteCarloRunner(
        scenario,
        base_config=base_cfg,
        price_history=_price_history(),
    )

    captured_returns: list[pd.DataFrame] = []

    def fake_run_simulation(config: Any, returns: pd.DataFrame) -> RunResult:
        captured_returns.append(returns.copy())
        metrics = pd.DataFrame({"annual_return": [0.1]}, index=["user_weight"])
        return RunResult(metrics=metrics, details={}, seed=0, environment={})

    monkeypatch.setattr(runner_module, "run_simulation", fake_run_simulation)

    with caplog.at_level(logging.WARNING, logger="trend_analysis.monte_carlo"):
        _ = runner.run(jobs=1)

    assert captured_returns
    for returns in captured_returns:
        assert (
            "CASH" not in returns.columns
        ), "CASH should not be injected when allow_risk_free_fallback is False"

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    fallback_warnings = [m for m in warning_messages if "allow_risk_free_fallback" in m]
    assert fallback_warnings, (
        f"Expected a warning about allow_risk_free_fallback, got: {warning_messages}"
    )
    # P2 fix: The warning should be emitted only once, not per path.
    assert len(fallback_warnings) == 1, (
        f"Expected exactly 1 fallback warning but got {len(fallback_warnings)}"
    )


def test_scenario_data_override_merges_into_base_config(
    monkeypatch: Any,
) -> None:
    """Scenario-level data overrides (e.g. allow_risk_free_fallback) should be
    merged into the runner's base config even when the base config file sets
    a different value."""
    scenario = load_scenario("cost_regime_example")
    scenario.monte_carlo.n_paths = 10
    scenario.strategy_set = {"curated": [StrategyVariant(name="StrategyA")]}
    scenario.costs = None

    # Do NOT pass base_config — let the runner load it from the scenario's
    # base_config path.  The scenario YAML sets data.allow_risk_free_fallback:
    # true which should override defaults.yml's false.
    runner = MonteCarloRunner(
        scenario,
        price_history=_price_history(),
    )

    # The merged base config should now have the override applied.
    assert runner.base_config["data"]["allow_risk_free_fallback"] is True, (
        "Scenario-level data.allow_risk_free_fallback should override defaults"
    )
