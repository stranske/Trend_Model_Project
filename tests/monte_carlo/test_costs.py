from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import trend_analysis.monte_carlo.runner as runner_module
from trend_analysis.api import RunResult
from trend_analysis.monte_carlo.costs import CostProcess
from trend_analysis.monte_carlo.registry import list_scenarios, load_scenario
from trend_analysis.monte_carlo.runner import MonteCarloRunner
from trend_analysis.monte_carlo.scenario import MonteCarloScenario
from trend_analysis.monte_carlo.strategy import StrategyVariant
from trend_analysis.risk import periods_per_year_from_code


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
        "portfolio": {
            "selection_mode": "all",
            "weighting": {"name": "equal"},
            "cost_model": {"per_trade_bps": 0.0, "half_spread_bps": 0.0},
        },
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


def _cost_fixture_frame() -> pd.DataFrame:
    fixture_path = Path("tests/fixtures/costs/cost_regime_fixture.csv")
    frame = pd.read_csv(fixture_path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")


def test_cost_process_fixed_distribution_applies_slippage() -> None:
    config = {
        "kind": "regime_stochastic",
        "default_regime": "calm",
        "calm": {
            "trade_cost_bps": {"kind": "fixed", "value": 5.0},
            "slippage_multiplier": 1.5,
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
        "kind": "regime_stochastic",
        "default_regime": "base",
        "base": {"trade_cost_bps": {"kind": "normal", "mean": 8.0, "std": 0.5}},
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
        "kind": "regime_stochastic",
        "default_regime": "calm",
        "calm": {"trade_cost_bps": {"kind": "lognormal", "mean": 1.0, "sigma": 0.1}},
        "stress": {"trade_cost_bps": {"kind": "lognormal", "mean": 1.3, "sigma": 0.4}},
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


def test_cost_process_lognormal_mean_is_arithmetic_bps_target() -> None:
    config = {
        "kind": "regime_stochastic",
        "default_regime": "base",
        "base": {"trade_cost_bps": {"kind": "lognormal", "mean": 20.0, "sigma": 0.35}},
    }
    process = CostProcess.from_config(config)
    assert process is not None

    rng = np.random.default_rng(123)
    regimes = pd.Series(["base"] * 20000)
    output = process.sample(regimes=regimes, turnover=0.1, index=None, rng=rng)

    mean = float(output.cost_bps.mean())
    assert 19.5 < mean < 20.5
    assert float(output.cost_bps.quantile(0.99)) < 50.0


def test_cost_process_lognormal_rejects_non_positive_arithmetic_mean() -> None:
    with np.testing.assert_raises_regex(ValueError, "lognormal mean must be > 0"):
        CostProcess.from_config(
            {
                "kind": "regime_stochastic",
                "base": {
                    "trade_cost_bps": {
                        "kind": "lognormal",
                        "mean": 0.0,
                        "sigma": 0.35,
                    }
                },
            }
        )


def test_cost_process_canonical_regime_stochastic_trade_cost_bps_kind() -> None:
    config = {
        "kind": "regime_stochastic",
        "default_regime": "calm",
        "calm": {"trade_cost_bps": {"kind": "fixed", "value": 6.0}},
        "stress": {
            "trade_cost_bps": {"kind": "fixed", "value": 12.0},
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


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"calm": {"trade_cost_bps": {"kind": "fixed", "value": 5}}}, "costs.kind"),
        ({"kind": "regime_stochastic", "regimes": {"calm": 5}}, "directly under costs"),
        ({"kind": "regime_stochastic", "calm": 5}, "regime mapping"),
        (
            {
                "kind": "regime_stochastic",
                "calm": {"distribution": {"kind": "fixed", "value": 5}},
            },
            "unsupported key",
        ),
        (
            {
                "kind": "regime_stochastic",
                "calm": {"trade_cost_bps": {"dist": "fixed", "value": 5}},
            },
            "use kind",
        ),
        (
            {
                "kind": "regime_stochastic",
                "calm": {"trade_cost_bps": 5},
            },
            "mapping with a kind",
        ),
        (
            {
                "kind": "regime_stochastic",
                "calm": {"trade_cost_bps": {"kind": "fixed", "bps": 5}},
            },
            "unsupported key",
        ),
        (
            {
                "kind": "regime_stochastic",
                "calm": {"trade_cost_bps": {"kind": "normal", "sigma": 0.5}},
            },
            "unsupported key",
        ),
        (
            {
                "kind": "regime_stochastic",
                "calm": {
                    "trade_cost_bps": {
                        "kind": "lognormal",
                        "mean": 5,
                        "sigma": 0.2,
                        "mu": 1.0,
                    }
                },
            },
            "unsupported key",
        ),
    ],
)
def test_cost_process_rejects_legacy_cost_shapes(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CostProcess.from_config(config)


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
    assert costs["slippage_multiplier"].tolist() == [1.0, 1.8, 1.0]
    cost_bps = pd.Series(costs["cost_bps"], index=transaction_costs.index, dtype=float)
    assert (cost_bps > 0.0).all()
    assert float(cost_bps.iloc[1]) > float(max(cost_bps.iloc[0], cost_bps.iloc[2]))
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
    # Repository-supported invocation for this scenario coverage:
    # python -m pytest tests/monte_carlo/test_costs.py -q --no-cov
    scenario = load_scenario("cost_regime_example")
    assert isinstance(scenario, MonteCarloScenario)
    assert scenario.name == "cost_regime_example"
    assert scenario.costs is not None
    assert scenario.costs["kind"] == "regime_stochastic"
    assert scenario.costs["default_regime"] == "calm"
    assert "regimes" not in scenario.costs
    for regime in ("calm", "stress"):
        assert isinstance(scenario.costs[regime], dict)
        assert "trade_cost_bps" in scenario.costs[regime]


def test_cost_regime_scenario_is_filterable_by_example_and_cost_tags() -> None:
    for tag in ("example", "cost"):
        names = {entry.name for entry in list_scenarios(tags=[tag])}
        assert "cost_regime_example" in names


def test_cost_regime_scenario_documents_exact_dry_run_command() -> None:
    scenario_file = Path("config/scenarios/monte_carlo/cost_regime_example.yml")
    scenario_text = scenario_file.read_text(encoding="utf-8")
    assert (
        "# - Run with: trend mc run --scenario cost_regime_example --dry-run --n-paths 10"
        in scenario_text
    )
    assert (
        "# - Validate with: python -m pytest tests/monte_carlo/test_costs.py "
        "tests/monte_carlo/test_registry.py -q --no-cov" in scenario_text
    )


def test_cost_regime_example_fixture_produces_exact_deterministic_outputs() -> None:
    scenario = load_scenario("cost_regime_example")
    assert scenario.costs is not None

    fixture = _cost_fixture_frame()
    regimes = fixture["regime"].astype("string")
    turnover = fixture["turnover"].astype(float)
    process = CostProcess.from_config(scenario.costs)
    assert process is not None
    sampled = process.sample(
        regimes=regimes,
        turnover=turnover,
        index=None,
        rng=np.random.default_rng(2026),
    )

    assert sampled.slippage_multiplier.tolist() == [1.0, 1.8, 1.0]
    np.testing.assert_allclose(
        sampled.cost_bps.to_numpy(dtype=float, copy=False),
        np.array([4.1820996013544685, 9.686837945111462, 5.142565961196969]),
        rtol=0.0,
        atol=1e-12,
    )
    assert float(sampled.cost_bps.max()) < 100.0
    np.testing.assert_allclose(
        sampled.transaction_costs.to_numpy(dtype=float, copy=False),
        np.array([4.182099601354469e-05, 0.0003487261660240126, 0.00015427697883590907]),
        rtol=0.0,
        atol=1e-12,
    )


def test_runner_injects_cash_series_when_override_enabled(monkeypatch: Any) -> None:
    scenario = load_scenario("cost_regime_example")
    scenario.monte_carlo.n_paths = 10
    scenario.strategy_set = {"curated": [StrategyVariant(name="StrategyA")]}
    scenario.costs = None
    captured_returns: list[pd.DataFrame] = []

    base_cfg = _base_config()
    base_cfg["metrics"]["rf_override_enabled"] = True

    runner = MonteCarloRunner(
        scenario,
        base_config=base_cfg,
        price_history=_price_history(),
    )

    def fake_run_simulation(config: Any, returns: pd.DataFrame) -> RunResult:
        captured_returns.append(returns.copy())
        metrics = pd.DataFrame({"annual_return": [0.1]}, index=["user_weight"])
        return RunResult(metrics=metrics, details={}, seed=0, environment={})

    monkeypatch.setattr(runner_module, "run_simulation", fake_run_simulation)

    _ = runner.run(jobs=1)

    assert captured_returns
    frequency = str(runner.base_config["data"].get("frequency", "M"))
    expected_periodic_rf = (1.0 + 0.03) ** (
        1.0 / float(periods_per_year_from_code(frequency))
    ) - 1.0
    for returns in captured_returns:
        assert "CASH" in returns.columns
        cash_series = returns["CASH"]
        assert pd.api.types.is_numeric_dtype(cash_series)
        assert np.isfinite(cash_series.to_numpy(dtype=float, copy=False)).all()
        expected = pd.Series(expected_periodic_rf, index=returns.index, name="CASH", dtype=float)
        pd.testing.assert_series_equal(cash_series, expected)


def test_inject_cash_warns_when_override_disabled(monkeypatch: Any, caplog: Any) -> None:
    """When rf_override_enabled is False the runner should log a skip warning."""
    scenario = load_scenario("cost_regime_example")
    scenario.monte_carlo.n_paths = 10
    scenario.strategy_set = {"curated": [StrategyVariant(name="StrategyA")]}
    scenario.costs = None
    # Remove scenario-level overrides so the base_config value wins.
    if isinstance(scenario.raw, dict):
        scenario.raw.pop("data", None)
        scenario.raw.pop("metrics", None)

    base_cfg = _base_config()
    base_cfg["metrics"]["rf_override_enabled"] = False

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
        ), "CASH should not be injected when metrics.rf_override_enabled is False"

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    fallback_warnings = [m for m in warning_messages if "metrics.rf_override_enabled" in m]
    assert (
        fallback_warnings
    ), f"Expected a warning about metrics.rf_override_enabled, got: {warning_messages}"
    # P2 fix: The warning should be emitted only once, not per path.
    assert (
        len(fallback_warnings) == 1
    ), f"Expected exactly 1 fallback warning but got {len(fallback_warnings)}"


def test_scenario_data_and_metrics_overrides_merge_into_base_config() -> None:
    """Scenario-level risk-free overrides should reach the runner config."""
    scenario = load_scenario("cost_regime_example")
    scenario.monte_carlo.n_paths = 10
    scenario.strategy_set = {"curated": [StrategyVariant(name="StrategyA")]}
    scenario.costs = None

    # Do NOT pass base_config — let the runner load it from the scenario's
    # base_config path. The scenario YAML pins both the fallback source and the
    # explicit CASH-injection gate.
    runner = MonteCarloRunner(
        scenario,
        price_history=_price_history(),
    )

    # The merged base config should now have the override applied.
    assert (
        runner.base_config["data"]["allow_risk_free_fallback"] is True
    ), "Scenario-level data.allow_risk_free_fallback should override defaults"
    assert runner.base_config["metrics"]["rf_override_enabled"] is True
    assert abs(runner.base_config["metrics"]["rf_rate_annual"] - 0.03) < 1e-12
    frequency = str(runner.base_config["data"].get("frequency", "M"))
    expected_periodic_rf = (1.0 + 0.03) ** (
        1.0 / float(periods_per_year_from_code(frequency))
    ) - 1.0
    returns = runner._apply_cash_handling(
        pd.DataFrame({"AssetA": [0.01, 0.02]}, index=_price_history().index[:2])
    )
    assert "CASH" in returns.columns
    assert returns["CASH"].tolist() == [expected_periodic_rf, expected_periodic_rf]
