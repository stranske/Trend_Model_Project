from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
import pytest

import trend_analysis.monte_carlo.runner as runner_module
from trend_analysis.api import RunResult
from trend_analysis.monte_carlo.results import (
    MonteCarloPathError,
    StrategyEvaluation,
    build_results_frame,
)
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


def _scenario_with_folds(
    *,
    mode: str,
    folds: dict[str, Any],
    strategies: list[StrategyVariant] | None = None,
    outputs: dict[str, Any] | None = None,
) -> MonteCarloScenario:
    curated = strategies or [StrategyVariant(name="StrategyA")]
    return MonteCarloScenario(
        name="mc_test_folds",
        base_config="config/defaults.yml",
        monte_carlo={
            "mode": mode,
            "n_paths": 2,
            "horizon_years": 1.0,
            "frequency": "M",
            "seed": 123,
            "jobs": 1,
        },
        strategy_set={"curated": curated},
        return_model={"kind": "stationary_bootstrap", "params": {"block_size": 3}},
        folds=folds,
        outputs=outputs,
    )


def _sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["path_id", "strategy"]).reset_index(drop=True)


def _returns_with_rf() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "A": [0.02, 0.01, 0.03, -0.01, 0.01, 0.02],
            "B": [0.005, 0.004, 0.006, 0.004, 0.005, 0.004],
            "RF": [0.002] * len(dates),
        }
    )


def _returns_without_rf() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "A": [0.03, -0.02, 0.04, -0.01, 0.05, -0.02],
            "B": [0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
        }
    )


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


def test_two_layer_strategies_share_path_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario("two_layer")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )

    def _fake_run_simulation(*_args: Any, **_kwargs: Any) -> RunResult:
        metrics = pd.DataFrame({"metric": [1.0]}, index=["equal_weight"])
        return RunResult(metrics=metrics, details={}, seed=0, environment={})

    monkeypatch.setattr(runner_module, "run_simulation", _fake_run_simulation)

    seen_prices: dict[int, pd.DataFrame] = {}
    original = runner._evaluate_strategy

    def _wrapped(strategy: StrategyVariant, context: runner_module._PathContext) -> Any:
        existing = seen_prices.get(context.path_id)
        if existing is None:
            seen_prices[context.path_id] = context.prices.copy()
        else:
            pd.testing.assert_frame_equal(existing, context.prices)
        return original(strategy, context)

    monkeypatch.setattr(runner, "_evaluate_strategy", _wrapped)

    results = runner.run(jobs=1)

    path_counts = results.results_frame["path_id"].value_counts()
    assert path_counts.nunique() == 1
    assert path_counts.iloc[0] == results.results_frame["strategy"].nunique()
    assert len(seen_prices) == scenario.monte_carlo.n_paths


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
    assert results.results_frame["path_id"].nunique() == 5


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


def test_runner_uses_fold_calibration_window(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _scenario_with_folds(
        mode="two_layer",
        folds={
            "mode": "explicit",
            "fold_starts": ["2022-01-31"],
            "calibration_lookback_years": 1.0,
        },
    )
    history = _price_history()
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=history,
    )
    captured: list[pd.DataFrame] = []

    def _fake_build_price_model(self: MonteCarloRunner, history_slice: pd.DataFrame) -> object:
        captured.append(history_slice.copy())
        return object()

    def _fake_run_mode(self: MonteCarloRunner, **_kwargs: Any) -> tuple[list[Any], list[Any]]:
        return [], []

    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)
    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)

    runner.run(jobs=1)

    assert len(captured) == 1
    assert captured[0].index.min() == pd.Timestamp("2020-12-31")
    assert captured[0].index.max() == pd.Timestamp("2021-12-31")


def test_runner_respects_fold_enabled_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _scenario_with_folds(
        mode="two_layer",
        folds={"enabled": False, "mode": "explicit", "fold_starts": ["2022-01-31"]},
    )
    history = _price_history()
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=history,
    )
    captured: list[pd.DataFrame] = []
    seen_fold_ids: list[int | None] = []

    def _fake_build_price_model(self: MonteCarloRunner, history_slice: pd.DataFrame) -> object:
        captured.append(history_slice.copy())
        return object()

    def _fake_run_mode(self: MonteCarloRunner, **kwargs: Any) -> tuple[list[Any], list[Any]]:
        seen_fold_ids.append(kwargs.get("fold_id"))
        return [], []

    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)
    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)

    runner.run(jobs=1)

    assert seen_fold_ids == [None]
    assert len(captured) == 1
    assert captured[0].index.min() == history.index.min()
    assert captured[0].index.max() == history.index.max()


def test_runner_builds_pooled_summary_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario_with_folds(
        mode="two_layer",
        folds={"mode": "explicit", "fold_starts": ["2022-01-31"]},
        outputs={"pooled_distributions": True},
    )
    history = _price_history()
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=history,
    )

    def _fake_build_price_model(self: MonteCarloRunner, _history_slice: pd.DataFrame) -> object:
        return object()

    def _fake_run_mode(self: MonteCarloRunner, **kwargs: Any) -> tuple[list[Any], list[Any]]:
        fold_id = kwargs.get("fold_id")
        evaluation = StrategyEvaluation(
            fold_id=fold_id,
            path_id=0,
            strategy_name="StrategyA",
            metrics={"metric": 1.0},
            metric_source="unit_test",
            path_hash="hash",
            seed=0,
        )
        return [evaluation], []

    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)
    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)

    results = runner.run(jobs=1)

    assert results.pooled_summary_frame is not None
    assert results.pooled_summary_frame.loc[0, "scope"] == "pooled"
    assert results.pooled_summary_frame.loc[0, "pooled_scope"] == "summary"
    assert results.cross_fold_summary_frame is not None
    assert results.metadata.get("pooled_distributions") is True


def test_runner_includes_fold_ids_in_results_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario_with_folds(
        mode="two_layer",
        folds={
            "mode": "explicit",
            "fold_starts": ["2022-01-31", "2023-01-31"],
            "calibration_lookback_years": 1.0,
        },
    )
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )

    def _fake_build_price_model(self: MonteCarloRunner, _history_slice: pd.DataFrame) -> object:
        return object()

    def _fake_run_mode(
        self: MonteCarloRunner,
        *,
        fold_id: int | None,
        **_kwargs: Any,
    ) -> tuple[list[StrategyEvaluation], list[Any]]:
        evaluation = StrategyEvaluation(
            fold_id=fold_id,
            path_id=0,
            strategy_name="StrategyA",
            metrics={"metric": 1.0},
            metric_source="unit_test",
            path_hash=f"hash-{fold_id}",
            seed=123,
        )
        return [evaluation], []

    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)
    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)

    results = runner.run(jobs=1)

    assert results.results_frame["fold_id"].dropna().tolist() == [1, 2]


def test_resolve_strategies_includes_sampled_turnover_caps() -> None:
    scenario = MonteCarloScenario(
        name="mc_sampled",
        base_config="config/defaults.yml",
        monte_carlo={
            "mode": "two_layer",
            "n_paths": 3,
            "horizon_years": 1.0,
            "frequency": "M",
            "seed": 99,
        },
        strategy_set={
            "sampled": {
                "enabled": True,
                "n_strategies": 3,
                "sampling": {
                    "portfolio.max_turnover": {
                        "dist": "uniform",
                        "low": 0.05,
                        "high": 0.10,
                    }
                },
            }
        },
        return_model={"kind": "stationary_bootstrap", "params": {"block_size": 3}},
    )
    base_config = _base_config()
    base_config["portfolio"]["max_turnover"] = 0.2
    runner = MonteCarloRunner(
        scenario,
        base_config=base_config,
    )

    strategies = runner._resolve_strategies()

    assert len(strategies) == 3
    for strategy in strategies:
        assert strategy.name.startswith("sampled_")
        turnover = strategy.overrides["portfolio"]["max_turnover"]
        assert 0.05 <= turnover <= 0.10


def test_guard_turnover_distribution_applies_per_strategy_seed() -> None:
    scenario = MonteCarloScenario(
        name="mc_guarded_turnover",
        base_config="config/defaults.yml",
        monte_carlo={
            "mode": "two_layer",
            "n_paths": 2,
            "horizon_years": 1.0,
            "frequency": "M",
            "seed": 11,
        },
        strategy_set={
            "curated": ["trend_basic"],
            "guards": {
                "max_turnover": {"dist": "discrete", "values": [0.05, 0.15]},
            },
        },
        return_model={"kind": "stationary_bootstrap", "params": {"block_size": 3}},
    )
    base_config = _base_config()
    base_config["portfolio"]["max_turnover"] = 0.2
    runner = MonteCarloRunner(
        scenario,
        base_config=base_config,
    )

    strategy = StrategyVariant(name="trend_basic")
    seed_a = runner._strategy_seed(0, strategy.name)
    seed_b = runner._strategy_seed(1, strategy.name)
    config_a = runner._build_strategy_config(strategy, seed_a)
    config_b = runner._build_strategy_config(strategy, seed_b)

    rng_a = random.Random(seed_a)
    rng_b = random.Random(seed_b)
    expected_a = rng_a.choice([0.05, 0.15])
    expected_b = rng_b.choice([0.05, 0.15])

    assert config_a.portfolio["max_turnover"] == expected_a
    assert config_b.portfolio["max_turnover"] == expected_b


def test_guard_turnover_distribution_respects_strategy_override() -> None:
    scenario = MonteCarloScenario(
        name="mc_guarded_turnover_override",
        base_config="config/defaults.yml",
        monte_carlo={
            "mode": "two_layer",
            "n_paths": 1,
            "horizon_years": 1.0,
            "frequency": "M",
            "seed": 5,
        },
        strategy_set={
            "curated": ["trend_basic"],
            "guards": {
                "max_turnover": {"dist": "uniform", "low": 0.1, "high": 0.3},
            },
        },
        return_model={"kind": "stationary_bootstrap", "params": {"block_size": 3}},
    )
    base_config = _base_config()
    base_config["portfolio"]["max_turnover"] = 0.2
    runner = MonteCarloRunner(
        scenario,
        base_config=base_config,
    )

    strategy = StrategyVariant(
        name="trend_basic",
        overrides={"portfolio": {"max_turnover": 0.42}},
    )
    seed = runner._strategy_seed(0, strategy.name)
    config = runner._build_strategy_config(strategy, seed)

    assert config.portfolio["max_turnover"] == 0.42


def test_guard_turnover_distribution_rejects_non_numeric_sample() -> None:
    scenario = MonteCarloScenario(
        name="mc_guarded_turnover_invalid",
        base_config="config/defaults.yml",
        monte_carlo={
            "mode": "two_layer",
            "n_paths": 1,
            "horizon_years": 1.0,
            "frequency": "M",
            "seed": 21,
        },
        strategy_set={
            "curated": ["trend_basic"],
            "guards": {
                "max_turnover": {"dist": "discrete", "values": ["invalid"]},
            },
        },
        return_model={"kind": "stationary_bootstrap", "params": {"block_size": 3}},
    )
    base_config = _base_config()
    base_config["portfolio"]["max_turnover"] = 0.2
    runner = MonteCarloRunner(
        scenario,
        base_config=base_config,
    )

    strategy = StrategyVariant(name="trend_basic")
    seed = runner._strategy_seed(0, strategy.name)

    with pytest.raises(ValueError, match="distribution must sample numeric values"):
        runner._build_strategy_config(strategy, seed)


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


def test_run_two_layer_parallel_jobs_matches_serial() -> None:
    scenario = _scenario("two_layer")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )

    serial = _sorted_frame(runner.run(jobs=1).results_frame)
    parallel = _sorted_frame(runner.run(jobs=2).results_frame)

    pd.testing.assert_frame_equal(serial, parallel)


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


def test_score_frame_uses_rf_override(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_config()
    cfg["metrics"] = {
        "registry": ["sharpe_ratio"],
        "rf_override_enabled": True,
        "rf_rate_annual": 0.12,
    }
    returns = _returns_with_rf()
    expected = (1.0 + 0.12) ** (1.0 / 12.0) - 1.0

    def _fake_single_period_run(*_args, **kwargs) -> pd.DataFrame:
        assert kwargs["risk_free"] == pytest.approx(expected)
        return pd.DataFrame({"sharpe_ratio": [1.0]}, index=["A"])

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.single_period_run",
        _fake_single_period_run,
    )

    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)
    frame = runner._compute_score_frame(returns)
    assert not frame.empty


def test_score_frame_uses_configured_rf_series(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_config()
    cfg["metrics"] = {"registry": ["sharpe_ratio"]}
    cfg["data"] = {
        "date_column": "Date",
        "frequency": "M",
        "risk_free_column": "RF",
        "allow_risk_free_fallback": True,
    }
    returns = _returns_with_rf()
    expected = pd.Series(
        returns["RF"].to_numpy(),
        index=pd.to_datetime(returns["Date"].values),
        name="RF",
    )

    def _fake_single_period_run(*_args, **kwargs) -> pd.DataFrame:
        rf = kwargs["risk_free"]
        assert isinstance(rf, pd.Series)
        pd.testing.assert_series_equal(rf, expected)
        return pd.DataFrame({"sharpe_ratio": [1.0]}, index=["A"])

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.single_period_run",
        _fake_single_period_run,
    )

    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)
    frame = runner._compute_score_frame(returns)
    assert not frame.empty


def test_score_frame_uses_fallback_rf_series(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_config()
    cfg["metrics"] = {"registry": ["sharpe_ratio"]}
    cfg["data"] = {
        "date_column": "Date",
        "frequency": "M",
        "risk_free_column": None,
        "allow_risk_free_fallback": True,
    }
    returns = _returns_without_rf()
    expected = pd.Series(
        returns["B"].to_numpy(),
        index=pd.to_datetime(returns["Date"].values),
        name="B",
    )

    def _fake_single_period_run(*_args, **kwargs) -> pd.DataFrame:
        rf = kwargs["risk_free"]
        assert isinstance(rf, pd.Series)
        pd.testing.assert_series_equal(rf, expected)
        return pd.DataFrame({"sharpe_ratio": [1.0]}, index=["A"])

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.single_period_run",
        _fake_single_period_run,
    )

    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)
    frame = runner._compute_score_frame(returns)
    assert not frame.empty


def test_run_mixture_requires_matching_seed_lengths() -> None:
    scenario = _scenario("mixture")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )
    model = runner._build_price_model()
    n_periods = runner._compute_n_periods()
    strategies = runner._resolve_strategies()

    with pytest.raises(ValueError, match="strategy_seeds must align with path_seeds"):
        runner._run_mixture(
            model=model,
            n_periods=n_periods,
            strategies=strategies,
            path_seeds=[101, 202],
            strategy_seeds=[303],
            progress_callback=None,
            jobs=1,
        )


def test_execute_paths_handles_unexpected_failure() -> None:
    scenario = _scenario("two_layer")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )
    path_seeds = [101, 202]

    def _boom(path_id: int, seed: int | None) -> tuple[list[Any], list[MonteCarloPathError]]:
        if path_id == 1:
            raise RuntimeError("boom")
        return [], []

    results = list(runner._execute_paths(path_seeds, _boom, jobs=1))

    assert results[0][1] == []
    assert results[0][2] == []
    assert results[1][1] == []
    assert len(results[1][2]) == 1
    assert results[1][2][0].error_type == "RuntimeError"


def test_evaluate_strategy_uses_score_frame_mean_when_metrics_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario("two_layer")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )

    def _fake_run_simulation(config: Any, returns: pd.DataFrame) -> RunResult:
        return RunResult(metrics=pd.DataFrame(), details={}, seed=0, environment={})

    monkeypatch.setattr(runner_module, "run_simulation", _fake_run_simulation)

    score_frame = pd.DataFrame(
        {"metric_a": [1.0, 3.0], "metric_b": [2.0, 4.0]},
        index=["AssetA", "AssetB"],
    )
    context = runner_module._PathContext(
        path_id=0,
        prices=pd.DataFrame({"AssetA": [1.0], "AssetB": [1.0]}),
        returns=pd.DataFrame(
            {
                "Date": [pd.Timestamp("2020-01-31")],
                "AssetA": [0.01],
                "AssetB": [0.02],
            }
        ),
        score_frame=score_frame,
        path_hash="abc",
        seed=11,
    )

    evaluation = runner._evaluate_strategy(StrategyVariant(name="StrategyA"), context)

    assert evaluation.metric_source == "score_frame_mean"
    assert evaluation.metrics["metric_a"] == pytest.approx(2.0)
    assert evaluation.metrics["metric_b"] == pytest.approx(3.0)
