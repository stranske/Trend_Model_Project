from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import trend_analysis.monte_carlo.folds as folds_module
import trend_analysis.monte_carlo.runner as runner_module
from trend_analysis.api import RunResult
from trend_analysis.monte_carlo.config import RiskFreeResolution
from trend_analysis.monte_carlo.results import (
    MonteCarloPathError,
    MonteCarloResults,
    StrategyEvaluation,
    build_results_frame,
)
from trend_analysis.monte_carlo.runner import MonteCarloRunner
from trend_analysis.monte_carlo.scenario import MonteCarloScenario
from trend_analysis.monte_carlo.strategy import StrategyVariant


def _price_history() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=48, freq="ME")
    base = np.linspace(100.0, 140.0, len(dates))
    prices = pd.DataFrame(
        {
            "AssetA": base,
            "AssetB": base * 1.05,
        },
        index=dates,
    )
    return prices


def _daily_price_history() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", "2020-03-31", freq="D")
    base = np.linspace(100.0, 115.0, len(dates))
    prices = pd.DataFrame(
        {
            "AssetA": base,
            "AssetB": base * 1.02,
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
    enable_fold_runs: bool | None = None,
) -> MonteCarloScenario:
    curated = strategies or [StrategyVariant(name="StrategyA")]
    extra: dict[str, Any] = {}
    if enable_fold_runs is not None:
        extra["enable_fold_runs"] = enable_fold_runs
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
        **extra,
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


def test_runner_exports_aggregation_outputs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = MonteCarloScenario(
        name="mc_export",
        base_config="config/defaults.yml",
        monte_carlo={
            "mode": "two_layer",
            "n_paths": 1,
            "horizon_years": 1.0,
            "frequency": "M",
            "seed": 42,
            "jobs": 1,
        },
        strategy_set={"curated": [StrategyVariant(name="StrategyA")]},
        return_model={"kind": "stationary_bootstrap", "params": {"block_size": 3}},
        outputs={
            "directory": str(tmp_path / "mc_out"),
            "format": "csv",
            "aggregation": {
                "quantiles": [0.1, 0.9],
                "breach": {"metric": [1.5]},
                "expected_shortfall": {"metric": 0.2},
            },
        },
    )
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )
    results_frame = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "path_id": 0,
                "strategy": "StrategyA",
                "metric": 1.0,
            },
            {
                "fold_id": 0,
                "path_id": 1,
                "strategy": "StrategyA",
                "metric": 2.0,
            },
        ]
    )
    results = MonteCarloResults(
        mode="two_layer",
        evaluations=[],
        errors=[],
        results_frame=results_frame,
        summary_frame=pd.DataFrame(),
    )

    captured: dict[str, Any] = {}

    def _fake_export_results(
        _results: Any, output_dir: Any, *, formats: Any = None
    ) -> dict[str, Any]:
        captured["results_output_dir"] = output_dir
        captured["results_formats"] = formats
        return {}

    def _fake_export_aggregation(
        aggregation: Any, output_dir: Any, *, formats: Any = None
    ) -> dict[str, Any]:
        captured["aggregation"] = aggregation
        captured["aggregation_output_dir"] = output_dir
        captured["aggregation_formats"] = formats
        return {}

    monkeypatch.setattr(runner_module, "export_results", _fake_export_results)
    monkeypatch.setattr(runner_module, "export_aggregation_results", _fake_export_aggregation)

    runner._maybe_export(results)

    assert captured["results_output_dir"] == captured["aggregation_output_dir"]
    assert captured["results_formats"] == captured["aggregation_formats"]
    aggregation = captured["aggregation"]
    assert {"strategy", "path", "fold"}.issubset(set(aggregation.path_frame.columns))
    assert sorted(aggregation.quantiles_frame["quantile"].unique()) == pytest.approx([0.1, 0.9])
    assert not aggregation.breach_frame.empty
    assert not aggregation.expected_shortfall_frame.empty


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
    captured: list[dict[str, Any]] = []

    def _fake_build_price_model(
        self: MonteCarloRunner,
        history_slice: pd.DataFrame,
        *,
        calibration_start: pd.Timestamp | None = None,
        calibration_end: pd.Timestamp | None = None,
    ) -> object:
        captured.append(
            {
                "history": history_slice.copy(),
                "calibration_start": calibration_start,
                "calibration_end": calibration_end,
            }
        )
        return object()

    def _fake_run_mode(self: MonteCarloRunner, **_kwargs: Any) -> tuple[list[Any], list[Any]]:
        return [], []

    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)
    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)

    runner.run(jobs=1)

    assert len(captured) == 1
    assert captured[0]["history"].index.min() == history.index.min()
    assert captured[0]["history"].index.max() == history.index.max()
    assert captured[0]["calibration_start"] == pd.Timestamp("2020-12-31")
    assert captured[0]["calibration_end"] == pd.Timestamp("2021-12-31")


def test_runner_uses_rolling_fold_calibration_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario_with_folds(
        mode="two_layer",
        folds={
            "mode": "rolling",
            "start": "2021-03-15",
            "end": "2021-09-30",
            "step_months": 3,
            "calibration_lookback_years": 1.0,
        },
    )
    history = _price_history()
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=history,
    )
    captured: list[tuple[pd.Timestamp | None, pd.Timestamp | None]] = []

    def _fake_build_price_model(
        self: MonteCarloRunner,
        _history_slice: pd.DataFrame,
        *,
        calibration_start: pd.Timestamp | None = None,
        calibration_end: pd.Timestamp | None = None,
    ) -> object:
        captured.append((calibration_start, calibration_end))
        return object()

    def _fake_run_mode(self: MonteCarloRunner, **_kwargs: Any) -> tuple[list[Any], list[Any]]:
        return [], []

    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)
    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)

    runner.run(jobs=1)

    assert captured == [
        (pd.Timestamp("2020-02-29"), pd.Timestamp("2021-02-28")),
        (pd.Timestamp("2020-05-31"), pd.Timestamp("2021-05-31")),
        (pd.Timestamp("2020-08-31"), pd.Timestamp("2021-08-31")),
    ]


def test_runner_uses_count_spaced_fold_calibration_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario_with_folds(
        mode="two_layer",
        folds={
            "mode": "count_spaced",
            "start": "2021-01-31",
            "end": "2021-12-31",
            "n_folds": 2,
            "calibration_lookback_years": 1.0,
        },
    )
    history = _price_history()
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=history,
    )
    captured: list[tuple[pd.Timestamp | None, pd.Timestamp | None]] = []

    def _fake_build_price_model(
        self: MonteCarloRunner,
        _history_slice: pd.DataFrame,
        *,
        calibration_start: pd.Timestamp | None = None,
        calibration_end: pd.Timestamp | None = None,
    ) -> object:
        captured.append((calibration_start, calibration_end))
        return object()

    def _fake_run_mode(self: MonteCarloRunner, **_kwargs: Any) -> tuple[list[Any], list[Any]]:
        return [], []

    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)
    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)

    runner.run(jobs=1)

    assert captured == [
        (pd.Timestamp("2020-01-31"), pd.Timestamp("2020-12-31")),
        (pd.Timestamp("2020-11-30"), pd.Timestamp("2021-11-30")),
    ]


def test_build_price_model_applies_calibration_window_after_normalization() -> None:
    scenario = _scenario("two_layer")
    history = _daily_price_history()
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=history,
    )

    model = runner._build_price_model(
        history,
        calibration_start=pd.Timestamp("2020-01-15"),
        calibration_end=pd.Timestamp("2020-03-15"),
    )

    log_returns = model.historical_log_returns
    assert log_returns.index.max() == pd.Timestamp("2020-02-29")
    assert log_returns.index.min() == pd.Timestamp("2020-02-29")


def test_build_price_model_rejects_empty_calibration_window() -> None:
    scenario = _scenario("two_layer")
    history = _daily_price_history()
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=history,
    )

    with pytest.raises(ValueError, match="fold calibration window produced no history data"):
        runner._build_price_model(
            history,
            calibration_start=pd.Timestamp("2019-01-01"),
            calibration_end=pd.Timestamp("2019-06-30"),
        )


def test_runner_respects_enable_fold_runs_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _scenario_with_folds(
        mode="two_layer",
        folds={"enabled": True, "mode": "explicit", "fold_starts": ["2022-01-31"]},
        enable_fold_runs=False,
    )
    history = _price_history()
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=history,
    )
    captured: list[pd.DataFrame] = []
    captured_calibration: list[tuple[pd.Timestamp | None, pd.Timestamp | None]] = []
    seen_fold_ids: list[int | None] = []

    def _fake_build_price_model(
        self: MonteCarloRunner,
        history_slice: pd.DataFrame,
        *,
        calibration_start: pd.Timestamp | None = None,
        calibration_end: pd.Timestamp | None = None,
    ) -> object:
        captured.append(history_slice.copy())
        captured_calibration.append((calibration_start, calibration_end))
        return object()

    def _fake_run_mode(self: MonteCarloRunner, **kwargs: Any) -> tuple[list[Any], list[Any]]:
        seen_fold_ids.append(kwargs.get("fold_id"))
        return [], []

    def _raise_from_config(cls: type[folds_module.FoldGenerator], _config: Any) -> Any:
        raise AssertionError("FoldGenerator.from_config should not be called when disabled")

    monkeypatch.setattr(
        folds_module.FoldGenerator,
        "from_config",
        classmethod(_raise_from_config),
    )
    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)
    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)

    runner.run(jobs=1)

    assert seen_fold_ids == [None]
    assert len(captured) == 1
    assert captured[0].index.min() == history.index.min()
    assert captured[0].index.max() == history.index.max()
    assert captured_calibration == [(None, None)]


def test_runner_errors_include_fold_label(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _scenario_with_folds(
        mode="two_layer",
        folds={"mode": "explicit", "fold_starts": ["2022-01-31"]},
    )
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "_generate_path_context", _boom)

    results = runner.run(jobs=1)

    assert results.errors
    assert {error.fold_id for error in results.errors} == {1}
    assert {error.fold_label for error in results.errors} == {"2022-01"}


def test_runner_logs_fold_context_for_errors(caplog: pytest.LogCaptureFixture) -> None:
    scenario = _scenario(mode="two_layer")
    logger = logging.getLogger("trend_analysis.monte_carlo")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
        logger=logger,
    )

    with caplog.at_level(logging.ERROR, logger="trend_analysis.monte_carlo"):
        runner._log_path_error(
            3,
            "StrategyA",
            RuntimeError("boom"),
            fold_id=2,
            fold_label="2022-01",
        )

    assert any(
        "fold 2 (2022-01) path 3 strategy StrategyA" in record.message for record in caplog.records
    )


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

    def _fake_build_price_model(
        self: MonteCarloRunner,
        _history_slice: pd.DataFrame,
        *,
        calibration_start: pd.Timestamp | None = None,
        calibration_end: pd.Timestamp | None = None,
    ) -> object:
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
    assert results.pooled_distribution_frame is not None
    assert results.pooled_distribution_frame.loc[0, "scope"] == "pooled"
    assert results.pooled_distribution_frame.loc[0, "pooled_scope"] == "distribution"
    assert results.cross_fold_summary_frame is not None
    assert results.metadata.get("pooled_distributions") is True
    assert results.metadata.get("pooled_scope") == "summary+distribution"
    assert results.metadata.get("pooled_outputs") == ["summary", "distribution"]


def test_runner_builds_cross_fold_summary_without_pooled_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario_with_folds(
        mode="two_layer",
        folds={"mode": "explicit", "fold_starts": ["2022-01-31", "2023-01-31"]},
    )
    history = _price_history()
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=history,
    )
    seen_fold_ids: list[int] = []

    def _fake_build_price_model(
        self: MonteCarloRunner,
        _history_slice: pd.DataFrame,
        *,
        calibration_start: pd.Timestamp | None = None,
        calibration_end: pd.Timestamp | None = None,
    ) -> object:
        return object()

    def _fake_run_mode(self: MonteCarloRunner, **kwargs: Any) -> tuple[list[Any], list[Any]]:
        fold_id = int(kwargs.get("fold_id") or 0)
        seen_fold_ids.append(fold_id)
        evaluation = StrategyEvaluation(
            fold_id=fold_id,
            path_id=0,
            strategy_name="StrategyA",
            metrics={"metric": 1.0 + (fold_id * 2.0)},
            metric_source="unit_test",
            path_hash=f"hash-{fold_id}",
            seed=0,
        )
        return [evaluation], []

    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)
    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)

    results = runner.run(jobs=1)

    assert results.pooled_summary_frame is None
    assert results.metadata.get("pooled_distributions") is False
    assert results.metadata.get("pooled_scope") == "none"
    assert results.metadata.get("pooled_outputs") == []
    cross_fold = results.cross_fold_summary_frame
    assert cross_fold is not None
    assert cross_fold.loc[0, "scope"] == "cross_fold"
    assert pd.isna(cross_fold.loc[0, "fold_id"])
    assert cross_fold.loc[0, "folds"] == 2
    metric_values = [1.0 + (fold_id * 2.0) for fold_id in seen_fold_ids]
    assert cross_fold.loc[0, "metric_mean"] == pytest.approx(float(np.mean(metric_values)))
    assert cross_fold.loc[0, "metric_min"] == pytest.approx(float(np.min(metric_values)))
    assert cross_fold.loc[0, "metric_max"] == pytest.approx(float(np.max(metric_values)))
    assert cross_fold.loc[0, "metric_median"] == pytest.approx(float(np.median(metric_values)))


def test_runner_populates_nav_paths_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario("two_layer")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )
    nav_index = pd.date_range("2021-01-31", periods=3, freq="ME")

    def _fake_build_price_model(
        self: MonteCarloRunner,
        _history_slice: pd.DataFrame,
        *,
        calibration_start: pd.Timestamp | None = None,
        calibration_end: pd.Timestamp | None = None,
    ) -> object:
        return object()

    def _fake_run_mode(self: MonteCarloRunner, **_kwargs: Any) -> tuple[list[Any], list[Any]]:
        evals = [
            StrategyEvaluation(
                fold_id=None,
                path_id=0,
                strategy_name="StrategyA",
                metrics={"metric": 1.0},
                metric_source="unit_test",
                path_hash="hash-0",
                seed=0,
                nav_series=pd.Series([1.0, 1.02, 1.05], index=nav_index, name="NAV"),
            ),
            StrategyEvaluation(
                fold_id=None,
                path_id=1,
                strategy_name="StrategyA",
                metrics={"metric": 2.0},
                metric_source="unit_test",
                path_hash="hash-1",
                seed=1,
                nav_series=pd.Series([1.0, 0.98, 1.01], index=nav_index, name="NAV"),
            ),
        ]
        return evals, []

    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)
    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)

    results = runner.run(jobs=1)

    nav_paths = results.metadata.get("nav_paths")
    assert isinstance(nav_paths, pd.DataFrame)
    assert not nav_paths.empty
    assert isinstance(nav_paths.index, pd.DatetimeIndex)
    assert nav_paths.index.equals(nav_index)
    assert set(nav_paths.columns) == {0, 1}
    assert nav_paths.columns.name == "path"
    assert nav_paths.loc[nav_index[-1], 0] == pytest.approx(1.05)


def test_runner_coerces_nav_paths_multiindex_asset_level() -> None:
    scenario = _scenario("two_layer")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )
    columns = pd.MultiIndex.from_arrays([[0, 1], ["foo", "bar"]])
    frame = pd.DataFrame([[1.0, 1.0], [1.1, 0.9]], columns=columns)

    coerced = runner._coerce_nav_paths_columns(frame)

    assert isinstance(coerced.columns, pd.MultiIndex)
    assert coerced.columns.names == ["path", "asset"]
    assert set(coerced.columns.get_level_values("path")) == {0, 1}
    assert set(coerced.columns.get_level_values("asset")) == {"NAV"}


def test_runner_populates_nav_paths_by_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario_with_folds(
        mode="two_layer",
        folds={"mode": "explicit", "fold_starts": ["2022-01-31", "2023-01-31"]},
    )
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )
    nav_index = pd.date_range("2021-01-31", periods=2, freq="ME")

    def _fake_build_price_model(
        self: MonteCarloRunner,
        _history_slice: pd.DataFrame,
        *,
        calibration_start: pd.Timestamp | None = None,
        calibration_end: pd.Timestamp | None = None,
    ) -> object:
        return object()

    def _fake_run_mode(self: MonteCarloRunner, **kwargs: Any) -> tuple[list[Any], list[Any]]:
        fold_id = int(kwargs.get("fold_id") or 0)
        evals = [
            StrategyEvaluation(
                fold_id=fold_id,
                path_id=0,
                strategy_name="StrategyA",
                metrics={"metric": 1.0},
                metric_source="unit_test",
                path_hash=f"hash-{fold_id}-0",
                seed=fold_id,
                nav_series=pd.Series([1.0, 1.0 + (0.01 * fold_id)], index=nav_index, name="NAV"),
            ),
            StrategyEvaluation(
                fold_id=fold_id,
                path_id=1,
                strategy_name="StrategyA",
                metrics={"metric": 1.5},
                metric_source="unit_test",
                path_hash=f"hash-{fold_id}-1",
                seed=fold_id + 1,
                nav_series=pd.Series([1.0, 1.0 + (0.02 * fold_id)], index=nav_index, name="NAV"),
            ),
        ]
        return evals, []

    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)
    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)

    results = runner.run(jobs=1)

    nav_paths_by_fold = results.metadata.get("nav_paths_by_fold")
    assert isinstance(nav_paths_by_fold, dict)
    assert set(nav_paths_by_fold) == {1, 2}
    fold_one = nav_paths_by_fold[1]
    fold_two = nav_paths_by_fold[2]
    assert isinstance(fold_one, pd.DataFrame)
    assert isinstance(fold_two, pd.DataFrame)
    assert fold_one.index.equals(nav_index)
    assert fold_two.index.equals(nav_index)
    assert fold_one.loc[nav_index[-1], 0] == pytest.approx(1.01)
    assert fold_two.loc[nav_index[-1], 0] == pytest.approx(1.02)


def test_runner_cross_fold_summary_stats_include_pooled_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario_with_folds(
        mode="two_layer",
        folds={"mode": "explicit", "fold_starts": ["2022-01-31", "2023-01-31"]},
        outputs={"pooled_distributions": True},
    )
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )
    seen_fold_ids: list[int] = []

    def _fake_build_price_model(
        self: MonteCarloRunner,
        _history_slice: pd.DataFrame,
        *,
        calibration_start: pd.Timestamp | None = None,
        calibration_end: pd.Timestamp | None = None,
    ) -> object:
        return object()

    def _fake_run_mode(self: MonteCarloRunner, **kwargs: Any) -> tuple[list[Any], list[Any]]:
        fold_id = int(kwargs.get("fold_id") or 0)
        seen_fold_ids.append(fold_id)
        evaluation = StrategyEvaluation(
            fold_id=fold_id,
            path_id=0,
            strategy_name="StrategyA",
            metrics={"metric": 10.0 + fold_id},
            metric_source="unit_test",
            path_hash=f"hash-{fold_id}",
            seed=0,
        )
        return [evaluation], []

    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)
    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)

    results = runner.run(jobs=1)

    cross_fold = results.cross_fold_summary_frame
    assert cross_fold is not None
    assert cross_fold.loc[0, "scope"] == "cross_fold"
    assert cross_fold.loc[0, "folds"] == 2
    metric_values = [10.0 + fold_id for fold_id in seen_fold_ids]
    assert cross_fold.loc[0, "metric_mean"] == pytest.approx(float(np.mean(metric_values)))
    assert cross_fold.loc[0, "metric_min"] == pytest.approx(float(np.min(metric_values)))
    assert cross_fold.loc[0, "metric_max"] == pytest.approx(float(np.max(metric_values)))
    assert cross_fold.loc[0, "metric_median"] == pytest.approx(float(np.median(metric_values)))

    pooled = results.pooled_summary_frame
    assert pooled is not None
    assert pooled.loc[0, "scope"] == "pooled"
    assert pooled.loc[0, "pooled_scope"] == "summary"
    assert results.metadata.get("pooled_distributions") is True
    assert results.pooled_distribution_frame is not None
    assert results.pooled_distribution_frame.loc[0, "pooled_scope"] == "distribution"


def test_runner_pooled_summary_includes_fold_count_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario_with_folds(
        mode="two_layer",
        folds={"mode": "explicit", "fold_starts": ["2022-01-31", "2023-01-31"]},
        outputs={"pooled_distributions": True},
    )
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )

    def _fake_build_price_model(
        self: MonteCarloRunner,
        _history_slice: pd.DataFrame,
        *,
        calibration_start: pd.Timestamp | None = None,
        calibration_end: pd.Timestamp | None = None,
    ) -> object:
        return object()

    def _fake_run_mode(self: MonteCarloRunner, **kwargs: Any) -> tuple[list[Any], list[Any]]:
        fold_id = int(kwargs.get("fold_id") or 0)
        evaluation = StrategyEvaluation(
            fold_id=fold_id,
            path_id=0,
            strategy_name="StrategyA",
            metrics={"metric": 5.0 + fold_id},
            metric_source="unit_test",
            path_hash=f"hash-{fold_id}",
            seed=0,
        )
        return [evaluation], []

    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)
    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)

    results = runner.run(jobs=1)

    pooled = results.pooled_summary_frame
    assert pooled is not None
    assert pooled.loc[0, "scope"] == "pooled"
    assert pooled.loc[0, "pooled_scope"] == "summary"
    assert pooled.loc[0, "folds"] == 2
    assert results.metadata.get("pooled_distributions") is True
    assert results.pooled_distribution_frame is not None


def test_runner_exports_pooled_summary_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "mc_outputs"
    scenario = _scenario_with_folds(
        mode="two_layer",
        folds={"mode": "explicit", "fold_starts": ["2022-01-31"]},
        outputs={
            "directory": str(output_dir),
            "formats": ["csv"],
            "pooled_distributions": True,
        },
    )
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )

    def _fake_build_price_model(
        self: MonteCarloRunner,
        _history_slice: pd.DataFrame,
        *,
        calibration_start: pd.Timestamp | None = None,
        calibration_end: pd.Timestamp | None = None,
    ) -> object:
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

    runner.run(jobs=1)

    pooled_path = output_dir / "pooled_summary.csv"
    assert pooled_path.exists()
    pooled_frame = pd.read_csv(pooled_path)
    assert pooled_frame.loc[0, "scope"] == "pooled"
    assert pooled_frame.loc[0, "pooled_scope"] == "summary"

    pooled_dist_path = output_dir / "pooled_distributions.csv"
    assert pooled_dist_path.exists()
    pooled_dist = pd.read_csv(pooled_dist_path)
    assert pooled_dist.loc[0, "scope"] == "pooled"
    assert pooled_dist.loc[0, "pooled_scope"] == "distribution"


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

    def _fake_build_price_model(
        self: MonteCarloRunner,
        _history_slice: pd.DataFrame,
        *,
        calibration_start: pd.Timestamp | None = None,
        calibration_end: pd.Timestamp | None = None,
    ) -> object:
        return object()

    def _fake_run_mode(
        self: MonteCarloRunner,
        *,
        fold_id: int | None,
        fold_label: str | None,
        **_kwargs: Any,
    ) -> tuple[list[StrategyEvaluation], list[Any]]:
        evaluation = StrategyEvaluation(
            fold_id=fold_id,
            fold_label=fold_label,
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
    assert results.results_frame["fold_label"].dropna().tolist() == ["2022-01", "2023-01"]


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


def test_run_mixture_uses_shared_bulk_generation_when_path_seeds_match_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario("mixture")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )
    n_periods = runner._compute_n_periods()
    total = scenario.monte_carlo.n_paths
    assert total is not None
    strategies = runner._resolve_strategies()
    path_seeds, strategy_seeds = runner._build_seeds()
    call_log: list[dict[str, Any]] = []

    class _RecordingModel:
        def sample_prices(
            self,
            *,
            n_periods: int,
            n_paths: int,
            frequency: str,
            seed: int | None,
        ) -> Any:
            call_log.append(
                {
                    "n_periods": n_periods,
                    "n_paths": n_paths,
                    "frequency": frequency,
                    "seed": seed,
                }
            )
            index = pd.date_range("2024-01-31", periods=n_periods, freq="ME")
            columns = pd.MultiIndex.from_product(
                [range(n_paths), ["AssetA", "AssetB"]],
                names=["path", "asset"],
            )
            prices = pd.DataFrame(100.0, index=index, columns=columns)
            log_returns = pd.DataFrame(0.0, index=index, columns=columns)
            return type("PathResult", (), {"prices": prices, "log_returns": log_returns})()

    monkeypatch.setattr(
        runner,
        "_compute_score_frame",
        lambda _returns: pd.DataFrame({"score": [1.0]}, index=["AssetA"]),
    )

    def _fake_evaluate_strategy(
        strategy: StrategyVariant, context: runner_module._PathContext
    ) -> StrategyEvaluation:
        return StrategyEvaluation(
            fold_id=None,
            path_id=context.path_id,
            strategy_name=strategy.name,
            metrics={"metric": 1.0},
            metric_source="stub",
            path_hash=context.path_hash,
            seed=context.seed,
        )

    monkeypatch.setattr(runner, "_evaluate_strategy", _fake_evaluate_strategy)

    evals, errors = runner._run_mixture(
        model=_RecordingModel(),
        n_periods=n_periods,
        strategies=strategies,
        path_seeds=path_seeds,
        strategy_seeds=strategy_seeds,
        progress_callback=None,
        jobs=1,
    )

    assert not errors
    assert len(evals) == total
    assert len(call_log) == 1
    assert call_log[0]["n_paths"] == total
    assert call_log[0]["seed"] == scenario.monte_carlo.seed


def test_run_mixture_uses_per_path_generation_when_path_seeds_diverge(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scenario = _scenario("mixture")
    runner = MonteCarloRunner(
        scenario,
        base_config=_base_config(),
        price_history=_price_history(),
    )
    n_periods = runner._compute_n_periods()
    total = scenario.monte_carlo.n_paths
    assert total is not None
    strategies = runner._resolve_strategies()
    path_seeds, strategy_seeds = runner._build_seeds()
    assert path_seeds[0] is not None
    divergent_seeds = list(path_seeds)
    divergent_seeds[0] = int(path_seeds[0]) + 1
    call_log: list[dict[str, Any]] = []

    class _RecordingModel:
        def sample_prices(
            self,
            *,
            n_periods: int,
            n_paths: int,
            frequency: str,
            seed: int | None,
        ) -> Any:
            call_log.append(
                {
                    "n_periods": n_periods,
                    "n_paths": n_paths,
                    "frequency": frequency,
                    "seed": seed,
                }
            )
            index = pd.date_range("2024-01-31", periods=n_periods, freq="ME")
            columns = pd.MultiIndex.from_product(
                [range(n_paths), ["AssetA", "AssetB"]],
                names=["path", "asset"],
            )
            prices = pd.DataFrame(100.0, index=index, columns=columns)
            log_returns = pd.DataFrame(0.0, index=index, columns=columns)
            return type("PathResult", (), {"prices": prices, "log_returns": log_returns})()

    monkeypatch.setattr(
        runner,
        "_compute_score_frame",
        lambda _returns: pd.DataFrame({"score": [1.0]}, index=["AssetA"]),
    )

    def _fake_evaluate_strategy(
        strategy: StrategyVariant, context: runner_module._PathContext
    ) -> StrategyEvaluation:
        return StrategyEvaluation(
            fold_id=None,
            path_id=context.path_id,
            strategy_name=strategy.name,
            metrics={"metric": 1.0},
            metric_source="stub",
            path_hash=context.path_hash,
            seed=context.seed,
        )

    monkeypatch.setattr(runner, "_evaluate_strategy", _fake_evaluate_strategy)

    with caplog.at_level(logging.DEBUG, logger="trend_analysis.monte_carlo"):
        evals, errors = runner._run_mixture(
            model=_RecordingModel(),
            n_periods=n_periods,
            strategies=strategies,
            path_seeds=divergent_seeds,
            strategy_seeds=strategy_seeds,
            progress_callback=None,
            jobs=1,
        )

    assert not errors
    assert len(evals) == total
    assert len(call_log) == total
    assert all(call["n_paths"] == 1 for call in call_log)
    assert [call["seed"] for call in call_log] == divergent_seeds
    assert "Mixture mode selected per-path generation" in caplog.text


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


def test_should_resolve_risk_free_gate_truth_table() -> None:
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=_base_config())

    assert (
        runner._should_resolve_risk_free(
            data_settings={"risk_free_column": None, "allow_risk_free_fallback": False},
            metrics_settings={"rf_override_enabled": True},
        )
        is True
    )
    assert (
        runner._should_resolve_risk_free(
            data_settings={"risk_free_column": "RF", "allow_risk_free_fallback": False},
            metrics_settings={"rf_override_enabled": False},
        )
        is True
    )
    assert (
        runner._should_resolve_risk_free(
            data_settings={"risk_free_column": None, "allow_risk_free_fallback": True},
            metrics_settings={"rf_override_enabled": False},
        )
        is True
    )
    assert (
        runner._should_resolve_risk_free(
            data_settings={"risk_free_column": None, "allow_risk_free_fallback": False},
            metrics_settings={"rf_override_enabled": False},
        )
        is False
    )


def test_should_resolve_risk_free_treats_blank_column_as_unset() -> None:
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=_base_config())

    assert (
        runner._should_resolve_risk_free(
            data_settings={"risk_free_column": "   ", "allow_risk_free_fallback": False},
            metrics_settings={"rf_override_enabled": False},
        )
        is False
    )


def test_should_inject_cash_is_gated_by_override_flag() -> None:
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=_base_config())

    assert runner._should_inject_cash(metrics_settings={"rf_override_enabled": True}) is True
    assert runner._should_inject_cash(metrics_settings={"rf_override_enabled": False}) is False
    assert runner._should_inject_cash(metrics_settings={}) is False
    assert runner._should_inject_cash(metrics_settings={"rf_override_enabled": None}) is False


def test_apply_cash_handling_injects_cash_when_override_gate_enabled() -> None:
    cfg = _base_config()
    cfg["metrics"] = {
        "registry": ["sharpe_ratio"],
        "rf_override_enabled": True,
        "rf_rate_annual": 0.12,
    }
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)
    returns = _returns_without_rf()

    injected = runner._apply_cash_handling(returns)

    expected_periodic = (1.0 + 0.12) ** (1.0 / 12.0) - 1.0
    assert "CASH" in injected.columns
    np.testing.assert_allclose(
        injected["CASH"].to_numpy(dtype=float, copy=False),
        np.full(len(returns), expected_periodic, dtype=float),
    )


def test_cash_injection_when_condition_met() -> None:
    cfg = _base_config()
    cfg["metrics"] = {
        "registry": ["sharpe_ratio"],
        "rf_override_enabled": True,
        "rf_rate_annual": 0.06,
    }
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)

    injected = runner._apply_cash_handling(_returns_without_rf())

    assert "CASH" in injected.columns


def test_apply_cash_handling_skips_when_override_disabled_even_with_risk_free_column() -> None:
    cfg = _base_config()
    cfg["metrics"] = {
        "registry": ["sharpe_ratio"],
        "rf_override_enabled": False,
    }
    cfg["data"]["risk_free_column"] = "RF"
    cfg["data"]["allow_risk_free_fallback"] = False
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)
    returns = _returns_with_rf()

    injected = runner._apply_cash_handling(returns)

    assert "CASH" not in injected.columns


def test_apply_cash_handling_skips_when_override_disabled_even_with_fallback_enabled() -> None:
    cfg = _base_config()
    cfg["metrics"] = {
        "registry": ["sharpe_ratio"],
        "rf_override_enabled": False,
    }
    cfg["data"]["risk_free_column"] = None
    cfg["data"]["allow_risk_free_fallback"] = True
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)

    injected = runner._apply_cash_handling(_returns_without_rf())

    assert "CASH" not in injected.columns


def test_apply_cash_handling_does_not_resolve_rf_when_override_gate_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_config()
    cfg["metrics"] = {
        "registry": ["sharpe_ratio"],
        "rf_override_enabled": False,
        "rf_rate_annual": 0.12,
    }
    cfg["data"]["risk_free_column"] = "RF"
    cfg["data"]["allow_risk_free_fallback"] = True
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)

    resolver_called = False

    def _resolver_should_not_run(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal resolver_called
        resolver_called = True
        return RiskFreeResolution(source="override", risk_free=0.01)

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.resolve_risk_free_source",
        _resolver_should_not_run,
    )

    injected = runner._apply_cash_handling(_returns_with_rf())

    assert "CASH" not in injected.columns
    assert resolver_called is False


def test_apply_cash_handling_skips_when_override_missing_even_with_rf_sources() -> None:
    cfg = _base_config()
    cfg["metrics"] = {"registry": ["sharpe_ratio"]}
    cfg["data"]["risk_free_column"] = "RF"
    cfg["data"]["allow_risk_free_fallback"] = True
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)

    injected = runner._apply_cash_handling(_returns_with_rf())

    assert "CASH" not in injected.columns


def test_cash_injection_when_condition_not_met() -> None:
    cfg = _base_config()
    cfg["metrics"] = {"registry": ["sharpe_ratio"], "rf_override_enabled": False}
    cfg["data"]["allow_risk_free_fallback"] = False
    cfg["data"]["risk_free_column"] = None
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)

    injected = runner._apply_cash_handling(_returns_without_rf())

    assert "CASH" not in injected.columns


def test_apply_cash_handling_skips_when_override_gate_disabled() -> None:
    cfg = _base_config()
    cfg["metrics"] = {
        "registry": ["sharpe_ratio"],
        "rf_override_enabled": False,
        "rf_rate_annual": 0.12,
    }
    cfg["data"]["allow_risk_free_fallback"] = False
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)

    injected = runner._apply_cash_handling(_returns_without_rf())

    assert "CASH" not in injected.columns


def test_apply_cash_handling_logs_gate_components_when_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _base_config()
    cfg["metrics"] = {
        "registry": ["sharpe_ratio"],
        "rf_override_enabled": False,
    }
    cfg["data"]["risk_free_column"] = "   "
    cfg["data"]["allow_risk_free_fallback"] = False
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)

    with caplog.at_level(logging.WARNING):
        injected = runner._apply_cash_handling(_returns_without_rf())

    assert "CASH" not in injected.columns
    assert "gate=false" in caplog.text
    assert "metrics.rf_override_enabled=False" in caplog.text


def test_cash_uses_correct_risk_free_rate() -> None:
    cfg = _base_config()
    cfg["metrics"] = {
        "registry": ["sharpe_ratio"],
        "rf_override_enabled": True,
        "rf_rate_annual": 0.12,
    }
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)

    injected = runner._apply_cash_handling(_returns_without_rf())

    expected = np.full(6, (1.12 ** (1.0 / 12.0)) - 1.0, dtype=float)
    np.testing.assert_allclose(injected["CASH"].to_numpy(dtype=float, copy=False), expected)


def test_apply_cash_handling_handles_missing_metrics_nulls_and_empty_inputs() -> None:
    cfg = _base_config()
    cfg.pop("metrics", None)
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)

    injected_missing_metrics = runner._apply_cash_handling(_returns_without_rf())
    assert "CASH" not in injected_missing_metrics.columns

    null_returns = _returns_without_rf()
    null_returns.loc[1, "B"] = np.nan
    injected_null = runner._apply_cash_handling(null_returns)
    assert "CASH" not in injected_null.columns

    empty_returns = _returns_without_rf().iloc[:0].copy()
    injected_empty = runner._apply_cash_handling(empty_returns)
    assert injected_empty.empty
    assert "CASH" not in injected_empty.columns


def test_apply_cash_handling_returns_unchanged_when_cash_already_present() -> None:
    cfg = _base_config()
    cfg["metrics"] = {"registry": ["sharpe_ratio"], "rf_override_enabled": True}
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)
    returns = _returns_without_rf().copy()
    returns["CASH"] = 0.001

    injected = runner._apply_cash_handling(returns)

    pd.testing.assert_frame_equal(injected, returns)


def test_apply_cash_handling_returns_unchanged_when_legacy_lowercase_cash_present() -> None:
    cfg = _base_config()
    cfg["metrics"] = {"registry": ["sharpe_ratio"], "rf_override_enabled": True}
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)
    returns = _returns_without_rf().copy()
    returns["cash"] = 0.001

    injected = runner._apply_cash_handling(returns)

    pd.testing.assert_frame_equal(injected, returns)


def test_apply_cash_handling_skips_when_base_config_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_config()
    cfg["metrics"] = {"registry": ["sharpe_ratio"], "rf_override_enabled": True}
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)

    class _BrokenConfig:
        def __init__(self, **_kwargs: Any) -> None:
            raise ValueError("bad config")

    monkeypatch.setattr(runner_module, "Config", _BrokenConfig)

    injected = runner._apply_cash_handling(_returns_without_rf())

    assert "CASH" not in injected.columns


def test_apply_cash_handling_skips_without_parsing_config_when_override_gate_disabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=_base_config())

    class _BrokenConfig:
        def __init__(self, **_kwargs: Any) -> None:
            raise ValueError("bad config")

    monkeypatch.setattr(runner_module, "Config", _BrokenConfig)

    with caplog.at_level(logging.WARNING):
        injected = runner._apply_cash_handling(_returns_without_rf())

    assert "CASH" not in injected.columns
    assert "gate=false" in caplog.text
    assert "failed to parse base config" not in caplog.text


def test_apply_cash_handling_skips_when_rf_resolution_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_config()
    cfg["metrics"] = {"registry": ["sharpe_ratio"], "rf_override_enabled": True}
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("resolver failed")

    monkeypatch.setattr("trend_analysis.monte_carlo.runner.resolve_risk_free_source", _boom)

    injected = runner._apply_cash_handling(_returns_without_rf())

    assert "CASH" not in injected.columns


def test_apply_cash_handling_skips_when_rf_resolution_returns_unsupported_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_config()
    cfg["metrics"] = {"registry": ["sharpe_ratio"], "rf_override_enabled": True}
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)
    returns = _returns_without_rf()

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.resolve_risk_free_source",
        lambda *_args, **_kwargs: RiskFreeResolution(source="test", risk_free=["bad"]),
    )

    injected = runner._apply_cash_handling(returns)

    assert "CASH" not in injected.columns


def test_apply_cash_handling_skips_when_rf_series_alignment_introduces_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_config()
    cfg["metrics"] = {"registry": ["sharpe_ratio"], "rf_override_enabled": True}
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)
    returns = _returns_without_rf()
    bad_series = pd.Series([0.001] * (len(returns) - 1), name="RF")

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.resolve_risk_free_source",
        lambda *_args, **_kwargs: RiskFreeResolution(source="test", risk_free=bad_series),
    )

    injected = runner._apply_cash_handling(returns)

    assert "CASH" not in injected.columns


def test_apply_cash_handling_skips_when_rf_series_contains_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_config()
    cfg["metrics"] = {"registry": ["sharpe_ratio"], "rf_override_enabled": True}
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)
    returns = _returns_without_rf()
    nan_series = pd.Series([0.001, np.nan, 0.001, 0.001, 0.001, 0.001], name="RF")

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.resolve_risk_free_source",
        lambda *_args, **_kwargs: RiskFreeResolution(source="test", risk_free=nan_series),
    )

    injected = runner._apply_cash_handling(returns)

    assert "CASH" not in injected.columns


def test_apply_cash_handling_aligns_rf_series_by_returns_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_config()
    cfg["metrics"] = {"registry": ["sharpe_ratio"], "rf_override_enabled": True}
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)
    returns = _returns_without_rf()

    rf_values = pd.Series(
        [0.002, 0.003, 0.004, 0.005, 0.006, 0.007],
        index=pd.Index([5, 4, 3, 2, 1, 0]),
        name="RF",
    )

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.resolve_risk_free_source",
        lambda *_args, **_kwargs: RiskFreeResolution(source="test", risk_free=rf_values),
    )

    injected = runner._apply_cash_handling(returns)

    np.testing.assert_allclose(
        injected["CASH"].to_numpy(dtype=float, copy=False),
        np.array([0.007, 0.006, 0.005, 0.004, 0.003, 0.002], dtype=float),
    )


def test_inject_cash_returns_aliases_apply_cash_handling() -> None:
    cfg = _base_config()
    cfg["metrics"] = {
        "registry": ["sharpe_ratio"],
        "rf_override_enabled": True,
        "rf_rate_annual": 0.06,
    }
    runner = MonteCarloRunner(_scenario("two_layer"), base_config=cfg)
    returns = _returns_without_rf()

    expected = runner._apply_cash_handling(returns)
    actual = runner._inject_cash_returns(returns)

    pd.testing.assert_frame_equal(actual, expected)


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
