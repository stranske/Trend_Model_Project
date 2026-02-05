from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from trend_analysis.api import RunResult
from trend_analysis.monte_carlo.results import (
    StrategyEvaluation,
    build_diagnostics_frame,
)
from trend_analysis.monte_carlo.runner import MonteCarloRunner, _PathContext
from trend_analysis.monte_carlo.scenario import MonteCarloScenario
from trend_analysis.monte_carlo.strategy import StrategyVariant


def _base_config(max_turnover: object | None = None) -> dict[str, object]:
    portfolio: dict[str, object] = {}
    if max_turnover is not None:
        portfolio["max_turnover"] = max_turnover
    return {
        "version": "1",
        "data": {
            "date_column": "Date",
            "frequency": "M",
            "risk_free_column": None,
            "allow_risk_free_fallback": True,
        },
        "preprocessing": {},
        "vol_adjust": {},
        "sample_split": {},
        "portfolio": portfolio,
        "metrics": {},
        "export": {},
        "run": {},
        "benchmarks": {},
    }


def _scenario() -> MonteCarloScenario:
    return MonteCarloScenario(
        name="diagnostic_test",
        base_config="base.yml",
        monte_carlo={
            "mode": "two_layer",
            "n_paths": 1,
            "horizon_years": 1.0,
            "frequency": "M",
        },
        return_model={"kind": "stationary_bootstrap"},
        enable_fold_runs=False,
    )


def test_runner_records_turnover_and_binding(monkeypatch) -> None:
    dates = pd.date_range("2021-01-31", periods=2, freq="ME")
    returns = pd.DataFrame({"Date": dates, "Asset": [0.01, 0.02]})
    turnover = pd.Series([0.1, 0.25], index=dates, name="turnover")
    out_scaled = pd.DataFrame({"Asset": [0.01, 0.02]}, index=dates)
    metrics = pd.DataFrame({"cagr": [0.1]}, index=["user_weight"])
    run_result = RunResult(
        metrics=metrics,
        details={"out_sample_scaled": out_scaled},
        seed=0,
        environment={},
        turnover=turnover,
    )

    def _fake_run_simulation(*_args, **_kwargs):
        return run_result

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.run_simulation",
        _fake_run_simulation,
    )

    runner = MonteCarloRunner(_scenario(), base_config=_base_config(max_turnover=0.2))
    context = _PathContext(
        path_id=0,
        prices=pd.DataFrame(),
        returns=returns,
        score_frame=pd.DataFrame(),
        path_hash="hash",
        seed=123,
    )

    evaluation = runner._evaluate_strategy(StrategyVariant(name="base"), context)
    diagnostic = evaluation.diagnostic or {}

    pdt.assert_series_equal(diagnostic["turnover"], turnover)
    expected_binding = pd.Series(
        [False, True],
        index=dates,
        name="turnover_cap_binding",
    )
    pdt.assert_series_equal(diagnostic["turnover_cap_binding"], expected_binding)
    pdt.assert_series_equal(evaluation.turnover, turnover)
    pdt.assert_series_equal(evaluation.turnover_cap_binding, expected_binding)


def test_runner_records_turnover_per_path(monkeypatch) -> None:
    dates = pd.date_range("2021-01-31", periods=2, freq="ME")

    def _fake_generate_path_context(
        self,
        *,
        path_id: int,
        fold_id: int | None = None,
        **_kwargs,
    ) -> _PathContext:
        if path_id == 0:
            values = [0.01, 0.02]
        else:
            values = [0.03, 0.04]
        returns = pd.DataFrame({"Date": dates, "Asset": values})
        return _PathContext(
            fold_id=fold_id,
            path_id=path_id,
            prices=pd.DataFrame(),
            returns=returns,
            score_frame=pd.DataFrame(),
            path_hash=f"hash-{path_id}",
            seed=path_id,
        )

    def _fake_run_simulation(*_args, **_kwargs):
        returns = _kwargs.get("returns")
        if returns is None and len(_args) > 1:
            returns = _args[1]
        series = pd.Series(
            returns["Asset"].to_numpy(),
            index=pd.to_datetime(returns["Date"].values),
            name="turnover",
        )
        out_scaled = pd.DataFrame(
            {"Asset": returns["Asset"].to_numpy()}, index=series.index
        )
        metrics = pd.DataFrame({"cagr": [0.1]}, index=["user_weight"])
        return RunResult(
            metrics=metrics,
            details={"out_sample_scaled": out_scaled},
            seed=0,
            environment={},
            turnover=series,
        )

    monkeypatch.setattr(
        MonteCarloRunner, "_generate_path_context", _fake_generate_path_context
    )
    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.run_simulation",
        _fake_run_simulation,
    )

    runner = MonteCarloRunner(_scenario(), base_config=_base_config(max_turnover=1.0))
    strategies = [StrategyVariant(name="base")]
    evaluations, errors = runner._run_two_layer(
        model=object(),
        n_periods=2,
        strategies=strategies,
        path_seeds=[1, 2],
        progress_callback=None,
        jobs=1,
        fold_id=None,
    )

    assert not errors
    assert len(evaluations) == 2
    evaluation_by_path = {evaluation.path_id: evaluation for evaluation in evaluations}
    expected_a = pd.Series([0.01, 0.02], index=dates, name="turnover")
    expected_b = pd.Series([0.03, 0.04], index=dates, name="turnover")
    pdt.assert_series_equal(
        evaluation_by_path[0].turnover, expected_a, check_freq=False
    )
    pdt.assert_series_equal(
        evaluation_by_path[1].turnover, expected_b, check_freq=False
    )


def test_runner_resolves_regime_turnover_caps(monkeypatch) -> None:
    dates = pd.date_range("2021-01-31", periods=3, freq="ME")
    returns = pd.DataFrame({"Date": dates, "Asset": [0.01, 0.02, 0.03]})
    turnover = pd.Series([0.1, 0.08, 0.2], index=dates, name="turnover")
    regimes = pd.Series(["risk_on", "risk_off", "unknown"], index=dates, name="regime")
    out_scaled = pd.DataFrame({"Asset": [0.01, 0.02, 0.03]}, index=dates)
    metrics = pd.DataFrame({"cagr": [0.1]}, index=["user_weight"])
    run_result = RunResult(
        metrics=metrics,
        details={"out_sample_scaled": out_scaled, "regime_labels_out": regimes},
        seed=0,
        environment={},
        turnover=turnover,
    )
    captured: dict[str, object] = {}

    def _fake_run_simulation(config, *_args, **_kwargs):
        captured["max_turnover"] = config.portfolio.get("max_turnover")
        return run_result

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.run_simulation",
        _fake_run_simulation,
    )

    runner = MonteCarloRunner(
        _scenario(),
        base_config=_base_config(max_turnover={"risk_on": 0.15, "risk_off": 0.05}),
    )
    context = _PathContext(
        path_id=0,
        prices=pd.DataFrame(),
        returns=returns,
        score_frame=pd.DataFrame(),
        path_hash="hash",
        seed=123,
    )

    evaluation = runner._evaluate_strategy(StrategyVariant(name="base"), context)
    diagnostic = evaluation.diagnostic or {}

    assert captured["max_turnover"] == {"risk_on": 0.15, "risk_off": 0.05}
    expected_binding = pd.Series(
        [False, True, False],
        index=dates,
        name="turnover_cap_binding",
    )
    pdt.assert_series_equal(diagnostic["turnover_cap_binding"], expected_binding)


def test_runner_uses_risk_diagnostics_turnover_value(monkeypatch) -> None:
    dates = pd.date_range("2021-01-31", periods=2, freq="ME")
    returns = pd.DataFrame({"Date": dates, "Asset": [0.01, 0.02]})
    out_scaled = pd.DataFrame({"Asset": [0.01, 0.02]}, index=dates)
    metrics = pd.DataFrame({"cagr": [0.1]}, index=["user_weight"])
    run_result = RunResult(
        metrics=metrics,
        details={
            "out_sample_scaled": out_scaled,
            "risk_diagnostics": {"turnover_value": 0.3},
        },
        seed=0,
        environment={},
    )

    def _fake_run_simulation(*_args, **_kwargs):
        return run_result

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.run_simulation",
        _fake_run_simulation,
    )

    runner = MonteCarloRunner(_scenario(), base_config=_base_config(max_turnover=0.25))
    context = _PathContext(
        path_id=0,
        prices=pd.DataFrame(),
        returns=returns,
        score_frame=pd.DataFrame(),
        path_hash="hash",
        seed=123,
    )

    evaluation = runner._evaluate_strategy(StrategyVariant(name="base"), context)
    diagnostic = evaluation.diagnostic or {}

    expected_turnover = pd.Series([0.3, 0.3], index=dates, name="turnover")
    expected_binding = pd.Series(
        [True, True],
        index=dates,
        name="turnover_cap_binding",
    )
    pdt.assert_series_equal(diagnostic["turnover"], expected_turnover)
    pdt.assert_series_equal(diagnostic["turnover_cap_binding"], expected_binding)


def test_runner_expands_turnover_series_across_periods(monkeypatch) -> None:
    dates = pd.date_range("2021-01-31", periods=3, freq="ME")
    returns = pd.DataFrame({"Date": dates, "Asset": [0.01, 0.02, 0.03]})
    turnover = pd.Series([0.12], index=[dates[0]], name="turnover")
    out_scaled = pd.DataFrame({"Asset": [0.01, 0.02, 0.03]}, index=dates)
    metrics = pd.DataFrame({"cagr": [0.1]}, index=["user_weight"])
    run_result = RunResult(
        metrics=metrics,
        details={"out_sample_scaled": out_scaled},
        seed=0,
        environment={},
        turnover=turnover,
    )

    def _fake_run_simulation(*_args, **_kwargs):
        return run_result

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.run_simulation",
        _fake_run_simulation,
    )

    runner = MonteCarloRunner(_scenario(), base_config=_base_config(max_turnover=0.2))
    context = _PathContext(
        path_id=1,
        prices=pd.DataFrame(),
        returns=returns,
        score_frame=pd.DataFrame(),
        path_hash="hash",
        seed=321,
    )

    evaluation = runner._evaluate_strategy(StrategyVariant(name="base"), context)
    diagnostic = evaluation.diagnostic or {}

    expected_turnover = pd.Series([0.12, 0.12, 0.12], index=dates, name="turnover")
    pdt.assert_series_equal(diagnostic["turnover"], expected_turnover)


def test_runner_expands_scalar_turnover_from_date_index(monkeypatch) -> None:
    dates = pd.date_range("2021-05-31", periods=3, freq="ME")
    returns = pd.DataFrame({"Date": dates, "Asset": [0.01, 0.02, 0.03]})
    metrics = pd.DataFrame({"cagr": [0.1]}, index=["user_weight"])
    run_result = RunResult(
        metrics=metrics,
        details={},
        seed=0,
        environment={},
        turnover=0.2,
    )

    def _fake_run_simulation(*_args, **_kwargs):
        return run_result

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.run_simulation",
        _fake_run_simulation,
    )

    runner = MonteCarloRunner(_scenario(), base_config=_base_config(max_turnover=0.15))
    context = _PathContext(
        path_id=2,
        prices=pd.DataFrame(),
        returns=returns,
        score_frame=pd.DataFrame(),
        path_hash="hash",
        seed=999,
    )

    evaluation = runner._evaluate_strategy(StrategyVariant(name="base"), context)
    diagnostic = evaluation.diagnostic or {}

    expected_turnover = pd.Series([0.2, 0.2, 0.2], index=dates, name="turnover")
    expected_turnover.index.name = "Date"
    expected_binding = pd.Series(
        [True, True, True],
        index=dates,
        name="turnover_cap_binding",
    )
    expected_binding.index.name = "Date"
    pdt.assert_series_equal(diagnostic["turnover"], expected_turnover, check_freq=False)
    pdt.assert_series_equal(
        diagnostic["turnover_cap_binding"], expected_binding, check_freq=False
    )
    pdt.assert_series_equal(evaluation.turnover, expected_turnover, check_freq=False)
    pdt.assert_series_equal(
        evaluation.turnover_cap_binding, expected_binding, check_freq=False
    )


def test_results_include_turnover_binding_diagnostics(monkeypatch) -> None:
    dates = pd.date_range("2021-01-31", periods=2, freq="ME")
    turnover = pd.Series([0.1, 0.25], index=dates, name="turnover")
    binding = pd.Series([False, True], index=dates, name="turnover_cap_binding")
    evaluation = StrategyEvaluation(
        fold_id=None,
        path_id=0,
        strategy_name="base",
        metrics={"cagr": 0.1},
        metric_source="metrics",
        path_hash="hash",
        seed=123,
        diagnostic={"turnover": turnover, "turnover_cap_binding": binding},
    )

    def _fake_run_mode(*_args, **_kwargs):
        return [evaluation], []

    def _fake_build_price_model(*_args, **_kwargs):
        return object()

    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)
    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)

    history = pd.DataFrame({"Asset": [100.0, 101.0]}, index=dates)
    runner = MonteCarloRunner(
        _scenario(), base_config=_base_config(), price_history=history
    )
    results = runner.run()

    diagnostics = results.diagnostics_frame
    assert diagnostics is not None

    expected = pd.DataFrame(
        {
            "fold_id": [None, None],
            "path_id": [0, 0],
            "strategy": ["base", "base"],
            "path_hash": ["hash", "hash"],
            "seed": [123, 123],
            "period": list(dates),
            "turnover": [0.1, 0.25],
            "turnover_cap_binding": [False, True],
        }
    )
    pdt.assert_frame_equal(diagnostics.reset_index(drop=True), expected)


def test_runner_diagnostics_frame_reports_binding(monkeypatch) -> None:
    dates = pd.date_range("2021-04-30", periods=2, freq="ME")
    returns = pd.DataFrame({"Date": dates, "Asset": [0.01, 0.02]})
    turnover = pd.Series([0.1, 0.3], index=dates, name="turnover")
    out_scaled = pd.DataFrame({"Asset": [0.01, 0.02]}, index=dates)
    metrics = pd.DataFrame({"cagr": [0.1]}, index=["user_weight"])
    run_result = RunResult(
        metrics=metrics,
        details={"out_sample_scaled": out_scaled},
        seed=0,
        environment={},
        turnover=turnover,
    )

    def _fake_run_simulation(*_args, **_kwargs):
        return run_result

    monkeypatch.setattr(
        "trend_analysis.monte_carlo.runner.run_simulation",
        _fake_run_simulation,
    )

    runner = MonteCarloRunner(_scenario(), base_config=_base_config(max_turnover=0.2))
    context = _PathContext(
        path_id=0,
        prices=pd.DataFrame(),
        returns=returns,
        score_frame=pd.DataFrame(),
        path_hash="hash",
        seed=123,
    )

    evaluation = runner._evaluate_strategy(StrategyVariant(name="base"), context)
    diagnostics = build_diagnostics_frame([evaluation]).reset_index(drop=True)

    expected = pd.DataFrame(
        {
            "fold_id": [None, None],
            "path_id": [0, 0],
            "strategy": ["base", "base"],
            "path_hash": ["hash", "hash"],
            "seed": [123, 123],
            "period": list(dates),
            "turnover": [0.1, 0.3],
            "turnover_cap_binding": [False, True],
        }
    )
    pdt.assert_frame_equal(diagnostics, expected)


def test_results_include_binding_for_multiple_paths(monkeypatch) -> None:
    dates = pd.date_range("2021-02-28", periods=2, freq="ME")
    turnover_a = pd.Series([0.05, 0.15], index=dates, name="turnover")
    binding_a = pd.Series([False, True], index=dates, name="turnover_cap_binding")
    turnover_b = pd.Series([0.08, 0.04], index=dates, name="turnover")
    binding_b = pd.Series([True, False], index=dates, name="turnover_cap_binding")
    evaluation_a = StrategyEvaluation(
        fold_id=None,
        path_id=0,
        strategy_name="base",
        metrics={"cagr": 0.1},
        metric_source="metrics",
        path_hash="hash-a",
        seed=123,
        diagnostic={"turnover": turnover_a, "turnover_cap_binding": binding_a},
    )
    evaluation_b = StrategyEvaluation(
        fold_id=None,
        path_id=1,
        strategy_name="alt",
        metrics={"cagr": 0.2},
        metric_source="metrics",
        path_hash="hash-b",
        seed=321,
        diagnostic={"turnover": turnover_b, "turnover_cap_binding": binding_b},
    )

    def _fake_run_mode(*_args, **_kwargs):
        return [evaluation_a, evaluation_b], []

    def _fake_build_price_model(*_args, **_kwargs):
        return object()

    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)
    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)

    history = pd.DataFrame({"Asset": [100.0, 101.0]}, index=dates)
    runner = MonteCarloRunner(
        _scenario(), base_config=_base_config(), price_history=history
    )
    results = runner.run()

    diagnostics = results.diagnostics_frame
    assert diagnostics is not None

    expected = pd.DataFrame(
        {
            "fold_id": [None, None, None, None],
            "path_id": [0, 0, 1, 1],
            "strategy": ["base", "base", "alt", "alt"],
            "path_hash": ["hash-a", "hash-a", "hash-b", "hash-b"],
            "seed": [123, 123, 321, 321],
            "period": list(dates) + list(dates),
            "turnover": [0.05, 0.15, 0.08, 0.04],
            "turnover_cap_binding": [False, True, True, False],
        }
    )
    pdt.assert_frame_equal(diagnostics.reset_index(drop=True), expected)


def test_results_expose_turnover_series_on_evaluations(monkeypatch) -> None:
    dates = pd.date_range("2021-01-31", periods=2, freq="ME")
    turnover = pd.Series([0.12, 0.18], index=dates, name="turnover")
    evaluation = StrategyEvaluation(
        fold_id=None,
        path_id=0,
        strategy_name="base",
        metrics={"cagr": 0.1},
        metric_source="metrics",
        path_hash="hash",
        seed=123,
        diagnostic={"turnover": turnover},
    )

    def _fake_run_mode(*_args, **_kwargs):
        return [evaluation], []

    def _fake_build_price_model(*_args, **_kwargs):
        return object()

    monkeypatch.setattr(MonteCarloRunner, "_run_mode", _fake_run_mode)
    monkeypatch.setattr(MonteCarloRunner, "_build_price_model", _fake_build_price_model)

    history = pd.DataFrame({"Asset": [100.0, 101.0]}, index=dates)
    runner = MonteCarloRunner(
        _scenario(), base_config=_base_config(), price_history=history
    )
    results = runner.run()

    assert results.evaluations
    diagnostic = results.evaluations[0].diagnostic or {}
    pdt.assert_series_equal(diagnostic["turnover"], turnover)


def test_build_diagnostics_frame_expands_binding_indicator() -> None:
    dates = pd.date_range("2021-01-31", periods=3, freq="ME")
    turnover = pd.Series([0.05, 0.2, 0.1], index=dates, name="turnover")
    evaluation = StrategyEvaluation(
        fold_id=None,
        path_id=1,
        strategy_name="base",
        metrics={"cagr": 0.1},
        metric_source="metrics",
        path_hash="hash",
        seed=7,
        diagnostic={"turnover": turnover, "turnover_cap_binding": True},
    )

    diagnostics = build_diagnostics_frame([evaluation])

    expected = pd.DataFrame(
        {
            "fold_id": [None, None, None],
            "path_id": [1, 1, 1],
            "strategy": ["base", "base", "base"],
            "path_hash": ["hash", "hash", "hash"],
            "seed": [7, 7, 7],
            "period": list(dates),
            "turnover": [0.05, 0.2, 0.1],
            "turnover_cap_binding": [True, True, True],
        }
    )
    pdt.assert_frame_equal(diagnostics.reset_index(drop=True), expected)


def test_build_diagnostics_frame_preserves_fold_id_and_binding() -> None:
    dates = pd.date_range("2021-03-31", periods=2, freq="ME")
    turnover = pd.Series([0.08, 0.12], index=dates, name="turnover")
    binding = pd.Series([True, False], index=dates, name="turnover_cap_binding")
    evaluation = StrategyEvaluation(
        fold_id=1,
        path_id=7,
        strategy_name="base",
        metrics={"cagr": 0.1},
        metric_source="metrics",
        path_hash="hash",
        seed=21,
        diagnostic={"turnover": turnover, "turnover_cap_binding": binding},
    )

    diagnostics = build_diagnostics_frame([evaluation])

    expected = pd.DataFrame(
        {
            "fold_id": [1, 1],
            "path_id": [7, 7],
            "strategy": ["base", "base"],
            "path_hash": ["hash", "hash"],
            "seed": [21, 21],
            "period": list(dates),
            "turnover": [0.08, 0.12],
            "turnover_cap_binding": [True, False],
        }
    )
    pdt.assert_frame_equal(diagnostics.reset_index(drop=True), expected)


def test_build_diagnostics_frame_includes_binding_without_turnover() -> None:
    dates = pd.date_range("2022-01-31", periods=2, freq="ME")
    binding = pd.Series([True, False], index=dates, name="turnover_cap_binding")
    evaluation = StrategyEvaluation(
        fold_id=None,
        path_id=2,
        strategy_name="base",
        metrics={"cagr": 0.1},
        metric_source="metrics",
        path_hash="hash",
        seed=11,
        diagnostic={"turnover_cap_binding": binding},
    )

    diagnostics = build_diagnostics_frame([evaluation])

    expected = pd.DataFrame(
        {
            "fold_id": [None, None],
            "path_id": [2, 2],
            "strategy": ["base", "base"],
            "path_hash": ["hash", "hash"],
            "seed": [11, 11],
            "period": list(dates),
            "turnover": [None, None],
            "turnover_cap_binding": [True, False],
        }
    )
    pdt.assert_frame_equal(diagnostics.reset_index(drop=True), expected)


def test_build_diagnostics_frame_includes_turnover_without_binding() -> None:
    dates = pd.date_range("2022-06-30", periods=2, freq="ME")
    turnover = pd.Series([0.11, 0.09], index=dates, name="turnover")
    evaluation = StrategyEvaluation(
        fold_id=None,
        path_id=4,
        strategy_name="base",
        metrics={"cagr": 0.1},
        metric_source="metrics",
        path_hash="hash",
        seed=17,
        diagnostic={"turnover": turnover},
    )

    diagnostics = build_diagnostics_frame([evaluation])

    expected = pd.DataFrame(
        {
            "fold_id": [None, None],
            "path_id": [4, 4],
            "strategy": ["base", "base"],
            "path_hash": ["hash", "hash"],
            "seed": [17, 17],
            "period": list(dates),
            "turnover": [0.11, 0.09],
            "turnover_cap_binding": [None, None],
        }
    )
    pdt.assert_frame_equal(diagnostics.reset_index(drop=True), expected)


def test_build_diagnostics_frame_includes_binding_for_multiple_paths() -> None:
    dates = pd.date_range("2023-01-31", periods=2, freq="ME")
    turnover_a = pd.Series([0.05, 0.1], index=dates, name="turnover")
    turnover_b = pd.Series([0.2, 0.3], index=dates, name="turnover")
    evaluation_a = StrategyEvaluation(
        fold_id=None,
        path_id=0,
        strategy_name="base",
        metrics={"cagr": 0.1},
        metric_source="metrics",
        path_hash="hash-a",
        seed=5,
        diagnostic={"turnover": turnover_a, "turnover_cap_binding": False},
    )
    evaluation_b = StrategyEvaluation(
        fold_id=None,
        path_id=1,
        strategy_name="alt",
        metrics={"cagr": 0.2},
        metric_source="metrics",
        path_hash="hash-b",
        seed=6,
        diagnostic={
            "turnover": turnover_b,
            "turnover_cap_binding": pd.Series([True, False], index=dates),
        },
    )

    diagnostics = build_diagnostics_frame([evaluation_a, evaluation_b])

    expected = pd.DataFrame(
        {
            "fold_id": [None, None, None, None],
            "path_id": [0, 0, 1, 1],
            "strategy": ["base", "base", "alt", "alt"],
            "path_hash": ["hash-a", "hash-a", "hash-b", "hash-b"],
            "seed": [5, 5, 6, 6],
            "period": list(dates) + list(dates),
            "turnover": [0.05, 0.1, 0.2, 0.3],
            "turnover_cap_binding": [False, False, True, False],
        }
    )
    pdt.assert_frame_equal(diagnostics.reset_index(drop=True), expected)


def test_build_diagnostics_frame_uses_evaluation_fields() -> None:
    dates = pd.date_range("2024-06-30", periods=2, freq="ME")
    turnover = pd.Series([0.07, 0.09], index=dates, name="turnover")
    binding = pd.Series([False, True], index=dates, name="turnover_cap_binding")
    evaluation = StrategyEvaluation(
        fold_id=None,
        path_id=9,
        strategy_name="base",
        metrics={"cagr": 0.1},
        metric_source="metrics",
        path_hash="hash",
        seed=19,
        turnover=turnover,
        turnover_cap_binding=binding,
    )

    diagnostics = build_diagnostics_frame([evaluation])

    expected = pd.DataFrame(
        {
            "fold_id": [None, None],
            "path_id": [9, 9],
            "strategy": ["base", "base"],
            "path_hash": ["hash", "hash"],
            "seed": [19, 19],
            "period": list(dates),
            "turnover": [0.07, 0.09],
            "turnover_cap_binding": [False, True],
        }
    )
    pdt.assert_frame_equal(diagnostics.reset_index(drop=True), expected)


def test_build_diagnostics_frame_uses_evaluation_binding_with_diagnostic_turnover() -> (
    None
):
    dates = pd.date_range("2024-09-30", periods=2, freq="ME")
    turnover = pd.Series([0.12, 0.18], index=dates, name="turnover")
    binding = pd.Series([True, False], index=dates, name="turnover_cap_binding")
    evaluation = StrategyEvaluation(
        fold_id=None,
        path_id=10,
        strategy_name="base",
        metrics={"cagr": 0.1},
        metric_source="metrics",
        path_hash="hash",
        seed=29,
        diagnostic={"turnover": turnover},
        turnover_cap_binding=binding,
    )

    diagnostics = build_diagnostics_frame([evaluation])

    expected = pd.DataFrame(
        {
            "fold_id": [None, None],
            "path_id": [10, 10],
            "strategy": ["base", "base"],
            "path_hash": ["hash", "hash"],
            "seed": [29, 29],
            "period": list(dates),
            "turnover": [0.12, 0.18],
            "turnover_cap_binding": [True, False],
        }
    )
    pdt.assert_frame_equal(diagnostics.reset_index(drop=True), expected)


def test_build_diagnostics_frame_expands_scalar_turnover_with_binding_series() -> None:
    dates = pd.date_range("2024-01-31", periods=2, freq="ME")
    binding = pd.Series([True, False], index=dates, name="turnover_cap_binding")
    evaluation = StrategyEvaluation(
        fold_id=None,
        path_id=3,
        strategy_name="base",
        metrics={"cagr": 0.1},
        metric_source="metrics",
        path_hash="hash",
        seed=13,
        diagnostic={"turnover": 0.2, "turnover_cap_binding": binding},
    )

    diagnostics = build_diagnostics_frame([evaluation])

    expected = pd.DataFrame(
        {
            "fold_id": [None, None],
            "path_id": [3, 3],
            "strategy": ["base", "base"],
            "path_hash": ["hash", "hash"],
            "seed": [13, 13],
            "period": list(dates),
            "turnover": [0.2, 0.2],
            "turnover_cap_binding": [True, False],
        }
    )
    pdt.assert_frame_equal(diagnostics.reset_index(drop=True), expected)
