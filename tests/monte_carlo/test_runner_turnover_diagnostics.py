from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from trend_analysis.api import RunResult
from trend_analysis.monte_carlo.results import StrategyEvaluation
from trend_analysis.monte_carlo.runner import MonteCarloRunner, _PathContext
from trend_analysis.monte_carlo.scenario import MonteCarloScenario
from trend_analysis.monte_carlo.strategy import StrategyVariant


def _base_config(max_turnover: float | None = None) -> dict[str, object]:
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
    runner = MonteCarloRunner(_scenario(), base_config=_base_config(), price_history=history)
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
