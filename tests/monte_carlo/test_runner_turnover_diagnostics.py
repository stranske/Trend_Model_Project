from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from trend_analysis.api import RunResult
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
