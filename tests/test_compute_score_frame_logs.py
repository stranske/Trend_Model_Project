from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from trend_analysis.monte_carlo.runner import MonteCarloRunner
from trend_analysis.monte_carlo.scenario import MonteCarloScenario
from trend_analysis.monte_carlo.strategy import StrategyVariant


def _scenario() -> MonteCarloScenario:
    return MonteCarloScenario(
        name="invalid_config_log_test",
        base_config="config/defaults.yml",
        monte_carlo={
            "mode": "two_layer",
            "n_paths": 2,
            "horizon_years": 1.0,
            "frequency": "M",
            "seed": 123,
            "jobs": 1,
        },
        strategy_set={"curated": [StrategyVariant(name="StrategyA")]},
        return_model={"kind": "stationary_bootstrap", "params": {"block_size": 3}},
    )


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


def _returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "AssetA": [0.02, 0.01, 0.03, -0.01, 0.01, 0.02],
            "AssetB": [0.005, 0.004, 0.006, 0.004, 0.005, 0.004],
        }
    )


def test_compute_score_frame_logs_invalid_base_config(caplog) -> None:
    config = _base_config()
    config["sample_split"] = None
    runner = MonteCarloRunner(_scenario(), base_config=config)

    with caplog.at_level(logging.WARNING, logger="trend_analysis.monte_carlo"):
        frame = runner._compute_score_frame(_returns())

    assert frame.empty
    assert "Invalid Monte Carlo base config; score frame omitted" in caplog.text
