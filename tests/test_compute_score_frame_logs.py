from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pytest

from trend_analysis.monte_carlo.runner import MonteCarloRunner
from trend_analysis.monte_carlo.scenario import MonteCarloScenario
from trend_analysis.monte_carlo.strategy import StrategyVariant


def _scenario() -> MonteCarloScenario:
    return MonteCarloScenario(
        name="score_frame_logging",
        base_config="config/defaults.yml",
        monte_carlo={
            "mode": "two_layer",
            "n_paths": 1,
            "horizon_years": 1.0,
            "frequency": "M",
            "seed": 1,
            "jobs": 1,
        },
        strategy_set={"curated": [StrategyVariant(name="StrategyA")]},
        return_model={"kind": "stationary_bootstrap", "params": {"block_size": 3}},
    )


def _invalid_base_config() -> dict[str, Any]:
    return {
        "version": "0.1.0",
        "data": [],
        "preprocessing": {},
        "vol_adjust": {"enabled": False},
        "sample_split": {},
        "portfolio": {},
        "metrics": {},
        "export": {},
        "run": {},
        "benchmarks": {},
    }


def test_compute_score_frame_logs_invalid_base_config(caplog: pytest.LogCaptureFixture) -> None:
    runner = MonteCarloRunner(_scenario(), base_config=_invalid_base_config())
    returns = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-31", periods=3, freq="ME"),
            "AssetA": [0.01, 0.02, -0.01],
        }
    )

    with caplog.at_level(logging.WARNING, logger="trend_analysis.monte_carlo"):
        frame = runner._compute_score_frame(returns)

    assert frame.empty
    assert "Failed to parse base config for score frame" in caplog.text
    assert "data must be a dictionary" in caplog.text
