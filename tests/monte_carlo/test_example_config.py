from __future__ import annotations

from pathlib import Path
from typing import Mapping

from trend_analysis.monte_carlo import MonteCarloScenario, MonteCarloSettings, load_scenario


def test_example_config_validates_against_schema() -> None:
    scenario = load_scenario("example_scenario")

    assert scenario.name == "example_scenario"
    assert scenario.description == "Example Monte Carlo scenario configuration."
    assert isinstance(scenario.monte_carlo, MonteCarloSettings)
    assert isinstance(scenario.monte_carlo.n_paths, int)
    assert scenario.monte_carlo.n_paths >= 1
    assert isinstance(scenario.monte_carlo.horizon_years, float)
    assert scenario.monte_carlo.horizon_years > 0.0
    assert scenario.monte_carlo.frequency in {"D", "W", "M", "Q", "Y"}
    assert isinstance(scenario.base_config, Path)
    assert scenario.return_model is not None
    assert isinstance(scenario.return_model, Mapping)
    assert scenario.return_model.get("kind") == "stationary_bootstrap"
    assert scenario.folds is not None
    assert isinstance(scenario.folds, Mapping)
    assert scenario.folds.get("enabled") is True
    assert scenario.outputs is not None
    assert isinstance(scenario.outputs, Mapping)
    assert scenario.outputs.get("format") == "parquet"
