from __future__ import annotations

from pathlib import Path

import yaml

from trend_analysis.monte_carlo import MonteCarloScenario, MonteCarloSettings


def test_monte_carlo_settings_validates_and_normalizes() -> None:
    settings = MonteCarloSettings(
        mode="Two_Layer",
        n_paths=250,
        horizon_years=7.5,
        frequency="m",
        seed=42,
        jobs=4,
    )

    assert settings.mode == "two_layer"
    assert settings.n_paths == 250
    assert settings.horizon_years == 7.5
    assert settings.frequency == "M"
    assert settings.seed == 42
    assert settings.jobs == 4


def test_monte_carlo_scenario_accepts_valid_config() -> None:
    settings = MonteCarloSettings(
        mode="mixture",
        n_paths=100,
        horizon_years=5,
        frequency="Q",
        seed=None,
        jobs=None,
    )

    scenario = MonteCarloScenario(
        name="demo_scenario",
        description="Demo scenario",
        base_config="config/defaults.yml",
        monte_carlo=settings,
        return_model={"kind": "stationary_bootstrap"},
        strategy_set={"curated": []},
        folds={"enabled": False},
        outputs={"directory": "outputs/monte_carlo/demo"},
    )

    assert scenario.monte_carlo is settings
    assert scenario.return_model["kind"] == "stationary_bootstrap"


def test_monte_carlo_scenario_builds_nested_configs_from_mappings() -> None:
    scenario = MonteCarloScenario(
        name="nested_demo",
        description="Nested mapping inputs",
        base_config="config/defaults.yml",
        monte_carlo={
            "mode": "two_layer",
            "n_paths": 150,
            "horizon_years": 3.0,
            "frequency": "q",
            "seed": 7,
            "jobs": 2,
        },
        return_model={"kind": "stationary_bootstrap", "params": {"block_size": 12}},
        strategy_set={"curated": ["trend_basic"], "guards": {"max_turnover": 0.2}},
        folds={"enabled": True, "n_folds": 4},
        outputs={"directory": "outputs/monte_carlo/nested", "format": "parquet"},
    )

    assert isinstance(scenario.monte_carlo, MonteCarloSettings)
    assert scenario.monte_carlo.mode == "two_layer"
    assert scenario.monte_carlo.frequency == "Q"
    assert scenario.return_model["params"]["block_size"] == 12
    assert scenario.strategy_set["guards"]["max_turnover"] == 0.2
    assert scenario.folds["n_folds"] == 4
    assert scenario.outputs["format"] == "parquet"


def test_monte_carlo_scenario_validates_full_schema_from_yaml() -> None:
    payload = yaml.safe_load("""
name: example_scenario
description: Example schema payload for validation
base_config: config/defaults.yml
monte_carlo:
  mode: mixture
  n_paths: 500
  horizon_years: 4.0
  frequency: M
  seed: 123
  jobs: 8
return_model:
  kind: stationary_bootstrap
  params:
    block_size: 6
strategy_set:
  curated:
    - trend_basic
  guards:
    max_turnover: 0.15
folds:
  enabled: true
  n_folds: 3
outputs:
  directory: outputs/monte_carlo/example
  format: parquet
""")

    scenario = MonteCarloScenario(**payload)

    assert scenario.name == "example_scenario"
    assert scenario.monte_carlo.n_paths == 500
    assert scenario.monte_carlo.frequency == "M"
    assert scenario.return_model["params"]["block_size"] == 6
    assert scenario.strategy_set["guards"]["max_turnover"] == 0.15
    assert scenario.folds["n_folds"] == 3
    assert scenario.outputs["directory"] == "outputs/monte_carlo/example"


def test_example_scenario_file_loads_and_validates() -> None:
    root = Path(__file__).resolve().parents[2]
    scenario_path = root / "config" / "scenarios" / "monte_carlo" / "example.yml"

    payload = yaml.safe_load(scenario_path.read_text())
    scenario = MonteCarloScenario(**payload)

    assert scenario.name == "example_scenario"
    assert scenario.monte_carlo.n_paths == 500
