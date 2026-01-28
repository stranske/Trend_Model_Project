from __future__ import annotations

from pathlib import Path

import pytest
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
    if "scenario" in payload:
        scenario_meta = payload.pop("scenario")
        payload = {**scenario_meta, **payload}
    scenario = MonteCarloScenario(**payload)

    assert scenario.name == "example_scenario"
    assert scenario.monte_carlo.n_paths == 500


def test_example_scenario_file_invalid_monte_carlo_raises_clear_error() -> None:
    root = Path(__file__).resolve().parents[2]
    scenario_path = root / "config" / "scenarios" / "monte_carlo" / "example.yml"

    payload = yaml.safe_load(scenario_path.read_text())
    if "scenario" in payload:
        scenario_meta = payload.pop("scenario")
        payload = {**scenario_meta, **payload}
    payload["monte_carlo"]["n_paths"] = 0

    with pytest.raises(ValueError, match="monte_carlo.n_paths must be >= 1"):
        MonteCarloScenario(**payload)


def test_example_scenario_file_missing_base_config_raises_clear_error() -> None:
    root = Path(__file__).resolve().parents[2]
    scenario_path = root / "config" / "scenarios" / "monte_carlo" / "example.yml"

    payload = yaml.safe_load(scenario_path.read_text())
    if "scenario" in payload:
        scenario_meta = payload.pop("scenario")
        payload = {**scenario_meta, **payload}
    payload.pop("base_config", None)

    with pytest.raises(ValueError, match="base_config is required"):
        MonteCarloScenario(**payload)


def test_monte_carlo_settings_missing_required_fields_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="mode is required"):
        MonteCarloSettings()


@pytest.mark.parametrize(
    ("field", "payload", "message"),
    [
        (
            "mode",
            {"mode": "invalid", "n_paths": 10, "horizon_years": 1.0, "frequency": "M"},
            "mode must be one of",
        ),
        (
            "n_paths",
            {"mode": "mixture", "n_paths": 0, "horizon_years": 1.0, "frequency": "M"},
            "n_paths must be >= 1",
        ),
        (
            "horizon_years",
            {"mode": "mixture", "n_paths": 10, "horizon_years": 0.0, "frequency": "M"},
            "horizon_years must be > 0",
        ),
        (
            "frequency",
            {"mode": "mixture", "n_paths": 10, "horizon_years": 1.0, "frequency": "Z"},
            "frequency must be one of",
        ),
        (
            "seed",
            {"mode": "mixture", "n_paths": 10, "horizon_years": 1.0, "frequency": "M", "seed": -1},
            "seed must be >= 0",
        ),
        (
            "jobs",
            {"mode": "mixture", "n_paths": 10, "horizon_years": 1.0, "frequency": "M", "jobs": 0},
            "jobs must be >= 1",
        ),
    ],
)
def test_monte_carlo_settings_invalid_fields_raise_clear_errors(
    field: str, payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MonteCarloSettings(**payload)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -float("inf"), "nan", "inf", "-inf"],
)
def test_monte_carlo_settings_rejects_non_finite_horizon_years(value: object) -> None:
    with pytest.raises(ValueError, match="horizon_years must be a finite number"):
        MonteCarloSettings(
            mode="mixture",
            n_paths=10,
            horizon_years=value,
            frequency="M",
        )


def test_monte_carlo_settings_coerces_individual_fields() -> None:
    settings = MonteCarloSettings(
        mode=" mixture ",
        n_paths="250",
        horizon_years="2.5",
        frequency=" q ",
        seed="12",
        jobs="3",
    )

    assert settings.mode == "mixture"
    assert settings.n_paths == 250
    assert settings.horizon_years == 2.5
    assert settings.frequency == "Q"
    assert settings.seed == 12
    assert settings.jobs == 3


def test_monte_carlo_settings_allows_optional_seed_and_jobs() -> None:
    settings = MonteCarloSettings(
        mode="two_layer",
        n_paths=10,
        horizon_years=1.0,
        frequency="M",
        seed=None,
        jobs=None,
    )

    assert settings.seed is None
    assert settings.jobs is None


def test_monte_carlo_scenario_missing_required_fields_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="name is required"):
        MonteCarloScenario()


def test_monte_carlo_scenario_missing_base_config_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="base_config is required"):
        MonteCarloScenario(
            name="missing_base_config",
            description="Missing base_config",
            monte_carlo={
                "mode": "mixture",
                "n_paths": 10,
                "horizon_years": 1.0,
                "frequency": "M",
            },
        )


def test_monte_carlo_scenario_missing_monte_carlo_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="monte_carlo is required"):
        MonteCarloScenario(
            name="missing_monte_carlo",
            description="Missing monte_carlo",
            base_config="config/defaults.yml",
        )


def test_monte_carlo_scenario_missing_mapping_fields_raise_clear_errors() -> None:
    settings = MonteCarloSettings(
        mode="mixture",
        n_paths=10,
        horizon_years=1.0,
        frequency="M",
        seed=None,
        jobs=None,
    )

    with pytest.raises(ValueError, match="return_model is required"):
        MonteCarloScenario(
            name="missing_mappings",
            description="Missing return_model mapping",
            base_config="config/defaults.yml",
            monte_carlo=settings,
            return_model=None,
            strategy_set={"curated": []},
            folds={"enabled": False},
            outputs={"directory": "outputs/monte_carlo/demo"},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("return_model", "bootstrap", "return_model must be a mapping"),
        ("strategy_set", ["trend_basic"], "strategy_set must be a mapping"),
        ("folds", ["2020-01-01"], "folds must be a mapping"),
        ("outputs", ["outputs/dir"], "outputs must be a mapping"),
    ],
)
def test_monte_carlo_scenario_invalid_nested_configs_raise_clear_errors(
    field: str, value: object, message: str
) -> None:
    payload = {
        "name": "invalid_nested",
        "description": "Invalid nested config",
        "base_config": "config/defaults.yml",
        "monte_carlo": {
            "mode": "mixture",
            "n_paths": 10,
            "horizon_years": 1.0,
            "frequency": "M",
        },
        "return_model": {"kind": "stationary_bootstrap"},
        "strategy_set": {"curated": []},
        "folds": {"enabled": False},
        "outputs": {"directory": "outputs/monte_carlo/demo"},
    }
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        MonteCarloScenario(**payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "", "name must be a non-empty string"),
        ("name", 123, "name must be a string"),
        ("base_config", " ", "base_config must be a non-empty string"),
        ("base_config", 456, "base_config must be a string"),
        ("version", "", "version must be a non-empty string"),
        ("version", 789, "version must be a string"),
        ("monte_carlo", "invalid", "monte_carlo must be a mapping"),
        ("raw", "payload", "raw must be a mapping"),
    ],
)
def test_monte_carlo_scenario_invalid_field_values_raise_clear_errors(
    field: str, value: object, message: str
) -> None:
    payload = {
        "name": "invalid_field_values",
        "description": "Invalid field values",
        "version": "v1",
        "base_config": "config/defaults.yml",
        "monte_carlo": {
            "mode": "mixture",
            "n_paths": 10,
            "horizon_years": 1.0,
            "frequency": "M",
        },
        "return_model": {"kind": "stationary_bootstrap"},
        "strategy_set": {"curated": []},
        "folds": {"enabled": False},
        "outputs": {"directory": "outputs/monte_carlo/demo"},
    }
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        MonteCarloScenario(**payload)
