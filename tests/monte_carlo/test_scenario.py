from __future__ import annotations

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
