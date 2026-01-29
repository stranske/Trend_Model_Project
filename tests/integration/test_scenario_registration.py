from __future__ import annotations

import pytest

from trend_analysis.monte_carlo.registry import (
    MonteCarloScenario,
    list_scenarios,
    load_scenario,
)


@pytest.mark.integration
def test_registry_includes_new_scenario_and_loads() -> None:
    names = {entry.name for entry in list_scenarios()}
    assert "credit_stress_15y" in names

    scenario = load_scenario("credit_stress_15y")
    assert isinstance(scenario, MonteCarloScenario)
    assert scenario.name == "credit_stress_15y"
    assert scenario.base_config.name == "defaults.yml"
    assert scenario.monte_carlo.mode == "two_layer"
    assert scenario.monte_carlo.frequency == "Q"
    assert scenario.outputs is not None
    assert scenario.outputs["directory"] == "outputs/monte_carlo/credit_stress_15y"
