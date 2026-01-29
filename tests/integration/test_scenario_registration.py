from __future__ import annotations

import pytest
import yaml

from trend_analysis.monte_carlo.registry import (
    MonteCarloScenario,
    list_scenarios,
    load_scenario,
)
from utils.paths import proj_path


@pytest.mark.integration
def test_registry_includes_new_scenario_and_loads() -> None:
    registry_path = proj_path("config", "scenarios", "monte_carlo", "index.yml")
    scenarios_dir = registry_path.parent
    scenario_name = "integration_dynamic_scenario"
    scenario_file = scenarios_dir / "integration_dynamic_scenario.yml"
    original_registry = registry_path.read_text(encoding="utf-8")

    scenario_payload = (
        "scenario:\n"
        f"  name: {scenario_name}\n"
        "  description: Integration test scenario.\n"
        "  version: \"1.0\"\n"
        "base_config: config/defaults.yml\n"
        "monte_carlo:\n"
        "  mode: two_layer\n"
        "  n_paths: 250\n"
        "  horizon_years: 3.0\n"
        "  frequency: Q\n"
        "outputs:\n"
        f"  directory: outputs/monte_carlo/{scenario_name}\n"
        "  format: parquet\n"
    )

    try:
        scenario_file.write_text(scenario_payload, encoding="utf-8")
        payload = yaml.safe_load(original_registry) or {}
        scenarios = list(payload.get("scenarios") or [])
        scenarios.append(
            {
                "name": scenario_name,
                "path": scenario_file.name,
                "description": "Integration test scenario.",
                "tags": ["integration", "test"],
            }
        )
        payload["scenarios"] = scenarios
        registry_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

        names = {entry.name for entry in list_scenarios()}
        assert scenario_name in names

        scenario = load_scenario(scenario_name)
        assert isinstance(scenario, MonteCarloScenario)
        assert scenario.name == scenario_name
        assert scenario.base_config.name == "defaults.yml"
        assert scenario.monte_carlo.mode == "two_layer"
        assert scenario.monte_carlo.frequency == "Q"
        assert scenario.outputs is not None
        assert scenario.outputs["directory"] == f"outputs/monte_carlo/{scenario_name}"
    finally:
        registry_path.write_text(original_registry, encoding="utf-8")
        if scenario_file.exists():
            scenario_file.unlink()
