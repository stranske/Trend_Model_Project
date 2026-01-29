from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trend_analysis.monte_carlo.registry import (
    MonteCarloScenario,
    list_scenarios,
    load_scenario,
)


@pytest.mark.integration
def test_registry_includes_new_scenario_and_loads(tmp_path: Path) -> None:
    scenario_name = "integration_dynamic_scenario"
    scenario_file = tmp_path / "integration_dynamic_scenario.yml"

    scenario_payload = (
        "scenario:\n"
        f"  name: {scenario_name}\n"
        "  description: Integration test scenario.\n"
        '  version: "1.0"\n'
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

    scenario_file.write_text(scenario_payload, encoding="utf-8")
    payload = {
        "scenarios": [
            {
                "name": scenario_name,
                "path": scenario_file.as_posix(),
                "description": "Integration test scenario.",
                "tags": ["integration", "test"],
            }
        ]
    }
    temp_registry = tmp_path / "index.yml"
    temp_registry.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    names = {entry.name for entry in list_scenarios(registry_path=temp_registry)}
    assert scenario_name in names

    scenario = load_scenario(scenario_name, registry_path=temp_registry)
    assert isinstance(scenario, MonteCarloScenario)
    assert scenario.name == scenario_name
    assert scenario.base_config.name == "defaults.yml"
    assert scenario.monte_carlo.mode == "two_layer"
    assert scenario.monte_carlo.frequency == "Q"
    assert scenario.outputs is not None
    assert scenario.outputs["directory"] == f"outputs/monte_carlo/{scenario_name}"
