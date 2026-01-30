from __future__ import annotations

import uuid
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
    scenario_name = f"integration_dynamic_scenario_{tmp_path.name}_{uuid.uuid4().hex}"
    scenario_dir = tmp_path / "config" / "scenarios" / "monte_carlo"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    registry_path = scenario_dir / "index.yml"
    scenario_file = scenario_dir / f"{scenario_name}.yml"
    assert scenario_file.is_relative_to(tmp_path)

    base_config = scenario_dir / "base.yml"
    base_config.write_text("{}", encoding="utf-8")

    scenario_payload = (
        "scenario:\n"
        f"  name: {scenario_name}\n"
        "  description: Integration test scenario.\n"
        '  version: "1.0"\n'
        "base_config: base.yml\n"
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

    registry_data = {
        "scenarios": [
            {
                "name": scenario_name,
                "path": scenario_file.name,
                "description": "Integration test scenario.",
                "tags": ["integration", "test"],
            }
        ]
    }
    entry_path = Path(registry_data["scenarios"][0]["path"])
    assert not entry_path.is_absolute()
    registry_root = registry_path.parent.resolve()
    assert (registry_root / entry_path).resolve().is_relative_to(registry_root)

    registry_path.write_text(
        yaml.safe_dump(registry_data, sort_keys=False),
        encoding="utf-8",
    )

    names = {entry.name for entry in list_scenarios(registry_path=registry_path)}
    assert scenario_name in names

    scenario = load_scenario(scenario_name, registry_path=registry_path)
    assert isinstance(scenario, MonteCarloScenario)
    assert scenario.name == scenario_name
    assert scenario.base_config == base_config.resolve()
    assert scenario.monte_carlo.mode == "two_layer"
    assert scenario.monte_carlo.frequency == "Q"
    assert scenario.outputs is not None
    assert scenario.outputs["directory"] == f"outputs/monte_carlo/{scenario_name}"
