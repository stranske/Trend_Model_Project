from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from tests.path_utils import is_relative_to
from trend_analysis.monte_carlo.registry import (
    MonteCarloScenario,
    list_scenarios,
    load_scenario,
)


@pytest.mark.integration
def test_registry_includes_new_scenario_and_loads(tmp_path: Path) -> None:
    scenario_name = "integration_fixture_scenario"
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures" / "scenario_registration"
    scenario_dir = tmp_path / "config" / "scenarios" / "monte_carlo"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    fixture_files = ("index.yml", "integration_fixture.yml", "defaults.yml")
    for filename in fixture_files:
        shutil.copy2(fixtures_dir / filename, scenario_dir / filename)

    registry_path = scenario_dir / "index.yml"
    scenario_file = scenario_dir / "integration_fixture.yml"
    assert is_relative_to(scenario_file, tmp_path)

    registry_data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert isinstance(registry_data, dict)
    entry_path = Path(registry_data["scenarios"][0]["path"])
    assert not entry_path.is_absolute()
    registry_root = registry_path.parent.resolve()
    assert is_relative_to((registry_root / entry_path).resolve(), registry_root)

    names = {entry.name for entry in list_scenarios(registry_path=registry_path)}
    assert scenario_name in names

    scenario = load_scenario(scenario_name, registry_path=registry_path)
    assert isinstance(scenario, MonteCarloScenario)
    assert scenario.name == scenario_name
    assert scenario.base_config.name == "defaults.yml"
    assert scenario.monte_carlo.mode == "two_layer"
    assert scenario.monte_carlo.frequency == "Q"
    assert scenario.outputs is not None
    assert scenario.outputs["directory"] == f"outputs/monte_carlo/{scenario_name}"
