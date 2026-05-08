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
    assert scenario.base_config == base_config.resolve()
    assert scenario.monte_carlo.mode == "two_layer"
    assert scenario.monte_carlo.frequency == "Q"
    assert scenario.outputs is not None
    assert scenario.outputs["directory"] == f"outputs/monte_carlo/{scenario_name}"


@pytest.mark.integration
def test_registry_scenario_allows_null_optional_sections(tmp_path: Path) -> None:
    scenario_name = "integration_null_optional"
    scenario_dir = tmp_path / "config" / "scenarios" / "monte_carlo"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    registry_path = scenario_dir / "index.yml"
    scenario_file = scenario_dir / "integration_null_optional.yml"
    base_config = scenario_dir / "base.yml"

    base_config.write_text("{}", encoding="utf-8")
    scenario_file.write_text(
        "scenario:\n"
        f"  name: {scenario_name}\n"
        "base_config: base.yml\n"
        "monte_carlo:\n"
        "  mode: mixture\n"
        "  n_paths: 10\n"
        "  horizon_years: 1.0\n"
        "  frequency: M\n"
        "strategy_set: null\n"
        "outputs: null\n"
        "costs: null\n",
        encoding="utf-8",
    )
    registry_path.write_text(
        "scenarios:\n" f"  - name: {scenario_name}\n" "    path: integration_null_optional.yml\n",
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_name, registry_path=registry_path)
    assert scenario.strategy_set is None
    assert scenario.outputs is None
    assert scenario.costs is None


@pytest.mark.integration
def test_registry_scenario_rejects_null_required_monte_carlo(tmp_path: Path) -> None:
    scenario_name = "integration_null_required"
    scenario_dir = tmp_path / "config" / "scenarios" / "monte_carlo"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    registry_path = scenario_dir / "index.yml"
    scenario_file = scenario_dir / "integration_null_required.yml"
    base_config = scenario_dir / "base.yml"

    base_config.write_text("{}", encoding="utf-8")
    scenario_file.write_text(
        "scenario:\n"
        f"  name: {scenario_name}\n"
        "base_config: base.yml\n"
        "monte_carlo: null\n"
        "strategy_set: null\n"
        "outputs: null\n"
        "costs: null\n",
        encoding="utf-8",
    )
    registry_path.write_text(
        "scenarios:\n" f"  - name: {scenario_name}\n" "    path: integration_null_required.yml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Scenario config must define monte_carlo"):
        load_scenario(scenario_name, registry_path=registry_path)
