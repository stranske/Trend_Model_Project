from __future__ import annotations

import re
from pathlib import Path

import pytest

from trend_analysis.monte_carlo.registry import (
    MonteCarloScenario,
    ScenarioRegistryEntry,
    get_scenario_path,
    list_scenarios,
    load_scenario,
)
from trend_analysis.monte_carlo.scenario import MonteCarloSettings
from utils.paths import repo_root


def test_list_scenarios_basic() -> None:
    scenarios = list_scenarios()
    names = {entry.name for entry in scenarios}
    assert "hf_equity_ls_10y" in names
    assert "hf_macro_20y" in names
    assert "hf_diversified_5y" in names
    assert "hf_credit_liquidity_7y" in names
    assert "example_scenario" in names


def test_list_scenarios_returns_registry_entries() -> None:
    scenarios = list_scenarios()
    assert scenarios
    assert all(isinstance(entry, ScenarioRegistryEntry) for entry in scenarios)
    assert all(entry.name for entry in scenarios)


def test_list_scenarios_returns_entries_with_paths() -> None:
    scenarios = {entry.name: entry for entry in list_scenarios()}
    equity = scenarios["hf_equity_ls_10y"]
    macro = scenarios["hf_macro_20y"]

    for entry in (equity, macro):
        assert entry.path.exists()
        assert entry.path.suffix == ".yml"
        assert isinstance(entry.tags, tuple)


def test_list_scenarios_normalizes_tags(tmp_path: Path) -> None:
    scenario_a = tmp_path / "alpha.yml"
    scenario_a.write_text("{}", encoding="utf-8")

    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n"
        "  - name: alpha\n"
        "    path: alpha.yml\n"
        "    tags: [' Core ', 'Stress']\n",
        encoding="utf-8",
    )

    scenarios = list_scenarios(registry_path=registry)
    assert scenarios[0].tags == ("core", "stress")


def test_list_scenarios_filters_by_tags(tmp_path: Path) -> None:
    scenario_a = tmp_path / "alpha.yml"
    scenario_b = tmp_path / "beta.yml"
    scenario_a.write_text("{}", encoding="utf-8")
    scenario_b.write_text("{}", encoding="utf-8")

    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n"
        "  - name: alpha\n"
        "    path: alpha.yml\n"
        "    tags: [core, beta]\n"
        "  - name: beta\n"
        "    path: beta.yml\n"
        "    tags: [stress]\n",
        encoding="utf-8",
    )

    filtered = list_scenarios(tags=["stress"], registry_path=registry)
    assert [entry.name for entry in filtered] == ["beta"]


def test_list_scenarios_filters_by_tags_or(tmp_path: Path) -> None:
    scenario_a = tmp_path / "alpha.yml"
    scenario_b = tmp_path / "beta.yml"
    scenario_c = tmp_path / "gamma.yml"
    scenario_a.write_text("{}", encoding="utf-8")
    scenario_b.write_text("{}", encoding="utf-8")
    scenario_c.write_text("{}", encoding="utf-8")

    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n"
        "  - name: alpha\n"
        "    path: alpha.yml\n"
        "    tags: [core]\n"
        "  - name: beta\n"
        "    path: beta.yml\n"
        "    tags: [stress]\n"
        "  - name: gamma\n"
        "    path: gamma.yml\n"
        "    tags: [other]\n",
        encoding="utf-8",
    )

    filtered = list_scenarios(tags=["core", "stress"], registry_path=registry)
    assert {entry.name for entry in filtered} == {"alpha", "beta"}


def test_list_scenarios_filters_by_tags_or_does_not_require_all(tmp_path: Path) -> None:
    scenario_a = tmp_path / "alpha.yml"
    scenario_b = tmp_path / "beta.yml"
    scenario_c = tmp_path / "gamma.yml"
    scenario_a.write_text("{}", encoding="utf-8")
    scenario_b.write_text("{}", encoding="utf-8")
    scenario_c.write_text("{}", encoding="utf-8")

    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n"
        "  - name: alpha\n"
        "    path: alpha.yml\n"
        "    tags: [core]\n"
        "  - name: beta\n"
        "    path: beta.yml\n"
        "    tags: [stress]\n"
        "  - name: gamma\n"
        "    path: gamma.yml\n"
        "    tags: [core, stress]\n",
        encoding="utf-8",
    )

    filtered = list_scenarios(tags=["core", "stress"], registry_path=registry)
    assert {entry.name for entry in filtered} == {"alpha", "beta", "gamma"}


def test_list_scenarios_filters_by_tags_ignores_case_and_whitespace(tmp_path: Path) -> None:
    scenario_a = tmp_path / "alpha.yml"
    scenario_b = tmp_path / "beta.yml"
    scenario_a.write_text("{}", encoding="utf-8")
    scenario_b.write_text("{}", encoding="utf-8")

    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n"
        "  - name: alpha\n"
        "    path: alpha.yml\n"
        "    tags: [Core]\n"
        "  - name: beta\n"
        "    path: beta.yml\n"
        "    tags: [stress]\n",
        encoding="utf-8",
    )

    filtered = list_scenarios(tags=["  core  "], registry_path=registry)
    assert [entry.name for entry in filtered] == ["alpha"]


def test_list_scenarios_missing_registry(tmp_path: Path) -> None:
    registry = tmp_path / "missing.yml"
    with pytest.raises(FileNotFoundError, match="Scenario registry"):
        list_scenarios(registry_path=registry)


def test_list_scenarios_missing_file(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n" "  - name: alpha\n" "    path: alpha.yml\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Could not locate"):
        list_scenarios(registry_path=registry)


def test_list_scenarios_rejects_non_list_entries(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scenarios.*list"):
        list_scenarios(registry_path=registry)


def test_list_scenarios_rejects_non_mapping_entry(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios:\n  - 42\n", encoding="utf-8")

    with pytest.raises(ValueError, match="entries must be mappings"):
        list_scenarios(registry_path=registry)


def test_list_scenarios_rejects_missing_name(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios:\n  - path: alpha.yml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'name'"):
        list_scenarios(registry_path=registry)


def test_list_scenarios_rejects_duplicate_names(tmp_path: Path) -> None:
    scenario_path = tmp_path / "alpha.yml"
    scenario_path.write_text("{}", encoding="utf-8")
    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n"
        "  - name: alpha\n"
        "    path: alpha.yml\n"
        "  - name: alpha\n"
        "    path: alpha.yml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicated"):
        list_scenarios(registry_path=registry)


def test_load_scenario_returns_model() -> None:
    scenario = load_scenario("hf_equity_ls_10y")
    assert isinstance(scenario, MonteCarloScenario)
    assert scenario.name == "hf_equity_ls_10y"
    assert scenario.base_config.name == "defaults.yml"
    assert "mode" in scenario.monte_carlo
    assert scenario.strategy_set is not None
    assert "curated" in scenario.strategy_set
    assert scenario.outputs is not None
    assert "directory" in scenario.outputs


def test_load_scenario_diversified_projection() -> None:
    scenario = load_scenario("hf_diversified_5y")
    assert isinstance(scenario, MonteCarloScenario)
    assert scenario.name == "hf_diversified_5y"
    assert scenario.base_config.name == "defaults.yml"
    assert scenario.monte_carlo.n_paths == 300
    assert scenario.monte_carlo.horizon_years == 5.0
    assert scenario.monte_carlo.frequency == "Q"
    assert scenario.outputs is not None
    assert scenario.outputs["directory"] == "outputs/monte_carlo/hf_diversified_5y"


def test_load_scenario_credit_liquidity_projection() -> None:
    scenario = load_scenario("hf_credit_liquidity_7y")
    assert isinstance(scenario, MonteCarloScenario)
    assert scenario.name == "hf_credit_liquidity_7y"
    assert scenario.base_config.name == "defaults.yml"
    assert scenario.monte_carlo.mode == "mixture"
    assert scenario.monte_carlo.n_paths == 400
    assert scenario.monte_carlo.horizon_years == 7.0
    assert scenario.monte_carlo.frequency == "Q"
    assert scenario.return_model is not None
    assert scenario.return_model["kind"] == "stationary_bootstrap"
    assert scenario.outputs is not None
    assert scenario.outputs["directory"] == "outputs/monte_carlo/hf_credit_liquidity_7y"


def test_load_scenario_includes_optional_sections() -> None:
    scenario = load_scenario("example_scenario")
    assert scenario.return_model is not None
    assert scenario.return_model["kind"] == "stationary_bootstrap"
    assert scenario.folds is not None
    assert scenario.folds["enabled"] is True


def test_load_scenario_example_config_path() -> None:
    scenario = load_scenario("example_scenario")
    assert scenario.path is not None
    assert scenario.path.name == "example.yml"
    assert scenario.path.exists()
    assert scenario.base_config.name == "defaults.yml"


def test_example_scenario_conforms_to_schema() -> None:
    scenario = load_scenario("example_scenario")
    assert isinstance(scenario.monte_carlo, MonteCarloSettings)
    assert scenario.monte_carlo.mode == "mixture"
    assert scenario.monte_carlo.n_paths == 500
    assert scenario.monte_carlo.horizon_years == 4.0
    assert scenario.monte_carlo.frequency == "M"
    assert scenario.return_model is not None
    assert scenario.folds is not None
    assert scenario.outputs is not None


def test_load_scenario_rejects_invalid(tmp_path: Path) -> None:
    scenario_path = tmp_path / "broken.yml"
    scenario_path.write_text("monte_carlo: {}\n", encoding="utf-8")
    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n" "  - name: broken\n" "    path: broken.yml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scenario"):
        load_scenario("broken", registry_path=registry)


def test_load_scenario_supports_top_level_metadata(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yml"
    base_config.write_text("{}", encoding="utf-8")
    scenario_path = tmp_path / "legacy.yml"
    scenario_path.write_text(
        "name: legacy\n"
        "description: Legacy scenario\n"
        "version: '1'\n"
        "base_config: base.yml\n"
        "monte_carlo:\n"
        "  mode: mixture\n"
        "  n_paths: 10\n"
        "  horizon_years: 1\n"
        "  frequency: M\n"
        "return_model:\n"
        "  kind: stationary_bootstrap\n"
        "  params:\n"
        "    block_size: 3\n",
        encoding="utf-8",
    )
    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n" "  - name: legacy\n" "    path: legacy.yml\n",
        encoding="utf-8",
    )

    scenario = load_scenario("legacy", registry_path=registry)
    assert scenario.name == "legacy"
    assert scenario.description == "Legacy scenario"
    assert scenario.version == "1"
    assert scenario.return_model is not None


def test_load_scenario_from_registry_entry_only(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    base_config = scenario_dir / "base.yml"
    base_config.write_text("{}", encoding="utf-8")
    scenario_path = scenario_dir / "new.yml"
    scenario_path.write_text(
        "scenario:\n"
        "  name: new\n"
        "  version: '1'\n"
        "base_config: base.yml\n"
        "monte_carlo:\n"
        "  mode: mixture\n"
        "  n_paths: 5\n"
        "  horizon_years: 1\n"
        "  frequency: M\n",
        encoding="utf-8",
    )
    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n" "  - name: new\n" "    path: scenarios/new.yml\n",
        encoding="utf-8",
    )

    scenarios = {entry.name: entry for entry in list_scenarios(registry_path=registry)}
    assert scenarios["new"].path == scenario_path.resolve()

    loaded = load_scenario("new", registry_path=registry)
    assert loaded.name == "new"
    assert loaded.base_config == base_config.resolve()
    assert loaded.path == scenario_path.resolve()


def test_load_scenario_accepts_folds_mapping(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yml"
    base_config.write_text("{}", encoding="utf-8")
    scenario_path = tmp_path / "folds.yml"
    scenario_path.write_text(
        "scenario:\n"
        "  name: folds\n"
        "  version: '1'\n"
        "base_config: base.yml\n"
        "monte_carlo:\n"
        "  mode: mixture\n"
        "  n_paths: 10\n"
        "  horizon_years: 1\n"
        "  frequency: M\n"
        "folds:\n"
        "  train_years: 5\n"
        "  test_years: 2\n",
        encoding="utf-8",
    )
    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n" "  - name: folds\n" "    path: folds.yml\n",
        encoding="utf-8",
    )

    scenario = load_scenario("folds", registry_path=registry)
    assert scenario.folds == {"train_years": 5, "test_years": 2}


def test_load_scenario_accepts_return_model_mapping(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yml"
    base_config.write_text("{}", encoding="utf-8")
    scenario_path = tmp_path / "return_model.yml"
    scenario_path.write_text(
        "scenario:\n"
        "  name: return_model\n"
        "  version: '1'\n"
        "base_config: base.yml\n"
        "monte_carlo:\n"
        "  mode: mixture\n"
        "  n_paths: 10\n"
        "  horizon_years: 1\n"
        "  frequency: M\n"
        "return_model:\n"
        "  kind: stationary_bootstrap\n"
        "  params:\n"
        "    block_size: 4\n",
        encoding="utf-8",
    )
    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n" "  - name: return_model\n" "    path: return_model.yml\n",
        encoding="utf-8",
    )

    scenario = load_scenario("return_model", registry_path=registry)
    assert scenario.return_model == {
        "kind": "stationary_bootstrap",
        "params": {"block_size": 4},
    }


def test_load_scenario_rejects_null_folds_mapping(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yml"
    base_config.write_text("{}", encoding="utf-8")
    scenario_path = tmp_path / "null_folds.yml"
    scenario_path.write_text(
        "scenario:\n"
        "  name: null_folds\n"
        "  version: '1'\n"
        "base_config: base.yml\n"
        "monte_carlo:\n"
        "  mode: mixture\n"
        "  n_paths: 10\n"
        "  horizon_years: 1\n"
        "  frequency: M\n"
        "folds: null\n",
        encoding="utf-8",
    )
    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n" "  - name: null_folds\n" "    path: null_folds.yml\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="Scenario config 'folds' must be a mapping \\(null provided\\)"
    ):
        load_scenario("null_folds", registry_path=registry)


def test_load_scenario_rejects_null_return_model_mapping(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yml"
    base_config.write_text("{}", encoding="utf-8")
    scenario_path = tmp_path / "null_return_model.yml"
    scenario_path.write_text(
        "scenario:\n"
        "  name: null_return_model\n"
        "  version: '1'\n"
        "base_config: base.yml\n"
        "monte_carlo:\n"
        "  mode: mixture\n"
        "  n_paths: 10\n"
        "  horizon_years: 1\n"
        "  frequency: M\n"
        "return_model: null\n",
        encoding="utf-8",
    )
    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n" "  - name: null_return_model\n" "    path: null_return_model.yml\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="Scenario config 'return_model' must be a mapping \\(null provided\\)"
    ):
        load_scenario("null_return_model", registry_path=registry)


def test_load_scenario_rejects_null_scenario_block(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yml"
    base_config.write_text("{}", encoding="utf-8")
    scenario_path = tmp_path / "null_scenario.yml"
    scenario_path.write_text(
        "scenario: null\n"
        "name: null_scenario\n"
        "base_config: base.yml\n"
        "monte_carlo:\n"
        "  mode: mixture\n"
        "  n_paths: 10\n"
        "  horizon_years: 1\n"
        "  frequency: M\n",
        encoding="utf-8",
    )
    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n" "  - name: null_scenario\n" "    path: null_scenario.yml\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="Scenario config 'scenario' must be a mapping \\(null provided\\)"
    ):
        load_scenario("null_scenario", registry_path=registry)


def test_load_scenario_missing(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios: []\n", encoding="utf-8")

    pattern = re.escape("registry 'index.yml'")
    with pytest.raises(ValueError, match=pattern):
        load_scenario("missing", registry_path=registry)


def test_load_scenario_missing_custom_registry_uses_basename(tmp_path: Path) -> None:
    registry = tmp_path / "custom_registry.yml"
    registry.write_text("scenarios: []\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_scenario("missing", registry_path=registry)

    message = str(excinfo.value)
    assert "registry 'custom_registry.yml'" in message
    assert str(tmp_path) not in message


def test_load_scenario_missing_default_registry_uses_stable_label() -> None:
    with pytest.raises(ValueError) as excinfo:
        load_scenario("missing_default_registry")

    message = str(excinfo.value)
    assert "config/scenarios/monte_carlo/index.yml" in message
    assert str(repo_root()) not in message


def test_load_scenario_missing_registry(tmp_path: Path) -> None:
    registry = tmp_path / "missing.yml"
    with pytest.raises(FileNotFoundError, match="Scenario registry"):
        load_scenario("alpha", registry_path=registry)


def test_load_scenario_rejects_registry_format(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scenarios.*list"):
        load_scenario("alpha", registry_path=registry)


def test_load_scenario_rejects_non_mapping_entry(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios:\n  - 42\n", encoding="utf-8")

    with pytest.raises(ValueError, match="entries must be mappings"):
        load_scenario("alpha", registry_path=registry)


def test_load_scenario_rejects_missing_name(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios:\n  - path: alpha.yml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'name'"):
        load_scenario("alpha", registry_path=registry)


def test_load_scenario_rejects_missing_path(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios:\n  - name: alpha\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'path'"):
        load_scenario("alpha", registry_path=registry)


def test_load_scenario_missing_file(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n" "  - name: alpha\n" "    path: alpha.yml\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Could not locate"):
        load_scenario("alpha", registry_path=registry)


def test_load_scenario_missing_includes_available(tmp_path: Path) -> None:
    scenario_path = tmp_path / "alpha.yml"
    scenario_path.write_text(
        "scenario:\n"
        "  name: alpha\n"
        "  version: '1'\n"
        "base_config: alpha.yml\n"
        "monte_carlo: {}\n",
        encoding="utf-8",
    )
    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n" "  - name: alpha\n" "    path: alpha.yml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Available: alpha"):
        load_scenario("missing", registry_path=registry)


def test_load_scenario_requires_name() -> None:
    with pytest.raises(ValueError, match="Scenario name is required"):
        load_scenario(" ")


def test_get_scenario_path() -> None:
    path = get_scenario_path("hf_macro_20y")
    assert path.name == "hf_macro_20y.yml"


def test_get_scenario_path_resolves_from_registry(tmp_path: Path) -> None:
    scenario_path = tmp_path / "alpha.yml"
    scenario_path.write_text("{}", encoding="utf-8")
    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n" "  - name: alpha\n" "    path: alpha.yml\n",
        encoding="utf-8",
    )

    path = get_scenario_path("alpha", registry_path=registry)
    assert path == scenario_path.resolve()


def test_get_scenario_path_requires_name() -> None:
    with pytest.raises(ValueError, match="Scenario name is required"):
        get_scenario_path("")


def test_get_scenario_path_missing(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios: []\n", encoding="utf-8")

    pattern = re.escape("registry 'index.yml'")
    with pytest.raises(ValueError, match=pattern):
        get_scenario_path("missing", registry_path=registry)


def test_get_scenario_path_missing_registry(tmp_path: Path) -> None:
    registry = tmp_path / "missing.yml"
    with pytest.raises(FileNotFoundError, match="Scenario registry"):
        get_scenario_path("alpha", registry_path=registry)


def test_get_scenario_path_missing_file(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text(
        "scenarios:\n" "  - name: alpha\n" "    path: alpha.yml\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Could not locate"):
        get_scenario_path("alpha", registry_path=registry)


def test_get_scenario_path_rejects_registry_format(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scenarios.*list"):
        get_scenario_path("alpha", registry_path=registry)


def test_get_scenario_path_rejects_non_mapping_entry(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios:\n  - 42\n", encoding="utf-8")

    with pytest.raises(ValueError, match="entries must be mappings"):
        get_scenario_path("alpha", registry_path=registry)


def test_get_scenario_path_rejects_missing_name(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios:\n  - path: alpha.yml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'name'"):
        get_scenario_path("alpha", registry_path=registry)
