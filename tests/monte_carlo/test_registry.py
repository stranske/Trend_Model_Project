from __future__ import annotations

from pathlib import Path

import pytest

from trend_analysis.monte_carlo.registry import (
    MonteCarloScenario,
    get_scenario_path,
    list_scenarios,
    load_scenario,
)


def test_list_scenarios_basic() -> None:
    scenarios = list_scenarios()
    names = {entry.name for entry in scenarios}
    assert "hf_equity_ls_10y" in names
    assert "hf_macro_20y" in names


def test_list_scenarios_returns_entries_with_paths() -> None:
    scenarios = {entry.name: entry for entry in list_scenarios()}
    equity = scenarios["hf_equity_ls_10y"]
    macro = scenarios["hf_macro_20y"]

    for entry in (equity, macro):
        assert entry.path.exists()
        assert entry.path.suffix == ".yml"
        assert isinstance(entry.tags, tuple)


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


def test_load_scenario_missing(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown scenario"):
        load_scenario("missing", registry_path=registry)


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

    with pytest.raises(ValueError, match="Unknown scenario"):
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
