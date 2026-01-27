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


def test_load_scenario_returns_model() -> None:
    scenario = load_scenario("hf_equity_ls_10y")
    assert isinstance(scenario, MonteCarloScenario)
    assert scenario.name == "hf_equity_ls_10y"
    assert scenario.base_config.name == "defaults.yml"
    assert "mode" in scenario.monte_carlo


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


def test_load_scenario_missing(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown scenario"):
        load_scenario("missing", registry_path=registry)


def test_load_scenario_requires_name() -> None:
    with pytest.raises(ValueError, match="Scenario name is required"):
        load_scenario(" ")


def test_get_scenario_path() -> None:
    path = get_scenario_path("hf_macro_20y")
    assert path.name == "hf_macro_20y.yml"


def test_get_scenario_path_requires_name() -> None:
    with pytest.raises(ValueError, match="Scenario name is required"):
        get_scenario_path("")


def test_get_scenario_path_missing(tmp_path: Path) -> None:
    registry = tmp_path / "index.yml"
    registry.write_text("scenarios: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown scenario"):
        get_scenario_path("missing", registry_path=registry)
