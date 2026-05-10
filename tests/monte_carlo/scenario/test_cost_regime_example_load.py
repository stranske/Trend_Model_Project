from __future__ import annotations

from typing import Mapping

from trend_analysis.monte_carlo.registry import load_scenario


def test_cost_regime_example_loads_without_exception() -> None:
    scenario = load_scenario("cost_regime_example")

    assert scenario.name == "cost_regime_example"
    assert scenario.costs is not None
    assert isinstance(scenario.costs, Mapping)
    assert scenario.costs.get("kind") == "regime_stochastic"
