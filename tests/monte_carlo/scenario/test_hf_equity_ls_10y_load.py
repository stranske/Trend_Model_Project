from __future__ import annotations

from typing import Mapping

from trend_analysis.monte_carlo.registry import load_scenario


def test_hf_equity_ls_10y_loads_with_canonical_costs_shape() -> None:
    scenario = load_scenario("hf_equity_ls_10y")

    assert scenario.costs is not None
    assert isinstance(scenario.costs, Mapping)
    assert scenario.costs.get("kind") == "regime_stochastic"

    calm = scenario.costs.get("calm")
    stress = scenario.costs.get("stress")
    assert isinstance(calm, Mapping)
    assert isinstance(stress, Mapping)

    calm_trade_cost = calm.get("trade_cost_bps")
    stress_trade_cost = stress.get("trade_cost_bps")
    assert isinstance(calm_trade_cost, Mapping)
    assert isinstance(stress_trade_cost, Mapping)
    assert calm_trade_cost.get("kind") == "lognormal"
    assert stress_trade_cost.get("kind") == "lognormal"
