from __future__ import annotations

from trend_analysis.monte_carlo.registry import load_scenario
from trend_analysis.monte_carlo.strategy import StrategyVariant


def test_scenario_yaml_strategy_set_curated_variants() -> None:
    scenario = load_scenario("hf_equity_ls_10y")

    assert scenario.strategy_set is not None
    curated = scenario.strategy_set["curated"]

    assert isinstance(curated, list)
    assert curated

    variant = curated[0]
    assert isinstance(variant, StrategyVariant)
    assert variant.name == "Rank_12_Equal_TightTurnover"
    assert variant.overrides["portfolio"]["selection_mode"] == "rank"
    assert variant.overrides["portfolio"]["rank"]["n"] == 12
