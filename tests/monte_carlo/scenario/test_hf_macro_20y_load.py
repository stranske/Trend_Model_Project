from __future__ import annotations

from pathlib import Path

import yaml

from trend_analysis.config.model import TrendConfig
from trend_analysis.monte_carlo.registry import load_scenario
from trend_analysis.monte_carlo.strategy import StrategyVariant


def test_hf_macro_20y_loads_and_instantiates_curated_pack() -> None:
    scenario = load_scenario("hf_macro_20y")

    assert scenario.strategy_set is not None
    curated = scenario.strategy_set.get("curated")
    assert isinstance(curated, list)
    assert all(isinstance(variant, StrategyVariant) for variant in curated)

    base_path = scenario.base_config if isinstance(scenario.base_config, Path) else Path(scenario.base_config)
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    instantiated_strategies = [
        variant.to_trend_config(base_config, base_path=base_path.parent) for variant in curated
    ]

    assert all(isinstance(config, TrendConfig) for config in instantiated_strategies)
    assert len(instantiated_strategies) == 10
