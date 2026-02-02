from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from trend_analysis.config.model import TrendConfig
from trend_analysis.monte_carlo.strategy import StrategyVariant


def test_hf_equity_curated_strategies_validate_against_schema() -> None:
    base_path = Path("config/defaults.yml")
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    strategy_path = Path("config/scenarios/monte_carlo/strategies/hf_equity_curated.yml")
    payload = yaml.safe_load(strategy_path.read_text(encoding="utf-8"))

    curated = payload.get("curated")
    assert isinstance(curated, list)
    assert len(curated) == 12

    for entry in curated:
        variant = StrategyVariant(
            name=entry["name"],
            overrides=entry.get("overrides", {}),
            tags=entry.get("tags", ()),
        )
        validated = variant.to_trend_config(base_config, base_path=base_path.parent)
        assert isinstance(validated, TrendConfig)


def test_hf_equity_curated_strategies_compatible_with_strategy_guards() -> None:
    base_path = Path("config/defaults.yml")
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    strategy_path = Path("config/scenarios/monte_carlo/strategies/hf_equity_curated.yml")
    payload = yaml.safe_load(strategy_path.read_text(encoding="utf-8"))

    curated = payload.get("curated")
    assert isinstance(curated, list)
    assert len(curated) == 12

    guards = {"max_turnover": 0.12}

    for entry in curated:
        variant = StrategyVariant(
            name=entry["name"],
            overrides=entry.get("overrides", {}),
            tags=entry.get("tags", ()),
        )
        merged = variant.apply_to(base_config)
        guarded = deepcopy(merged)
        portfolio = guarded.setdefault("portfolio", {})
        if isinstance(portfolio, dict) and "max_turnover" in guards:
            portfolio["max_turnover"] = guards["max_turnover"]
        validated = TrendConfig(**guarded)
        assert isinstance(validated, TrendConfig)
