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


def test_hf_equity_curated_strategies_cover_major_axes() -> None:
    strategy_path = Path("config/scenarios/monte_carlo/strategies/hf_equity_curated.yml")
    payload = yaml.safe_load(strategy_path.read_text(encoding="utf-8"))

    curated = payload.get("curated")
    assert isinstance(curated, list)
    assert len(curated) == 12

    selection_modes = set()
    rank_inclusion = set()
    holding_buckets = set()
    weighting_schemes = set()
    turnover_buckets = set()
    constraint_flags = set()

    for entry in curated:
        overrides = entry.get("overrides", {})
        portfolio = overrides.get("portfolio", {})
        selection_mode = portfolio.get("selection_mode")
        selection_modes.add(selection_mode)

        if selection_mode == "rank":
            rank = portfolio.get("rank", {})
            rank_inclusion.add(rank.get("inclusion_approach"))
            n = rank.get("n")
            if isinstance(n, int):
                if n <= 10:
                    holding_buckets.add("small")
                elif n <= 15:
                    holding_buckets.add("mid")
                else:
                    holding_buckets.add("large")
            if "pct" in rank:
                holding_buckets.add("pct")
            if "threshold" in rank:
                holding_buckets.add("threshold")
        elif selection_mode == "random":
            n = portfolio.get("random_n")
            if isinstance(n, int):
                if n <= 10:
                    holding_buckets.add("small")
                elif n <= 15:
                    holding_buckets.add("mid")
                else:
                    holding_buckets.add("large")
        elif selection_mode == "manual":
            manual_list = portfolio.get("manual_list", [])
            if isinstance(manual_list, list):
                count = len(manual_list)
                if count <= 10:
                    holding_buckets.add("small")
                elif count <= 15:
                    holding_buckets.add("mid")
                else:
                    holding_buckets.add("large")
        elif selection_mode == "all":
            holding_buckets.add("all")

        weighting_schemes.add(portfolio.get("weighting_scheme"))
        weighting = portfolio.get("weighting", {})
        if isinstance(weighting, dict) and weighting.get("name"):
            weighting_schemes.add(weighting.get("name"))

        max_turnover = portfolio.get("max_turnover")
        if isinstance(max_turnover, (int, float)):
            if max_turnover <= 0.1:
                turnover_buckets.add("tight")
            elif max_turnover <= 0.2:
                turnover_buckets.add("moderate")
            else:
                turnover_buckets.add("loose")

        constraints = portfolio.get("constraints", {})
        if isinstance(constraints, dict):
            if "max_weight" in constraints:
                constraint_flags.add("max_weight")
            if "max_active_positions" in constraints:
                constraint_flags.add("max_active_positions")
            if constraints.get("long_only") is False:
                constraint_flags.add("long_short")

        vol_adjust = overrides.get("vol_adjust", {})
        if isinstance(vol_adjust, dict) and vol_adjust.get("enabled") is True:
            constraint_flags.add("vol_target")

    assert selection_modes == {"rank", "random", "manual", "all"}
    assert rank_inclusion == {"top_n", "top_pct", "threshold"}
    assert holding_buckets.issuperset({"small", "mid", "large", "pct", "threshold", "all"})
    assert weighting_schemes.issuperset(
        {
            "equal",
            "risk_parity",
            "hrp",
            "erc",
            "robust_mv",
            "robust_risk_parity",
            "score_prop",
            "score_prop_bayes",
            "adaptive_bayes",
        }
    )
    assert turnover_buckets == {"tight", "moderate", "loose"}
    assert constraint_flags.issuperset(
        {"max_weight", "max_active_positions", "long_short", "vol_target"}
    )


def test_hf_equity_curated_docs_cover_all_strategies() -> None:
    strategy_path = Path("config/scenarios/monte_carlo/strategies/hf_equity_curated.yml")
    payload = yaml.safe_load(strategy_path.read_text(encoding="utf-8"))
    curated = payload.get("curated")
    assert isinstance(curated, list)
    assert len(curated) == 12
    curated_names = {entry["name"] for entry in curated}

    readme_path = Path("config/scenarios/monte_carlo/strategies/README.md")
    readme_lines = readme_path.read_text(encoding="utf-8").splitlines()

    table_rows: list[tuple[str, str, str]] = []
    in_table = False
    for line in readme_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        columns = [column.strip() for column in stripped.strip("|").split("|")]
        if columns[:3] == ["Strategy", "Intent", "Rationale"]:
            in_table = True
            continue
        if not in_table or len(columns) < 3:
            continue
        if all(set(column) <= {"-"} for column in columns[:3]):
            continue
        strategy, intent, rationale = columns[:3]
        if strategy and intent and rationale:
            table_rows.append((strategy, intent, rationale))

    documented_names = {row[0] for row in table_rows}
    assert documented_names == curated_names
    assert len(documented_names) == len(curated_names)
