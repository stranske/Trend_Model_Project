from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path

import yaml

from trend_analysis.config.model import TrendConfig
from trend_analysis.config.schema_validation import load_schema, validate_config_data
from trend_analysis.monte_carlo.registry import load_scenario
from trend_analysis.monte_carlo.strategy import StrategyVariant


def _load_curated_entries(path: Path, *, expected_count: int) -> list[dict[str, object]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    curated = payload.get("curated")
    if curated is None:
        curated = payload.get("strategies")
    assert isinstance(curated, list)
    assert len(curated) == expected_count
    assert all(isinstance(entry, dict) for entry in curated)
    return curated


@dataclass(frozen=True)
class ParsedCuratedStrategies:
    strategies: list[TrendConfig]
    identifiers: list[str]
    names: list[str]


def _parse_hf_macro_curated_via_trendconfig() -> ParsedCuratedStrategies:
    base_path = Path("config/defaults.yml")
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    strategy_path = Path("config/scenarios/monte_carlo/strategies/hf_macro_curated.yml")
    entries = _load_curated_entries(strategy_path, expected_count=10)

    parsed: list[TrendConfig] = []
    identifiers: list[str] = []
    names: list[str] = []

    for entry in entries:
        identifier = str(entry.get("identifier", "")).strip()
        name = str(entry.get("name", "")).strip()
        variant = StrategyVariant(
            name=name,
            overrides=entry.get("overrides", {}),
            tags=entry.get("tags", ()),
            curated=True,
        )
        parsed.append(variant.to_trend_config(base_config, base_path=base_path.parent))
        identifiers.append(identifier)
        names.append(name)

    return ParsedCuratedStrategies(strategies=parsed, identifiers=identifiers, names=names)


def _readme_table_rows_for_pack(pack_filename: str) -> list[tuple[str, str, str]]:
    readme_path = Path("config/scenarios/monte_carlo/strategies/README.md")
    lines = readme_path.read_text(encoding="utf-8").splitlines()

    heading = f"({pack_filename})"
    in_section = False
    in_table = False
    rows: list[tuple[str, str, str]] = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("## "):
            if in_section:
                break
            in_section = heading in stripped
            continue

        if not in_section or not stripped.startswith("|"):
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
        rows.append((strategy, intent, rationale))

    assert rows, f"No strategy rows found for pack: {pack_filename}"
    return rows


def test_hf_equity_curated_strategies_validate_against_schema() -> None:
    base_path = Path("config/defaults.yml")
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    baseline = deepcopy(base_config)
    schema = load_schema()

    strategy_path = Path("config/scenarios/monte_carlo/strategies/hf_equity_curated.yml")
    curated = _load_curated_entries(strategy_path, expected_count=12)

    for entry in curated:
        variant = StrategyVariant(
            name=entry["name"],
            overrides=entry.get("overrides", {}),
            tags=entry.get("tags", ()),
            curated=True,
        )
        working_copy = deepcopy(base_config)
        merged = variant.apply_to(working_copy)
        schema_errors = validate_config_data(merged, schema)
        assert schema_errors == []
        validated = variant.to_trend_config(base_config, base_path=base_path.parent)
        assert isinstance(validated, TrendConfig)
        assert base_config == baseline


def test_hf_macro_curated_validates_against_trendconfig() -> None:
    base_path = Path("config/defaults.yml")
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    baseline = deepcopy(base_config)
    schema = load_schema()

    strategy_path = Path("config/scenarios/monte_carlo/strategies/hf_macro_curated.yml")
    curated = _load_curated_entries(strategy_path, expected_count=10)

    for entry in curated:
        variant = StrategyVariant(
            name=entry["name"],
            overrides=entry.get("overrides", {}),
            tags=entry.get("tags", ()),
            curated=True,
        )
        working_copy = deepcopy(base_config)
        merged = variant.apply_to(working_copy)
        schema_errors = validate_config_data(merged, schema)
        assert schema_errors == []
        validated = variant.to_trend_config(base_config, base_path=base_path.parent)
        assert isinstance(validated, TrendConfig)
        assert base_config == baseline


def test_hf_macro_curated_has_10_strategies() -> None:
    parsed_config = _parse_hf_macro_curated_via_trendconfig()
    assert len(parsed_config.strategies) == 10


def test_hf_macro_curated_has_unique_id_and_name() -> None:
    parsed_config = _parse_hf_macro_curated_via_trendconfig()
    assert all(identifier for identifier in parsed_config.identifiers)
    assert all(name for name in parsed_config.names)
    assert len(set(parsed_config.identifiers)) == len(parsed_config.identifiers)
    assert len(set(parsed_config.names)) == len(parsed_config.names)


def test_hf_equity_curated_strategies_do_not_mutate_defaults() -> None:
    base_path = Path("config/defaults.yml")
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    base_snapshot = deepcopy(base_config)

    strategy_path = Path("config/scenarios/monte_carlo/strategies/hf_equity_curated.yml")
    curated = _load_curated_entries(strategy_path, expected_count=12)

    for entry in curated:
        variant = StrategyVariant(
            name=entry["name"],
            overrides=entry.get("overrides", {}),
            tags=entry.get("tags", ()),
            curated=True,
        )
        _ = variant.apply_to(base_config)
        assert base_config == base_snapshot


def test_hf_equity_curated_to_trend_config_does_not_mutate_defaults() -> None:
    base_path = Path("config/defaults.yml")
    before_text = base_path.read_text(encoding="utf-8")
    base_config = yaml.safe_load(before_text)
    base_snapshot = deepcopy(base_config)

    strategy_path = Path("config/scenarios/monte_carlo/strategies/hf_equity_curated.yml")
    curated = _load_curated_entries(strategy_path, expected_count=12)

    for entry in curated:
        variant = StrategyVariant(
            name=entry["name"],
            overrides=entry.get("overrides", {}),
            tags=entry.get("tags", ()),
            curated=True,
        )
        validated = variant.to_trend_config(base_config, base_path=base_path.parent)
        assert isinstance(validated, TrendConfig)
        assert base_config == base_snapshot

    after_text = base_path.read_text(encoding="utf-8")
    assert after_text == before_text


def test_hf_equity_curated_strategies_compatible_with_strategy_guards() -> None:
    base_path = Path("config/defaults.yml")
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    strategy_path = Path("config/scenarios/monte_carlo/strategies/hf_equity_curated.yml")
    curated = _load_curated_entries(strategy_path, expected_count=12)

    guards = {"max_turnover": 0.12}

    for entry in curated:
        variant = StrategyVariant(
            name=entry["name"],
            overrides=entry.get("overrides", {}),
            tags=entry.get("tags", ()),
            curated=True,
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
    curated = _load_curated_entries(strategy_path, expected_count=12)

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
    curated = _load_curated_entries(strategy_path, expected_count=12)
    curated_names = {entry["name"] for entry in curated}

    rows = _readme_table_rows_for_pack("hf_equity_curated.yml")
    documented_names = {row[0] for row in rows}

    assert documented_names == curated_names
    assert len(documented_names) == len(curated_names)


def test_hf_macro_curated_docs_cover_all_strategies() -> None:
    strategy_path = Path("config/scenarios/monte_carlo/strategies/hf_macro_curated.yml")
    curated = _load_curated_entries(strategy_path, expected_count=10)
    curated_names = {entry["name"] for entry in curated}

    rows = _readme_table_rows_for_pack("hf_macro_curated.yml")
    documented_names = {row[0] for row in rows}

    assert documented_names == curated_names
    assert len(documented_names) == len(curated_names)


def test_hf_macro_curated_docs_has_10_rows_with_intent_and_rationale() -> None:
    rows = _readme_table_rows_for_pack("hf_macro_curated.yml")

    assert len(rows) == 10
    for strategy, intent, rationale in rows:
        assert strategy.strip()
        assert intent.strip()
        assert rationale.strip()


def test_hf_macro_20y_loads_curated_pack_variants() -> None:
    scenario = load_scenario("hf_macro_20y")
    assert scenario.strategy_set is not None
    curated = scenario.strategy_set["curated"]
    assert isinstance(curated, list)
    assert len(curated) == 10
    assert all(isinstance(variant, StrategyVariant) for variant in curated)

    strategy_path = Path("config/scenarios/monte_carlo/strategies/hf_macro_curated.yml")
    pack_entries = _load_curated_entries(strategy_path, expected_count=10)
    expected_names = [entry["name"] for entry in pack_entries]
    loaded_names = [variant.name for variant in curated]
    assert loaded_names == expected_names
