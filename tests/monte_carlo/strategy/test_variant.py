from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from trend_analysis.config.model import validate_trend_config
from trend_analysis.monte_carlo.strategy import StrategyVariant


def _base_config(tmp_path: Path) -> dict[str, object]:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    return {
        "version": "1",
        "data": {
            "csv_path": str(csv_file),
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "cost_model": {"per_trade_bps": 10, "half_spread_bps": 0},
            "rank": {"n": 5, "metric": "Sharpe"},
            "weighting": {
                "name": "equal",
                "params": {"column": "Sharpe", "shrink_tau": 0.25},
            },
        },
        "vol_adjust": {"target_vol": 0.1},
        "extra": {"list": [1, 2, 3]},
    }


def test_apply_to_simple_override(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="Rank_5",
        overrides={"portfolio": {"max_turnover": 0.2}},
        tags=["low_turnover"],
    )

    merged = variant.apply_to(base)

    assert merged["portfolio"]["max_turnover"] == 0.2
    assert base["portfolio"]["max_turnover"] == 0.5
    assert variant.tags == ("low_turnover",)


def test_apply_to_nested_override(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(name="Rank_12", overrides={"portfolio": {"rank": {"n": 12}}})

    merged = variant.apply_to(base)

    assert merged["portfolio"]["rank"] == {"n": 12, "metric": "Sharpe"}
    assert base["portfolio"]["rank"]["n"] == 5


def test_apply_to_accepts_trend_config(tmp_path: Path) -> None:
    cfg = validate_trend_config(_base_config(tmp_path), base_path=tmp_path)
    variant = StrategyVariant(name="Rank_6", overrides={"portfolio": {"max_turnover": 0.4}})

    assert variant.apply_to(cfg)["portfolio"]["max_turnover"] == 0.4
    assert variant.to_trend_config(cfg, base_path=tmp_path).portfolio.max_turnover == 0.4


def test_apply_to_accepts_regime_turnover_mapping(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="RegimeCaps",
        overrides={"portfolio": {"max_turnover": {"calm": 0.15, "stress": 0.08}}},
    )

    assert variant.apply_to(base)["portfolio"]["max_turnover"] == {
        "calm": 0.15,
        "stress": 0.08,
    }


def test_canonical_weighting_name_and_params_merge_without_mutation(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    snapshot = deepcopy(base)
    variant = StrategyVariant(
        name="WeightedParams",
        overrides={
            "portfolio": {
                "weighting": {
                    "name": "score_prop_bayes",
                    "params": {"column": "Return", "half_life": 12},
                }
            }
        },
        curated=True,
    )

    merged = variant.apply_to(base)

    assert merged["portfolio"]["weighting"]["name"] == "score_prop_bayes"
    assert merged["portfolio"]["weighting"]["params"] == {
        "column": "Return",
        "shrink_tau": 0.25,
        "half_life": 12,
    }
    assert base == snapshot
    assert variant.to_trend_config(base, base_path=tmp_path).portfolio.rebalance_calendar == "NYSE"


def test_canonical_weighting_override_works_with_shipped_defaults() -> None:
    defaults_path = Path("config/defaults.yml")
    base = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    snapshot = deepcopy(base)
    variant = StrategyVariant(
        name="DefaultsWeighted",
        overrides={"portfolio": {"weighting": {"name": "risk_parity"}}},
    )

    cfg = variant.to_trend_config(base, base_path=defaults_path.parent)

    assert cfg.portfolio.rebalance_calendar == "NYSE"
    assert variant.apply_to(base)["portfolio"]["weighting"]["name"] == "risk_parity"
    assert base == snapshot


def test_curated_variant_can_extend_weighting_params(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="ExtendParams",
        overrides={"portfolio": {"weighting": {"params": {"half_life": 30}}}},
        curated=True,
    )

    merged = variant.apply_to(base)
    assert merged["portfolio"]["weighting"]["params"]["half_life"] == 30
    assert "half_life" not in base["portfolio"]["weighting"]["params"]


def test_non_curated_variant_rejects_weighting_param_extension(tmp_path: Path) -> None:
    variant = StrategyVariant(
        name="ExtendParams",
        overrides={"portfolio": {"weighting": {"params": {"half_life": 30}}}},
    )

    with pytest.raises(ValueError, match="portfolio.weighting.params.half_life"):
        variant.apply_to(_base_config(tmp_path))


@pytest.mark.parametrize("bad_name", [None, ["risk_parity"], "not_a_scheme"])
def test_to_trend_config_rejects_invalid_weighting_name(tmp_path: Path, bad_name: object) -> None:
    variant = StrategyVariant(
        name="BadName",
        overrides={"portfolio": {"weighting": {"name": bad_name}}},
    )

    with pytest.raises(ValueError, match="portfolio.weighting.name"):
        variant.to_trend_config(_base_config(tmp_path), base_path=tmp_path)


def test_apply_to_curated_rejects_unknown_path_outside_freeform(tmp_path: Path) -> None:
    variant = StrategyVariant(
        name="CuratedInvalid",
        overrides={"portfolio": {"rank": {"unknown": 3}}},
        curated=True,
    )
    with pytest.raises(ValueError, match="portfolio.rank.unknown"):
        variant.apply_to(_base_config(tmp_path))


def test_to_trend_config_error_includes_strategy_name(tmp_path: Path) -> None:
    variant = StrategyVariant(
        name="Broken",
        overrides={"portfolio": {"max_turnover": "fast"}},
    )
    with pytest.raises(ValueError, match="Strategy 'Broken' overrides invalid"):
        variant.to_trend_config(_base_config(tmp_path), base_path=tmp_path)


def test_apply_to_type_mismatch_raises(tmp_path: Path) -> None:
    variant = StrategyVariant(
        name="BadType",
        overrides={"portfolio": {"max_turnover": "high"}},
    )
    with pytest.raises(TypeError, match="portfolio.max_turnover"):
        variant.apply_to(_base_config(tmp_path))


def test_apply_to_missing_path_raises(tmp_path: Path) -> None:
    variant = StrategyVariant(name="Missing", overrides={"portfolio": {"rank": {"missing": 3}}})
    with pytest.raises(ValueError, match="portfolio.rank.missing"):
        variant.apply_to(_base_config(tmp_path))


def test_apply_to_rejects_unsupported_override(tmp_path: Path) -> None:
    variant = StrategyVariant(name="BadOverride", overrides={"extra": {"list": {"oops": 1}}})
    with pytest.raises(TypeError, match="extra.list"):
        variant.apply_to(_base_config(tmp_path))
