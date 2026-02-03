from __future__ import annotations

from pathlib import Path

import pytest

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
            "transaction_cost_bps": 10,
            "rank": {"n": 5, "metric": "Sharpe"},
            "weighting_scheme": "equal",
            "weighting": {"name": "equal", "params": {"column": "Sharpe", "shrink_tau": 0.25}},
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
    variant = StrategyVariant(
        name="Rank_12",
        overrides={"portfolio": {"rank": {"n": 12}}},
    )

    merged = variant.apply_to(base)

    assert merged["portfolio"]["rank"]["n"] == 12
    assert merged["portfolio"]["rank"]["metric"] == "Sharpe"
    assert base["portfolio"]["rank"]["n"] == 5


def test_apply_to_accepts_trend_config(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    cfg = validate_trend_config(base, base_path=tmp_path)
    variant = StrategyVariant(
        name="Rank_6",
        overrides={"portfolio": {"max_turnover": 0.4}},
    )

    merged = variant.apply_to(cfg)

    assert merged["portfolio"]["max_turnover"] == 0.4


def test_to_trend_config_validates_merge(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="Rank_8",
        overrides={"portfolio": {"max_turnover": 0.35}},
    )

    cfg = variant.to_trend_config(base, base_path=tmp_path)

    assert cfg.portfolio.max_turnover == 0.35


def test_to_trend_config_accepts_trend_config(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    base_cfg = validate_trend_config(base, base_path=tmp_path)
    variant = StrategyVariant(
        name="Rank_4",
        overrides={"portfolio": {"max_turnover": 0.4}},
    )

    cfg = variant.to_trend_config(base_cfg, base_path=tmp_path)

    assert cfg.portfolio.max_turnover == 0.4


def test_to_trend_config_accepts_weighting_scheme_and_name(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="Weighted",
        overrides={
            "portfolio": {
                "weighting_scheme": "risk_parity",
                "weighting": {"name": "score_prop_bayes", "params": {"column": "Sharpe"}},
            }
        },
    )

    merged = variant.apply_to(base)
    assert merged["portfolio"]["weighting_scheme"] == "risk_parity"
    assert merged["portfolio"]["weighting"]["name"] == "score_prop_bayes"
    assert merged["portfolio"]["weighting"]["params"]["column"] == "Sharpe"

    cfg = variant.to_trend_config(base, base_path=tmp_path)
    assert cfg.portfolio.rebalance_calendar == "NYSE"


def test_to_trend_config_allows_only_weighting_scheme_override(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="SchemeOnly",
        overrides={"portfolio": {"weighting_scheme": "hrp"}},
    )

    merged = variant.apply_to(base)
    assert merged["portfolio"]["weighting_scheme"] == "hrp"
    assert merged["portfolio"]["weighting"]["name"] == "equal"

    cfg = variant.to_trend_config(base, base_path=tmp_path)
    assert cfg.portfolio.max_turnover == 0.5


def test_to_trend_config_allows_only_weighting_name_override(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="NameOnly",
        overrides={"portfolio": {"weighting": {"name": "score_prop"}}},
    )

    merged = variant.apply_to(base)
    assert merged["portfolio"]["weighting_scheme"] == "equal"
    assert merged["portfolio"]["weighting"]["name"] == "score_prop"

    cfg = variant.to_trend_config(base, base_path=tmp_path)
    assert cfg.portfolio.max_turnover == 0.5


def test_apply_to_allows_weighting_params_extension(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="ExtendParams",
        overrides={
            "portfolio": {
                "weighting": {"params": {"column": "Sharpe", "half_life": 30, "obs_sigma": 0.15}}
            }
        },
        curated=True,
    )

    merged = variant.apply_to(base)

    assert merged["portfolio"]["weighting"]["params"]["shrink_tau"] == 0.25
    assert merged["portfolio"]["weighting"]["params"]["column"] == "Sharpe"
    assert merged["portfolio"]["weighting"]["params"]["half_life"] == 30
    assert merged["portfolio"]["weighting"]["params"]["obs_sigma"] == 0.15
    assert "half_life" not in base["portfolio"]["weighting"]["params"]
    assert "obs_sigma" not in base["portfolio"]["weighting"]["params"]


def test_apply_to_rejects_weighting_params_extension_for_non_curated(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="ExtendParams",
        overrides={
            "portfolio": {"weighting": {"params": {"half_life": 30}}},
        },
    )

    with pytest.raises(ValueError, match="portfolio.weighting.params.half_life"):
        variant.apply_to(base)


def test_to_trend_config_rejects_invalid_weighting_scheme_type(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="BadScheme",
        overrides={"portfolio": {"weighting_scheme": ["risk_parity"]}},
    )

    with pytest.raises(ValueError, match="portfolio.weighting_scheme"):
        variant.to_trend_config(base, base_path=tmp_path)


def test_to_trend_config_rejects_invalid_weighting_name_type(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="BadName",
        overrides={"portfolio": {"weighting": {"name": None}}},
    )

    with pytest.raises(ValueError, match="portfolio.weighting.name"):
        variant.to_trend_config(base, base_path=tmp_path)


def test_apply_to_type_mismatch_raises(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="BadType",
        overrides={"portfolio": {"max_turnover": "high"}},
    )

    with pytest.raises(TypeError, match="portfolio.max_turnover"):
        variant.apply_to(base)


def test_apply_to_missing_top_level_path_raises(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="MissingTopLevel",
        overrides={"missing": {"value": 1}},
    )

    with pytest.raises(ValueError, match="missing"):
        variant.apply_to(base)


def test_to_trend_config_reports_invalid_path(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="MissingPath",
        overrides={"portfolio": {"rank": {"missing": 3}}},
    )

    with pytest.raises(ValueError, match="portfolio.rank.missing"):
        variant.to_trend_config(base, base_path=tmp_path)


def test_apply_to_rejects_unsupported_override(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    variant = StrategyVariant(
        name="BadOverride",
        overrides={"extra": {"list": {"oops": 1}}},
    )

    with pytest.raises(TypeError, match="extra.list"):
        variant.apply_to(base)
