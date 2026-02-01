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
