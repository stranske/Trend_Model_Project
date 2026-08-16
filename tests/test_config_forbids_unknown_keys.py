"""Regression coverage for strict top-level and portfolio config keys."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from trend_analysis.config import models
from trend_analysis.config.model import validate_trend_config


def _valid_payload(tmp_path: Path) -> dict[str, object]:
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Date,FundA\\n2024-01-31,0.01\\n", encoding="utf-8")
    return {
        "data": {"csv_path": str(csv_path), "date_column": "Date", "frequency": "M"},
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 1.0,
            "cost_model": {"per_trade_bps": 0, "half_spread_bps": 0},
        },
        "vol_adjust": {"target_vol": 0.1},
    }


def test_unknown_top_level_and_nested_keys_are_rejected(tmp_path: Path) -> None:
    production_payload = {
        "version": "1",
        "data": {},
        "preprocessing": {},
        "vol_adjust": {},
        "sample_split": {},
        "portfolio": {},
        "metrics": {},
        "export": {},
        "run": {},
        "bogus_section": {},
    }
    with pytest.raises(ValidationError, match="bogus_section"):
        models.Config.model_validate(production_payload)

    top_level = _valid_payload(tmp_path)
    top_level["bogus_section"] = {}
    with pytest.raises(ValueError, match="bogus_section"):
        validate_trend_config(top_level, base_path=tmp_path)

    nested = _valid_payload(tmp_path)
    nested["portfolio"] = {**nested["portfolio"], "bogus_key": 1}  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"portfolio\.bogus_key"):
        validate_trend_config(nested, base_path=tmp_path)

    preprocessing = _valid_payload(tmp_path)
    preprocessing["preprocessing"] = {"misspelled_step": True}
    with pytest.raises(ValueError, match=r"preprocessing\.misspelled_step"):
        validate_trend_config(preprocessing, base_path=tmp_path)

    for retired_key in ("cooldown_periods", "cooldown_months"):
        retired_cooldown = _valid_payload(tmp_path)
        retired_cooldown["multi_period"] = {retired_key: 2}
        with pytest.raises(ValueError, match=rf"multi_period\.{retired_key}"):
            models.load_config(retired_cooldown)

    retired_max_active = _valid_payload(tmp_path)
    retired_max_active["portfolio"] = {
        **retired_max_active["portfolio"],  # type: ignore[arg-type]
        "constraints": {"max_active": 3},
    }
    with pytest.raises(ValueError, match=r"portfolio\.constraints\.max_active"):
        validate_trend_config(retired_max_active, base_path=tmp_path)

    production_payload["portfolio"] = {"constraints": {"max_active": 3}}
    with pytest.raises(ValidationError, match=r"portfolio\.constraints\.max_active"):
        models.Config.model_validate(production_payload)

    # The closed production model still accepts every declared section shipped
    # by the two canonical configurations.
    for config_path in ("config/defaults.yml", "config/demo.yml"):
        assert models.load(config_path).version


def test_declared_legacy_sections_remain_optional(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload.update(
        {
            "version": "1",
            "identity": {"name": "legacy-identity"},
            "extra": {"note": "legacy-extra"},
            "strategy": {"name": "legacy-strategy"},
            "walk_forward": {"enabled": False},
        }
    )

    validated = validate_trend_config(payload, base_path=tmp_path)
    model = models.Config.model_validate(payload)

    assert validated.data.csv_path.name == "returns.csv"
    assert not (
        {"identity", "extra", "strategy", "walk_forward"} & set(models.Config.REQUIRED_DICT_FIELDS)
    )
    assert model.identity == {"name": "legacy-identity"}
    assert model.extra == {"note": "legacy-extra"}
    assert model.strategy == {"name": "legacy-strategy"}
    assert model.walk_forward == {"enabled": False}


def test_active_config_guidance_uses_canonical_cost_keys() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    active_guidance = (
        "docs/UserGuide.md",
        "docs/config.md",
        "docs/phase-2/multi_period_types.md",
        "tools/prompt_dataset.yml",
    )
    retired = (
        "portfolio.transaction_cost_bps",
        "portfolio.cost_model.bps_per_trade",
        "portfolio.cost_model.slippage_bps",
    )

    for relative in active_guidance:
        text = (repo_root / relative).read_text(encoding="utf-8")
        assert not [key for key in retired if key in text], relative

    user_guide = (repo_root / "docs/UserGuide.md").read_text(encoding="utf-8")
    prompt_dataset = (repo_root / "tools/prompt_dataset.yml").read_text(encoding="utf-8")
    assert "portfolio.cost_model.per_trade_bps" in user_guide
    assert "portfolio.cost_model.half_spread_bps" in user_guide
    assert "path: portfolio.cost_model.half_spread_bps" in prompt_dataset
