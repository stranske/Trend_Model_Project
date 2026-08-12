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
            "transaction_cost_bps": 0,
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
