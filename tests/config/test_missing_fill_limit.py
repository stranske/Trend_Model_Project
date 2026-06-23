from __future__ import annotations

from pathlib import Path

from trend_analysis.config import model as config_model
from trend_analysis.config.models import load_config


def _returns_csv(tmp_path: Path) -> Path:
    path = tmp_path / "returns.csv"
    path.write_text("Date,Asset\n2024-01-01,1\n", encoding="utf-8")
    return path


def test_missing_fill_limit_alias_populates_missing_limit(tmp_path: Path) -> None:
    csv_path = _returns_csv(tmp_path)

    settings = config_model.DataSettings.model_validate(
        {
            "csv_path": str(csv_path),
            "date_column": "Date",
            "frequency": "D",
            "missing_policy": "ffill",
            "missing_fill_limit": "2",
        },
        context={"base_path": tmp_path},
    )

    assert settings.missing_limit == 2


def test_missing_limit_wins_when_both_alias_and_canonical_are_set(tmp_path: Path) -> None:
    csv_path = _returns_csv(tmp_path)

    settings = config_model.DataSettings.model_validate(
        {
            "csv_path": str(csv_path),
            "date_column": "Date",
            "frequency": "D",
            "missing_policy": "ffill",
            "missing_fill_limit": 2,
            "missing_limit": 4,
        },
        context={"base_path": tmp_path},
    )

    assert settings.missing_limit == 4


def _config_payload(tmp_path: Path, data_overrides: dict[str, object]) -> dict[str, object]:
    csv_path = _returns_csv(tmp_path)
    data = {
        "csv_path": str(csv_path),
        "date_column": "Date",
        "frequency": "D",
        "missing_policy": "ffill",
    }
    data.update(data_overrides)
    return {
        "version": "1",
        "data": data,
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.1},
        "sample_split": {},
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "transaction_cost_bps": 0.0,
        },
        "metrics": {},
        "export": {},
        "run": {},
    }


def test_load_config_mapping_canonicalizes_missing_fill_limit(tmp_path: Path) -> None:
    cfg = load_config(_config_payload(tmp_path, {"missing_fill_limit": 2}))

    assert cfg.data["missing_limit"] == 2
    assert "missing_fill_limit" not in cfg.data


def test_load_config_mapping_preserves_canonical_missing_limit(tmp_path: Path) -> None:
    cfg = load_config(_config_payload(tmp_path, {"missing_fill_limit": 2, "missing_limit": 4}))

    assert cfg.data["missing_limit"] == 4
    assert "missing_fill_limit" not in cfg.data
