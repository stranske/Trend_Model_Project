from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from trend_analysis.config import model as config_model
from trend_analysis.config.models import load_config


def _returns_csv(tmp_path: Path) -> Path:
    path = tmp_path / "returns.csv"
    path.write_text("Date,Asset\n2024-01-01,1\n", encoding="utf-8")
    return path


def _data_payload(tmp_path: Path, **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "csv_path": str(_returns_csv(tmp_path)),
        "date_column": "Date",
        "frequency": "D",
        "missing_policy": "ffill",
    }
    data.update(overrides)
    return data


def _config_payload(tmp_path: Path, data_overrides: dict[str, object]) -> dict[str, object]:
    return {
        "version": "1",
        "data": _data_payload(tmp_path, **data_overrides),
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.1},
        "sample_split": {},
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "cost_model": {"per_trade_bps": 0.0, "half_spread_bps": 0.0},
        },
        "metrics": {},
        "export": {},
        "run": {},
    }


@pytest.mark.parametrize(
    "removed_key,value",
    [
        ("missing_fill_limit", 2),
        ("indices_glob", "unused"),
        ("price_column", "unused"),
        ("currency", "unused"),
        ("lookback_required", 10),
    ],
)
def test_removed_data_keys_are_rejected(tmp_path: Path, removed_key: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        config_model.DataSettings.model_validate(
            _data_payload(tmp_path, **{removed_key: value}),
            context={"base_path": tmp_path},
        )

    message = str(exc_info.value)
    assert removed_key in message
    assert "Extra inputs are not permitted" in message


@pytest.mark.parametrize(
    "removed_key,value",
    [
        ("missing_fill_limit", 2),
        ("indices_glob", "unused"),
        ("price_column", "unused"),
        ("currency", "unused"),
        ("lookback_required", 10),
    ],
)
def test_load_config_rejects_removed_data_keys(
    tmp_path: Path, removed_key: str, value: object
) -> None:
    with pytest.raises(ValueError, match=removed_key):
        load_config(_config_payload(tmp_path, {removed_key: value}))


def test_load_config_preserves_canonical_missing_limit(tmp_path: Path) -> None:
    cfg = load_config(_config_payload(tmp_path, {"missing_limit": 4}))

    assert cfg.data["missing_limit"] == 4
    assert "missing_fill_limit" not in cfg.data


def test_current_timezone_survives_strict_validation(tmp_path: Path) -> None:
    settings = config_model.DataSettings.model_validate(
        _data_payload(tmp_path, timezone="America/Chicago"),
        context={"base_path": tmp_path},
    )

    assert settings.timezone == "America/Chicago"


def test_current_timezone_survives_runtime_loading(tmp_path: Path) -> None:
    cfg = load_config(_config_payload(tmp_path, {"timezone": "America/Chicago"}))

    assert cfg.data["timezone"] == "America/Chicago"


def test_invalid_timezone_fails_at_startup(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not a valid IANA timezone"):
        config_model.DataSettings.model_validate(
            _data_payload(tmp_path, timezone="Mars/Olympus_Mons"),
            context={"base_path": tmp_path},
        )


def test_current_volatility_window_survives_runtime_loading(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path, {})
    payload["vol_adjust"] = {
        "target_vol": 0.1,
        "window": {"length": 63, "decay": "ewma", "lambda": 0.94},
    }

    cfg = load_config(payload)

    assert cfg.vol_adjust["window"] == {
        "length": 63,
        "decay": "ewma",
        "lambda": 0.94,
    }


@pytest.mark.parametrize(
    "window",
    [
        {"length": 0},
        {"length": 63, "decay": "unsupported"},
        {"length": 63, "decay": "ewma", "lambda": 1.0},
        {"length": 63, "unknown": True},
    ],
)
def test_invalid_volatility_window_fails_at_startup(
    tmp_path: Path, window: dict[str, object]
) -> None:
    payload = _config_payload(tmp_path, {})
    payload["vol_adjust"] = {"target_vol": 0.1, "window": window}

    with pytest.raises(ValueError, match="window"):
        load_config(payload)
