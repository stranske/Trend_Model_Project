from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trend_analysis import config
from trend_analysis.config.validation import validate_config


def _write_cfg(
    path: Path,
    version: str,
    *,
    csv_path: Path,
    extra: dict[str, object] | None = None,
    portfolio_extra: dict[str, object] | None = None,
) -> None:
    try:
        csv_value = str(csv_path.relative_to(path.parent))
    except ValueError:
        csv_value = str(csv_path)

    payload: dict[str, object] = {
        "version": version,
        "data": {
            "csv_path": csv_value,
            "date_column": "Date",
            "frequency": "M",
            "allow_risk_free_fallback": False,
            "missing_policy": "drop",
        },
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.15},
        "sample_split": {},
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.25,
            "cost_model": {
                "per_trade_bps": 10,
                "half_spread_bps": 0,
            },
        },
        "metrics": {},
        "export": {},
        "run": {},
    }

    if extra:
        payload.update(extra)
    if portfolio_extra:
        payload["portfolio"].update(portfolio_extra)

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _make_csv(path: Path) -> Path:
    path.write_text("Date,RF\n2020-01-31,0.0\n", encoding="utf-8")
    return path


def test_load_default():
    cfg = config.load()
    assert isinstance(cfg, config.Config)
    assert cfg.version


def test_load_custom(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yml"
    csv_path = _make_csv(tmp_path / "data.csv")
    _write_cfg(cfg_path, "99", csv_path=csv_path)
    cfg = config.load(str(cfg_path))
    assert cfg.version == "99"


def test_load_preserves_rebalance_freq(tmp_path: Path) -> None:
    cfg_path = tmp_path / "rebalance.yml"
    csv_path = _make_csv(tmp_path / "data.csv")
    _write_cfg(
        cfg_path,
        "1",
        csv_path=csv_path,
        portfolio_extra={"rebalance_freq": "Q"},
    )
    cfg = config.load(str(cfg_path))
    assert cfg.portfolio["rebalance_freq"] == "Q"


def test_env_var_override(tmp_path: Path, monkeypatch) -> None:
    cfg_file = tmp_path / "env.yml"
    csv_path = _make_csv(tmp_path / "data.csv")
    _write_cfg(cfg_file, "42", csv_path=csv_path)
    monkeypatch.setenv("TREND_CFG", str(cfg_file))
    cfg = config.load()
    assert cfg.version == "42"
    monkeypatch.delenv("TREND_CFG", raising=False)


def test_removed_output_alias_is_rejected(tmp_path: Path) -> None:
    cfg_file = tmp_path / "alias.yml"
    csv_path = _make_csv(tmp_path / "data.csv")
    _write_cfg(
        cfg_file,
        "1",
        csv_path=csv_path,
        extra={
            "output": {"format": "csv", "path": str(tmp_path / "res")},
        },
    )
    with pytest.raises(ValueError, match="output: unexpected or inert config key"):
        config.load(str(cfg_file))


def test_empty_version_rejected(tmp_path: Path) -> None:
    cfg_file = tmp_path / "empty_version.yml"
    csv_path = _make_csv(tmp_path / "data.csv")
    _write_cfg(cfg_file, "", csv_path=csv_path)
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        config.load(str(cfg_file))
    assert "String should have at least 1 character" in str(exc_info.value)


def test_whitespace_version_rejected(tmp_path: Path) -> None:
    cfg_file = tmp_path / "whitespace_version.yml"
    csv_path = _make_csv(tmp_path / "data.csv")
    _write_cfg(cfg_file, "   ", csv_path=csv_path)
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        config.load(str(cfg_file))
    assert "Version field cannot be empty" in str(exc_info.value)


def test_validate_config_rejects_misspelled_rf_override_enabled(tmp_path: Path) -> None:
    payload = {
        "version": "1",
        "data": {"csv_path": "data.csv"},
        "preprocessing": {},
        "vol_adjust": {},
        "sample_split": {},
        "portfolio": {},
        "metrics": {"rf_override_enbaled": True},
        "export": {},
        "run": {},
    }

    result = validate_config(payload, base_path=tmp_path, skip_required_fields=True)

    assert not result.valid
    assert any(error.path == "metrics.rf_override_enbaled" for error in result.errors)
