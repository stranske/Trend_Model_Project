"""Additional branch coverage for ``trend_analysis.config.model``.""" 

from __future__ import annotations

from pathlib import Path

import pytest

from trend_analysis.config import model


def _base_data_settings() -> dict[str, object]:
    return {
        "csv_path": "data.csv",
        "frequency": "M",
        "date_column": "Date",
    }


def _portfolio_settings() -> dict[str, object]:
    return {
        "rebalance_calendar": "NYSE",
        "max_turnover": 0.5,
        "transaction_cost_bps": 10,
    }


def _risk_settings() -> dict[str, object]:
    return {"target_vol": 0.1, "floor_vol": 0.02, "warmup_periods": 0}


def _minimal_config() -> dict[str, object]:
    return {
        "data": _base_data_settings(),
        "portfolio": _portfolio_settings(),
        "vol_adjust": _risk_settings(),
    }


def test_resolve_path_missing_raises(tmp_path: Path) -> None:
    missing = tmp_path / "absent.csv"
    with pytest.raises(ValueError):
        model._resolve_path(str(missing), base_dir=tmp_path)


def test_validate_managers_glob_requires_string() -> None:
    data = _base_data_settings()
    data["managers_glob"] = 123
    with pytest.raises(ValueError):
        model.DataSettings.model_validate(data, context={"base_path": Path.cwd()})


def test_validate_date_column_rejects_empty() -> None:
    data = _base_data_settings()
    data["date_column"] = ""
    with pytest.raises(ValueError):
        model.DataSettings.model_validate(data, context={"base_path": Path.cwd()})


def test_normalize_frequency_requires_supported_value() -> None:
    data = _base_data_settings()
    data["frequency"] = "Hourly"
    with pytest.raises(ValueError):
        model.DataSettings.model_validate(data, context={"base_path": Path.cwd()})


def test_validate_missing_limit_rejects_non_numeric() -> None:
    data = _base_data_settings()
    data["missing_limit"] = "invalid"
    with pytest.raises(ValueError):
        model.DataSettings.model_validate(data, context={"base_path": Path.cwd()})


def test_ensure_source_requires_csv_or_glob() -> None:
    data = _base_data_settings()
    data["csv_path"] = None
    data["managers_glob"] = None
    with pytest.raises(ValueError):
        model.DataSettings.model_validate(data, context={"base_path": Path.cwd()})


def test_portfolio_turnover_bounds() -> None:
    base = _portfolio_settings()
    base["max_turnover"] = -0.1
    with pytest.raises(ValueError):
        model.PortfolioSettings.model_validate(base)
    base["max_turnover"] = 1.5
    with pytest.raises(ValueError):
        model.PortfolioSettings.model_validate(base)


def test_portfolio_cost_non_negative() -> None:
    base = _portfolio_settings()
    base["transaction_cost_bps"] = -5
    with pytest.raises(ValueError):
        model.PortfolioSettings.model_validate(base)


def test_risk_settings_validators() -> None:
    risk = _risk_settings()
    risk["target_vol"] = 0
    with pytest.raises(ValueError):
        model.RiskSettings.model_validate(risk)
    risk = _risk_settings()
    risk["floor_vol"] = -0.1
    with pytest.raises(ValueError):
        model.RiskSettings.model_validate(risk)
    risk = _risk_settings()
    risk["warmup_periods"] = -1
    with pytest.raises(ValueError):
        model.RiskSettings.model_validate(risk)


def test_validate_trend_config_reports_location(tmp_path: Path) -> None:
    bad = _minimal_config()
    data_file = tmp_path / "data.csv"
    data_file.write_text("Date,Fund\n2024-01-31,0.01\n", encoding="utf-8")
    bad["data"]["csv_path"] = str(data_file)
    bad["portfolio"]["rebalance_calendar"] = ""
    with pytest.raises(ValueError) as excinfo:
        model.validate_trend_config(bad, base_path=tmp_path)
    assert "portfolio.rebalance_calendar" in str(excinfo.value)


def test_load_trend_config_rejects_non_mapping(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(TypeError):
        model.load_trend_config(str(cfg_file))
