"""Additional coverage for configuration model validation edge cases."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from trend_analysis.config import model as config_model


def test_resolve_path_considers_base_dir_parent(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    base_dir = workspace / "configs"
    base_dir.mkdir()

    data_file = workspace / "returns.csv"
    data_file.write_text("date,manager,value\n", encoding="utf-8")

    leaf = workspace / "leaf"
    leaf.mkdir()
    monkeypatch.chdir(leaf)

    resolved = config_model._resolve_path("returns.csv", base_dir=base_dir)

    assert resolved == data_file.resolve()


def test_resolve_path_falls_back_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("date,nav\n", encoding="utf-8")

    resolved = config_model._resolve_path("data.csv", base_dir=None)

    assert resolved == csv_path.resolve()


def test_resolve_path_rejects_wildcard_patterns(tmp_path):
    with pytest.raises(ValueError, match="contains wildcard characters"):
        config_model._resolve_path("*.csv", base_dir=tmp_path)


def test_resolve_path_rejects_directory(tmp_path):
    directory = tmp_path / "inputs"
    directory.mkdir()

    with pytest.raises(ValueError, match="points to a directory"):
        config_model._resolve_path(directory, base_dir=None)


def test_ensure_glob_matches_reports_missing_files(tmp_path):
    with pytest.raises(ValueError, match="did not match any CSV files"):
        config_model._ensure_glob_matches("missing/*.csv", base_dir=tmp_path)


def test_ensure_glob_matches_rejects_non_csv(tmp_path):
    storage = tmp_path / "managers"
    storage.mkdir()
    txt_file = storage / "managers.txt"
    txt_file.write_text("manager\n", encoding="utf-8")

    pattern = os.path.join(storage.name, "*.txt")

    with pytest.raises(ValueError, match="must resolve to CSV files"):
        config_model._ensure_glob_matches(pattern, base_dir=tmp_path)


def test_data_settings_rejects_non_string_glob(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("date,nav\n", encoding="utf-8")

    with pytest.raises(ValidationError) as exc:
        config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_path),
                "managers_glob": object(),
                "date_column": "Date",
                "frequency": "D",
            },
            context={"base_path": tmp_path},
        )

    assert "data.managers_glob must be a string" in str(exc.value)


def test_data_settings_rejects_invalid_frequency(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("date,nav\n", encoding="utf-8")

    with pytest.raises(ValidationError) as exc:
        config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_path),
                "managers_glob": None,
                "date_column": "Date",
                "frequency": "quarterly",
            },
            context={"base_path": tmp_path},
        )

    message = exc.value.errors()[0]["msg"]
    assert "data.frequency" in message


def test_data_settings_invalid_frequency_lists_allowed_values(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("date,nav\n", encoding="utf-8")

    with pytest.raises(ValidationError) as exc:
        config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_path),
                "managers_glob": None,
                "date_column": "Date",
                "frequency": "quarterlyish",
            },
            context={"base_path": tmp_path},
        )

    message = exc.value.errors()[0]["msg"]
    assert "Choose one of" in message
    assert "quarterlyish" in message


def test_data_settings_requires_frequency_value(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("date,nav\n", encoding="utf-8")

    with pytest.raises(ValidationError) as exc:
        config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_path),
                "managers_glob": None,
                "date_column": "Date",
                "frequency": None,
            },
            context={"base_path": tmp_path},
        )

    message = exc.value.errors()[0]["msg"]
    assert "data.frequency must be provided" in message


def test_data_settings_normalises_frequency(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("date,nav\n", encoding="utf-8")

    cfg = config_model.DataSettings.model_validate(
        {
            "csv_path": str(csv_path),
            "managers_glob": None,
            "date_column": "Date",
            "frequency": " m ",
        },
        context={"base_path": tmp_path},
    )

    assert cfg.frequency == "M"

    cfg_me = config_model.DataSettings.model_validate(
        {
            "csv_path": str(csv_path),
            "managers_glob": None,
            "date_column": "Date",
            "frequency": "me",
        },
        context={"base_path": tmp_path},
    )

    assert cfg_me.frequency == "ME"


def test_data_settings_require_source(tmp_path):
    with pytest.raises(ValidationError) as exc:
        config_model.DataSettings.model_validate(
            {
                "csv_path": None,
                "managers_glob": "",
                "date_column": "Date",
                "frequency": "D",
            },
            context={"base_path": tmp_path},
        )

    message = exc.value.errors()[0]["msg"]
    assert "data.csv_path must point" in message


def test_validate_trend_config_formats_error_messages(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("date,nav\n", encoding="utf-8")

    raw = {
        "data": {
            "csv_path": str(csv_path),
            "managers_glob": None,
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "transaction_cost_bps": -1,
        },
        "vol_adjust": {"target_vol": 1.0},
    }

    with pytest.raises(ValueError) as exc:
        config_model.validate_trend_config(raw, base_path=tmp_path)

    assert "portfolio.transaction_cost_bps" in str(exc.value)


def test_portfolio_settings_enforce_turnover_bounds():
    with pytest.raises(ValidationError) as exc:
        config_model.PortfolioSettings.model_validate(
            {
                "rebalance_calendar": "NYSE",
                "max_turnover": 1.5,
                "transaction_cost_bps": 5,
            }
        )

    assert "portfolio.max_turnover" in str(exc.value)


def test_portfolio_settings_reject_negative_cost():
    with pytest.raises(ValidationError) as exc:
        config_model.PortfolioSettings.model_validate(
            {
                "rebalance_calendar": "NYSE",
                "max_turnover": 0.5,
                "transaction_cost_bps": -1,
            }
        )

    assert "portfolio.transaction_cost_bps cannot be negative" in str(exc.value)


def test_risk_settings_require_positive_target():
    with pytest.raises(ValidationError) as exc:
        config_model.RiskSettings.model_validate({"target_vol": 0})

    assert "vol_adjust.target_vol must be greater than zero" in str(exc.value)


def test_validate_trend_config_reports_first_error(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("date,nav\n", encoding="utf-8")

    data = {
        "data": {
            "csv_path": str(csv_path),
            "managers_glob": None,
            "date_column": "Date",
            "frequency": "D",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.2,
            "transaction_cost_bps": "not-a-number",
        },
        "vol_adjust": {"target_vol": 0.1},
    }

    with pytest.raises(ValueError) as exc:
        config_model.validate_trend_config(data, base_path=tmp_path)

    assert "portfolio.transaction_cost_bps must be numeric" in str(exc.value)


def test_resolve_config_path_reports_missing_file():
    with pytest.raises(FileNotFoundError) as exc:
        config_model._resolve_config_path("nonexistent_configuration")

    assert "Configuration file" in str(exc.value)


def test_load_trend_config_requires_mapping(tmp_path):
    cfg_path = tmp_path / "invalid.yml"
    cfg_path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must contain a mapping"):
        config_model.load_trend_config(cfg_path)


def test_candidate_roots_includes_parent(tmp_path, monkeypatch):
    base = tmp_path / "config"
    base.mkdir()
    monkeypatch.chdir(tmp_path)
    roots = list(config_model._candidate_roots(base))
    assert roots[0] == base
    assert roots[1] == tmp_path
    assert Path.cwd() in roots


def test_data_settings_missing_policy_invalid_type(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("date,nav\n", encoding="utf-8")
    with pytest.raises(ValidationError) as exc:
        config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_path),
                "managers_glob": None,
                "date_column": "Date",
                "frequency": "D",
                "missing_policy": 123,
            },
            context={"base_path": tmp_path},
        )
    assert "data.missing_policy" in str(exc.value)


def test_data_settings_missing_limit_invalid_type(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("date,nav\n", encoding="utf-8")
    with pytest.raises(ValidationError) as exc:
        config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_path),
                "managers_glob": None,
                "date_column": "Date",
                "frequency": "D",
                "missing_limit": "not-int",
            },
            context={"base_path": tmp_path},
        )
    assert "data.missing_limit" in str(exc.value)


def test_risk_settings_reject_negative_floor():
    with pytest.raises(ValidationError) as exc:
        config_model.RiskSettings.model_validate({"target_vol": 0.1, "floor_vol": -0.1})
    assert "vol_adjust.floor_vol cannot be negative" in str(exc.value)

    with pytest.raises(ValidationError):
        config_model.RiskSettings.model_validate({"target_vol": 0.1, "warmup_periods": -1})


def test_resolve_config_path_uses_environment(tmp_path, monkeypatch):
    cfg = tmp_path / "demo.yml"
    cfg.write_text("data: {}\n", encoding="utf-8")
    monkeypatch.setenv("TREND_CONFIG", str(cfg))
    resolved = config_model._resolve_config_path(None)
    assert resolved == cfg.resolve()


def test_validate_trend_config_reports_location(tmp_path):
    cfg_path = tmp_path / "demo.yml"
    cfg_path.write_text("data: {}\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        config_model.validate_trend_config({}, base_path=tmp_path)
    assert "data" in str(exc.value)
