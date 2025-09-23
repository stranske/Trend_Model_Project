from __future__ import annotations

from pathlib import Path

import pytest

import yaml
from trend_analysis.config import (
    TrendConfig,
    load_config,
    load_trend_config,
    validate_trend_config,
)


def _write_config(tmp_path: Path, csv_path: Path, **overrides: object) -> Path:
    data = {
        "version": "1",
        "data": {
            "csv_path": str(csv_path),
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "transaction_cost_bps": 10,
        },
        "vol_adjust": {"target_vol": 0.1},
    }
    for key, value in overrides.items():
        if isinstance(value, dict):
            data.setdefault(key, {}).update(value)
        else:
            data[key] = value
    cfg_path = tmp_path / "test.yml"
    cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return cfg_path


def test_load_trend_config_defaults() -> None:
    cfg, resolved = load_trend_config("demo")
    assert resolved.name == "demo.yml"
    # The demo dataset is checked into the repository so the path should exist.
    assert cfg.data.csv_path.exists()
    assert cfg.data.date_column == "Date"
    assert isinstance(cfg, TrendConfig)


def test_load_trend_config_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    cfg_path = _write_config(tmp_path, csv_file)
    monkeypatch.setenv("TREND_CONFIG", str(cfg_path))

    cfg, resolved = load_trend_config()
    assert resolved == cfg_path
    assert cfg.data.csv_path == csv_file.resolve()


def test_trend_config_rejects_invalid_frequency(tmp_path: Path) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    cfg_path = _write_config(
        tmp_path,
        csv_file,
        data={"frequency": "dailyish"},
    )

    with pytest.raises(ValueError) as exc:
        load_trend_config(cfg_path)
    assert "frequency" in str(exc.value)


def test_trend_config_requires_existing_paths(tmp_path: Path) -> None:
    csv_file = tmp_path / "missing.csv"
    cfg_path = _write_config(tmp_path, csv_file)

    with pytest.raises(ValueError) as exc:
        load_trend_config(cfg_path)
    assert "does not exist" in str(exc.value)


def test_load_config_mapping_requires_source(tmp_path: Path) -> None:
    cfg = {
        "version": "1",
        "data": {"date_column": "Date", "frequency": "M"},
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "transaction_cost_bps": 10,
        },
        "vol_adjust": {"target_vol": 0.1},
        "preprocessing": {},
        "sample_split": {},
        "metrics": {},
        "export": {},
        "run": {},
    }

    with pytest.raises(ValueError) as exc:
        load_config(cfg)
    assert "data.csv_path" in str(exc.value)


def test_trend_config_accepts_valid_managers_glob(tmp_path: Path) -> None:
    managers_dir = tmp_path / "managers"
    managers_dir.mkdir()
    (managers_dir / "fund_a.csv").write_text(
        "Date,A\n2020-01-31,0.1\n", encoding="utf-8"
    )

    cfg = {
        "version": "1",
        "data": {
            "managers_glob": str(managers_dir / "*.csv"),
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "transaction_cost_bps": 10,
        },
        "vol_adjust": {"target_vol": 0.1},
    }

    validated = validate_trend_config(cfg, base_path=tmp_path)
    assert validated.data.managers_glob == str(managers_dir / "*.csv")


def test_trend_config_requires_matching_managers_glob(tmp_path: Path) -> None:
    cfg = {
        "version": "1",
        "data": {
            "managers_glob": str(tmp_path / "missing" / "*.csv"),
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "transaction_cost_bps": 10,
        },
        "vol_adjust": {"target_vol": 0.1},
    }

    with pytest.raises(ValueError) as exc:
        validate_trend_config(cfg, base_path=tmp_path)
    assert "managers_glob" in str(exc.value)


def test_trend_config_managers_glob_requires_csv_extension(tmp_path: Path) -> None:
    data_dir = tmp_path / "inputs"
    data_dir.mkdir()
    (data_dir / "fund_a.txt").write_text("This is not a CSV file.", encoding="utf-8")

    cfg = {
        "version": "1",
        "data": {
            "managers_glob": str(data_dir / "*"),
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "transaction_cost_bps": 10,
        },
        "vol_adjust": {"target_vol": 0.1},
    }

    with pytest.raises(ValueError) as exc:
        validate_trend_config(cfg, base_path=tmp_path)
    message = str(exc.value)
    assert "CSV" in message
    assert "fund_a.txt" in message


def test_validate_trend_config_normalises_month_end_frequency(tmp_path: Path) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    cfg = {
        "version": "1",
        "data": {
            "csv_path": str(csv_file),
            "date_column": "Date",
            "frequency": "me",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "transaction_cost_bps": 10,
        },
        "vol_adjust": {"target_vol": 0.1},
    }

    validated = validate_trend_config(cfg, base_path=tmp_path)
    assert validated.data.frequency == "ME"


def test_validate_trend_config_normalises_weekly_frequency(tmp_path: Path) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    cfg = {
        "version": "1",
        "data": {
            "csv_path": str(csv_file),
            "date_column": "Date",
            "frequency": "w",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "transaction_cost_bps": 10,
        },
        "vol_adjust": {"target_vol": 0.1},
    }

    validated = validate_trend_config(cfg, base_path=tmp_path)
    assert validated.data.frequency == "W"


def test_validate_trend_config_reports_frequency_error_message(tmp_path: Path) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    cfg = {
        "version": "1",
        "data": {
            "csv_path": str(csv_file),
            "date_column": "Date",
            "frequency": "quarterlyish",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "transaction_cost_bps": 10,
        },
        "vol_adjust": {"target_vol": 0.1},
    }

    with pytest.raises(ValueError) as exc:
        validate_trend_config(cfg, base_path=tmp_path)

    assert "data.frequency 'quarterlyish'" in str(exc.value)
