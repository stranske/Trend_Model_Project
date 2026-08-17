from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trend_analysis.config import (
    Config,
    SignalSettings,
    TrendConfig,
    load_config,
    load_trend_config,
    validate_trend_config,
)
from trend_analysis.pipeline_helpers import _build_trend_spec


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
            "cost_model": {"per_trade_bps": 10, "half_spread_bps": 0},
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
    assert _build_trend_spec(cfg, cfg.vol_adjust) is None


@pytest.mark.parametrize(
    "cost_model",
    [
        {},
        {"per_trade_bps": 1.0},
        {"half_spread_bps": 0.5},
    ],
)
def test_public_load_config_requires_complete_cost_model(
    tmp_path: Path,
    cost_model: dict[str, float],
) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    cfg_path = _write_config(
        tmp_path,
        csv_file,
        portfolio={"cost_model": cost_model},
    )
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="per_trade_bps|half_spread_bps"):
        load_config(payload)


def test_load_trend_config_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    cfg_path = _write_config(tmp_path, csv_file)
    monkeypatch.setenv("TREND_CONFIG", str(cfg_path))

    cfg, resolved = load_trend_config()
    assert resolved == cfg_path
    assert cfg.data.csv_path == csv_file.resolve()


def test_load_trend_config_preserves_canonical_signals(tmp_path: Path) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    signals = {
        "kind": "tsmom",
        "window": 63,
        "lag": 1,
        "min_periods": None,
        "zscore": False,
        "vol_adjust": False,
        "vol_target": 0.10,
    }
    cfg_path = _write_config(tmp_path, csv_file, signals=signals)

    cfg, _ = load_trend_config(cfg_path)

    assert cfg.signals.window == 63
    assert cfg.signals.lag == 1
    assert cfg.signals.vol_target == pytest.approx(0.10)
    assert cfg.model_dump(exclude_none=True)["signals"] == {
        key: value for key, value in signals.items() if value is not None
    }
    spec = _build_trend_spec(cfg, cfg.vol_adjust)
    assert spec is not None
    assert spec.window == 63


@pytest.mark.parametrize("signals", [{"trend": {"window": 20}}, {"windw": 20}])
def test_load_trend_config_rejects_unknown_signal_shapes(
    tmp_path: Path,
    signals: dict[str, object],
) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    cfg_path = _write_config(tmp_path, csv_file, signals=signals)

    with pytest.raises(ValueError, match="signals"):
        load_trend_config(cfg_path)


@pytest.mark.parametrize("signals", [{"trend": {"window": 20}}, {"windw": 20}])
def test_runtime_config_rejects_unknown_signal_shapes(signals: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="signals"):
        Config(
            version="1",
            portfolio={"cost_model": {"per_trade_bps": 0, "half_spread_bps": 0}},
            signals=signals,
        )


def test_signal_min_periods_cannot_exceed_window() -> None:
    with pytest.raises(ValueError, match="min_periods.*window"):
        SignalSettings(window=5, min_periods=6)

    with pytest.raises(ValueError, match="min_periods.*window"):
        Config(
            version="1",
            portfolio={"cost_model": {"per_trade_bps": 0, "half_spread_bps": 0}},
            signals={"window": 5, "min_periods": 6},
        )


def test_signal_settings_match_canonical_trend_spec_bounds() -> None:
    settings = SignalSettings(
        kind="tsmom",
        window=1,
        lag=11,
        min_periods=1,
        vol_target=0.001,
    )

    assert settings.kind == "tsmom"
    assert settings.window == 1
    assert settings.lag == 11
    assert settings.vol_target == pytest.approx(0.001)

    with pytest.raises(ValueError, match="kind"):
        SignalSettings(kind="cross_sectional")  # type: ignore[arg-type]


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
            "cost_model": {"per_trade_bps": 10, "half_spread_bps": 0},
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
    (managers_dir / "fund_a.csv").write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

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
            "cost_model": {"per_trade_bps": 10, "half_spread_bps": 0},
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
            "cost_model": {"per_trade_bps": 10, "half_spread_bps": 0},
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
            "cost_model": {"per_trade_bps": 10, "half_spread_bps": 0},
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
            "cost_model": {"per_trade_bps": 10, "half_spread_bps": 0},
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
            "cost_model": {"per_trade_bps": 10, "half_spread_bps": 0},
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
            "cost_model": {"per_trade_bps": 10, "half_spread_bps": 0},
        },
        "vol_adjust": {"target_vol": 0.1},
    }

    with pytest.raises(ValueError) as exc:
        validate_trend_config(cfg, base_path=tmp_path)

    assert "data.frequency 'quarterlyish'" in str(exc.value)


def test_validate_trend_config_locates_csv_relative_to_parent(tmp_path: Path) -> None:
    base_dir = tmp_path / "configs"
    base_dir.mkdir()
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    relative_path = csv_file.relative_to(base_dir.parent)
    cfg_path = _write_config(
        base_dir,
        relative_path,
    )

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    validated = validate_trend_config(cfg, base_path=base_dir)
    assert validated.data.csv_path == csv_file.resolve()


def test_validate_trend_config_rejects_directory_csv_path(tmp_path: Path) -> None:
    csv_dir = tmp_path / "inputs"
    csv_dir.mkdir()

    cfg = {
        "version": "1",
        "data": {
            "csv_path": str(csv_dir),
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "cost_model": {"per_trade_bps": 10, "half_spread_bps": 0},
        },
        "vol_adjust": {"target_vol": 0.1},
    }

    with pytest.raises(ValueError) as exc:
        validate_trend_config(cfg, base_path=tmp_path)

    assert "points to a directory" in str(exc.value)


def test_validate_trend_config_accepts_pathlike_managers_glob(tmp_path: Path) -> None:
    manager_file = tmp_path / "fund.csv"
    manager_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    cfg = {
        "version": "1",
        "data": {
            "managers_glob": manager_file,
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "cost_model": {"per_trade_bps": 10, "half_spread_bps": 0},
        },
        "vol_adjust": {"target_vol": 0.1},
    }

    validated = validate_trend_config(cfg, base_path=tmp_path)
    assert validated.data.managers_glob == str(manager_file.resolve())


def test_validate_trend_config_reports_validation_location(tmp_path: Path) -> None:
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    cfg = {
        "version": "1",
        "data": {
            "csv_path": str(csv_file),
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 2,
            "cost_model": {"per_trade_bps": 10, "half_spread_bps": 0},
        },
        "vol_adjust": {"target_vol": 0.1},
    }

    with pytest.raises(ValueError) as exc:
        validate_trend_config(cfg, base_path=tmp_path)

    assert str(exc.value).startswith("portfolio.max_turnover")


def test_load_trend_config_accepts_relative_file_without_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    csv_file = tmp_path / "returns.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    cfg_path = cfg_dir / "custom.yml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "version": "1",
                "data": {
                    "csv_path": str(csv_file),
                    "date_column": "Date",
                    "frequency": "M",
                },
                "portfolio": {
                    "rebalance_calendar": "NYSE",
                    "max_turnover": 0.5,
                    "cost_model": {"per_trade_bps": 10, "half_spread_bps": 0},
                },
                "vol_adjust": {"target_vol": 0.1},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(cfg_dir)
    cfg, resolved = load_trend_config("custom")
    assert resolved == cfg_path.resolve()
    assert cfg.data.csv_path == csv_file.resolve()


def test_load_trend_config_rejects_non_mapping(tmp_path: Path) -> None:
    cfg_path = tmp_path / "list.yml"
    cfg_path.write_text("- item\n- other\n", encoding="utf-8")

    with pytest.raises(TypeError) as exc:
        load_trend_config(cfg_path)

    assert "mapping" in str(exc.value)
