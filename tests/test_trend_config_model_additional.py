from __future__ import annotations

from pathlib import Path

import pytest

from trend_analysis.config.model import (
    DataSettings,
    PortfolioSettings,
    RiskSettings,
    _ensure_glob_matches,
    _expand_pattern,
    _resolve_config_path,
    _resolve_path,
    validate_trend_config,
)


def test_resolve_path_uses_base_dir_and_validates(tmp_path: Path) -> None:
    base_dir = tmp_path / "configs"
    base_dir.mkdir()
    csv_parent = tmp_path / "returns.csv"
    csv_parent.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")

    resolved = _resolve_path("returns.csv", base_dir=base_dir)
    assert resolved == csv_parent.resolve()

    with pytest.raises(ValueError, match="does not exist"):
        _resolve_path("missing.csv", base_dir=base_dir)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with pytest.raises(ValueError, match="points to a directory"):
        _resolve_path("data", base_dir=tmp_path)

    with pytest.raises(ValueError, match="wildcard characters"):
        _resolve_path("*.csv", base_dir=base_dir)


def test_expand_pattern_deduplicates_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    base_dir = Path.cwd()
    pattern = "inputs/*.csv"
    result = _expand_pattern(pattern, base_dir=base_dir)
    assert result[0] == base_dir / Path(pattern)
    assert result[1] == base_dir.parent / Path(pattern)
    assert len(result) == 2

    absolute = _expand_pattern(str(tmp_path / "absolute.csv"), base_dir=base_dir)
    assert absolute == [tmp_path / "absolute.csv"]


def test_ensure_glob_matches_error_branches(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="did not match any CSV files"):
        _ensure_glob_matches("missing/*.csv", base_dir=tmp_path)

    pattern_dir = tmp_path / "managers"
    pattern_dir.mkdir()
    (pattern_dir / "fund.txt").write_text("not csv", encoding="utf-8")

    with pytest.raises(ValueError, match="non-CSV inputs"):
        _ensure_glob_matches(str(pattern_dir / "*"), base_dir=None)

    (pattern_dir / "fund.csv").write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    _ensure_glob_matches(str(pattern_dir / "*.csv"), base_dir=None)


def test_data_settings_validators_optional_and_invalid(tmp_path: Path) -> None:
    base_dir = tmp_path / "configs"
    base_dir.mkdir()
    managers_dir = tmp_path / "managers"
    managers_dir.mkdir()
    csv_file = managers_dir / "fund.csv"
    csv_file.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    pattern = str(managers_dir / "*.csv")

    settings = DataSettings.model_validate(
        {
            "csv_path": "",
            "managers_glob": pattern,
            "date_column": "Date",
            "frequency": "d",
            "missing_policy": "",
            "missing_limit": "",
        },
        context={"base_path": base_dir},
    )
    assert settings.csv_path is None
    assert settings.managers_glob == pattern
    assert settings.frequency == "D"
    assert settings.missing_policy is None
    assert settings.missing_limit is None

    empty_managers = DataSettings.model_validate(
        {
            "csv_path": str(csv_file),
            "managers_glob": "",
            "date_column": "Date",
            "frequency": "M",
        },
        context={"base_path": base_dir},
    )
    assert empty_managers.managers_glob is None

    with pytest.raises(ValueError, match="managers_glob must be a string"):
        DataSettings.model_validate(
            {
                "csv_path": str(csv_file),
                "managers_glob": 123,
                "date_column": "Date",
                "frequency": "M",
            },
            context={"base_path": base_dir},
        )

    with pytest.raises(ValueError, match="missing_policy"):
        DataSettings.model_validate(
            {
                "csv_path": str(csv_file),
                "managers_glob": pattern,
                "date_column": "Date",
                "frequency": "M",
                "missing_policy": 123,
            },
            context={"base_path": base_dir},
        )

    with pytest.raises(ValueError, match="missing_limit"):
        DataSettings.model_validate(
            {
                "csv_path": str(csv_file),
                "managers_glob": pattern,
                "date_column": "Date",
                "frequency": "M",
                "missing_limit": object(),
            },
            context={"base_path": base_dir},
        )

    mapping_settings = DataSettings.model_validate(
        {
            "csv_path": str(csv_file),
            "managers_glob": pattern,
            "date_column": "Date",
            "frequency": "M",
            "missing_limit": {"fund": None},
        },
        context={"base_path": base_dir},
    )
    assert mapping_settings.missing_limit == {"fund": None}

    numeric_settings = DataSettings.model_validate(
        {
            "csv_path": str(csv_file),
            "managers_glob": pattern,
            "date_column": "Date",
            "frequency": "M",
            "missing_limit": "5",
        },
        context={"base_path": base_dir},
    )
    assert numeric_settings.missing_limit == 5

    with pytest.raises(ValueError, match="data.date_column"):
        DataSettings.model_validate(
            {
                "csv_path": str(csv_file),
                "date_column": " ",
                "frequency": "M",
            },
            context={"base_path": base_dir},
        )

    with pytest.raises(ValueError, match="must be provided"):
        DataSettings.model_validate(
            {
                "csv_path": str(csv_file),
                "date_column": "Date",
                "frequency": None,
            },
            context={"base_path": base_dir},
        )

    with pytest.raises(ValueError, match="data.csv_path must point"):
        DataSettings.model_validate(
            {
                "date_column": "Date",
                "frequency": "M",
            },
            context={"base_path": base_dir},
        )


def test_portfolio_settings_validator_errors() -> None:
    with pytest.raises(ValueError, match="trading calendar"):
        PortfolioSettings.model_validate(
            {
                "rebalance_calendar": " ",
                "max_turnover": 0.5,
                "transaction_cost_bps": 10,
            }
        )

    with pytest.raises(ValueError, match="cannot be negative"):
        PortfolioSettings.model_validate(
            {
                "rebalance_calendar": "NYSE",
                "max_turnover": -0.1,
                "transaction_cost_bps": 10,
            }
        )

    with pytest.raises(ValueError, match="between 0 and 1.0"):
        PortfolioSettings.model_validate(
            {
                "rebalance_calendar": "NYSE",
                "max_turnover": 1.5,
                "transaction_cost_bps": 10,
            }
        )

    with pytest.raises(ValueError, match="transaction_cost_bps cannot be negative"):
        PortfolioSettings.model_validate(
            {
                "rebalance_calendar": "NYSE",
                "max_turnover": 0.5,
                "transaction_cost_bps": -5,
            }
        )


def test_risk_settings_validator_errors() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        RiskSettings.model_validate(
            {"target_vol": 0, "floor_vol": 0.01, "warmup_periods": 0}
        )

    with pytest.raises(ValueError, match="floor_vol cannot be negative"):
        RiskSettings.model_validate(
            {"target_vol": 0.1, "floor_vol": -0.1, "warmup_periods": 0}
        )

    with pytest.raises(ValueError, match="warmup_periods cannot be negative"):
        RiskSettings.model_validate(
            {"target_vol": 0.1, "floor_vol": 0.01, "warmup_periods": -1}
        )


def test_resolve_config_path_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    default_path = _resolve_config_path("demo")
    assert default_path.name == "demo.yml"

    monkeypatch.delenv("TREND_CONFIG", raising=False)
    monkeypatch.delenv("TREND_CFG", raising=False)
    expected_default = _resolve_config_path(None)
    assert expected_default.name == "demo.yml"

    custom = tmp_path / "custom.yml"
    custom.write_text("version: 1\n", encoding="utf-8")
    monkeypatch.setenv("TREND_CFG", str(custom))
    resolved = _resolve_config_path(None)
    assert resolved == custom.resolve()
    monkeypatch.delenv("TREND_CFG", raising=False)

    with pytest.raises(FileNotFoundError):
        _resolve_config_path(tmp_path / "missing.yml")


def test_validate_trend_config_formats_first_error(tmp_path: Path) -> None:
    data = {
        "data": {
            "csv_path": str(tmp_path / "missing.csv"),
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {
            "rebalance_calendar": "",
            "max_turnover": 0.5,
            "transaction_cost_bps": 10,
        },
        "vol_adjust": {"target_vol": 0.1},
    }

    with pytest.raises(ValueError) as exc:
        validate_trend_config(data, base_path=tmp_path)

    assert str(exc.value).startswith("data.csv_path")
