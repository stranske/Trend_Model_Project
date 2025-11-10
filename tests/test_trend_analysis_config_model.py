"""Unit tests for :mod:`trend_analysis.config.model`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from trend_analysis.config import model as config_model


def test_resolve_path_prefers_base_directory(tmp_path: Path) -> None:
    csv_path = tmp_path / "data" / "returns.csv"
    csv_path.parent.mkdir()
    csv_path.write_text("date,asset\n2020-01-01,1.0\n")

    resolved = config_model._resolve_path("returns.csv", base_dir=csv_path.parent)

    assert resolved == csv_path.resolve()


def test_resolve_path_falls_back_to_parent_directory(tmp_path: Path) -> None:
    base_dir = tmp_path / "configs" / "nested"
    base_dir.mkdir(parents=True)
    csv_path = tmp_path / "configs" / "returns.csv"
    csv_path.write_text("date,asset\n2020-01-01,1.0\n")

    resolved = config_model._resolve_path("returns.csv", base_dir=base_dir)

    assert resolved == csv_path.resolve()


def test_resolve_path_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        config_model._resolve_path("missing.csv", base_dir=tmp_path)


def test_resolve_path_rejects_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="points to a directory"):
        config_model._resolve_path(tmp_path, base_dir=None)


def test_resolve_path_rejects_glob_patterns(tmp_path: Path) -> None:
    file_path = tmp_path / "returns.csv"
    file_path.write_text("date,asset\n2020-01-01,1.0\n")

    with pytest.raises(ValueError, match="wildcard characters"):
        config_model._resolve_path("*.csv", base_dir=file_path.parent)


def test_expand_pattern_handles_absolute_path(tmp_path: Path) -> None:
    absolute = tmp_path / "pattern.csv"

    result = config_model._expand_pattern(str(absolute), base_dir=None)

    assert result == [absolute]


def test_expand_pattern_deduplicates_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    base_dir = tmp_path

    expanded = config_model._expand_pattern("data/*.csv", base_dir=base_dir)

    # When base_dir == cwd, the cwd entry should be deduplicated.
    assert expanded.count(base_dir / "data" / "*.csv") == 1


def test_ensure_glob_matches_returns_csv_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "returns.csv").write_text("date,asset\n2020-01-01,1.0\n")

    config_model._ensure_glob_matches("data/*.csv", base_dir=None)


def test_ensure_glob_matches_requires_csv_extension(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "returns.txt").write_text("placeholder")

    with pytest.raises(ValueError, match="must resolve to CSV"):
        config_model._ensure_glob_matches("data/*.txt", base_dir=tmp_path)


def test_ensure_glob_matches_raises_when_no_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="did not match any CSV files"):
        config_model._ensure_glob_matches("data/*.csv", base_dir=tmp_path)


def _basic_data_settings_dict(tmp_path: Path) -> dict[str, object]:
    returns = tmp_path / "returns.csv"
    returns.write_text("date,asset\n2020-01-01,1.0\n")
    return {
        "csv_path": "returns.csv",
        "date_column": "date",
        "frequency": "D",
    }


def test_data_settings_resolves_paths_with_context(tmp_path: Path) -> None:
    payload = _basic_data_settings_dict(tmp_path)

    result = config_model.DataSettings.model_validate(
        payload,
        context={"base_path": tmp_path},
    )

    assert result.csv_path == (tmp_path / "returns.csv")


def test_data_settings_supports_managers_glob(tmp_path: Path) -> None:
    (tmp_path / "managers").mkdir()
    manager_csv = tmp_path / "managers" / "manager_a.csv"
    manager_csv.write_text("id,value\n1,42\n")
    payload = _basic_data_settings_dict(tmp_path)
    payload["csv_path"] = None
    payload["managers_glob"] = "managers/*.csv"

    result = config_model.DataSettings.model_validate(
        payload,
        context={"base_path": tmp_path},
    )

    assert result.managers_glob == "managers/*.csv"
    assert result.csv_path is None


def test_data_settings_accepts_concrete_manager_path(tmp_path: Path) -> None:
    manager_csv = tmp_path / "manager.csv"
    manager_csv.write_text("id,value\n1,42\n")
    payload = _basic_data_settings_dict(tmp_path)
    payload["managers_glob"] = "manager.csv"

    result = config_model.DataSettings.model_validate(
        payload,
        context={"base_path": tmp_path},
    )

    assert result.managers_glob == str(manager_csv)


def test_data_settings_accepts_pathlike_manager_pattern(tmp_path: Path) -> None:
    manager_csv = tmp_path / "manager.csv"
    manager_csv.write_text("id,value\n1,42\n")
    payload = _basic_data_settings_dict(tmp_path)
    payload["managers_glob"] = manager_csv

    result = config_model.DataSettings.model_validate(
        payload,
        context={"base_path": tmp_path},
    )

    assert result.managers_glob == str(manager_csv)


def test_data_settings_rejects_empty_date_column(tmp_path: Path) -> None:
    payload = _basic_data_settings_dict(tmp_path)
    payload["date_column"] = "  "

    with pytest.raises(ValueError, match="date_column must be a non-empty string"):
        config_model.DataSettings.model_validate(
            payload,
            context={"base_path": tmp_path},
        )


@pytest.mark.parametrize("invalid", [None, [], 42])
def test_data_settings_requires_string_date_column(
    invalid: object, tmp_path: Path
) -> None:
    payload = _basic_data_settings_dict(tmp_path)
    payload["date_column"] = invalid

    with pytest.raises(ValidationError):
        config_model.DataSettings.model_validate(
            payload,
            context={"base_path": tmp_path},
        )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("d", "D"),
        ("ME", "ME"),
    ],
)
def test_data_settings_normalises_frequency(
    value: str, expected: str, tmp_path: Path
) -> None:
    payload = _basic_data_settings_dict(tmp_path)
    payload["frequency"] = value

    result = config_model.DataSettings.model_validate(
        payload,
        context={"base_path": tmp_path},
    )

    assert result.frequency == expected


def test_data_settings_rejects_unknown_frequency(tmp_path: Path) -> None:
    payload = _basic_data_settings_dict(tmp_path)
    payload["frequency"] = "quarterly"

    with pytest.raises(ValueError, match="is not supported"):
        config_model.DataSettings.model_validate(
            payload,
            context={"base_path": tmp_path},
        )


def test_data_settings_requires_frequency(tmp_path: Path) -> None:
    payload = _basic_data_settings_dict(tmp_path)
    payload["frequency"] = None

    with pytest.raises(ValueError, match="must be provided"):
        config_model.DataSettings.model_validate(
            payload,
            context={"base_path": tmp_path},
        )


def test_data_settings_requires_source(tmp_path: Path) -> None:
    payload = _basic_data_settings_dict(tmp_path)
    payload["csv_path"] = None

    with pytest.raises(ValueError, match="must point to the returns CSV file"):
        config_model.DataSettings.model_validate(
            payload,
            context={"base_path": tmp_path},
        )


def test_data_settings_missing_policy_and_limit_validation(tmp_path: Path) -> None:
    payload = _basic_data_settings_dict(tmp_path)
    payload["missing_policy"] = {"asset": "forward_fill"}
    payload["missing_limit"] = {"asset": 2}

    result = config_model.DataSettings.model_validate(
        payload,
        context={"base_path": tmp_path},
    )

    assert result.missing_policy == {"asset": "forward_fill"}
    assert result.missing_limit == {"asset": 2}


def test_data_settings_missing_policy_defaults_to_none(tmp_path: Path) -> None:
    payload = _basic_data_settings_dict(tmp_path)
    payload["missing_policy"] = ""

    result = config_model.DataSettings.model_validate(
        payload,
        context={"base_path": tmp_path},
    )

    assert result.missing_policy is None


def test_data_settings_missing_limit_from_string(tmp_path: Path) -> None:
    payload = _basic_data_settings_dict(tmp_path)
    payload["missing_limit"] = "3"

    result = config_model.DataSettings.model_validate(
        payload,
        context={"base_path": tmp_path},
    )

    assert result.missing_limit == 3


def test_data_settings_missing_limit_null_string(tmp_path: Path) -> None:
    payload = _basic_data_settings_dict(tmp_path)
    payload["missing_limit"] = "null"

    result = config_model.DataSettings.model_validate(
        payload,
        context={"base_path": tmp_path},
    )

    assert result.missing_limit is None


def test_data_settings_missing_limit_invalid(tmp_path: Path) -> None:
    payload = _basic_data_settings_dict(tmp_path)
    payload["missing_limit"] = "many"

    with pytest.raises(ValueError, match="missing_limit must be an integer"):
        config_model.DataSettings.model_validate(
            payload,
            context={"base_path": tmp_path},
        )


def test_data_settings_missing_policy_invalid(tmp_path: Path) -> None:
    payload = _basic_data_settings_dict(tmp_path)
    payload["missing_policy"] = 123

    with pytest.raises(ValueError, match="missing_policy must be a string or mapping"):
        config_model.DataSettings.model_validate(
            payload,
            context={"base_path": tmp_path},
        )


def test_portfolio_settings_validation(tmp_path: Path) -> None:
    payload = {
        "rebalance_calendar": "NYSE",
        "max_turnover": "0.5",
        "transaction_cost_bps": "15",
    }

    result = config_model.PortfolioSettings.model_validate(payload)

    assert result.max_turnover == pytest.approx(0.5)
    assert result.transaction_cost_bps == pytest.approx(15.0)


def test_portfolio_settings_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="turnover cannot be negative"):
        config_model.PortfolioSettings.model_validate(
            {
                "rebalance_calendar": "NYSE",
                "max_turnover": -0.1,
                "transaction_cost_bps": 5,
            }
        )
    with pytest.raises(ValueError, match="must be between 0 and 1.0"):
        config_model.PortfolioSettings.model_validate(
            {
                "rebalance_calendar": "NYSE",
                "max_turnover": 1.5,
                "transaction_cost_bps": 5,
            }
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        config_model.PortfolioSettings.model_validate(
            {
                "rebalance_calendar": "NYSE",
                "max_turnover": 0.4,
                "transaction_cost_bps": -1,
            }
        )


def test_portfolio_settings_requires_calendar() -> None:
    with pytest.raises(ValueError, match="must name a valid trading calendar"):
        config_model.PortfolioSettings.model_validate(
            {
                "rebalance_calendar": " ",
                "max_turnover": 0.1,
                "transaction_cost_bps": 5,
            }
        )


def test_risk_settings_validation() -> None:
    result = config_model.RiskSettings.model_validate(
        {
            "target_vol": "0.2",
            "floor_vol": "0.01",
            "warmup_periods": "3",
        }
    )

    assert result.target_vol == pytest.approx(0.2)
    assert result.floor_vol == pytest.approx(0.01)
    assert result.warmup_periods == 3


def test_risk_settings_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="must be greater than zero"):
        config_model.RiskSettings.model_validate(
            {"target_vol": 0, "floor_vol": 0.02, "warmup_periods": 1}
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        config_model.RiskSettings.model_validate(
            {"target_vol": 0.1, "floor_vol": -0.1, "warmup_periods": 1}
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        config_model.RiskSettings.model_validate(
            {"target_vol": 0.1, "floor_vol": 0.02, "warmup_periods": -1}
        )


def test_resolve_config_path_uses_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "custom.yml"
    config_file.write_text("demo: true\n")
    monkeypatch.setenv("TREND_CONFIG", str(config_file))

    resolved = config_model._resolve_config_path(None)

    assert resolved == config_file.resolve()


def test_resolve_config_path_adds_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "analysis.yml"
    config_file.write_text("demo: true\n")
    monkeypatch.chdir(tmp_path)

    resolved = config_model._resolve_config_path("analysis")

    assert resolved == config_file.resolve()


def test_resolve_config_path_prefers_repo_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_config = Path(__file__).resolve().parents[1] / "config" / "demo.yml"
    assert repo_config.exists(), "expected bundled demo config"
    monkeypatch.delenv("TREND_CONFIG", raising=False)
    monkeypatch.delenv("TREND_CFG", raising=False)

    resolved = config_model._resolve_config_path(None)

    assert resolved == repo_config.resolve()


def test_resolve_config_path_raises_for_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="was not found"):
        config_model._resolve_config_path("missing.yml")


def _valid_config_payload(tmp_path: Path) -> dict[str, object]:
    returns = tmp_path / "returns.csv"
    returns.write_text("date,asset\n2020-01-01,1.0\n")
    return {
        "data": {
            "csv_path": "returns.csv",
            "date_column": "date",
            "frequency": "D",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "transaction_cost_bps": 10,
        },
        "vol_adjust": {
            "target_vol": 0.2,
            "floor_vol": 0.01,
            "warmup_periods": 0,
        },
    }


def test_validate_trend_config_success(tmp_path: Path) -> None:
    payload = _valid_config_payload(tmp_path)

    cfg = config_model.validate_trend_config(payload, base_path=tmp_path)

    assert cfg.data.csv_path == (tmp_path / "returns.csv")
    assert cfg.portfolio.max_turnover == pytest.approx(0.5)
    assert cfg.vol_adjust.target_vol == pytest.approx(0.2)


def test_validate_trend_config_raises_value_error(tmp_path: Path) -> None:
    payload = _valid_config_payload(tmp_path)
    payload["data"]["csv_path"] = None

    with pytest.raises(ValueError, match="data.csv_path must point to the returns CSV"):
        config_model.validate_trend_config(payload, base_path=tmp_path)


def test_validate_trend_config_formats_nested_error(tmp_path: Path) -> None:
    payload = _valid_config_payload(tmp_path)
    payload["data"]["frequency"] = "invalid"

    with pytest.raises(
        ValueError,
        match=r"data\.frequency: Value error, data\.frequency 'invalid' is not supported",
    ):
        config_model.validate_trend_config(payload, base_path=tmp_path)


def test_load_trend_config_parses_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_config_payload(tmp_path)
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump(payload))
    monkeypatch.chdir(tmp_path)

    cfg, path = config_model.load_trend_config("config.yml")

    assert path == config_file.resolve()
    assert cfg.data.frequency == "D"


def test_load_trend_config_requires_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump([1, 2, 3]))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(TypeError, match="must contain a mapping"):
        config_model.load_trend_config("config.yml")
