from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from trend_analysis.config import model as config_model


def test_resolve_path_prefers_base_dir_and_errors(tmp_path: Path) -> None:
    base_dir = tmp_path / "inputs"
    base_dir.mkdir()
    csv_path = base_dir / "returns.csv"
    csv_path.write_text("Date,Value\n2024-01-31,1\n", encoding="utf-8")

    resolved = config_model._resolve_path("returns.csv", base_dir=base_dir)
    assert resolved == csv_path

    absolute = config_model._resolve_path(csv_path, base_dir=None)
    assert absolute == csv_path

    with pytest.raises(ValueError, match="does not exist"):
        config_model._resolve_path("missing.csv", base_dir=base_dir)

    with pytest.raises(ValueError, match="points to a directory"):
        config_model._resolve_path(base_dir, base_dir=None)

    with pytest.raises(ValueError, match="contains wildcard"):
        config_model._resolve_path("*.csv", base_dir=base_dir)


def test_expand_pattern_and_glob_validation(tmp_path: Path) -> None:
    managers_dir = tmp_path / "managers"
    managers_dir.mkdir()
    (managers_dir / "alpha.csv").write_text("", encoding="utf-8")
    (managers_dir / "beta.txt").write_text("", encoding="utf-8")

    pattern = str(managers_dir / "*.csv")
    config_model._ensure_glob_matches(pattern, base_dir=None)

    expanded = config_model._expand_pattern("managers/*.csv", base_dir=tmp_path)
    assert any(str(candidate).endswith("managers/*.csv") for candidate in expanded)

    with pytest.raises(ValueError, match="did not match any CSV files"):
        config_model._ensure_glob_matches("missing/*.csv", base_dir=tmp_path)

    with pytest.raises(ValueError, match="must resolve to CSV"):
        config_model._ensure_glob_matches(str(managers_dir / "*.txt"), base_dir=None)


def _write_returns_csv(base_dir: Path) -> Path:
    csv_path = base_dir / "returns.csv"
    csv_path.write_text("Date,FundA\n2024-01-31,0.01\n", encoding="utf-8")
    return csv_path


def test_data_settings_validators(tmp_path: Path) -> None:
    base_dir = tmp_path / "cfg"
    base_dir.mkdir()
    csv_path = _write_returns_csv(base_dir)

    settings = config_model.DataSettings.model_validate(
        {
            "csv_path": "returns.csv",
            "date_column": "Date",
            "frequency": "m",
            "missing_policy": {"*": "ffill"},
            "missing_limit": "5",
        },
        context={"base_path": base_dir},
    )

    assert settings.csv_path == csv_path
    assert settings.frequency == "M"
    assert settings.missing_limit == 5

    (base_dir / "managers.csv").write_text("", encoding="utf-8")
    glob_settings = config_model.DataSettings.model_validate(
        {
            "managers_glob": "*.csv",
            "date_column": "Date",
            "frequency": "W",
        },
        context={"base_path": base_dir},
    )
    assert glob_settings.managers_glob == "*.csv"

    with pytest.raises(ValueError, match="not supported"):
        config_model.DataSettings.model_validate(
            {
                "csv_path": csv_path,
                "date_column": "Date",
                "frequency": "hourly",
            },
            context={"base_path": base_dir},
        )

    with pytest.raises(ValueError, match="must point to the returns CSV file"):
        config_model.DataSettings.model_validate(
            {"date_column": "Date", "frequency": "D"},
            context={"base_path": base_dir},
        )

    with pytest.raises(ValueError, match="must be a string"):
        config_model.DataSettings.model_validate(
            {
                "managers_glob": 123,
                "date_column": "Date",
                "frequency": "D",
            },
            context={"base_path": base_dir},
        )


def test_portfolio_and_risk_settings_validation() -> None:
    with pytest.raises(ValueError, match="trading calendar"):
        config_model.PortfolioSettings.model_validate(
            {
                "rebalance_calendar": " ",
                "max_turnover": 0.5,
                "transaction_cost_bps": 5,
            }
        )

    with pytest.raises(ValueError, match="between 0 and 1"):
        config_model.PortfolioSettings.model_validate(
            {
                "rebalance_calendar": "NYSE",
                "max_turnover": 1.5,
                "transaction_cost_bps": 0,
            }
        )

    with pytest.raises(ValueError, match="cannot be negative"):
        config_model.PortfolioSettings.model_validate(
            {
                "rebalance_calendar": "NYSE",
                "max_turnover": 0.25,
                "transaction_cost_bps": -1,
            }
        )

    with pytest.raises(ValueError, match="must be greater than zero"):
        config_model.RiskSettings.model_validate(
            {"target_vol": 0.0, "floor_vol": 0.01, "warmup_periods": 0}
        )

    with pytest.raises(ValueError, match="cannot be negative"):
        config_model.RiskSettings.model_validate(
            {"target_vol": 0.1, "floor_vol": -0.01, "warmup_periods": 0}
        )

    with pytest.raises(ValueError, match="cannot be negative"):
        config_model.RiskSettings.model_validate(
            {"target_vol": 0.1, "floor_vol": 0.01, "warmup_periods": -1}
        )

    risk = config_model.RiskSettings.model_validate(
        {"target_vol": 0.1, "floor_vol": 0.02, "warmup_periods": 6}
    )
    assert risk.target_vol == pytest.approx(0.1)


def _minimal_config(base_dir: Path) -> dict[str, Any]:
    csv_path = _write_returns_csv(base_dir)
    return {
        "data": {
            "csv_path": str(csv_path),
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "transaction_cost_bps": 0.0,
        },
        "vol_adjust": {
            "target_vol": 0.1,
            "floor_vol": 0.02,
            "warmup_periods": 3,
        },
    }


def test_validate_trend_config_formats_errors(tmp_path: Path) -> None:
    data = _minimal_config(tmp_path)
    data["data"]["frequency"] = "invalid"

    with pytest.raises(ValueError, match="data.frequency"):
        config_model.validate_trend_config(data, base_path=tmp_path)


def test_load_trend_config_reads_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "trend.yml"
    config_data = _minimal_config(tmp_path)
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    cfg, loaded_path = config_model.load_trend_config(config_path)
    assert loaded_path == config_path
    assert cfg.data.csv_path == Path(config_data["data"]["csv_path"])

    bad_path = tmp_path / "bad.yml"
    bad_path.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")
    with pytest.raises(TypeError, match="must contain a mapping"):
        config_model.load_trend_config(bad_path)


def test_resolve_path_checks_parent_and_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = tmp_path / "nested" / "inner"
    base_dir.mkdir(parents=True)
    parent_file = base_dir.parent / "parent.csv"
    parent_file.write_text("", encoding="utf-8")

    # When the file lives in the parent directory the helper should find it.
    resolved_parent = config_model._resolve_path("parent.csv", base_dir=base_dir)
    assert resolved_parent == parent_file.resolve()

    cwd_file = tmp_path / "cwd.csv"
    cwd_file.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    resolved_cwd = config_model._resolve_path("cwd.csv", base_dir=None)
    assert resolved_cwd == cwd_file.resolve()


def test_expand_pattern_deduplicates_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = tmp_path / "dup"
    base_dir.mkdir()
    monkeypatch.chdir(base_dir)
    expanded = config_model._expand_pattern("example.csv", base_dir=base_dir)
    # Only two unique candidates should be produced (base_dir and cwd which are identical).
    assert len(expanded) == 2
    assert expanded[0] != expanded[1]


def test_data_settings_optional_branches(tmp_path: Path) -> None:
    base_dir = tmp_path / "cfg_optional"
    base_dir.mkdir()
    managers = base_dir / "m.csv"
    managers.write_text("", encoding="utf-8")

    settings = config_model.DataSettings.model_validate(
        {
            "csv_path": None,
            "managers_glob": "*.csv",
            "date_column": "Date",
            "frequency": "ME",
            "missing_policy": None,
            "missing_limit": {"A": 3},
        },
        context={"base_path": base_dir},
    )

    assert settings.csv_path is None
    assert settings.frequency == "ME"
    assert settings.missing_policy is None
    assert settings.missing_limit == {"A": 3}

    with pytest.raises(ValueError, match="must be an integer"):
        config_model.DataSettings.model_validate(
            {
                "csv_path": managers,
                "date_column": "Date",
                "frequency": "M",
                "missing_limit": "five",
            },
            context={"base_path": base_dir},
        )

    with pytest.raises(ValueError, match="must be a string or mapping"):
        config_model.DataSettings.model_validate(
            {
                "csv_path": managers,
                "date_column": "Date",
                "frequency": "M",
                "missing_policy": 123,
            },
            context={"base_path": base_dir},
        )


def test_portfolio_settings_negative_turnover_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        config_model.PortfolioSettings.model_validate(
            {
                "rebalance_calendar": "NYSE",
                "max_turnover": -0.1,
                "transaction_cost_bps": 0,
            }
        )


def test_resolve_config_path_uses_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    config_file = cfg_dir / "custom.yml"
    config_file.write_text(yaml.safe_dump(_minimal_config(tmp_path)), encoding="utf-8")

    monkeypatch.setenv("TREND_CONFIG", str(config_file))
    resolved = config_model._resolve_config_path(None)
    assert resolved == config_file.resolve()

    config_file_no_suffix = cfg_dir / "alias.yml"
    config_file_no_suffix.write_text("{}", encoding="utf-8")
    resolved_with_suffix = config_model._resolve_config_path(
        str(config_file_no_suffix.with_suffix(""))
    )
    assert resolved_with_suffix == config_file_no_suffix.resolve()


def test_candidate_roots_includes_base_and_parent(tmp_path: Path) -> None:
    base_dir = tmp_path / "roots"
    base_dir.mkdir()
    roots = list(config_model._candidate_roots(base_dir))
    assert base_dir in roots
    assert base_dir.parent in roots
    assert Path.cwd() in roots


def test_data_settings_pathlike_managers_glob(tmp_path: Path) -> None:
    base_dir = tmp_path / "mgr_path"
    base_dir.mkdir()
    managers_file = base_dir / "managers.csv"
    managers_file.write_text("", encoding="utf-8")

    settings = config_model.DataSettings.model_validate(
        {
            "csv_path": managers_file,
            "managers_glob": managers_file,
            "date_column": "Date",
            "frequency": "M",
        },
        context={"base_path": base_dir},
    )
    assert settings.managers_glob == str(managers_file)

    with pytest.raises(ValueError, match="must be a non-empty string"):
        config_model.DataSettings.model_validate(
            {
                "csv_path": managers_file,
                "date_column": " ",
                "frequency": "M",
            },
            context={"base_path": base_dir},
        )

    with pytest.raises(ValueError, match="must be provided"):
        config_model.DataSettings.model_validate(
            {
                "csv_path": managers_file,
                "date_column": "Date",
                "frequency": None,
            },
            context={"base_path": base_dir},
        )


def test_resolve_config_path_defaults_to_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TREND_CONFIG", raising=False)
    monkeypatch.delenv("TREND_CFG", raising=False)
    resolved = config_model._resolve_config_path(None)
    assert resolved.name == "demo.yml"


def test_resolve_config_path_prefers_repo_relative(tmp_path: Path) -> None:
    repo_config = config_model._CONFIG_DIR  # type: ignore[attr-defined]
    custom = repo_config / "temporary_test_config.yml"
    custom.write_text("{}", encoding="utf-8")
    try:
        resolved = config_model._resolve_config_path("temporary_test_config")
        assert resolved == custom.resolve()
    finally:
        custom.unlink()


def test_validate_trend_config_error_message_contains_location(tmp_path: Path) -> None:
    data = _minimal_config(tmp_path)
    data["portfolio"]["max_turnover"] = 2  # type: ignore[index]
    with pytest.raises(ValueError, match="portfolio.max_turnover"):
        config_model.validate_trend_config(data, base_path=tmp_path)
