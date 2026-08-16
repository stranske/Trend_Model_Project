from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from trend_analysis.config import model as config_model
from trend_analysis.config.lint_keys import _load_defaults, lint_portfolio_keys


def _write_returns_csv(base_dir: Path) -> Path:
    csv_path = base_dir / "returns.csv"
    csv_path.write_text("Date,FundA\n2024-01-31,0.01\n", encoding="utf-8")
    return csv_path


def test_unknown_data_key_rejected(tmp_path: Path) -> None:
    csv_path = _write_returns_csv(tmp_path)

    with pytest.raises(ValidationError) as exc_info:
        config_model.DataSettings.model_validate(
            {
                "csv_path": str(csv_path),
                "date_column": "Date",
                "frequency": "M",
                "bogus": 1,
            },
            context={"base_path": tmp_path},
        )

    assert "Extra inputs are not permitted" in str(exc_info.value)
    assert "bogus" in str(exc_info.value)


def test_portfolio_keylint_flags_unknown() -> None:
    errors = lint_portfolio_keys(
        {
            "portfolio": {
                "selection_mode": "all",
                "rebalance_calendar": "NYSE",
                "max_turnover": 1.0,
                "cost_model": {"per_trade_bps": 0, "half_spread_bps": 0},
                "bogus_key": True,
            }
        }
    )

    assert errors == ["portfolio.bogus_key"]


def test_portfolio_keylint_accepts_canonical_threshold_controls() -> None:
    errors = lint_portfolio_keys(
        {
            "portfolio": {
                "constraints": {"min_weight_strikes": 2},
                "sticky_add_x": 2,
                "sticky_drop_y": 3,
            }
        }
    )

    assert errors == []


def test_validate_trend_config_rejects_unknown_portfolio_key(tmp_path: Path) -> None:
    csv_path = _write_returns_csv(tmp_path)
    payload = {
        "data": {
            "csv_path": str(csv_path),
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 1.0,
            "cost_model": {"per_trade_bps": 0, "half_spread_bps": 0},
            "bogus_key": True,
        },
        "vol_adjust": {
            "target_vol": 0.1,
        },
    }

    with pytest.raises(ValueError, match=r"portfolio\.bogus_key"):
        config_model.validate_trend_config(payload, base_path=tmp_path)


@pytest.mark.parametrize(
    "config_path",
    [
        Path("config/long_backtest.yml"),
        Path("config/trend_universe_2004.yml"),
        Path("config/trend_concentrated_2004.yml"),
    ],
)
def test_existing_backtest_configs_have_no_strict_key_lint_errors(config_path: Path) -> None:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert lint_portfolio_keys(payload) == []


def test_data_settings_preserves_na_as_zero(tmp_path: Path) -> None:
    csv_path = _write_returns_csv(tmp_path)

    config = config_model.DataSettings.model_validate(
        {
            "csv_path": str(csv_path),
            "date_column": "Date",
            "frequency": "M",
            "na_as_zero": {
                "enabled": True,
                "max_missing_per_window": 1,
                "max_consecutive_gap": 1,
            },
        },
        context={"base_path": tmp_path},
    )

    assert config.na_as_zero == {
        "enabled": True,
        "max_missing_per_window": 1,
        "max_consecutive_gap": 1,
    }


def test_load_defaults_is_cached() -> None:
    lint_portfolio_keys({"portfolio": {"rebalance_calendar": "NYSE"}})
    before = _load_defaults.cache_info()

    lint_portfolio_keys({"portfolio": {"rebalance_calendar": "NYSE"}})
    after = _load_defaults.cache_info()

    assert after.hits == before.hits + 1


def _valid_payload(base_dir: Path) -> dict:
    """Minimal payload that ``validate_trend_config`` accepts."""

    csv_path = _write_returns_csv(base_dir)
    return {
        "data": {
            "csv_path": str(csv_path),
            "date_column": "Date",
            "frequency": "M",
        },
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 1.0,
            "cost_model": {"per_trade_bps": 0, "half_spread_bps": 0},
        },
        "vol_adjust": {
            "target_vol": 0.1,
        },
    }


def test_unknown_metrics_key_rejected(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload["metrics"] = {"registry": "core", "bogus": True}

    with pytest.raises(ValueError, match=r"metrics\.bogus"):
        config_model.validate_trend_config(payload, base_path=tmp_path)


def test_unknown_run_key_rejected(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload["run"] = {"jobs": 1, "bogus": True}

    with pytest.raises(ValueError, match=r"run\.bogus"):
        config_model.validate_trend_config(payload, base_path=tmp_path)


def test_dead_run_n_jobs_alias_rejected(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload["run"] = {"n_jobs": 1}

    with pytest.raises(ValueError, match=r"run\.n_jobs"):
        config_model.validate_trend_config(payload, base_path=tmp_path)


def test_unknown_export_key_rejected(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload["export"] = {"formats": ["csv"], "bogus": True}

    with pytest.raises(ValueError, match=r"export\.bogus"):
        config_model.validate_trend_config(payload, base_path=tmp_path)


def test_unknown_top_level_section_rejected(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload["bogus"] = {"anything": 1}

    with pytest.raises(ValueError, match=r"bogus"):
        config_model.validate_trend_config(payload, base_path=tmp_path)


def test_consumed_optional_top_level_sections_allowed(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload["signals"] = {"window": 10, "lag": 1}
    payload["output"] = {"path": "report.html", "format": "csv"}
    payload["extra"] = {"scenario": "smoke"}

    config_model.validate_trend_config(payload, base_path=tmp_path)


def test_shipped_demo_and_defaults_have_no_section_lint_errors() -> None:
    for config_path in (Path("config/demo.yml"), Path("config/defaults.yml")):
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config_model.lint_config_sections(payload) == [], config_path
