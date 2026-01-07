from __future__ import annotations

from pathlib import Path
from typing import Any

from trend_analysis.config.validation import format_validation_messages, validate_config


def _base_config(tmp_path: Path) -> dict[str, Any]:
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Date,A,B\n2020-01-31,0.0,0.0\n", encoding="utf-8")
    return {
        "version": "1",
        "data": {
            "csv_path": str(csv_path),
            "date_column": "Date",
            "frequency": "M",
        },
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.1},
        "sample_split": {},
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 1.0,
            "transaction_cost_bps": 0.0,
        },
        "metrics": {},
        "export": {},
        "run": {},
    }


def _has_path(result, path: str) -> bool:
    return any(issue.path == path for issue in result.errors)


def test_validate_config_missing_required_section(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg.pop("data")

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "data")


def test_validate_config_missing_portfolio_section(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg.pop("portfolio")

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "portfolio")


def test_validate_config_missing_required_field(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["data"].pop("date_column")

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "data.date_column")


def test_validate_config_missing_data_source(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["data"].pop("csv_path")

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "data.csv_path")


def test_validate_config_csv_path_wrong_type(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["data"]["csv_path"] = 123

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "data.csv_path")
    issue = next((issue for issue in result.errors if issue.path == "data.csv_path"), None)
    assert issue is not None
    assert issue.expected == "string"


def test_validate_config_wrong_type(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["data"]["frequency"] = 12

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "data.frequency")


def test_validate_config_out_of_range_value(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["sample_split"] = {"ratio": 1.5}

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "sample_split.ratio")


def test_validate_config_invalid_enum(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["sample_split"] = {"method": "unsupported"}

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "sample_split.method")


def test_validate_config_date_range_violation(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["sample_split"] = {
        "in_start": "2020-03",
        "in_end": "2020-01",
        "out_start": "2020-04",
        "out_end": "2020-05",
    }

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert any(
        issue.path in {"sample_split.in_start", "sample_split.out_start"} for issue in result.errors
    )


def test_validate_config_reports_multiple_invalid_dates(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["sample_split"] = {
        "in_start": "bad-date",
        "in_end": "2020-02",
        "out_start": "2020-03",
        "out_end": "also-bad",
    }

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "sample_split.in_start")
    assert _has_path(result, "sample_split.out_end")


def test_validate_config_top_n_exceeds_fund_count(tmp_path: Path) -> None:
    managers_dir = tmp_path / "managers"
    managers_dir.mkdir()
    for name in ("alpha.csv", "beta.csv"):
        (managers_dir / name).write_text("Date,Return\n2020-01-31,0.1\n", encoding="utf-8")

    cfg = _base_config(tmp_path)
    cfg["data"].pop("csv_path")
    cfg["data"]["managers_glob"] = "managers/*.csv"
    cfg["portfolio"]["rank"] = {"inclusion_approach": "top_n", "n": 3}

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "portfolio.rank.n")


def test_validation_message_includes_expected_actual_suggestion(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg.pop("data")

    result = validate_config(cfg, base_path=tmp_path)
    messages = format_validation_messages(result, include_warnings=False)

    assert any(
        "data:" in message
        and "Expected section present" in message
        and "got missing" in message
        and "Suggestion:" in message
        for message in messages
    )


def test_validation_error_includes_expected_actual_suggestion_fields(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg.pop("data")

    result = validate_config(cfg, base_path=tmp_path)

    issue = next((issue for issue in result.errors if issue.path == "data"), None)
    assert issue is not None
    assert issue.expected == "section present"
    assert issue.actual == "missing"
    assert issue.suggestion is not None and "Add" in issue.suggestion


def test_validate_config_strict_mode_treats_warnings_as_errors(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["data"]["csv_path"] = str(tmp_path / "missing.csv")
    cfg["portfolio"]["rank"] = {"inclusion_approach": "top_n", "n": 3}

    result = validate_config(cfg, base_path=tmp_path, strict=False)
    strict_result = validate_config(cfg, base_path=tmp_path, strict=True)

    assert result.valid
    assert result.errors == []
    assert result.warnings
    assert not strict_result.valid
    assert strict_result.errors == []
    assert strict_result.warnings
