from __future__ import annotations

from pathlib import Path
from typing import Any

from trend_analysis.config.validation import (
    ValidationError,
    ValidationResult,
    format_validation_messages,
    validate_config,
)


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


def test_validate_config_missing_version(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg.pop("version")

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "version")


def test_validate_config_blank_version(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["version"] = "  "

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "version")


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
    assert "string" in issue.expected
    assert issue.actual == 123
    assert issue.suggestion is not None


def test_validate_config_managers_glob_wrong_type(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["data"].pop("csv_path")
    cfg["data"]["managers_glob"] = 456

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "data.managers_glob")
    issue = next(
        (
            issue
            for issue in result.errors
            if issue.path == "data.managers_glob"
            and issue.message == "Managers glob must be a string."
        ),
        None,
    )
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


def test_validate_config_top_n_requires_positive(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["portfolio"]["rank"] = {"inclusion_approach": "top_n", "n": 0}

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "portfolio.rank.n")


def test_validate_config_top_n_requires_value(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["portfolio"]["rank"] = {"inclusion_approach": "top_n"}

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "portfolio.rank.n")


def test_validate_config_rank_requires_inclusion_approach(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["portfolio"]["rank"] = {}

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "portfolio.rank.inclusion_approach")


def test_validate_config_top_pct_in_range(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["portfolio"]["rank"] = {"inclusion_approach": "top_pct", "pct": 1.5}

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "portfolio.rank.pct")


def test_validate_config_top_pct_requires_value(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["portfolio"]["rank"] = {"inclusion_approach": "top_pct"}

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "portfolio.rank.pct")


def test_validate_config_threshold_requires_value(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["portfolio"]["rank"] = {"inclusion_approach": "threshold"}

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "portfolio.rank.threshold")


def test_validate_config_rank_requires_settings(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["portfolio"]["selection_mode"] = "rank"
    cfg["portfolio"].pop("rank", None)

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "portfolio.rank")


def test_validate_config_manual_selection_requires_list(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["portfolio"]["selection_mode"] = "manual"
    cfg["portfolio"].pop("manual_list", None)

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "portfolio.manual_list")


def test_validate_config_manual_selection_rejects_non_list(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["portfolio"]["selection_mode"] = "manual"
    cfg["portfolio"]["manual_list"] = "FundA"

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    issue = next(
        (
            issue
            for issue in result.errors
            if issue.path == "portfolio.manual_list"
            and issue.message == "Manual list must be a list."
        ),
        None,
    )
    assert issue is not None
    assert issue.expected == "list of strings"
    assert issue.actual == "str"
    assert issue.suggestion is not None and "list of fund identifiers" in issue.suggestion


def test_validate_config_manual_selection_rejects_empty_list(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["portfolio"]["selection_mode"] = "manual"
    cfg["portfolio"]["manual_list"] = [""]

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "portfolio.manual_list[0]")


def test_validate_config_invalid_enum(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["sample_split"] = {"method": "unsupported"}

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "sample_split.method")


def test_validate_config_unexpected_field_is_error(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["unexpected"] = {"extra": True}

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert any(issue.path == "unexpected" for issue in result.errors)


def test_validation_error_includes_expected_actual_suggestion(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["unexpected"] = {"extra": True}

    result = validate_config(cfg, base_path=tmp_path)

    issue = next((issue for issue in result.errors if issue.path == "unexpected"), None)
    assert issue is not None
    assert issue.expected
    assert issue.actual is not None
    assert issue.suggestion is not None


def test_validation_messages_include_errors(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["unexpected"] = {"extra": True}

    result = validate_config(cfg, base_path=tmp_path)
    messages = format_validation_messages(result, include_warnings=False)

    assert any("unexpected:" in message and "Expected" in message for message in messages)


def test_unknown_key_error_suggests_valid_keys(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["datta"] = {"extra": True}

    result = validate_config(cfg, base_path=tmp_path)

    issue = next((issue for issue in result.errors if issue.path == "datta"), None)
    assert issue is not None
    assert issue.suggestion is not None
    assert "Valid keys" in issue.suggestion
    assert "data" in issue.suggestion


def test_nested_unknown_key_reports_full_path(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["portfolio"]["constraints"] = {"max_weightt": 0.2}

    result = validate_config(cfg, base_path=tmp_path)

    issue = next(
        (issue for issue in result.errors if issue.path == "portfolio.constraints.max_weightt"),
        None,
    )
    assert issue is not None
    assert issue.suggestion is not None
    assert "max_weight" in issue.suggestion


def test_deep_nested_unknown_key_suggests_typo(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["preprocessing"]["missing_data"] = {"polciy": "drop"}

    result = validate_config(cfg, base_path=tmp_path)

    issue = next(
        (issue for issue in result.errors if issue.path == "preprocessing.missing_data.polciy"),
        None,
    )
    assert issue is not None
    assert issue.suggestion is not None
    assert "policy" in issue.suggestion


def test_format_validation_messages_defaults_suggestion(tmp_path: Path) -> None:
    result = ValidationResult(
        errors=[
            ValidationError(
                path="sample_split.method",
                message="Invalid selection.",
                expected="date or ratio",
                actual="bad",
                suggestion=None,
            )
        ],
        warnings=[],
    )
    messages = format_validation_messages(result, include_warnings=False)

    assert any(
        "Suggestion: Update the configuration to match the expected value." in msg
        for msg in messages
    )


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


def test_validate_config_out_end_before_out_start(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["sample_split"] = {
        "in_start": "2020-01",
        "in_end": "2020-02",
        "out_start": "2020-05",
        "out_end": "2020-04",
    }

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "sample_split.out_end")


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


def test_validate_config_ratio_requires_value(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["sample_split"] = {"method": "ratio"}

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "sample_split.ratio")


def test_validate_config_date_method_requires_fields(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    cfg["sample_split"] = {"method": "date", "in_start": "2020-01", "in_end": "2020-02"}

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "sample_split.out_start")
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


def test_validate_config_top_n_skips_fund_count_for_non_integer(tmp_path: Path) -> None:
    managers_dir = tmp_path / "managers"
    managers_dir.mkdir()
    for name in ("alpha.csv", "beta.csv"):
        (managers_dir / name).write_text("Date,Return\n2020-01-31,0.1\n", encoding="utf-8")

    cfg = _base_config(tmp_path)
    cfg["data"].pop("csv_path")
    cfg["data"]["managers_glob"] = "managers/*.csv"
    cfg["portfolio"]["rank"] = {"inclusion_approach": "top_n", "n": "3"}

    result = validate_config(cfg, base_path=tmp_path)

    assert not result.valid
    assert _has_path(result, "portfolio.rank.n")
    assert not any(
        issue.message == "top_n exceeds the number of available funds." for issue in result.errors
    )


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
    assert strict_result.errors
    assert strict_result.warnings == []


def test_validation_result_infers_valid_from_errors() -> None:
    issue = ValidationError(
        path="data.csv_path",
        message="Data source is required.",
        expected="csv_path or managers_glob",
        actual="missing",
        suggestion="Set data.csv_path to a CSV file or data.managers_glob to a CSV glob.",
    )

    result = ValidationResult(errors=[issue], warnings=[])

    assert not result.valid


def test_validation_result_defaults_are_valid() -> None:
    result = ValidationResult()

    assert result.valid
    assert result.errors == []
    assert result.warnings == []


def test_validation_result_coerces_string_issues() -> None:
    result = ValidationResult(errors=["Missing data section."], warnings=["Heads up."])

    assert result.errors[0].path == "<root>"
    assert result.errors[0].expected == "valid value"
    assert result.errors[0].actual == "unknown"
    assert result.errors[0].suggestion == "Update the configuration to match the expected value."
    assert result.warnings[0].path == "<root>"
