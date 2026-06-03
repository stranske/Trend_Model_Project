"""Structured validation helpers for configuration payloads."""

from __future__ import annotations

import difflib
import glob
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field, field_validator, model_validator

from trend_analysis.config.model import validate_trend_config
from trend_analysis.config.models import Config
from trend_analysis.config.schema_validation import load_schema
from trend_analysis.time_utils import resolve_period_bound
from utils.paths import proj_path

_PATH_PATTERN = re.compile(r"^([A-Za-z0-9_.\[\]-]+)(:|\s+)(.+)$")


class ValidationError(BaseModel):
    path: str
    message: str
    expected: str = "valid value"
    actual: Any = "unknown"
    suggestion: str | None = None


class ValidationResult(BaseModel):
    valid: bool = True
    errors: list[ValidationError] = Field(default_factory=list)
    warnings: list[ValidationError] = Field(default_factory=list)

    @field_validator("errors", "warnings", mode="before")
    @classmethod
    def _coerce_issue_strings(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        coerced: list[ValidationError | Any] = []
        for item in value:
            if isinstance(item, str):
                coerced.append(
                    ValidationError(
                        path="<root>",
                        message=item,
                        expected="valid value",
                        actual="unknown",
                        suggestion="Update the configuration to match the expected value.",
                    )
                )
            else:
                coerced.append(item)
        return coerced

    @model_validator(mode="after")
    def _infer_valid(self) -> "ValidationResult":
        if "valid" not in self.model_fields_set:
            self.valid = not self.errors
        return self


def validate_config(
    config: dict[str, Any],
    *,
    base_path: Path | None = None,
    strict: bool = False,
    skip_required_fields: bool = False,
    include_model_validation: bool = False,
) -> ValidationResult:
    """Validate a configuration payload and return structured results.

    Args:
        config: Configuration dictionary to validate
        base_path: Base path for resolving relative paths
        strict: If True, warnings are treated as errors
        skip_required_fields: If True, skip validation of required fields.
            Useful for CLI configs that will be overridden with -i or --preset.
        include_model_validation: If True, run the minimal TrendConfig model
            validation (including file existence checks).
    """

    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []

    if not isinstance(config, Mapping):
        error = ValidationError(
            path="<root>",
            message="Configuration payload must be an object.",
            expected="mapping",
            actual=type(config).__name__,
            suggestion="Provide a top-level mapping of configuration keys.",
        )
        return ValidationResult(valid=False, errors=[error], warnings=[])

    base = base_path or proj_path()

    _run_schema_validation(config, errors, warnings)
    _run_required_validation(config, errors, skip_required_fields)
    _run_data_semantic_validation(config, errors)
    if include_model_validation:
        _collect_trend_model_errors(config, errors, base)
    _run_sample_split_validation(config, errors)
    _run_portfolio_validation(config, errors, warnings, base)

    if strict and warnings:
        errors.extend(warnings)
        warnings = []

    valid = not errors
    return ValidationResult(valid=valid, errors=errors, warnings=warnings)


def _run_schema_validation(
    config: Mapping[str, Any],
    errors: list[ValidationError],
    warnings: list[ValidationError],
) -> None:
    _collect_schema_errors(config, errors, warnings)


def _run_required_validation(
    config: Mapping[str, Any],
    errors: list[ValidationError],
    skip_required_fields: bool,
) -> None:
    _check_required_sections(config, errors)
    if not skip_required_fields:
        _check_required_fields(config, errors)
    _check_version_field(config, errors)


def _run_data_semantic_validation(
    config: Mapping[str, Any],
    errors: list[ValidationError],
) -> None:
    data = config.get("data")
    if isinstance(data, Mapping):
        _check_data_frequency_supported(data, errors)


def _run_sample_split_validation(
    config: Mapping[str, Any],
    errors: list[ValidationError],
) -> None:
    _check_sample_split_requirements(config, errors)
    _check_date_ranges(config, errors)


def _run_portfolio_validation(
    config: Mapping[str, Any],
    errors: list[ValidationError],
    warnings: list[ValidationError],
    base: Path,
) -> None:
    _check_portfolio_selection_requirements(config, errors)
    _check_manual_selection_requirements(config, errors)
    _check_rank_inclusion_requirements(config, errors)
    _check_rank_value_ranges(config, errors)
    _check_rank_fund_count(config, errors, warnings, base)
    _check_portfolio_feasibility(config, errors, warnings, base)


def format_validation_messages(
    result: ValidationResult,
    *,
    include_warnings: bool = True,
) -> list[str]:
    """Format validation issues into user-facing messages."""

    issues = list(result.errors)
    if include_warnings:
        issues.extend(result.warnings)
    return [_format_issue(issue) for issue in issues]


def _collect_schema_errors(
    config: Mapping[str, Any],
    errors: list[ValidationError],
    warnings: list[ValidationError],
) -> None:
    schema = load_schema()
    validator = Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(config), key=lambda err: list(err.absolute_path)):
        issues = _schema_error_to_issues(error)
        for issue in issues:
            _append_issue(errors, issue)


def _schema_error_to_issues(error: Any) -> list[ValidationError]:
    path = _format_path(error.absolute_path)
    validator = error.validator
    message = str(error.message)
    expected = _expected_for_error(error)
    actual = error.instance
    suggestion = _suggestion_for_error(error)

    if validator == "required":
        missing = _missing_required_field(message)
        if missing:
            path = _join_path(path, missing)
        message = "Missing required field."
        expected = "field present"
        actual = "missing"
        suggestion = f"Add '{missing}' to the configuration." if missing else suggestion

    if validator == "additionalProperties":
        unexpected = _unexpected_property(message)
        if unexpected:
            path = _join_path(path, unexpected)
        message = f"Unexpected field '{unexpected}'." if unexpected else "Unexpected field."
        expected = "no additional properties"
        actual = unexpected or "unknown"
        suggestion = _suggest_additional_property(unexpected, error) or suggestion

    return [
        ValidationError(
            path=path,
            message=message,
            expected=expected,
            actual=actual,
            suggestion=suggestion,
        )
    ]


def _suggest_additional_property(unexpected: str | None, error: Any) -> str | None:
    if not unexpected:
        return None
    schema = error.schema or {}
    properties = schema.get("properties") or {}
    if not isinstance(properties, Mapping) or not properties:
        return f"Remove '{unexpected}' or move it under the correct section."
    valid_keys = sorted(str(key) for key in properties.keys())
    suggestions = difflib.get_close_matches(unexpected, valid_keys, n=3, cutoff=0.6)
    hint = f"Did you mean: {', '.join(suggestions)}? " if suggestions else ""
    cap = 8
    listed = ", ".join(valid_keys[:cap])
    suffix = "..." if len(valid_keys) > cap else ""
    return f"{hint}Valid keys: {listed}{suffix}."


def _expected_for_error(error: Any) -> str:
    validator = error.validator
    schema = error.schema or {}
    if validator == "type":
        types = schema.get("type")
        if isinstance(types, list):
            return f"type {', '.join(types)}"
        if isinstance(types, str):
            return f"type {types}"
    if validator == "enum":
        values = schema.get("enum") or error.validator_value
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            return f"one of {list(values)}"
    if validator == "minimum":
        return f">= {schema.get('minimum')}"
    if validator == "maximum":
        return f"<= {schema.get('maximum')}"
    if validator == "minItems":
        return f"at least {schema.get('minItems')} items"
    if validator == "maxItems":
        return f"at most {schema.get('maxItems')} items"
    if validator == "pattern":
        pattern = schema.get("pattern")
        return f"match pattern {pattern}" if pattern else "matching value"
    return str(validator) if validator else "valid value"


def _suggestion_for_error(error: Any) -> str | None:
    validator = error.validator
    schema = error.schema or {}
    if validator == "type":
        expected = schema.get("type")
        if isinstance(expected, list):
            return f"Use one of the supported types: {', '.join(expected)}."
        if isinstance(expected, str):
            return f"Provide a {expected} value."
    if validator == "enum":
        values = schema.get("enum") or error.validator_value
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            return f"Use one of: {', '.join(map(str, values))}."
    if validator == "minimum":
        return f"Use a value >= {schema.get('minimum')}."
    if validator == "maximum":
        return f"Use a value <= {schema.get('maximum')}."
    if validator == "minItems":
        return f"Provide at least {schema.get('minItems')} entries."
    if validator == "maxItems":
        return f"Provide no more than {schema.get('maxItems')} entries."
    return None


def _missing_required_field(message: str) -> str | None:
    match = re.search(r"'([^']+)' is a required property", message)
    return match.group(1) if match else None


def _unexpected_property(message: str) -> str | None:
    match = re.search(r"\('([^']+)' was unexpected\)", message)
    return match.group(1) if match else None


def _check_required_sections(config: Mapping[str, Any], errors: list[ValidationError]) -> None:
    for field in Config.REQUIRED_DICT_FIELDS:
        if field not in config:
            issue = ValidationError(
                path=field,
                message="Required section is missing.",
                expected="section present",
                actual="missing",
                suggestion=f"Add the '{field}' section to the configuration.",
            )
            _append_issue(errors, issue)
            continue
        if not isinstance(config[field], Mapping):
            issue = ValidationError(
                path=field,
                message="Section must be an object.",
                expected="object",
                actual=type(config[field]).__name__,
                suggestion=f"Update '{field}' to be a mapping of settings.",
            )
            _append_issue(errors, issue)


def _check_required_fields(config: Mapping[str, Any], errors: list[ValidationError]) -> None:
    data = config.get("data")
    if isinstance(data, Mapping):
        _check_data_required_fields(data, errors)

    portfolio = config.get("portfolio")
    if isinstance(portfolio, Mapping):
        _check_portfolio_required_fields(portfolio, errors)

    vol_adjust = config.get("vol_adjust")
    if isinstance(vol_adjust, Mapping):
        _check_vol_adjust_required_fields(vol_adjust, errors)


def _check_version_field(config: Mapping[str, Any], errors: list[ValidationError]) -> None:
    if "version" not in config:
        issue = ValidationError(
            path="version",
            message="Version is required.",
            expected="non-empty string",
            actual="missing",
            suggestion="Provide a version string (e.g., '1').",
        )
        _append_issue(errors, issue)
        return
    version = config.get("version")
    if not isinstance(version, str):
        issue = ValidationError(
            path="version",
            message="Version must be a string.",
            expected="string",
            actual=type(version).__name__,
            suggestion="Wrap the version number in quotes.",
        )
        _append_issue(errors, issue)
        return
    if not version.strip():
        issue = ValidationError(
            path="version",
            message="Version cannot be blank.",
            expected="non-empty string",
            actual=version,
            suggestion="Provide a non-empty version string.",
        )
        _append_issue(errors, issue)


def _require_field(
    errors: list[ValidationError],
    section: Mapping[str, Any],
    section_name: str,
    field: str,
    *,
    expected: str,
    suggestion: str,
) -> None:
    if field not in section:
        issue = ValidationError(
            path=f"{section_name}.{field}",
            message="Required field is missing.",
            expected=expected,
            actual="missing",
            suggestion=suggestion,
        )
        _append_issue(errors, issue)
        return
    value = section.get(field)
    if not _is_present(value):
        issue = ValidationError(
            path=f"{section_name}.{field}",
            message="Required field is missing.",
            expected=expected,
            actual=value,
            suggestion=suggestion,
        )
        _append_issue(errors, issue)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _check_data_required_fields(data: Mapping[str, Any], errors: list[ValidationError]) -> None:
    csv_path = data.get("csv_path")
    if csv_path is not None and not isinstance(csv_path, str):
        issue = ValidationError(
            path="data.csv_path",
            message="CSV path must be a string.",
            expected="string",
            actual=type(csv_path).__name__,
            suggestion="Provide data.csv_path as a string path to a CSV file.",
        )
        _append_issue(errors, issue)
    managers_glob = data.get("managers_glob")
    if managers_glob is not None and not isinstance(managers_glob, str):
        issue = ValidationError(
            path="data.managers_glob",
            message="Managers glob must be a string.",
            expected="string",
            actual=type(managers_glob).__name__,
            suggestion="Provide data.managers_glob as a glob string to CSV files.",
        )
        _append_issue(errors, issue)
    _require_field(
        errors,
        data,
        "data",
        "date_column",
        expected="non-empty string",
        suggestion="Set data.date_column to the date column name (e.g., 'Date').",
    )
    _require_field(
        errors,
        data,
        "data",
        "frequency",
        expected="non-empty string",
        suggestion="Set data.frequency to 'M' or 'ME'.",
    )
    if not _is_present(csv_path) and not _is_present(managers_glob):
        issue = ValidationError(
            path="data.csv_path",
            message="Data source is required.",
            expected="csv_path or managers_glob",
            actual="missing",
            suggestion="Set data.csv_path to a CSV file or data.managers_glob to a CSV glob.",
        )
        _append_issue(errors, issue)


def _check_data_frequency_supported(
    data: Mapping[str, Any], errors: list[ValidationError]
) -> None:
    value = data.get("frequency")
    if not _is_present(value):
        return
    if not isinstance(value, str):
        issue = ValidationError(
            path="data.frequency",
            message="Frequency must be a string.",
            expected="'M' or 'ME'",
            actual=type(value).__name__,
            suggestion="Set data.frequency to 'M' or 'ME'.",
        )
        _append_issue(errors, issue)
        return

    frequency = value.strip().upper()
    if frequency not in {"M", "ME"}:
        issue = ValidationError(
            path="data.frequency",
            message=(
                "Only monthly data.frequency values are currently supported; "
                f"'{value}' would be silently resampled to monthly."
            ),
            expected="'M' or 'ME'",
            actual=value,
            suggestion=(
                "Use data.frequency: M for the current monthly pipeline, or add "
                "full non-monthly periods-per-year support before using D/W."
            ),
        )
        _append_issue(errors, issue)


def _check_portfolio_required_fields(
    portfolio: Mapping[str, Any], errors: list[ValidationError]
) -> None:
    _require_field(
        errors,
        portfolio,
        "portfolio",
        "selection_mode",
        expected="non-empty string",
        suggestion="Set portfolio.selection_mode (e.g., 'all').",
    )
    _require_field(
        errors,
        portfolio,
        "portfolio",
        "rebalance_calendar",
        expected="non-empty string",
        suggestion="Set portfolio.rebalance_calendar (e.g., 'NYSE').",
    )
    _require_field(
        errors,
        portfolio,
        "portfolio",
        "max_turnover",
        expected="number",
        suggestion="Set portfolio.max_turnover to a numeric value (e.g., 1.0).",
    )
    _require_field(
        errors,
        portfolio,
        "portfolio",
        "transaction_cost_bps",
        expected="number",
        suggestion="Set portfolio.transaction_cost_bps to a numeric value (e.g., 0).",
    )


def _check_vol_adjust_required_fields(
    vol_adjust: Mapping[str, Any], errors: list[ValidationError]
) -> None:
    _require_field(
        errors,
        vol_adjust,
        "vol_adjust",
        "target_vol",
        expected="number",
        suggestion="Set vol_adjust.target_vol to a numeric target (e.g., 0.1).",
    )


def _collect_trend_model_errors(
    config: Mapping[str, Any], errors: list[ValidationError], base: Path
) -> None:
    try:
        validate_trend_config(dict(config), base_path=base)
    except Exception as exc:
        parsed = _error_from_exception(exc, config)
        if parsed is not None:
            _append_issue(errors, parsed)


def _check_date_ranges(config: Mapping[str, Any], errors: list[ValidationError]) -> None:
    split = config.get("sample_split")
    if not isinstance(split, Mapping):
        return
    required = ("in_start", "in_end", "out_start", "out_end")
    if not all(key in split for key in required):
        return

    # Resolve each label to the same instant the window slicer uses
    # (stages/preprocessing.py): start labels -> month start, end labels ->
    # month end. Resolving uniformly to month start here would let the
    # validator and slicer disagree on what an `*_end` label denotes.
    bound_roles = {"in_start": "start", "in_end": "end", "out_start": "start", "out_end": "end"}
    parsed: dict[str, pd.Timestamp] = {}
    invalid_fields: set[str] = set()
    for key in required:
        raw = split.get(key)
        try:
            parsed[key] = resolve_period_bound(raw, bound=bound_roles[key])
        except Exception:
            issue = ValidationError(
                path=f"sample_split.{key}",
                message="Date must be a valid timestamp.",
                expected="ISO date string",
                actual=raw,
                suggestion="Use a YYYY-MM or YYYY-MM-DD formatted date.",
            )
            _append_issue(errors, issue)
            invalid_fields.add(key)

    if invalid_fields:
        return

    in_start = parsed["in_start"]
    in_end = parsed["in_end"]
    out_start = parsed["out_start"]
    out_end = parsed["out_end"]
    if in_start >= in_end:
        issue = ValidationError(
            path="sample_split.in_start",
            message="In-sample start must be before in-sample end.",
            expected="in_start < in_end",
            actual=f"{in_start.date()} >= {in_end.date()}",
            suggestion="Move in_start earlier than in_end.",
        )
        _append_issue(errors, issue)
    if in_end >= out_start:
        issue = ValidationError(
            path="sample_split.out_start",
            message="Out-of-sample start must be after in-sample end.",
            expected="in_end < out_start",
            actual=f"{in_end.date()} >= {out_start.date()}",
            suggestion="Move out_start after in_end.",
        )
        _append_issue(errors, issue)
    if out_start >= out_end:
        issue = ValidationError(
            path="sample_split.out_end",
            message="Out-of-sample end must be after out-of-sample start.",
            expected="out_start < out_end",
            actual=f"{out_start.date()} >= {out_end.date()}",
            suggestion="Move out_end after out_start.",
        )
        _append_issue(errors, issue)


def _check_sample_split_requirements(
    config: Mapping[str, Any], errors: list[ValidationError]
) -> None:
    split = config.get("sample_split")
    if not isinstance(split, Mapping):
        return
    method = split.get("method")
    if method == "ratio":
        if not _is_present(split.get("ratio")):
            issue = ValidationError(
                path="sample_split.ratio",
                message="Ratio is required when sample_split.method is ratio.",
                expected="number between 0 and 1",
                actual="missing",
                suggestion="Set sample_split.ratio to a decimal between 0 and 1.",
            )
            _append_issue(errors, issue)
        return
    if method == "date":
        _require_field(
            errors,
            split,
            "sample_split",
            "in_start",
            expected="non-empty string",
            suggestion="Set sample_split.in_start to a valid date string.",
        )
        _require_field(
            errors,
            split,
            "sample_split",
            "in_end",
            expected="non-empty string",
            suggestion="Set sample_split.in_end to a valid date string.",
        )
        _require_field(
            errors,
            split,
            "sample_split",
            "out_start",
            expected="non-empty string",
            suggestion="Set sample_split.out_start to a valid date string.",
        )
        _require_field(
            errors,
            split,
            "sample_split",
            "out_end",
            expected="non-empty string",
            suggestion="Set sample_split.out_end to a valid date string.",
        )


def _check_rank_fund_count(
    config: Mapping[str, Any],
    errors: list[ValidationError],
    warnings: list[ValidationError],
    base: Path,
) -> None:
    portfolio = config.get("portfolio")
    if not isinstance(portfolio, Mapping):
        return
    rank_cfg = portfolio.get("rank")
    if not isinstance(rank_cfg, Mapping):
        return
    approach = rank_cfg.get("inclusion_approach")
    if approach != "top_n":
        return
    n_value = rank_cfg.get("n")
    if n_value is None:
        return
    if not isinstance(n_value, int) or isinstance(n_value, bool):
        return
    top_n = n_value

    available = _count_available_funds(config, base)
    if available is None:
        issue = ValidationError(
            path="portfolio.rank.n",
            message="Unable to determine available fund count for top_n validation.",
            expected="fund count available",
            actual="unknown",
            suggestion="Ensure data.csv_path or data.managers_glob points to existing files.",
        )
        _append_issue(warnings, issue)
        return

    if top_n > available:
        issue = ValidationError(
            path="portfolio.rank.n",
            message="top_n exceeds the number of available funds.",
            expected=f"<= {available}",
            actual=top_n,
            suggestion=f"Reduce top_n to {available} or fewer.",
        )
        _append_issue(errors, issue)


def collect_feasibility_errors(
    config: Mapping[str, Any], *, base_path: Path | None = None
) -> list[ValidationError]:
    """Run ONLY the parameter-feasibility checks and return any errors.

    Lets non-CLI entry points (e.g. the Streamlit app, which validates a minimal
    payload subset) reuse the same infeasibility gate on a full config without
    running the rest of config validation.
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []
    _check_portfolio_feasibility(config, errors, warnings, base_path or Path("."))
    return errors


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_pos_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _effective_holding_count(
    config: Mapping[str, Any], portfolio: Mapping[str, Any], base: Path
) -> int | None:
    """Upper bound on the number of funds that will be held.

    Used to test whether a long-only, fully-invested book can satisfy weight
    caps. Prefer an explicitly configured selection cap (rank top_n,
    multi_period.max_funds); fall back to the available universe size.
    """
    rank_cfg = portfolio.get("rank")
    if portfolio.get("selection_mode") == "rank" and isinstance(rank_cfg, Mapping):
        if rank_cfg.get("inclusion_approach") == "top_n":
            top_n = _coerce_pos_int(rank_cfg.get("n"))
            if top_n is not None:
                return top_n
    multi_period = config.get("multi_period")
    if isinstance(multi_period, Mapping):
        max_funds = _coerce_pos_int(multi_period.get("max_funds"))
        if max_funds is not None:
            return max_funds
    return _count_available_funds(config, base)


def _check_portfolio_feasibility(
    config: Mapping[str, Any],
    errors: list[ValidationError],
    warnings: list[ValidationError],
    base: Path,
) -> None:
    """Reject internally contradictory / infeasible parameter sets.

    Policy: a simulation must not run on parameters that cannot describe a valid
    portfolio. The only escape for a capacity shortfall is an EXPLICIT cash
    allocation (constraints.cash_weight) -- never a silent residual.
    """
    portfolio = config.get("portfolio")
    if not isinstance(portfolio, Mapping):
        return
    constraints = portfolio.get("constraints")
    constraints = constraints if isinstance(constraints, Mapping) else {}

    long_only = bool(constraints.get("long_only", True))
    max_w = _coerce_float(constraints.get("max_weight"))
    min_w = _coerce_float(constraints.get("min_weight"))
    cash_w = _coerce_float(constraints.get("cash_weight")) or 0.0
    explicit_cash = cash_w > 1e-9

    # 1. max_weight must not be below min_weight.
    if max_w is not None and min_w is not None and max_w < min_w - 1e-12:
        _append_issue(
            errors,
            ValidationError(
                path="portfolio.constraints.max_weight",
                message="max_weight is below min_weight.",
                expected=f">= min_weight ({min_w})",
                actual=max_w,
                suggestion="Raise max_weight or lower min_weight.",
            ),
        )

    # 2. multi_period.min_funds must not exceed max_funds.
    multi_period = config.get("multi_period")
    if isinstance(multi_period, Mapping):
        min_funds = _coerce_pos_int(multi_period.get("min_funds"))
        max_funds = _coerce_pos_int(multi_period.get("max_funds"))
        if min_funds is not None and max_funds is not None and min_funds > max_funds:
            _append_issue(
                errors,
                ValidationError(
                    path="multi_period.min_funds",
                    message="min_funds exceeds max_funds.",
                    expected=f"<= max_funds ({max_funds})",
                    actual=min_funds,
                    suggestion="Set min_funds <= max_funds.",
                ),
            )

    # 3. vol_adjust.floor_vol must be below target_vol when targeting is enabled.
    vol_adjust = config.get("vol_adjust")
    if isinstance(vol_adjust, Mapping) and bool(vol_adjust.get("enabled", True)):
        floor_v = _coerce_float(vol_adjust.get("floor_vol"))
        target_v = _coerce_float(vol_adjust.get("target_vol"))
        if (
            floor_v is not None
            and target_v is not None
            and target_v > 0
            and floor_v >= target_v - 1e-12
        ):
            _append_issue(
                errors,
                ValidationError(
                    path="vol_adjust.floor_vol",
                    message="floor_vol must be below target_vol.",
                    expected=f"< target_vol ({target_v})",
                    actual=floor_v,
                    suggestion="Lower floor_vol below target_vol.",
                ),
            )

    # 4 & 5. Capacity vs weight bounds for a long-only, fully-invested book.
    # Only enforced when no explicit cash absorbs the slack.
    if not long_only or explicit_cash or (max_w is None and min_w is None):
        return
    n_funds = _effective_holding_count(config, portfolio, base)
    if n_funds is None or n_funds <= 0:
        return  # cannot determine capacity; do not guess

    if max_w is not None and max_w > 0 and max_w * n_funds < 1.0 - 1e-9:
        _append_issue(
            errors,
            ValidationError(
                path="portfolio.constraints.max_weight",
                message=(
                    "Infeasible: a long-only, fully-invested book of "
                    f"{n_funds} funds cannot satisfy max_weight "
                    f"(max_weight * n_funds = {max_w * n_funds:.3f} < 1)."
                ),
                expected=f">= {1.0 / n_funds:.4f} for {n_funds} funds",
                actual=max_w,
                suggestion=(
                    f"Raise max_weight to >= {1.0 / n_funds:.4f}, hold more funds, "
                    "or set an explicit constraints.cash_weight to absorb the remainder."
                ),
            ),
        )

    if min_w is not None and min_w > 0 and min_w * n_funds > 1.0 + 1e-9:
        _append_issue(
            errors,
            ValidationError(
                path="portfolio.constraints.min_weight",
                message=(
                    f"Infeasible: {n_funds} funds each >= min_weight cannot sum to "
                    f"<= 1 (min_weight * n_funds = {min_w * n_funds:.3f} > 1)."
                ),
                expected=f"<= {1.0 / n_funds:.4f} for {n_funds} funds",
                actual=min_w,
                suggestion=f"Lower min_weight to <= {1.0 / n_funds:.4f} or hold fewer funds.",
            ),
        )


def _check_portfolio_selection_requirements(
    config: Mapping[str, Any], errors: list[ValidationError]
) -> None:
    portfolio = config.get("portfolio")
    if not isinstance(portfolio, Mapping):
        return
    selection_mode = portfolio.get("selection_mode")
    if selection_mode != "rank":
        return
    rank_cfg = portfolio.get("rank")
    if rank_cfg is None:
        issue = ValidationError(
            path="portfolio.rank",
            message="Rank settings are required when selection_mode is rank.",
            expected="object",
            actual="missing",
            suggestion="Add portfolio.rank settings or change selection_mode.",
        )
        _append_issue(errors, issue)
        return
    if not isinstance(rank_cfg, Mapping):
        issue = ValidationError(
            path="portfolio.rank",
            message="Rank settings must be an object.",
            expected="object",
            actual=type(rank_cfg).__name__,
            suggestion="Provide portfolio.rank as a mapping of rank settings.",
        )
        _append_issue(errors, issue)


def _check_manual_selection_requirements(
    config: Mapping[str, Any], errors: list[ValidationError]
) -> None:
    portfolio = config.get("portfolio")
    if not isinstance(portfolio, Mapping):
        return
    if portfolio.get("selection_mode") != "manual":
        return
    manual_list = portfolio.get("manual_list")
    if manual_list is None:
        issue = ValidationError(
            path="portfolio.manual_list",
            message="Manual selection requires a manual_list.",
            expected="non-empty list of fund identifiers",
            actual="missing",
            suggestion="Set portfolio.manual_list to a list of fund identifiers.",
        )
        _append_issue(errors, issue)
        return
    if not isinstance(manual_list, list):
        issue = ValidationError(
            path="portfolio.manual_list",
            message="Manual list must be a list.",
            expected="list of strings",
            actual=type(manual_list).__name__,
            suggestion="Provide portfolio.manual_list as a list of fund identifiers.",
        )
        _append_issue(errors, issue)
        return
    if not manual_list:
        issue = ValidationError(
            path="portfolio.manual_list",
            message="Manual list cannot be empty.",
            expected="at least one fund identifier",
            actual="empty list",
            suggestion="Add at least one identifier to portfolio.manual_list.",
        )
        _append_issue(errors, issue)
        return
    for idx, value in enumerate(manual_list):
        if not isinstance(value, str) or not value.strip():
            issue = ValidationError(
                path=f"portfolio.manual_list[{idx}]",
                message="Manual list entries must be non-empty strings.",
                expected="non-empty string",
                actual=value,
                suggestion="Replace invalid entries with non-empty fund identifiers.",
            )
            _append_issue(errors, issue)


def _check_rank_inclusion_requirements(
    config: Mapping[str, Any], errors: list[ValidationError]
) -> None:
    portfolio = config.get("portfolio")
    if not isinstance(portfolio, Mapping):
        return
    rank_cfg = portfolio.get("rank")
    if not isinstance(rank_cfg, Mapping):
        return

    approach = rank_cfg.get("inclusion_approach")
    if not _is_present(approach):
        issue = ValidationError(
            path="portfolio.rank.inclusion_approach",
            message="Rank inclusion approach is required.",
            expected="one of top_n, top_pct, threshold",
            actual="missing" if approach is None else approach,
            suggestion="Set portfolio.rank.inclusion_approach to 'top_n', 'top_pct', or 'threshold'.",
        )
        _append_issue(errors, issue)
        return
    if approach == "top_n":
        if not _is_present(rank_cfg.get("n")):
            issue = ValidationError(
                path="portfolio.rank.n",
                message="top_n requires a rank count.",
                expected="positive integer",
                actual="missing",
                suggestion="Set portfolio.rank.n to a positive integer.",
            )
            _append_issue(errors, issue)
        return

    if approach == "top_pct":
        if not _is_present(rank_cfg.get("pct")):
            issue = ValidationError(
                path="portfolio.rank.pct",
                message="top_pct requires a percentile threshold.",
                expected="number between 0 and 1",
                actual="missing",
                suggestion="Set portfolio.rank.pct to a decimal between 0 and 1.",
            )
            _append_issue(errors, issue)
        return

    if approach == "threshold":
        if not _is_present(rank_cfg.get("threshold")):
            issue = ValidationError(
                path="portfolio.rank.threshold",
                message="threshold requires a cutoff value.",
                expected="number",
                actual="missing",
                suggestion="Set portfolio.rank.threshold to a numeric cutoff.",
            )
            _append_issue(errors, issue)


def _check_rank_value_ranges(config: Mapping[str, Any], errors: list[ValidationError]) -> None:
    portfolio = config.get("portfolio")
    if not isinstance(portfolio, Mapping):
        return
    rank_cfg = portfolio.get("rank")
    if not isinstance(rank_cfg, Mapping):
        return
    approach = rank_cfg.get("inclusion_approach")

    if approach == "top_n":
        n_value = rank_cfg.get("n")
        if isinstance(n_value, bool):
            return
        if isinstance(n_value, int):
            if n_value <= 0:
                issue = ValidationError(
                    path="portfolio.rank.n",
                    message="top_n must be a positive integer.",
                    expected=">= 1",
                    actual=n_value,
                    suggestion="Set portfolio.rank.n to a positive integer.",
                )
                _append_issue(errors, issue)
        return

    if approach == "top_pct":
        pct_value = rank_cfg.get("pct")
        if isinstance(pct_value, bool):
            return
        if isinstance(pct_value, (int, float)):
            if pct_value <= 0 or pct_value > 1:
                issue = ValidationError(
                    path="portfolio.rank.pct",
                    message="top_pct must be between 0 and 1.",
                    expected="> 0 and <= 1",
                    actual=pct_value,
                    suggestion="Set portfolio.rank.pct to a decimal between 0 and 1.",
                )
                _append_issue(errors, issue)


def _count_available_funds(config: Mapping[str, Any], base: Path) -> int | None:
    data = config.get("data")
    if not isinstance(data, Mapping):
        return None
    managers_glob = data.get("managers_glob")
    csv_path = data.get("csv_path")
    date_column = str(data.get("date_column") or "Date")
    risk_free_column = data.get("risk_free_column")

    if isinstance(managers_glob, str) and managers_glob.strip():
        pattern = _resolve_path(managers_glob, base)
        matches = glob.glob(str(pattern))
        files = [Path(match) for match in matches if Path(match).is_file()]
        return len([path for path in files if path.suffix.lower() == ".csv"])

    if isinstance(csv_path, str) and csv_path.strip():
        path = _resolve_path(csv_path, base)
        if not path.exists() or not path.is_file():
            return None
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            return None
        cols = [str(col) for col in header.columns]
        excluded = {date_column}
        if isinstance(risk_free_column, str):
            excluded.add(risk_free_column)
        return len([col for col in cols if col not in excluded])

    return None


def _resolve_path(value: str, base: Path) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw
    return (base / raw).resolve()


def _error_from_exception(exc: Exception, config: Mapping[str, Any]) -> ValidationError | None:
    message = str(exc).strip()
    if not message:
        return None
    path = "<root>"
    match = _PATH_PATTERN.match(message)
    if match:
        path = match.group(1)
    actual = _actual_from_path(config, path)
    suggestion = f"Update the value for '{path}'."
    return ValidationError(
        path=path,
        message=message,
        expected="valid value",
        actual=actual if actual is not None else "unknown",
        suggestion=suggestion,
    )


def _actual_from_path(config: Mapping[str, Any], path: str) -> Any:
    if path in {"<root>", ""}:
        return None
    segments: list[str | int] = []
    buffer = ""
    idx_buffer = ""
    in_index = False
    for char in path:
        if char == "." and not in_index:
            if buffer:
                segments.append(buffer)
                buffer = ""
            continue
        if char == "[":
            in_index = True
            if buffer:
                segments.append(buffer)
                buffer = ""
            continue
        if char == "]":
            in_index = False
            if idx_buffer:
                segments.append(int(idx_buffer))
                idx_buffer = ""
            continue
        if in_index:
            idx_buffer += char
        else:
            buffer += char
    if buffer:
        segments.append(buffer)

    current: Any = config
    for segment in segments:
        if isinstance(segment, int):
            if not isinstance(current, list) or segment >= len(current):
                return None
            current = current[segment]
        else:
            if not isinstance(current, Mapping) or segment not in current:
                return None
            current = current[segment]
    return current


def _format_path(parts: Iterable[Any]) -> str:
    segments: list[str] = []
    for part in parts:
        if isinstance(part, int):
            if not segments:
                segments.append(f"[{part}]")
            else:
                segments[-1] += f"[{part}]"
        else:
            segments.append(str(part))
    return ".".join(segments) if segments else "<root>"


def _join_path(base: str, leaf: str) -> str:
    if base in {"", "<root>"}:
        return leaf
    return f"{base}.{leaf}"


def _format_issue(issue: ValidationError) -> str:
    actual = _format_actual(issue.actual)
    suggestion = issue.suggestion or "Update the configuration to match the expected value."
    text = f"{issue.path}: {issue.message} Expected {issue.expected}, got {actual}."
    return f"{text} Suggestion: {suggestion}"


def _format_actual(actual: Any) -> str:
    if actual == "missing":
        return "missing"
    if actual is None:
        return "null"
    if isinstance(actual, str):
        return f'"{actual}"'
    return repr(actual)


def _append_issue(bucket: list[ValidationError], issue: ValidationError) -> None:
    for existing in bucket:
        if existing.path == issue.path and existing.message == issue.message:
            return
    bucket.append(issue)
