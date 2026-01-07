"""Structured validation helpers for configuration payloads."""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from jsonschema import Draft202012Validator
from pydantic import BaseModel

from trend_analysis.config.model import validate_trend_config
from trend_analysis.config.models import Config
from trend_analysis.config.schema_validation import load_schema
from utils.paths import proj_path

_PATH_PATTERN = re.compile(r"^([A-Za-z0-9_.\[\]-]+)(:|\s+)(.+)$")


class ValidationError(BaseModel):
    path: str
    message: str
    expected: str
    actual: Any
    suggestion: str | None = None


class ValidationResult(BaseModel):
    valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationError]


def validate_config(
    config: dict[str, Any],
    *,
    base_path: Path | None = None,
    strict: bool = False,
) -> ValidationResult:
    """Validate a configuration payload and return structured results."""

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

    _collect_schema_errors(config, errors)
    _check_required_sections(config, errors)
    _check_required_fields(config, errors)
    _check_version_field(config, errors)
    # Skip TrendConfig Pydantic validation as it checks file existence
    # which is too strict for CLI validation before input override
    # _collect_trend_model_errors(config, errors, base)
    _check_date_ranges(config, errors)
    _check_rank_fund_count(config, errors, warnings, base)

    valid = not errors and (not warnings or not strict)
    return ValidationResult(valid=valid, errors=errors, warnings=warnings)


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


def _collect_schema_errors(config: Mapping[str, Any], errors: list[ValidationError]) -> None:
    schema = load_schema()
    validator = Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(config), key=lambda err: list(err.absolute_path)):
        issues = _schema_error_to_issues(error)
        for issue in issues:
            _append_issue(errors, issue)


def _schema_error_to_issues(error: Any) -> list[ValidationError]:
    path = _format_path(error.absolute_path)
    validator = error.validator
    # Skip additionalProperties errors as CLI configs may have fields
    # that will be overridden (e.g., csv_path replaced by -i flag)
    if validator == "additionalProperties":
        return []
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
        message = "Unexpected field."
        expected = "no additional properties"
        actual = unexpected or "unknown"
        suggestion = (
            f"Remove '{unexpected}' or move it under the correct section."
            if unexpected
            else suggestion
        )

    return [
        ValidationError(
            path=path,
            message=message,
            expected=expected,
            actual=actual,
            suggestion=suggestion,
        )
    ]


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
            suggestion="Set data.frequency to one of the supported values (e.g., 'M').",
        )
        csv_path = data.get("csv_path")
        managers_glob = data.get("managers_glob")
        if not _is_present(csv_path) and not _is_present(managers_glob):
            issue = ValidationError(
                path="data.csv_path",
                message="Data source is required.",
                expected="csv_path or managers_glob",
                actual="missing",
                suggestion="Set data.csv_path to a CSV file or data.managers_glob to a CSV glob.",
            )
            _append_issue(errors, issue)

    portfolio = config.get("portfolio")
    if isinstance(portfolio, Mapping):
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

    vol_adjust = config.get("vol_adjust")
    if isinstance(vol_adjust, Mapping):
        _require_field(
            errors,
            vol_adjust,
            "vol_adjust",
            "target_vol",
            expected="number",
            suggestion="Set vol_adjust.target_vol to a numeric target (e.g., 0.1).",
        )


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

    parsed: dict[str, pd.Timestamp] = {}
    invalid_fields: set[str] = set()
    for key in required:
        raw = split.get(key)
        try:
            parsed[key] = pd.to_datetime(raw)
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
    try:
        top_n = int(n_value)
    except (TypeError, ValueError):
        return

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
    text = f"{issue.path}: {issue.message} Expected {issue.expected}, got {actual}."
    if issue.suggestion:
        text = f"{text} Suggestion: {issue.suggestion}"
    return text


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
