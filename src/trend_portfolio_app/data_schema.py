from __future__ import annotations

import io
import re
from typing import IO, Any, Dict, List, Optional, Tuple

import pandas as pd

from trend.input_validation import InputSchema, InputValidationError, validate_input
from trend_analysis.io.market_data import (
    MarketDataMetadata,
    MarketDataValidationError,
    ValidatedMarketData,
    validate_market_data,
)

DATE_COL = "Date"
UPLOAD_SCHEMA = InputSchema(
    date_column=DATE_COL,
    required_columns=(DATE_COL,),
    non_nullable=(DATE_COL,),
)


class SchemaMeta(Dict[str, Any]):
    """Lightweight metadata structure shared with the Streamlit app."""

    validation: Optional[Any]
    metadata: Optional[MarketDataMetadata]


def _build_validation_report(validated: ValidatedMarketData) -> Dict[str, Any]:
    metadata = validated.metadata
    frame = validated.frame
    warnings: list[str] = []
    rows = metadata.rows
    if rows < 12:
        warnings.append(
            f"Dataset is quite small ({rows} periods) – consider a longer history."
        )
    for column in frame.columns:
        valid = frame[column].notna().sum()
        if rows and valid / rows <= 0.5:
            warnings.append(
                f"Column '{column}' has >50% missing values ({valid}/{rows} valid)."
            )
    if metadata.frequency_missing_periods > 0:
        warnings.append(
            "Date index contains "
            f"{metadata.frequency_missing_periods} missing {metadata.frequency_label} periods "
            f"(max gap {metadata.frequency_max_gap_periods})."
        )
    if metadata.missing_policy_dropped:
        dropped = ", ".join(sorted(metadata.missing_policy_dropped))
        warnings.append(
            "Missing-data policy dropped columns: "
            f"{dropped} (policy={metadata.missing_policy})."
        )
    if metadata.missing_policy_summary and (
        metadata.frequency_missing_periods > 0
        or bool(metadata.missing_policy_filled)
        or bool(metadata.missing_policy_dropped)
    ):
        warnings.append(
            "Missing-data policy applied: " f"{metadata.missing_policy_summary}."
        )
    return {"issues": [], "warnings": warnings}


def _build_meta(
    validated: ValidatedMarketData, *, extra_warnings: Optional[List[str]] = None
) -> SchemaMeta:
    metadata = validated.metadata
    meta = SchemaMeta()
    meta["metadata"] = metadata
    validation = _build_validation_report(validated)
    if extra_warnings:
        validation["warnings"].extend(extra_warnings)
    meta["validation"] = validation
    meta["original_columns"] = list(metadata.columns or metadata.symbols)
    meta["symbols"] = list(metadata.symbols)
    meta["n_rows"] = metadata.rows
    meta["mode"] = metadata.mode.value
    meta["frequency"] = metadata.frequency_label
    meta["frequency_code"] = metadata.frequency
    meta["frequency_detected"] = metadata.frequency_detected
    meta["frequency_missing_periods"] = metadata.frequency_missing_periods
    meta["frequency_max_gap_periods"] = metadata.frequency_max_gap_periods
    meta["frequency_tolerance_periods"] = metadata.frequency_tolerance_periods
    meta["missing_policy"] = metadata.missing_policy
    meta["missing_policy_limit"] = metadata.missing_policy_limit
    meta["missing_policy_summary"] = metadata.missing_policy_summary
    meta["date_range"] = metadata.date_range
    meta["start"] = metadata.start
    meta["end"] = metadata.end
    return meta


def _validate_df(
    df: pd.DataFrame, *, extra_warnings: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, SchemaMeta]:
    try:
        normalised = validate_input(df, UPLOAD_SCHEMA)
    except InputValidationError as exc:
        raise MarketDataValidationError(exc.user_message, exc.issues) from exc
    validated = validate_market_data(normalised)
    meta = _build_meta(validated, extra_warnings=extra_warnings)
    return validated.frame, meta


def _sanitize_headers(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    sanitized: list[str] = []
    warnings: list[str] = []
    duplicates: dict[str, list[str]] = {}
    seen: dict[str, str] = {}

    for column in df.columns:
        text = str(column).strip()
        stripped = text
        while stripped.startswith(("=", "+", "-", "@")):
            stripped = stripped[1:]
        safe = stripped or "column"

        key = safe.casefold()
        if key in seen:
            duplicates.setdefault(key, [seen[key]]).append(text)
        seen[key] = text
        sanitized.append(safe)

    if duplicates:
        dup_values = {orig for items in duplicates.values() for orig in items}
        dup_display = ", ".join(sorted(dup_values))
        raise InputValidationError(
            "Column names must be unique.",
            issues=(
                [f"Duplicate column(s) detected: {dup_display}."]
                if dup_display
                else None
            ),
        )

    formula_like = [orig for orig, safe in zip(df.columns, sanitized) if orig != safe]
    if formula_like:
        formatted = ", ".join(sorted({str(col) for col in formula_like}))
        warnings.append(
            "Column names starting with =, +, -, or @ were cleaned to remove the prefix: "
            f"{formatted}."
        )

    cleaned = df.copy()
    cleaned.columns = sanitized
    return cleaned, warnings


def load_and_validate_csv(file_like: IO[Any]) -> Tuple[pd.DataFrame, SchemaMeta]:
    df = pd.read_csv(file_like)
    base_headers = [re.sub(r"\.\d+$", "", str(col)) for col in df.columns]
    df.columns = base_headers
    sanitized, warnings = _sanitize_headers(df)
    return _validate_df(sanitized, extra_warnings=warnings)


def load_and_validate_file(file_like: IO[Any]) -> Tuple[pd.DataFrame, SchemaMeta]:
    """Load CSV or Excel from an UploadedFile or file-like, then validate.

    Prefers file extension on the object (``.name``) to decide parser.
    Falls back to CSV when extension is missing or unrecognised.
    """
    name = getattr(file_like, "name", "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            # Ensure we pass a seekable buffer to pandas
            data = file_like.read()
            buf = io.BytesIO(data)
            df = pd.read_excel(buf)
        else:
            df = pd.read_csv(file_like)
    except Exception:
        raise
    else:
        try:
            file_like.seek(0)
        except Exception:
            pass
    base_headers = [re.sub(r"\.\d+$", "", str(col)) for col in df.columns]
    df.columns = base_headers
    sanitized, warnings = _sanitize_headers(df)
    return _validate_df(sanitized, extra_warnings=warnings)


def infer_benchmarks(columns: List[str]) -> List[str]:
    hints = [
        "SPX",
        "S&P",
        "SP500",
        "SP-500",
        "TSX",
        "AGG",
        "BOND",
        "BENCH",
        "IDX",
        "INDEX",
    ]
    cands = []
    for c in columns:
        uc = c.upper()
        if any(h in uc for h in hints):
            cands.append(c)
    return cands
