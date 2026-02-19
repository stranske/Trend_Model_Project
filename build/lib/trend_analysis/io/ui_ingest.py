"""UI-aligned dataset ingestion helpers (shared by CLI and Streamlit)."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from trend_analysis.io.date_correction import (
    DateCorrectionResult,
    analyze_date_column,
    apply_date_corrections,
)
from trend_analysis.io.market_data import (
    MarketDataMetadata,
    MarketDataValidationError,
    validate_market_data,
)

DANGEROUS_HEADER_PREFIXES = ("=", "+", "-", "@")
SAFE_HEADER_PREFIX = "safe_"


@dataclass(frozen=True)
class UiIngestSummary:
    """Summary of corrections applied during UI-style ingestion."""

    corrected_dates: int = 0
    dropped_rows: int = 0


def _normalise_header_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lstrip("\ufeff")
    if pd.isna(value):
        return ""
    return str(value)


def _read_binary_payload(file_like: str | Path) -> tuple[bytes, str]:
    path = Path(file_like)
    return path.read_bytes(), path.name


def _extract_headers_from_bytes(raw: bytes, *, is_excel: bool) -> list[str] | None:
    if not raw:
        return []
    if is_excel:
        try:
            preview = pd.read_excel(io.BytesIO(raw), header=None, nrows=1)
        except Exception:
            return None
        if preview.empty:
            return []
        return [_normalise_header_value(value) for value in preview.iloc[0].tolist()]

    lines = raw.splitlines()
    if not lines:
        return []
    try:
        decoded = lines[0].decode("utf-8-sig", errors="ignore")
    except Exception:
        decoded = lines[0].decode(errors="ignore")
    reader = csv.reader([decoded])
    try:
        header = next(reader)
    except StopIteration:
        return []
    return [_normalise_header_value(value) for value in header]


def _apply_original_headers(df: pd.DataFrame, headers: Sequence[str] | None) -> None:
    if not headers:
        return
    if len(headers) != len(df.columns):
        return
    df.columns = list(headers)


def _needs_formula_sanitization(name: str) -> bool:
    stripped = str(name).lstrip()
    return bool(stripped) and stripped.startswith(DANGEROUS_HEADER_PREFIXES)


def _allocate_unique_name(base: str, occupied: set[str]) -> str:
    candidate = base
    if candidate not in occupied:
        occupied.add(candidate)
        return candidate

    suffix = 2
    while True:
        candidate = f"{base}_{suffix}"
        if candidate not in occupied:
            occupied.add(candidate)
            return candidate
        suffix += 1


def _sanitize_formula_headers(df: pd.DataFrame) -> pd.DataFrame:
    occupied = {str(column) for column in df.columns}
    new_columns: list[Any] = list(df.columns)
    mutated = False

    for idx, column in enumerate(df.columns):
        original = str(column)
        if not _needs_formula_sanitization(original):
            continue

        stripped = original.lstrip()
        body = stripped.lstrip("=+-@").strip()
        base = body or f"{SAFE_HEADER_PREFIX}column"
        if base.casefold() == "date":
            base = "Date"
        candidate = _allocate_unique_name(base, occupied)
        new_columns[idx] = candidate
        mutated = True

    if not mutated:
        return df

    sanitized = df.copy()
    sanitized.columns = new_columns
    return sanitized


def _find_date_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        if str(column).strip().casefold() == "date":
            return str(column)
    return None


def inspect_ui_date_issues(path: str | Path) -> DateCorrectionResult:
    raw, name = _read_binary_payload(path)
    lowered = (name or "").lower()
    is_excel = lowered.endswith((".xlsx", ".xls"))
    headers = _extract_headers_from_bytes(raw, is_excel=is_excel)
    buffer = io.BytesIO(raw)
    buffer.name = name

    if is_excel:
        df = pd.read_excel(buffer)
    else:
        df = pd.read_csv(buffer)

    _apply_original_headers(df, headers)
    df = _sanitize_formula_headers(df)

    date_column = _find_date_column(df)
    if date_column is None:
        raise MarketDataValidationError(
            "Missing a 'Date' column or datetime index. "
            "Ensure the upload includes a timestamp column named 'Date'."
        )
    return analyze_date_column(df, date_column)


def _raise_date_issue(result: DateCorrectionResult, *, auto_fix_dates: bool) -> None:
    issues: list[str] = []
    total_fixable = len(result.corrections) + result.total_droppable_rows
    if total_fixable > 0:
        issues.append(f"{total_fixable} row(s) can be auto-corrected.")
    if result.unfixable:
        preview = ", ".join(f"row {row + 1}: {val!r}" for row, val in result.unfixable[:3])
        issues.append(f"Unfixable dates detected ({preview}).")

    if auto_fix_dates:
        message = "Date column contains values that cannot be auto-corrected."
    else:
        message = "Date column has fixable issues. Re-run with --auto-fix-dates to apply them."

    raise MarketDataValidationError(message, issues=issues)


def load_ui_dataset(
    path: str | Path,
    *,
    auto_fix_dates: bool = False,
    missing_policy: str | Mapping[str, str] = "zero",
    missing_limit: int | Mapping[str, int | None] | None = None,
) -> tuple[pd.DataFrame, MarketDataMetadata, UiIngestSummary]:
    raw, name = _read_binary_payload(path)
    lowered = (name or "").lower()
    is_excel = lowered.endswith((".xlsx", ".xls"))
    headers = _extract_headers_from_bytes(raw, is_excel=is_excel)
    buffer = io.BytesIO(raw)
    buffer.name = name

    if is_excel:
        df = pd.read_excel(buffer)
    else:
        df = pd.read_csv(buffer)

    _apply_original_headers(df, headers)
    df = _sanitize_formula_headers(df)

    date_column = _find_date_column(df)
    if date_column is None:
        raise MarketDataValidationError(
            "Missing a 'Date' column or datetime index. "
            "Ensure the upload includes a timestamp column named 'Date'."
        )

    summary = UiIngestSummary()
    correction_result = analyze_date_column(df, date_column)
    if correction_result.has_corrections or correction_result.total_droppable_rows > 0:
        if auto_fix_dates and correction_result.all_fixable:
            df = apply_date_corrections(
                df,
                date_column,
                correction_result.corrections,
                drop_rows=correction_result.trailing_empty_rows
                + correction_result.droppable_empty_rows,
            )
            summary = UiIngestSummary(
                corrected_dates=len(correction_result.corrections),
                dropped_rows=correction_result.total_droppable_rows,
            )
        else:
            _raise_date_issue(correction_result, auto_fix_dates=auto_fix_dates)
    elif correction_result.has_unfixable:
        _raise_date_issue(correction_result, auto_fix_dates=auto_fix_dates)

    validated = validate_market_data(
        df,
        source=str(path),
        missing_policy=missing_policy,
        missing_limit=missing_limit,
        auto_fix_dates=False,
    )

    return validated.frame, validated.metadata, summary
