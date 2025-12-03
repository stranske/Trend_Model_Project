from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from trend.input_validation import InputSchema, InputValidationError, validate_input
from trend_analysis.io.market_data import (
    MarketDataMetadata,
    MarketDataValidationError,
    ValidatedMarketData,
    validate_market_data,
)

DANGEROUS_HEADER_PREFIXES = ("=", "+", "-", "@")
SAFE_HEADER_PREFIX = "safe_"

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


def _normalise_header_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        # Remove the UTF-8 BOM (Byte Order Mark) character if present.
        return value.lstrip("\ufeff")
    if pd.isna(value):
        return ""
    return str(value)


def extract_headers_from_bytes(raw: bytes, *, is_excel: bool) -> list[str] | None:
    """Return the first-row headers without pandas mangling duplicate names."""

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


def apply_original_headers(
    df: pd.DataFrame, headers: Sequence[str] | None
) -> Sequence[str] | None:
    """Assign ``headers`` to ``df`` when lengths match, preserving duplicates."""

    if not headers:
        return None
    if len(headers) != len(df.columns):
        return None
    df.columns = list(headers)
    return headers


def _read_binary_payload(file_like: IO[Any] | str | Path) -> tuple[bytes, str]:
    if isinstance(file_like, (str, Path)):
        path = Path(file_like)
        return path.read_bytes(), path.name

    if hasattr(file_like, "read") and callable(file_like.read):
        try:
            current = file_like.tell()
            file_like.seek(0)
        except Exception:
            current = None
        data = file_like.read()
        if current is not None:
            try:
                file_like.seek(current)
            except Exception:
                # Seeking may fail for some file-like objects (e.g., streams); safe to ignore.
                pass
        if isinstance(data, bytes):
            raw = data
        else:
            raw = str(data or "").encode("utf-8")
        name = getattr(file_like, "name", "upload")
        return raw, name

    raise TypeError("Unsupported file-like object for data ingestion.")


def _build_validation_report(
    validated: ValidatedMarketData,
    sanitized_columns: Optional[list[Dict[str, str]]] = None,
) -> Dict[str, Any]:
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
    if sanitized_columns:
        formatted = ", ".join(
            f"{entry['original']!r} → {entry['sanitized']!r}"
            for entry in sanitized_columns
        )
        warnings.append(
            "Sanitized column headers (cleaned) to prevent Excel from running formulas: "
            + formatted
            + "."
        )
    return {"issues": [], "warnings": warnings}


def _build_meta(
    validated: ValidatedMarketData,
    sanitized_columns: Optional[list[Dict[str, str]]] = None,
) -> SchemaMeta:
    metadata = validated.metadata
    meta = SchemaMeta()
    meta["metadata"] = metadata
    meta["validation"] = _build_validation_report(
        validated, sanitized_columns=sanitized_columns
    )
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
    meta["sanitized_columns"] = sanitized_columns or []
    return meta


def _needs_formula_sanitization(name: str) -> bool:
    stripped = str(name).lstrip()
    return bool(stripped) and stripped.startswith(DANGEROUS_HEADER_PREFIXES)


def _allocate_unique_name(base: str, occupied: set[str]) -> str:
    """Return ``base`` or a suffixed variant that does not clash with ``occupied``."""

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


def _sanitize_formula_headers(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, list[Dict[str, str]]]:
    """Rename headers that could be interpreted as Excel formulas."""

    occupied = {str(column) for column in df.columns}
    new_columns: list[Any] = list(df.columns)
    changes: list[Dict[str, str]] = []
    mutated = False

    for idx, column in enumerate(df.columns):
        original = str(column)
        if not _needs_formula_sanitization(original):
            continue

        stripped = original.lstrip()
        body = stripped.lstrip("=+-@").strip()
        base = body or f"{SAFE_HEADER_PREFIX}column"
        if base.casefold() == DATE_COL.casefold():
            base = DATE_COL
        candidate = _allocate_unique_name(base, occupied)
        new_columns[idx] = candidate
        changes.append({"original": original, "sanitized": candidate})
        mutated = True

    if not mutated:
        return df, []

    sanitized = df.copy()
    sanitized.columns = new_columns
    return sanitized, changes


def _validate_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, SchemaMeta]:
    sanitized_source, sanitized_columns = _sanitize_formula_headers(df)
    try:
        normalised = validate_input(sanitized_source, UPLOAD_SCHEMA)
    except InputValidationError as exc:
        raise MarketDataValidationError(exc.user_message, exc.issues) from exc
    validated = validate_market_data(normalised)
    meta = _build_meta(validated, sanitized_columns=sanitized_columns)
    return validated.frame, meta


def load_and_validate_csv(
    file_like: IO[Any] | str | Path,
) -> Tuple[pd.DataFrame, SchemaMeta]:
    raw, name = _read_binary_payload(file_like)
    buffer = io.BytesIO(raw)
    buffer.name = name or "upload.csv"
    headers = extract_headers_from_bytes(raw, is_excel=False)
    df = pd.read_csv(buffer)
    apply_original_headers(df, headers)
    return _validate_df(df)


def load_and_validate_file(
    file_like: IO[Any] | str | Path,
) -> Tuple[pd.DataFrame, SchemaMeta]:
    """Load CSV or Excel from an UploadedFile or file-like, then validate."""

    raw, name = _read_binary_payload(file_like)
    lowered = (name or "").lower()
    is_excel = lowered.endswith((".xlsx", ".xls"))
    headers = extract_headers_from_bytes(raw, is_excel=is_excel)
    buffer = io.BytesIO(raw)
    buffer.name = name
    try:
        if is_excel:
            df = pd.read_excel(buffer)
        else:
            df = pd.read_csv(buffer)
    except Exception:
        raise
    apply_original_headers(df, headers)
    return _validate_df(df)


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
