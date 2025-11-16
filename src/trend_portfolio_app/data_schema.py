from __future__ import annotations

import io
from typing import IO, Any, Dict, List, Optional, Tuple

import pandas as pd

from trend.input_validation import InputSchema, validate_input
from trend_analysis.io.market_data import (
    MarketDataMetadata,
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


def _build_meta(validated: ValidatedMarketData) -> SchemaMeta:
    metadata = validated.metadata
    meta = SchemaMeta()
    meta["metadata"] = metadata
    meta["validation"] = _build_validation_report(validated)
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


def _validate_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, SchemaMeta]:
    normalised = validate_input(df, UPLOAD_SCHEMA)
    validated = validate_market_data(normalised)
    meta = _build_meta(validated)
    return validated.frame, meta


def load_and_validate_csv(file_like: IO[Any]) -> Tuple[pd.DataFrame, SchemaMeta]:
    df = pd.read_csv(file_like)
    return _validate_df(df)


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
