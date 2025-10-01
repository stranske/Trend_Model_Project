from __future__ import annotations

import io
from typing import IO, Any, Dict, List, Optional, Tuple

import pandas as pd

from trend_analysis.io.market_data import (
    MarketDataMetadata,
    MarketDataValidationError,
    ValidatedMarketData,
    validate_market_data,
)

DATE_COL = "Date"


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
    meta["date_range"] = metadata.date_range
    meta["start"] = metadata.start
    meta["end"] = metadata.end
    return meta


def _validate_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, SchemaMeta]:
    try:
        validated = validate_market_data(df)
    except MarketDataValidationError as exc:
        raise ValueError(exc.user_message) from exc
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
