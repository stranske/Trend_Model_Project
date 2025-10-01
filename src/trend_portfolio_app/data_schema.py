from __future__ import annotations

import io
from typing import IO, Any, Dict, List, Tuple

import pandas as pd

from trend_analysis.io.market_data import (
    MarketDataMetadata,
    MarketDataValidationError,
    validate_market_data,
)


class SchemaMeta(Dict[str, Any]):
    pass


def _build_meta(metadata: MarketDataMetadata) -> SchemaMeta:
    meta = SchemaMeta()
    meta["original_columns"] = list(metadata.columns)
    meta["n_rows"] = metadata.rows
    meta["mode"] = metadata.mode.value
    meta["frequency"] = metadata.frequency_label
    meta["date_range"] = metadata.date_range
    return meta


def _validate_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, SchemaMeta]:
    try:
        validated = validate_market_data(df)
    except MarketDataValidationError as exc:
        raise ValueError(exc.user_message) from exc
    meta = _build_meta(validated.metadata)
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
