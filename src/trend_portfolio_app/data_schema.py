from __future__ import annotations

import io
from typing import IO, Any, Dict, List, Tuple

import pandas as pd

from trend_analysis.io.market_data import (
    MarketDataValidationError,
    validate_market_data,
)

DATE_COL = "Date"


class SchemaMeta(Dict[str, Any]):
    pass


def _validate_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, SchemaMeta]:
    try:
        validated = validate_market_data(df, origin="streamlit upload")
    except MarketDataValidationError as exc:
        raise ValueError(str(exc)) from exc

    meta = SchemaMeta(
        original_columns=list(validated.columns),
        n_rows=len(validated),
    )
    metadata = dict(validated.attrs.get("market_data", {}))
    start = metadata.get("start")
    end = metadata.get("end")
    if isinstance(start, pd.Timestamp):
        metadata["start"] = start.isoformat()
    if isinstance(end, pd.Timestamp):
        metadata["end"] = end.isoformat()
    if metadata:
        meta.update(metadata)
    return validated, meta


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
