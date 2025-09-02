from __future__ import annotations
import io
from typing import Tuple, List
import pandas as pd

DATE_COL = "Date"


class SchemaMeta(dict):
    pass


def _validate_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, SchemaMeta]:
    if DATE_COL not in df.columns:
        raise ValueError("Missing required 'Date' column.")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.set_index(DATE_COL).sort_index()
    # Normalize to month-end timestamps
    idx = pd.to_datetime(df.index)
    df.index = pd.PeriodIndex(idx, freq="M").to_timestamp(how="end")
    df = df.dropna(axis=1, how="all")
    if df.shape[1] == 0:
        raise ValueError("No return columns found after dropping empty columns.")
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns: {dups}")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    meta = SchemaMeta(original_columns=list(df.columns), n_rows=len(df))
    return df, meta


def load_and_validate_csv(file_like) -> Tuple[pd.DataFrame, SchemaMeta]:
    df = pd.read_csv(file_like)
    return _validate_df(df)


def load_and_validate_file(file_like) -> Tuple[pd.DataFrame, SchemaMeta]:
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
