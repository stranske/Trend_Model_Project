from __future__ import annotations
import pandas as pd
from typing import Tuple, List

DATE_COL = "Date"


class SchemaMeta(dict):
    pass


def load_and_validate_csv(file_like) -> Tuple[pd.DataFrame, SchemaMeta]:
    df = pd.read_csv(file_like)
    if DATE_COL not in df.columns:
        raise ValueError("Missing required 'Date' column.")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.set_index(DATE_COL).sort_index()
    df.index = df.index.to_period("M").to_timestamp("M")
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
