"""Ingestion-time validation helpers for market price data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

PRICE_SCHEMA: Mapping[str, str] = {
    "symbol": "string",
    "date": "datetime64[ns]",
    "close": "float64",
}


def enforce_required_columns(
    df: pd.DataFrame, schema: Mapping[str, str]
) -> pd.DataFrame:
    """Ensure ``df`` contains ``schema`` columns with the requested dtypes."""

    missing = [column for column in schema if column not in df.columns]
    if missing:
        raise ValueError(
            "Price frame missing required columns: " + ", ".join(sorted(missing))
        )

    mismatched: list[str] = []
    for column, dtype in schema.items():
        expected = pd.api.types.pandas_dtype(dtype)
        actual = df[column].dtype
        if not pd.api.types.is_dtype_equal(actual, expected):
            mismatched.append(f"{column} (expected {expected}, found {actual})")

    if mismatched:
        raise ValueError(
            "Price frame column dtypes differ from expected schema: "
            + "; ".join(mismatched)
        )

    return df


def _coerce_price_schema(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["symbol"] = work["symbol"].astype("string")
    work["date"] = pd.to_datetime(work["date"], errors="raise")
    work["date"] = work["date"].dt.tz_localize(None)
    work["close"] = pd.to_numeric(work["close"], errors="raise").astype("float64")
    return work


def validate_prices_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and temporal ordering for ``df``."""

    if df is None:
        raise ValueError("Price frame cannot be None")

    work = _coerce_price_schema(df)
    enforce_required_columns(work, PRICE_SCHEMA)

    date_diffs = work.groupby("symbol", sort=False)["date"].diff()
    backwards = date_diffs.lt(pd.Timedelta(0))
    if backwards.any():
        monotonic_violations = work.loc[backwards.fillna(False), "symbol"].astype(str)
        offenders = sorted(monotonic_violations.unique())
        raise ValueError(
            "Price frame must be sorted by date within each symbol: "
            + ", ".join(offenders)
        )

    work = work.sort_values(["date", "symbol"]).reset_index(drop=True)

    duplicate_mask = work.duplicated(subset=["symbol", "date"])
    if duplicate_mask.any():
        dupes = work.loc[duplicate_mask, ["symbol", "date"]]
        examples = dupes.head().to_dict("records")
        raise ValueError(
            "Price frame contains duplicate symbol/date rows: " + str(examples)
        )

    indexed = work.set_index("date")
    indexed.index.name = "date"
    return indexed


def build_validation_frame(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """Return a tidy (symbol/date/close) view of ``df`` for validation."""

    if date_column not in df.columns:
        raise ValueError(f"DataFrame must include '{date_column}' for validation")

    value_columns = [col for col in df.columns if col != date_column]
    if not value_columns:
        raise ValueError("DataFrame must contain value columns to validate")

    melted = df.melt(
        id_vars=[date_column],
        value_vars=value_columns,
        var_name="symbol",
        value_name="close",
    )
    melted = melted.rename(columns={date_column: "date"})
    return melted.dropna(subset=["date"])


def _extract_date_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
        idx = pd.DatetimeIndex(df.index.get_level_values("date"))
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    elif "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("Unable to determine date index for lag validation")

    idx = idx.dropna()
    if idx.empty:
        raise ValueError("Price frame contains no valid dates for lag validation")
    return idx


def assert_execution_lag(
    df: pd.DataFrame, *, as_of: Any, max_lag_days: int | None
) -> None:
    """Raise if the freshest data trails ``as_of`` by more than ``max_lag_days``."""

    if max_lag_days is None or max_lag_days <= 0 or as_of in (None, ""):
        return

    idx = _extract_date_index(df)
    latest = idx.max().normalize()
    as_of_ts = pd.Timestamp(as_of).tz_localize(None).normalize()
    lag = (as_of_ts - latest).days

    if lag > max_lag_days:
        raise ValueError(
            (
                "Price data stale: latest observation {latest_date} is {lag} days "
                "behind requested as-of {as_of_date} (max allowed {max_days} days)."
            ).format(
                latest_date=latest.date(),
                lag=lag,
                as_of_date=as_of_ts.date(),
                max_days=max_lag_days,
            )
        )


__all__ = [
    "PRICE_SCHEMA",
    "assert_execution_lag",
    "build_validation_frame",
    "enforce_required_columns",
    "validate_prices_frame",
]
