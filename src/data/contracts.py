"""Validation helpers enforcing the price ingest contract."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from datetime import timedelta, timezone
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.tseries.frequencies import to_offset

__all__ = ["coerce_to_utc", "validate_prices"]

UTC = timezone.utc


def _datetime_index_from_frame(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return a DatetimeIndex extracted from ``df``.

    The helper accepts either an existing DatetimeIndex or a ``Date`` column
    that can be parsed into timestamps.  ``ValueError`` is raised when no
    timestamp information can be derived.
    """

    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    elif "Date" in df.columns:
        parsed = pd.to_datetime(df["Date"], errors="coerce")
        if parsed.isna().any():
            raise ValueError("Price data contains unparseable timestamps in 'Date'.")
        idx = pd.DatetimeIndex(parsed, name="Date")
    else:
        raise ValueError("Price data must include a DatetimeIndex or 'Date' column.")

    if idx.isna().any():
        raise ValueError("Price data contains missing timestamps.")
    return idx


def coerce_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` indexed by UTC timestamps.

    Loaders may emit timezone-naïve ``Date`` columns.  This helper promotes the
    index to ``datetime64[ns, UTC]`` so downstream validators can assert
    invariants on a consistent representation.
    """

    idx = _datetime_index_from_frame(df)
    if idx.tz is None:
        idx = idx.tz_localize(UTC)
    else:
        idx = idx.tz_convert(UTC)

    coerced = df.copy()
    coerced.index = idx
    coerced.index.name = idx.name or "Date"
    if "Date" in coerced.columns:
        coerced["Date"] = idx
    coerced.attrs = dict(df.attrs)
    return coerced


def _require_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Price data must be indexed by timestamps before validation.")
    if df.index.isna().any():
        raise ValueError("Price index contains missing timestamps.")
    if len(df.index) == 0:
        raise ValueError("Price data contains no timestamps to validate.")
    return df.index


def _ensure_utc_index(idx: pd.DatetimeIndex) -> None:
    if idx.tz is None:
        raise ValueError("Datetime index must be timezone-aware in UTC.")
    anchor = idx[0].to_pydatetime()
    offset = idx.tz.utcoffset(anchor)
    if offset != timedelta(0):
        raise ValueError("Datetime index must be normalised to UTC.")


def _check_monotonic(idx: pd.DatetimeIndex) -> None:
    if not idx.is_monotonic_increasing:
        first = None
        sorted_idx = idx.sort_values()
        for original, ordered in zip(idx, sorted_idx, strict=True):
            if original != ordered:
                first = original
                break
        detail = f" First unsorted timestamp: {first.isoformat()}" if first else ""
        raise ValueError("Datetime index must be sorted in ascending order." + detail)

    duplicates = idx[idx.duplicated()].unique()
    if len(duplicates) > 0:
        preview = ", ".join(ts.isoformat() for ts in duplicates[:5])
        raise ValueError(f"Duplicate (instrument, timestamp) keys detected: {preview}.")


def _normalise_frequency(freq: str | None) -> str | None:
    if freq is None:
        return None
    freq = freq.strip()
    return freq or None


def _check_frequency(idx: pd.DatetimeIndex, freq: str | None) -> None:
    expected = _normalise_frequency(freq)
    if expected is None or len(idx) < 3:
        return

    inferred = pd.infer_freq(idx)
    if inferred is None:
        raise ValueError(
            "Unable to infer a stable cadence from price data; mixed frequencies detected."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            expected_offset = to_offset(expected)
            inferred_offset = to_offset(inferred)
    except ValueError as exc:  # pragma: no cover - defensive guard for invalid freq
        raise ValueError(f"Unknown frequency alias '{freq}'.") from exc

    if expected_offset != inferred_offset:
        raise ValueError(
            "Detected mixed timestamp spacing; expected frequency "
            f"{expected_offset.rule_code}, observed {inferred_offset.rule_code}."
        )


def _is_price_mode(df: pd.DataFrame) -> bool:
    market_meta: Any = df.attrs.get("market_data", {})
    mode_candidate: Any = None

    if isinstance(market_meta, Mapping):
        metadata = market_meta.get("metadata")
        if metadata is not None:
            mode_attr = getattr(metadata, "mode", None)
            if mode_attr is not None and hasattr(mode_attr, "value"):
                mode_candidate = mode_attr.value
            elif mode_attr is not None:
                mode_candidate = str(mode_attr)
            elif isinstance(metadata, Mapping):
                mode_candidate = metadata.get("mode")
        if mode_candidate is None:
            mode_candidate = market_meta.get("mode")

    if mode_candidate is None:
        mode_candidate = df.attrs.get("market_data_mode")

    return isinstance(mode_candidate, str) and mode_candidate.lower() == "price"


def _check_non_negative_prices(df: pd.DataFrame) -> None:
    if not _is_price_mode(df):
        return

    value_columns = [col for col in df.columns if str(col).lower() != "date"]
    for column in value_columns:
        series = df[column]
        if not is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce")
        else:
            series = series.astype(float)
        bad_mask = (series <= 0).fillna(False)
        if bad_mask.any():
            first_idx = bad_mask[bad_mask].index[0]
            raise ValueError(
                "Price columns must contain strictly positive values. "
                f"Column '{column}' failed at {first_idx.isoformat()}"
            )


def validate_prices(df: pd.DataFrame, *, freq: str | None = "D") -> pd.DataFrame:
    """Validate price frames before ingest.

    Parameters
    ----------
    df:
        DataFrame indexed by timestamps.
    freq:
        Expected pandas offset alias (defaults to daily).  ``None`` skips the
        cadence check but still enforces ordering, duplication, timezone, and
        price constraints.
    """

    idx = _require_datetime_index(df)
    _ensure_utc_index(idx)
    _check_monotonic(idx)
    _check_frequency(idx, freq)
    _check_non_negative_prices(df)
    return df
