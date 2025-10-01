"""Market data validation helpers.

This module centralises the validation logic that backs every ingest entry
point (CSV, Parquet, and in-memory DataFrames).  The goal is to enforce a
single data contract so the application can provide deterministic feedback to
users regardless of how data is supplied.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------

_HUMAN_FREQUENCY_LABELS = {
    "D": "daily",
    "W": "weekly",
    "M": "monthly",
    "ME": "monthly",
    "Q": "quarterly",
    "QE": "quarterly",
    "Y": "annual",
}


class MarketDataMode(str, enum.Enum):
    """Supported representations for market data values."""

    RETURNS = "returns"
    PRICE = "price"


class MarketDataValidationError(ValueError):
    """Raised when uploaded market data fails validation checks."""

    def __init__(self, message: str, issues: Sequence[str] | None = None) -> None:
        formatted = message.strip()
        super().__init__(formatted)
        self.issues: list[str] = list(issues or [])
        self.user_message = formatted


class MarketDataMetadata(BaseModel):
    """Metadata captured during validation."""

    mode: MarketDataMode
    frequency: str
    frequency_label: str
    start: datetime
    end: datetime
    rows: int
    columns: List[str] = Field(default_factory=list)

    @property
    def date_range(self) -> Tuple[str, str]:
        return self.start.strftime("%Y-%m-%d"), self.end.strftime("%Y-%m-%d")


@dataclass(slots=True, frozen=True)
class ValidatedMarketData:
    """Container that pairs a validated frame with its metadata."""

    frame: pd.DataFrame
    metadata: MarketDataMetadata


def _format_issues(issues: Iterable[str]) -> str:
    lines = ["Data validation failed:"]
    for issue in issues:
        lines.append(f"• {issue}")
    return "\n".join(lines)


def _resolve_datetime_index(df: pd.DataFrame, *, source: str | None) -> pd.DataFrame:
    working = df.copy()

    if isinstance(working.index, pd.DatetimeIndex):
        idx = working.index.tz_localize(None)
    else:
        date_col = None
        for column in working.columns:
            if str(column).lower() == "date":
                date_col = column
                break
        if date_col is None:
            raise MarketDataValidationError(
                _format_issues(
                    [
                        "Missing a 'Date' column or datetime index. "
                        "Ensure the upload includes a timestamp column named 'Date'."
                    ]
                )
            )
        parsed = pd.to_datetime(working[date_col], errors="coerce")
        if parsed.isna().any():
            bad_values = working.loc[parsed.isna(), date_col].astype(str).tolist()
            preview = ", ".join(bad_values[:5])
            if len(bad_values) > 5:
                preview += " …"
            raise MarketDataValidationError(
                _format_issues(
                    [
                        "Found dates that could not be parsed. "
                        f"Examples: {preview}."
                    ]
                )
            )
        idx = pd.DatetimeIndex(parsed, name="Date")
        working = working.drop(columns=[date_col])

    if working.empty:
        raise MarketDataValidationError(
            _format_issues(["No data columns detected after extracting the Date index."])
        )

    idx = idx.tz_localize(None)
    working.index = idx
    working.index.name = "Date"
    return working


def _check_monotonic_index(index: pd.DatetimeIndex) -> list[str]:
    issues: list[str] = []
    if not index.is_monotonic_increasing:
        # Identify the first offending timestamp for actionable feedback
        sorted_index = index.sort_values()
        for original, ordered in zip(index, sorted_index, strict=False):
            if original != ordered:
                issues.append(
                    "Date index must be sorted ascending. "
                    f"First out-of-order timestamp: {original.strftime('%Y-%m-%d')}"
                )
                break
    duplicates = index[index.duplicated()].unique()
    if len(duplicates) > 0:
        preview = ", ".join(ts.strftime("%Y-%m-%d") for ts in duplicates[:5])
        if len(duplicates) > 5:
            preview += " …"
        issues.append(f"Found duplicate timestamps: {preview}")
    return issues


def _infer_frequency(index: pd.DatetimeIndex) -> Tuple[str, str]:
    if len(index) < 2:
        return "UNKNOWN", "unknown"

    freq = pd.infer_freq(index)
    if freq is None:
        diffs = index.to_series().diff().dropna()
        unique_deltas = diffs.unique()
        if len(unique_deltas) == 1:
            freq = pd.tseries.frequencies.to_offset(unique_deltas[0]).freqstr
        else:
            # Attempt common calendar-based frequencies (monthly/quarterly) even when
            # day deltas differ because of calendar length variations.
            for candidate in ("ME", "M", "QE", "Q", "A", "Y"):
                try:
                    reconstructed = index.to_period(candidate).to_timestamp(how="end")
                except Exception:  # pragma: no cover - defensive guard
                    continue
                if reconstructed.equals(index.sort_values()):
                    freq = candidate
                    break
        if freq is None:
            preview = ", ".join(str(delta) for delta in unique_deltas[:3])
            if len(unique_deltas) > 3:
                preview += " …"
            raise MarketDataValidationError(
                _format_issues(
                    [
                        "Unable to infer the sampling frequency. "
                        "Detected spacing values: "
                        + preview
                        + ". Ensure the Date index is evenly spaced."
                    ]
                )
            )

    canonical = pd.tseries.frequencies.to_offset(freq).freqstr.upper()
    label = _HUMAN_FREQUENCY_LABELS.get(canonical, canonical)

    expected = pd.date_range(index[0], index[-1], freq=canonical)
    missing = expected.difference(index)
    if len(missing) > 0:
        preview = ", ".join(ts.strftime("%Y-%m-%d") for ts in missing[:5])
        if len(missing) > 5:
            preview += " …"
        raise MarketDataValidationError(
            _format_issues(
                [
                    "Detected gaps in the Date index (missing timestamps: "
                    + preview
                    + ")."
                ]
            )
        )

    return canonical, label


def _coerce_numeric(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    numeric = pd.DataFrame(index=df.index)
    issues: list[str] = []

    for column in df.columns:
        series = df[column]
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().sum() == 0:
            issues.append(f"Column '{column}' contains no numeric data after coercion.")
        numeric[column] = coerced

    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] == 0:
        issues.append("No numeric data columns remain after validation.")

    return numeric, issues


def _column_mode(series: pd.Series) -> MarketDataMode | None:
    values = series.dropna().astype(float)
    if values.empty:
        return None

    abs_values = values.abs()
    median_abs = abs_values.median()
    max_abs = abs_values.max()
    neg_fraction = (values < 0).mean()

    returns_like = (median_abs <= 0.5 and max_abs <= 5) or (
        neg_fraction >= 0.05 and max_abs <= 10
    )

    price_like = values.min() >= 0 and (median_abs >= 1 or max_abs >= 10)

    if returns_like and not price_like:
        return MarketDataMode.RETURNS
    if price_like and not returns_like:
        return MarketDataMode.PRICE
    return None


def _infer_mode(df: pd.DataFrame) -> MarketDataMode:
    modes: list[MarketDataMode] = []
    ambiguous: list[str] = []
    for column in df.columns:
        if not is_numeric_dtype(df[column]):
            continue
        mode = _column_mode(df[column])
        if mode is None:
            ambiguous.append(column)
        else:
            modes.append(mode)

    if not modes:
        raise MarketDataValidationError(
            _format_issues(
                [
                    "Unable to determine whether the data are prices or returns. "
                    "Ensure numeric columns contain representative values."
                ]
            )
        )

    unique_modes = set(modes)
    if len(unique_modes) > 1:
        raise MarketDataValidationError(
            _format_issues(
                [
                    "Detected a mix of price-like and return-like columns. "
                    "Uploads must use a single representation."
                ]
            )
        )

    mode = modes[0]
    if ambiguous:
        preview = ", ".join(ambiguous[:5])
        raise MarketDataValidationError(
            _format_issues(
                [
                    "Could not classify columns as price or return series: "
                    + preview
                ]
            )
        )

    return mode


def validate_market_data(
    data: pd.DataFrame,
    *,
    source: str | None = None,
) -> ValidatedMarketData:
    """Validate market data according to the ingest contract."""

    frame = _resolve_datetime_index(data, source=source)
    issues = _check_monotonic_index(frame.index)
    if issues:
        raise MarketDataValidationError(_format_issues(issues), issues)

    numeric_frame, numeric_issues = _coerce_numeric(frame)
    if numeric_issues:
        raise MarketDataValidationError(_format_issues(numeric_issues), numeric_issues)

    frequency, label = _infer_frequency(numeric_frame.index)
    mode = _infer_mode(numeric_frame)

    metadata = MarketDataMetadata(
        mode=mode,
        frequency=frequency,
        frequency_label=label,
        start=numeric_frame.index.min().to_pydatetime(),
        end=numeric_frame.index.max().to_pydatetime(),
        rows=len(numeric_frame),
        columns=list(numeric_frame.columns),
    )

    validated = numeric_frame.sort_index()
    validated.attrs.setdefault("market_data", {})
    validated.attrs["market_data"]["metadata"] = metadata

    return ValidatedMarketData(frame=validated, metadata=metadata)


def load_market_data_csv(path: str) -> ValidatedMarketData:
    """Load a CSV file and validate its contents."""

    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise MarketDataValidationError(
            _format_issues([f"File not found: {path}"])
        ) from exc
    except PermissionError as exc:  # pragma: no cover - defensive guard
        raise MarketDataValidationError(
            _format_issues([f"Permission denied when reading: {path}"])
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise MarketDataValidationError(
            _format_issues([f"File contains no data: {path}"])
        ) from exc
    except pd.errors.ParserError as exc:
        raise MarketDataValidationError(
            _format_issues([f"Failed to parse file '{path}'"])
        ) from exc

    return validate_market_data(frame, source=path)


def load_market_data_parquet(path: str) -> ValidatedMarketData:
    """Load a Parquet file and validate its contents."""

    try:
        frame = pd.read_parquet(path)
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise MarketDataValidationError(
            _format_issues([f"File not found: {path}"])
        ) from exc
    except PermissionError as exc:  # pragma: no cover - defensive guard
        raise MarketDataValidationError(
            _format_issues([f"Permission denied when reading: {path}"])
        ) from exc

    return validate_market_data(frame, source=path)


def attach_metadata(frame: pd.DataFrame, metadata: MarketDataMetadata) -> pd.DataFrame:
    """Attach metadata to a DataFrame in-place and return it."""

    frame.attrs.setdefault("market_data", {})
    frame.attrs["market_data"]["metadata"] = metadata
    return frame

