from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Tuple

import pandas as pd
import pandera as pa
from pandas.tseries.frequencies import to_offset

FrequencyLabel = Literal[
    "daily",
    "business-daily",
    "weekly",
    "monthly",
    "quarterly",
    "annual",
    "irregular",
]


@dataclass(frozen=True)
class MarketDataMetadata:
    """Metadata describing a validated market data frame."""

    mode: Literal["returns", "prices"]
    frequency: FrequencyLabel
    frequency_code: str
    start: pd.Timestamp
    end: pd.Timestamp
    columns: Tuple[str, ...]
    origin: str | None = None


class MarketDataValidationError(ValueError):
    """Raised when market data does not satisfy the ingest contract."""

    def __init__(self, message: str, *, details: Dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


_DATE_SCHEMA = pa.SeriesSchema(pa.DateTime, coerce=True, nullable=False)
_NUMERIC_COLUMN_SCHEMA = pa.Column(pa.Float64, nullable=True, coerce=True)


_FRIENDLY_FREQUENCIES: dict[str, FrequencyLabel] = {
    "D": "daily",
    "B": "business-daily",
    "C": "business-daily",
    "W": "weekly",
    "W-SUN": "weekly",
    "W-MON": "weekly",
    "W-TUE": "weekly",
    "W-WED": "weekly",
    "W-THU": "weekly",
    "W-FRI": "weekly",
    "W-SAT": "weekly",
    "M": "monthly",
    "MS": "monthly",
    "ME": "monthly",
    "Q": "quarterly",
    "QS": "quarterly",
    "QE": "quarterly",
    "A": "annual",
    "AS": "annual",
    "Y": "annual",
}


def _normalise_datetime_index(
    series: pd.Series, origin: str | None
) -> pd.DatetimeIndex:
    try:
        coerced = _DATE_SCHEMA.validate(series)
    except pa.errors.SchemaError as exc:  # pragma: no cover - defensive
        # ``failure_cases`` holds rows where coercion failed. Surface a concise
        # error message with a preview of the offending values.
        cases = getattr(exc, "failure_cases", pd.DataFrame())
        if not cases.empty and {"index", "failure_case"}.issubset(cases.columns):
            preview = cases.iloc[:5]["failure_case"].tolist()
        else:
            preview = series.iloc[:5].tolist()
        tail = "…" if len(preview) == 5 and len(series) > 5 else ""
        raise MarketDataValidationError(
            (
                "Unable to parse Date values"
                + (f" in {origin}" if origin else "")
                + f": {preview}{tail}."
            )
        ) from exc

    if coerced.isna().any():
        bad_idx = coerced[coerced.isna()].index
        preview_vals = series.loc[bad_idx][:5].tolist()
        tail = "…" if len(preview_vals) == 5 and len(bad_idx) > 5 else ""
        raise MarketDataValidationError(
            (
                "Found date values that could not be parsed"
                + (f" in {origin}" if origin else "")
                + f": {preview_vals}{tail}"
            )
        )

    tz_aware = getattr(coerced.dt, "tz", None) is not None
    if tz_aware:
        coerced = coerced.dt.tz_convert(None)
    return pd.DatetimeIndex(coerced.dt.tz_localize(None))


def _ensure_monotonic(idx: pd.DatetimeIndex) -> None:
    if idx.is_monotonic_increasing:
        return
    # Identify the first inversion to give the user actionable feedback.
    for prev, nxt in zip(idx[:-1], idx[1:]):
        if prev > nxt:
            raise MarketDataValidationError(
                (
                    "Timestamps must be sorted in ascending order."
                    f" Found {prev.strftime('%Y-%m-%d')} before {nxt.strftime('%Y-%m-%d')}."
                )
            )
    # Fall back to a generic error if inversion detection failed.
    raise MarketDataValidationError("Timestamps must be in ascending order.")


def _ensure_unique(idx: pd.DatetimeIndex) -> None:
    if not idx.has_duplicates:
        return
    dupes = idx[idx.duplicated()].unique()
    preview = [ts.strftime("%Y-%m-%d") for ts in dupes[:5]]
    tail = "…" if len(dupes) > 5 else ""
    raise MarketDataValidationError(
        f"Duplicate timestamps detected: {', '.join(preview)}{tail}."
    )


def _infer_frequency(idx: pd.DatetimeIndex) -> Tuple[str, FrequencyLabel]:
    freq_code = idx.freqstr
    if freq_code is None:
        try:
            freq_code = pd.infer_freq(idx)
        except ValueError:
            freq_code = None

    if freq_code is None:
        diffs = idx.to_series().diff().dropna()
        if diffs.empty:
            raise MarketDataValidationError(
                "Need at least two rows to infer data frequency."
            )
        counts = diffs.value_counts(normalize=True)
        top_delta = counts.index[0]
        consistency = float(counts.iloc[0])
        if consistency < 0.8:
            preview = ", ".join(str(delta) for delta in counts.index[:3])
            raise MarketDataValidationError(
                (
                    "Mixed sampling cadence detected; unable to infer frequency."
                    f" Observed intervals: {preview}."
                )
            )
        freq_code = to_offset(top_delta).freqstr

    freq_code = freq_code or "irregular"
    label = _FRIENDLY_FREQUENCIES.get(freq_code, "irregular")
    return freq_code, label


def _validate_numeric_payload(df: pd.DataFrame, *, origin: str | None) -> pd.DataFrame:
    if df.empty:
        raise MarketDataValidationError("No data columns found after validation.")

    schema = pa.DataFrameSchema(
        {col: _NUMERIC_COLUMN_SCHEMA for col in df.columns}, coerce=True
    )
    try:
        validated = schema.validate(df)
    except pa.errors.SchemaError as exc:
        failure_cases = getattr(exc, "failure_cases", pd.DataFrame())
        if not failure_cases.empty and {"column", "failure_case"}.issubset(
            failure_cases.columns
        ):
            grouped: dict[str, List[Any]] = {}
            for col, value in zip(
                failure_cases["column"], failure_cases["failure_case"], strict=False
            ):
                grouped.setdefault(str(col), []).append(value)
            fragments = [
                f"{name}: {values[:3]}{'…' if len(values) > 3 else ''}"
                for name, values in grouped.items()
            ]
            msg = ", ".join(fragments)
        else:
            msg = "non-numeric values present"
        raise MarketDataValidationError(
            (
                "Failed to coerce numeric data"
                + (f" in {origin}" if origin else "")
                + f": {msg}."
            )
        ) from exc

    empty_columns = [col for col in validated.columns if validated[col].count() == 0]
    if empty_columns:
        raise MarketDataValidationError(
            "All values missing in column(s): " + ", ".join(empty_columns)
        )
    return validated.astype(float)


def _classify_column_mode(
    series: pd.Series,
) -> Literal["returns", "prices", "ambiguous"]:
    data = series.dropna()
    if data.empty:
        return "ambiguous"

    abs_vals = data.abs()
    returns_share = (abs_vals <= 1.2).mean()
    positive_share = (data >= 0).mean()
    high_value_share = (abs_vals >= 5).mean()
    median_val = float(abs_vals.median()) if not math.isnan(abs_vals.median()) else 0.0

    if returns_share >= 0.9:
        return "returns"
    if positive_share >= 0.98 and (high_value_share >= 0.6 or median_val >= 5.0):
        return "prices"
    return "ambiguous"


def _infer_mode(values: pd.DataFrame) -> Literal["returns", "prices"]:
    classifications: Dict[str, Literal["returns", "prices", "ambiguous"]] = {}
    for col in values.columns:
        classifications[col] = _classify_column_mode(values[col])

    resolved = {mode for mode in classifications.values() if mode != "ambiguous"}
    if not resolved:
        raise MarketDataValidationError(
            "Unable to determine if the dataset is in returns or price mode."
        )
    if len(resolved) > 1:
        returns_like = [
            col for col, mode in classifications.items() if mode == "returns"
        ]
        price_like = [col for col, mode in classifications.items() if mode == "prices"]
        raise MarketDataValidationError(
            (
                "Detected a mix of returns-like and price-like columns. "
                f"Returns-like: {returns_like or 'none'}; "
                f"Price-like: {price_like or 'none'}."
            )
        )
    mode = resolved.pop()
    ambiguous = [col for col, state in classifications.items() if state == "ambiguous"]
    if ambiguous:
        raise MarketDataValidationError(
            (
                "Unable to confidently classify column(s) as returns or prices: "
                + ", ".join(ambiguous)
            )
        )
    return mode


def _extract_date_series(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    if "Date" in df.columns:
        date_series = df["Date"]
        value_columns = [col for col in df.columns if col != "Date"]
    elif isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        if isinstance(df.index, pd.PeriodIndex):
            date_series = df.index.to_timestamp(how="end")
        else:
            date_series = pd.Series(df.index)
        value_columns = list(df.columns)
    else:
        raise MarketDataValidationError(
            "Expected a 'Date' column or a datetime-like index with market data."
        )
    return pd.Series(date_series, name="Date"), value_columns


def validate_market_data(
    data: pd.DataFrame, *, origin: str | None = None
) -> pd.DataFrame:
    """Validate market data and return a normalised frame.

    The returned DataFrame uses a ``DatetimeIndex`` named ``Date`` that is sorted,
    unique, and timezone-naive. Columns are coerced to ``float`` and the ingest
    mode (returns vs prices) together with the inferred sampling frequency are
    stored in ``frame.attrs['market_data']``.
    """

    if not isinstance(data, pd.DataFrame):
        raise MarketDataValidationError(
            f"Expected a pandas DataFrame, received {type(data).__name__}."
        )

    if data.empty:
        raise MarketDataValidationError("Input dataset contains no rows.")

    if data.columns.duplicated().any():
        duplicates = sorted({str(c) for c in data.columns[data.columns.duplicated()]})
        raise MarketDataValidationError(
            "Duplicate column names detected: " + ", ".join(duplicates)
        )

    date_series, value_columns = _extract_date_series(data)
    if not value_columns:
        raise MarketDataValidationError(
            "No data columns provided alongside Date column."
        )

    idx = _normalise_datetime_index(date_series, origin)
    _ensure_monotonic(idx)
    _ensure_unique(idx)
    freq_code, freq_label = _infer_frequency(idx)

    payload = data[value_columns].copy()
    payload = _validate_numeric_payload(payload, origin=origin)

    mode = _infer_mode(payload)

    payload.index = idx
    payload.index.name = "Date"
    payload = payload.sort_index()

    metadata = MarketDataMetadata(
        mode=mode,
        frequency=freq_label,
        frequency_code=freq_code,
        start=payload.index.min(),
        end=payload.index.max(),
        columns=tuple(str(c) for c in payload.columns),
        origin=origin,
    )
    payload.attrs.setdefault("market_data", asdict(metadata))
    payload.attrs["market_data_mode"] = mode
    payload.attrs["market_data_frequency"] = freq_label

    return payload


__all__ = [
    "MarketDataMetadata",
    "MarketDataValidationError",
    "validate_market_data",
]
