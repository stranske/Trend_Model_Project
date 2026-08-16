"""Market data validation helpers.

This module centralises the validation logic that backs every ingest
entry point (CSV, Parquet, and in-memory DataFrames).  The goal is to
enforce a single data contract so the application can provide
deterministic feedback to users regardless of how data is supplied.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, Field, model_validator

from trend_analysis.io.date_correction import (
    analyze_date_column,
    apply_date_corrections,
)
from trend_analysis.util.missing import (
    MissingPolicyResult,
    apply_missing_policy as _apply_missing_policy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------

_HUMAN_FREQUENCY_LABELS = {
    "D": "daily",
    "B": "daily",
    "W": "weekly",
    "M": "monthly",
    "ME": "monthly",
    "Q": "quarterly",
    "QE": "quarterly",
    "Y": "annual",
    "YE": "annual",
}


def _normalise_delta_days(delta_days: pd.Series) -> pd.Series:
    return _normalize_delta_days(delta_days)


_DEFAULT_MISSING_POLICY = "drop"


# ---------------------------------------------------------------------------
# Validation classes
# ---------------------------------------------------------------------------


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


class MissingPolicyFillDetails(BaseModel):
    """Details about how missing data were imputed for a column."""

    method: str
    count: int = 0


class MarketDataMetadata(BaseModel):
    """Metadata captured during validation."""

    mode: MarketDataMode
    frequency: str
    frequency_detected: str = ""
    frequency_label: str
    frequency_median_spacing_days: float = 0.0
    frequency_missing_periods: int = 0
    frequency_max_gap_periods: int = 0
    frequency_tolerance_periods: int = 0
    start: datetime
    end: datetime
    rows: int
    columns: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    missing_policy: str = Field(default=_DEFAULT_MISSING_POLICY)
    missing_policy_limit: int | None = None
    missing_policy_overrides: dict[str, str] = Field(default_factory=dict)
    missing_policy_limits: dict[str, int | None] = Field(default_factory=dict)
    missing_policy_filled: dict[str, MissingPolicyFillDetails] = Field(default_factory=dict)
    missing_policy_dropped: list[str] = Field(default_factory=list)
    missing_policy_summary: str | None = None

    @property
    def date_range(self) -> tuple[str, str]:
        return self.start.strftime("%Y-%m-%d"), self.end.strftime("%Y-%m-%d")

    @model_validator(mode="after")
    def _sync_symbols(self) -> MarketDataMetadata:
        """Keep the ``columns`` and ``symbols`` fields aligned."""

        if not self.symbols and self.columns:
            self.symbols = list(self.columns)
        elif self.symbols and not self.columns:
            self.columns = list(self.symbols)
        return self


@dataclass(slots=True, frozen=True)
class ValidatedMarketData:
    """Container that pairs a validated frame with its metadata."""

    frame: pd.DataFrame
    metadata: MarketDataMetadata

    def __getattr__(self, name: str) -> Any:
        # Delegate attribute access to the underlying DataFrame for
        # backwards compatibility with callers expecting the validated
        # payload itself.
        return getattr(self.frame, name)

    def __getitem__(self, key: Any) -> Any:
        return self.frame.__getitem__(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.frame)

    def __len__(self) -> int:  # pragma: no cover - passthrough delegation
        return len(self.frame)

    def __array__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return self.frame.__array__(*args, **kwargs)

    def to_frame(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""

        return self.frame


def _format_issues(issues: Iterable[str]) -> str:
    lines = ["Data validation failed:"]
    for issue in issues:
        lines.append(f"• {issue}")
    return "\n".join(lines)


def _missing_policy_fill_details(
    result: MissingPolicyResult,
) -> dict[str, MissingPolicyFillDetails]:
    """Adapt canonical diagnostics to the persisted market-data metadata model."""

    return {
        column: MissingPolicyFillDetails(
            method=result.policy[column],
            count=count,
        )
        for column, count in result.filled.items()
        if count > 0
    }


def _summarise_missing_policy(result: MissingPolicyResult) -> str:
    parts = [result.summary]
    filled = _missing_policy_fill_details(result)
    if filled:
        chunks = [
            f"{column} ({details.method}: {details.count})"
            for column, details in sorted(filled.items())
        ]
        parts.append("filled=" + ", ".join(chunks))
    if result.dropped_assets:
        parts.append("dropped=" + ", ".join(sorted(result.dropped_assets)))
    return "; ".join(parts)


def _normalize_delta_days(delta_days: pd.Series) -> pd.Series:
    cleaned = delta_days.replace([np.inf, -np.inf], np.nan).dropna()
    return cleaned.astype(float)


def classify_frequency(
    index: pd.DatetimeIndex,
    *,
    max_gap_limit: int | None = None,
) -> dict[str, Any]:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return {
            "canonical": "UNKNOWN",
            "code": "UNKNOWN",
            "label": "unknown",
            "median_days": 0.0,
            "max_missing_periods": 0,
            "total_missing_periods": 0,
            "tolerance_periods": 0,
        }

    idx = index.sort_values()
    diffs = idx.to_series().diff().dropna()
    if diffs.empty:
        return {
            "canonical": "UNKNOWN",
            "code": "UNKNOWN",
            "label": "unknown",
            "median_days": 0.0,
            "max_missing_periods": 0,
            "total_missing_periods": 0,
            "tolerance_periods": 0,
        }

    delta_days = diffs / pd.Timedelta(days=1)
    delta_days = _normalise_delta_days(delta_days)
    if delta_days.empty:
        return {
            "canonical": "UNKNOWN",
            "code": "UNKNOWN",
            "label": "unknown",
            "median_days": 0.0,
            "max_missing_periods": 0,
            "total_missing_periods": 0,
            "tolerance_periods": 0,
        }
    median_days = float(delta_days.median())

    if median_days <= 0:
        raise MarketDataValidationError(
            "Unable to infer frequency because date offsets are zero or negative.",
        )

    if median_days <= 2.5:
        code = "D"
        canonical = "D"
        label = _HUMAN_FREQUENCY_LABELS.get(canonical, "daily")
        tolerance_default = 3
        base_days = 1.0
    elif median_days <= 10.0:
        code = "W"
        canonical = "W"
        label = _HUMAN_FREQUENCY_LABELS.get(canonical, "weekly")
        tolerance_default = 1
        base_days = 7.0
    elif median_days <= 45.0:
        code = "M"
        canonical = "M"
        label = _HUMAN_FREQUENCY_LABELS.get(canonical, "monthly")
        tolerance_default = 1
        base_days = 30.0
    elif median_days <= 120.0:
        code = "Q"
        canonical = "Q"
        label = _HUMAN_FREQUENCY_LABELS.get(canonical, "quarterly")
        tolerance_default = 1
        base_days = 91.0
    elif median_days <= 500.0:
        code = "Y"
        canonical = "Y"
        label = _HUMAN_FREQUENCY_LABELS.get(canonical, "annual")
        tolerance_default = 0
        base_days = 365.0
    else:
        raise MarketDataValidationError(
            "Unable to infer frequency. Data spacing appears longer than annual.",
        )

    tolerance_limit = tolerance_default
    if max_gap_limit is not None:
        tolerance_limit = max(tolerance_default, max_gap_limit)

    raw_ratio = delta_days / base_days
    nearest = raw_ratio.round().clip(lower=1)

    deviation = (raw_ratio - nearest).abs()
    irregular_mask = (nearest == 1) & (deviation > 0.34)
    if irregular_mask.any():
        samples = delta_days[irregular_mask].sort_values()
        preview = ", ".join(f"{float(value):.1f}d" for value in samples.iloc[:3])
        if len(samples) > 3:
            preview += " …"
        issues = [
            "Detected irregular sampling intervals that do not align with the "
            f"identified {label} cadence (example gaps: {preview})."
        ]
        raise MarketDataValidationError(_format_issues(issues), issues)

    nearest_int = nearest.astype(int)
    missing_periods = (nearest_int - 1).clip(lower=0)
    max_missing_periods = int(missing_periods.max() or 0)
    total_missing_periods = int(missing_periods.sum())

    if max_missing_periods > tolerance_limit:
        raise MarketDataValidationError(
            "Detected gaps in the date index that exceed the configured tolerance.",
            issues=[
                (
                    f"Largest gap spans {max_missing_periods} {label} periods "
                    f"(allowed <= {tolerance_limit})."
                )
            ],
        )

    return {
        "canonical": canonical,
        "code": code,
        "label": label,
        "median_days": median_days,
        "max_missing_periods": max_missing_periods,
        "total_missing_periods": total_missing_periods,
        "tolerance_periods": tolerance_limit,
    }


def _resolve_datetime_index(
    df: pd.DataFrame, *, source: str | None, auto_fix_dates: bool = True
) -> pd.DataFrame:
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
            issues = [
                "Missing a 'Date' column or datetime index. "
                "Ensure the upload includes a timestamp column named 'Date'."
            ]
            raise MarketDataValidationError(_format_issues(issues), issues)

        # All ingest surfaces use the date-correction engine.  Validation keeps
        # the long-standing fail-soft behavior by dropping values that the
        # shared engine identifies as unfixable, while fixes and empty-row
        # handling come from the same analysis used by UI ingest.
        if auto_fix_dates:
            # The shared correction engine reports positional row numbers.
            # This ingest path consumes the Date column and replaces the index,
            # so normalizing the temporary index preserves that contract for
            # frames supplied with arbitrary labels.
            working = working.reset_index(drop=True)
            correction_result = analyze_date_column(working, str(date_col))
            drop_rows = (
                correction_result.trailing_empty_rows
                + correction_result.droppable_empty_rows
                + [row for row, _value in correction_result.unfixable]
            )
            if correction_result.corrections or drop_rows:
                working = apply_date_corrections(
                    working,
                    str(date_col),
                    correction_result.corrections,
                    drop_rows=drop_rows,
                )
            for correction in correction_result.corrections:
                logger.info(
                    "Auto-corrected invalid date at row %d: %r → %r",
                    correction.row_index + 1,
                    correction.original_value,
                    correction.corrected_value,
                )
            for row, value in correction_result.unfixable:
                logger.warning("Dropped row %d with unfixable date: %r", row + 1, value)
            for row in (
                correction_result.trailing_empty_rows + correction_result.droppable_empty_rows
            ):
                logger.warning("Dropped row %d with unfixable date: %r", row + 1, "empty date")

        try:
            parsed = pd.to_datetime(working[date_col], errors="coerce")
        except (TypeError, ValueError) as exc:
            sample_values = working[date_col].astype(str).tolist()
            preview = ", ".join(sample_values[:5])
            if len(sample_values) > 5:
                preview += " …"
            issues = [f"Found dates that could not be parsed. Examples: {preview or 'n/a'}."]
            raise MarketDataValidationError(_format_issues(issues), issues) from exc
        if parsed.isna().any():
            bad_values = working.loc[parsed.isna(), date_col].astype(str).tolist()
            preview = ", ".join(bad_values[:5])
            if len(bad_values) > 5:
                preview += " …"
            issues = [f"Found dates that could not be parsed. Examples: {preview}."]
            raise MarketDataValidationError(_format_issues(issues), issues)
        idx = pd.DatetimeIndex(parsed, name="Date")
        working = working.drop(columns=[date_col])

    if working.empty:
        issues = ["No data columns detected after extracting the Date index."]
        raise MarketDataValidationError(_format_issues(issues), issues)

    duplicated = working.columns[working.columns.duplicated()].unique()
    if len(duplicated) > 0:
        preview = ", ".join(str(col) for col in duplicated[:5])
        if len(duplicated) > 5:  # pragma: no cover - defensive guard
            preview += " …"
        issues = [
            "Detected duplicate column names after removing the Date column: "
            + preview
            + ". Each column must be uniquely labelled."
        ]
        raise MarketDataValidationError(_format_issues(issues), issues)

    idx = idx.tz_localize(None)
    working.index = idx
    working.index.name = "Date"

    # Auto-sort by date if not already in ascending order
    if not working.index.is_monotonic_increasing:
        logger.warning("Data not in ascending date order; auto-sorting by date index.")
        working = working.sort_index()

    return working


def _check_monotonic_index(index: pd.DatetimeIndex) -> list[str]:
    issues: list[str] = []
    if not index.is_monotonic_increasing:
        # Identify the first offending timestamp for actionable feedback
        sorted_index = index.sort_values()
        for original, ordered in zip(index, sorted_index, strict=True):
            if original != ordered:
                issues.append(
                    "Date index must be sorted in ascending order. "
                    f"First out-of-order timestamp: {original.strftime('%Y-%m-%d')}"
                )
                break
    duplicates = index[index.duplicated()].unique()
    if len(duplicates) > 0:
        preview = ", ".join(ts.strftime("%Y-%m-%d") for ts in duplicates[:5])
        if len(duplicates) > 5:
            preview += " …"
        issues.append(f"Duplicate timestamps detected: {preview}")
    return issues


def _infer_frequency(
    index: pd.DatetimeIndex,
    *,
    max_gap_limit: int | None = None,
) -> tuple[str, str, dict[str, Any]]:
    info = classify_frequency(index, max_gap_limit=max_gap_limit)
    return info["canonical"], info["label"], info


def _strip_percent(series: pd.Series) -> tuple[pd.Series, bool]:
    """Strip trailing '%' from string values and divide by 100 if needed.

    Returns
    -------
    Tuple[pd.Series, bool]
        The converted series and whether any '%' signs were stripped.
    """
    if not hasattr(series, "str"):
        return series, False

    str_series = series.astype(str)
    has_percent = str_series.str.endswith("%")
    if not has_percent.any():
        return series, False

    # Strip the '%' and convert to numeric, then divide by 100
    stripped = str_series.str.rstrip("%")
    numeric = pd.to_numeric(stripped, errors="coerce")

    # Only divide by 100 for values that originally had '%'
    result = numeric.where(~has_percent, numeric / 100)
    return result, True


def _coerce_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    numeric = pd.DataFrame(index=df.index)
    issues: list[str] = []

    for column in df.columns:
        series = df[column]
        # First, try stripping percentage signs
        stripped, had_percent = _strip_percent(series)
        if had_percent:
            coerced = stripped
        else:
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
    bounded_unit = max_abs <= 1

    returns_like = (
        bounded_unit
        or (median_abs <= 0.5 and max_abs <= 5)
        or (neg_fraction >= 0.05 and max_abs <= 10)
    )

    price_like = values.min() >= 0 and (median_abs >= 10 or max_abs >= 20)

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
        issues = [
            "Unable to determine whether the data are prices or returns. "
            "Ensure numeric columns contain representative values."
        ]
        raise MarketDataValidationError(_format_issues(issues), issues)

    unique_modes = set(modes)
    if len(unique_modes) > 1:
        issues = [
            "Detected a mix of returns-like and price-like columns. "
            "Uploads must use a single representation."
        ]
        raise MarketDataValidationError(_format_issues(issues), issues)

    mode = modes[0]
    if ambiguous:
        preview = ", ".join(ambiguous[:5])
        issues = ["Could not classify columns as price or return series: " + preview]
        raise MarketDataValidationError(_format_issues(issues), issues)

    return mode


def validate_market_data(
    data: pd.DataFrame,
    *,
    source: str | None = None,
    missing_policy: str | Mapping[str, str] | None = None,
    missing_limit: int | Mapping[str, int | None] | None = None,
    auto_fix_dates: bool = True,
) -> ValidatedMarketData:
    """Validate market data according to the ingest contract."""

    frame = _resolve_datetime_index(data, source=source, auto_fix_dates=auto_fix_dates)
    issues = _check_monotonic_index(frame.index)
    if issues:
        raise MarketDataValidationError(_format_issues(issues), issues)

    numeric_frame, numeric_issues = _coerce_numeric(frame)
    if numeric_issues:
        raise MarketDataValidationError(_format_issues(numeric_issues), numeric_issues)

    try:
        policy_frame, policy_result = _apply_missing_policy(
            numeric_frame, missing_policy, limit=missing_limit
        )
    except ValueError as exc:
        if str(exc).startswith("Unsupported missing-data policy"):
            raise ValueError(str(exc).replace("Unsupported", "Unknown", 1)) from exc
        raise

    if policy_frame.empty:
        dropped = list(policy_result.dropped_assets)
        detail = f" (dropped columns: {', '.join(dropped)})" if dropped else ""
        issues = [
            "Missing-data policy removed every column. "
            "Adjust the policy or limits to retain at least one series." + detail
        ]
        raise MarketDataValidationError(_format_issues(issues), issues)

    limit_candidates = [value for value in policy_result.limit.values() if value is not None]
    max_gap_limit = max(limit_candidates) if limit_candidates else None

    frequency, label, frequency_info = _infer_frequency(
        policy_frame.index, max_gap_limit=max_gap_limit
    )
    mode = _infer_mode(policy_frame)

    metadata = MarketDataMetadata(
        mode=mode,
        frequency=frequency,
        frequency_detected=frequency_info.get("code", ""),
        frequency_label=label,
        frequency_median_spacing_days=frequency_info.get("median_days", 0.0),
        frequency_missing_periods=frequency_info.get("total_missing_periods", 0),
        frequency_max_gap_periods=frequency_info.get("max_missing_periods", 0),
        frequency_tolerance_periods=frequency_info.get("tolerance_periods", 0),
        start=numeric_frame.index.min().to_pydatetime(),
        end=numeric_frame.index.max().to_pydatetime(),
        rows=len(policy_frame),
        columns=list(policy_frame.columns),
        symbols=list(policy_frame.columns),
        missing_policy=policy_result.default_policy,
        missing_policy_limit=policy_result.default_limit,
        # Record only per-column policies that differ from the default so the
        # metadata reflects explicit user overrides (instead of a full expansion).
        missing_policy_overrides={
            column: value
            for column, value in policy_result.policy.items()
            if value != policy_result.default_policy
        },
        missing_policy_limits=policy_result.limit,
        missing_policy_filled=_missing_policy_fill_details(policy_result),
        missing_policy_dropped=list(policy_result.dropped_assets),
        missing_policy_summary=_summarise_missing_policy(policy_result),
    )

    validated = policy_frame.sort_index()
    attach_metadata(validated, metadata)

    return ValidatedMarketData(frame=validated, metadata=metadata)


def load_market_data_csv(path: str) -> ValidatedMarketData:
    """Load a CSV file and validate its contents."""

    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        issues = [f"File not found: {path}"]
        raise MarketDataValidationError(_format_issues(issues), issues) from exc
    except PermissionError as exc:  # pragma: no cover - defensive guard
        issues = [f"Permission denied when reading: {path}"]
        raise MarketDataValidationError(_format_issues(issues), issues) from exc
    except pd.errors.EmptyDataError as exc:
        issues = [f"File contains no data: {path}"]
        raise MarketDataValidationError(_format_issues(issues), issues) from exc
    except pd.errors.ParserError as exc:
        issues = [f"Failed to parse file '{path}'"]
        raise MarketDataValidationError(_format_issues(issues), issues) from exc

    return validate_market_data(frame, source=path)


def load_market_data_parquet(path: str) -> ValidatedMarketData:
    """Load a Parquet file and validate its contents."""

    try:
        frame = pd.read_parquet(path)
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        issues = [f"File not found: {path}"]
        raise MarketDataValidationError(_format_issues(issues), issues) from exc
    except PermissionError as exc:  # pragma: no cover - defensive guard
        issues = [f"Permission denied when reading: {path}"]
        raise MarketDataValidationError(_format_issues(issues), issues) from exc

    return validate_market_data(frame, source=path)


def attach_metadata(frame: pd.DataFrame, metadata: MarketDataMetadata) -> pd.DataFrame:
    """Attach metadata to a DataFrame in-place and return it."""

    payload = frame.attrs.setdefault("market_data", {})
    payload.update(
        {
            "metadata": metadata.model_dump(mode="json"),
            "mode": "returns" if metadata.mode == MarketDataMode.RETURNS else "prices",
            "mode_enum": metadata.mode,
            "frequency": metadata.frequency_label,
            "frequency_code": metadata.frequency,
            "frequency_detected": metadata.frequency_detected,
            "frequency_median_spacing_days": metadata.frequency_median_spacing_days,
            "frequency_missing_periods": metadata.frequency_missing_periods,
            "frequency_max_gap_periods": metadata.frequency_max_gap_periods,
            "frequency_tolerance_periods": metadata.frequency_tolerance_periods,
            "start": metadata.start.isoformat(),
            "end": metadata.end.isoformat(),
            "rows": metadata.rows,
            "columns": list(metadata.columns),
            "symbols": list(metadata.symbols),
            "missing_policy": metadata.missing_policy,
            "missing_policy_limit": metadata.missing_policy_limit,
            "missing_policy_overrides": dict(metadata.missing_policy_overrides),
            "missing_policy_limits": dict(metadata.missing_policy_limits),
            "missing_policy_filled": {
                column: (details.model_dump() if hasattr(details, "model_dump") else dict(details))
                for column, details in metadata.missing_policy_filled.items()
            },
            "missing_policy_dropped": list(metadata.missing_policy_dropped),
            "missing_policy_summary": metadata.missing_policy_summary,
        }
    )
    return frame
