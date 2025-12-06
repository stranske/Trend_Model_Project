"""Market data validation helpers.

This module centralises the validation logic that backs every ingest
entry point (CSV, Parquet, and in-memory DataFrames).  The goal is to
enforce a single data contract so the application can provide
deterministic feedback to users regardless of how data is supplied.
"""

from __future__ import annotations

import calendar
import enum
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, Field, model_validator

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
_VALID_MISSING_POLICIES = {"drop", "ffill", "zero"}


# ---------------------------------------------------------------------------
# Date auto-correction helpers
# ---------------------------------------------------------------------------


def _fix_invalid_day(date_str: str) -> str | None:
    """Attempt to correct an invalid day-of-month in a date string.

    Common data entry errors include dates like 11/31/2017 (November only has
    30 days) or 9/31/2017 (September has 30 days). This function detects these
    patterns and corrects the day to the last valid day of the month.

    Returns the corrected date string, or None if the date cannot be fixed.
    """
    date_str = str(date_str).strip()

    # Try M/D/YYYY or MM/DD/YYYY format
    match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", date_str)
    if match:
        try:
            month, day, year = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
            if 1 <= month <= 12:
                max_day = calendar.monthrange(year, month)[1]
                if day > max_day:
                    return f"{month}/{max_day}/{year}"
        except (ValueError, IndexError):
            pass

    # Try YYYY-MM-DD format
    match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", date_str)
    if match:
        try:
            year, month, day = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
            if 1 <= month <= 12:
                max_day = calendar.monthrange(year, month)[1]
                if day > max_day:
                    return f"{year}-{month:02d}-{max_day}"
        except (ValueError, IndexError):
            # Invalid date format or out-of-range values; return None to indicate failure.
            pass

    return None


def _auto_fix_invalid_dates(
    df: pd.DataFrame, date_col: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Auto-correct or drop rows with invalid dates.

    Parameters
    ----------
    df:
        DataFrame with a date column.
    date_col:
        Name of the date column.

    Returns
    -------
    tuple[pd.DataFrame, list[dict]]
        Corrected DataFrame and list of corrections applied.
    """
    working = df.copy()
    raw_dates = working[date_col].astype(str)
    parsed = pd.to_datetime(raw_dates, errors="coerce")
    invalid_mask = parsed.isna()

    if not invalid_mask.any():
        return working, []

    corrections: list[dict[str, Any]] = []
    rows_to_drop: list[int] = []

    for idx in working.index[invalid_mask]:
        pos = working.index.get_loc(idx) if not isinstance(idx, int) else idx
        original_value = raw_dates.loc[idx]
        fixed = _fix_invalid_day(original_value)

        if fixed is not None:
            working.at[idx, date_col] = fixed
            corrections.append(
                {
                    "row": pos + 1,
                    "original": original_value,
                    "corrected": fixed,
                    "action": "fixed",
                }
            )
        else:
            rows_to_drop.append(idx)
            corrections.append(
                {
                    "row": pos + 1,
                    "original": original_value,
                    "corrected": None,
                    "action": "dropped",
                }
            )

    if rows_to_drop:
        working = working.drop(index=rows_to_drop)

    return working, corrections


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
    columns: List[str] = Field(default_factory=list)
    symbols: List[str] = Field(default_factory=list)
    missing_policy: str = Field(default=_DEFAULT_MISSING_POLICY)
    missing_policy_limit: Optional[int] = None
    missing_policy_overrides: Dict[str, str] = Field(default_factory=dict)
    missing_policy_limits: Dict[str, Optional[int]] = Field(default_factory=dict)
    missing_policy_filled: Dict[str, MissingPolicyFillDetails] = Field(
        default_factory=dict
    )
    missing_policy_dropped: List[str] = Field(default_factory=list)
    missing_policy_summary: Optional[str] = None

    @property
    def date_range(self) -> Tuple[str, str]:
        return self.start.strftime("%Y-%m-%d"), self.end.strftime("%Y-%m-%d")

    @model_validator(mode="after")
    def _sync_symbols(self) -> "MarketDataMetadata":
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


def _normalise_policy_value(value: str | None) -> str:
    policy = (value or _DEFAULT_MISSING_POLICY).strip().lower()
    if policy not in _VALID_MISSING_POLICIES:
        allowed = ", ".join(sorted(_VALID_MISSING_POLICIES))
        raise ValueError(
            f"Unknown missing-data policy '{value}'. Choose one of {allowed}."
        )
    return policy


def _coerce_limit_value(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        limit_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Missing-data limit must be an integer or null.") from exc
    if limit_int < 0:
        raise ValueError("Missing-data limit cannot be negative.")
    return limit_int


def _build_policy_maps(
    columns: Iterable[Any],
    policy: str | Mapping[str, str] | None,
    limit: int | Mapping[str, int | None] | None,
) -> tuple[Dict[str, str], str, Dict[str, Optional[int]], Optional[int]]:
    cols = [str(col) for col in columns]
    if isinstance(policy, Mapping):
        raw_policy = {str(k): v for k, v in policy.items()}
        default_policy = _normalise_policy_value(raw_policy.get("*"))
        policy_map = {
            col: _normalise_policy_value(raw_policy.get(col, default_policy))
            for col in cols
        }
    else:
        default_policy = _normalise_policy_value(policy)
        policy_map = {col: default_policy for col in cols}

    if isinstance(limit, Mapping):
        raw_limit = {str(k): v for k, v in limit.items()}
        default_limit = _coerce_limit_value(raw_limit.get("*"))
        limit_map = {
            col: _coerce_limit_value(raw_limit.get(col, default_limit)) for col in cols
        }
    else:
        default_limit = _coerce_limit_value(limit)
        limit_map = {col: default_limit for col in cols}

    return policy_map, default_policy, limit_map, default_limit


def _max_consecutive_nans(series: pd.Series) -> int:
    if series.isna().sum() == 0:
        return 0
    is_na = series.isna()
    groups = is_na.ne(is_na.shift()).cumsum()
    runs = (is_na.groupby(groups).cumcount() + 1) * is_na
    return int(runs.max() or 0)


def apply_missing_policy(
    frame: pd.DataFrame,
    policy: str | Mapping[str, str] | None,
    *,
    limit: int | Mapping[str, int | None] | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if frame.empty:
        return frame.copy(), {
            "policy": _DEFAULT_MISSING_POLICY,
            "policy_map": {},
            "limit": None,
            "limit_map": {},
            "filled": {},
            "dropped": [],
            "missing_counts": {},
            "max_consecutive_gaps": {},
        }

    policy_map, default_policy, limit_map, default_limit = _build_policy_maps(
        frame.columns, policy, limit
    )

    result = frame.copy()
    dropped: list[str] = []
    filled: dict[str, MissingPolicyFillDetails] = {}
    missing_counts: dict[str, int] = {}
    max_gaps: dict[str, int] = {}

    for column in frame.columns:
        col_policy = policy_map[column]
        col_limit = limit_map[column]
        series = result[column]
        na_mask = series.isna()
        missing_total = int(na_mask.sum())
        missing_counts[column] = missing_total
        max_gap = _max_consecutive_nans(series)
        max_gaps[column] = max_gap

        if missing_total == 0:
            continue

        if col_policy == "drop":
            dropped.append(column)
            continue

        limit_for_fill = col_limit if col_limit is not None else None

        if limit_for_fill is not None and max_gap > limit_for_fill:
            dropped.append(column)
            continue

        if col_policy == "ffill":
            filled_series = series.ffill(limit=limit_for_fill)
            # Handle leading NaNs that ffill cannot reach
            filled_series = filled_series.bfill(limit=limit_for_fill)
            if filled_series.isna().any():
                dropped.append(column)
                continue
            result[column] = filled_series
            filled[column] = MissingPolicyFillDetails(
                method="ffill", count=missing_total
            )
            continue

        if col_policy == "zero":
            result[column] = series.fillna(0.0)
            filled[column] = MissingPolicyFillDetails(
                method="zero", count=missing_total
            )
            continue

        raise ValueError(f"Unhandled missing-data policy '{col_policy}'.")

    if dropped:
        result = result.drop(columns=dropped, errors="ignore")

    summary = {
        "policy": default_policy,
        "policy_map": policy_map,
        "limit": default_limit,
        "limit_map": limit_map,
        "filled": filled,
        "dropped": dropped,
        "missing_counts": missing_counts,
        "max_consecutive_gaps": max_gaps,
    }

    return result, summary


def _summarise_missing_policy(info: Mapping[str, Any]) -> str:
    policy = info.get("policy", _DEFAULT_MISSING_POLICY)
    limit = info.get("limit")
    limit_text = f"limit={limit}" if limit is not None else "unlimited"

    overrides: Dict[str, str] = {}
    policy_map = cast(Mapping[str, str], info.get("policy_map", {}))
    default_policy = policy
    for column, value in policy_map.items():
        if value != default_policy:
            overrides[column] = value

    filled = cast(Mapping[str, Any], info.get("filled", {}))
    filled_chunks = []
    for column, details in filled.items():
        method: str
        count: int
        if isinstance(details, MissingPolicyFillDetails):
            method = details.method
            count = details.count
        elif isinstance(details, Mapping):
            raw_method = details.get("method", "fill")
            raw_count = details.get("count", 0)
            method = str(raw_method) if raw_method is not None else "fill"
            try:
                count = int(raw_count) if raw_count is not None else 0
            except (TypeError, ValueError):
                count = 0
        else:
            method = "fill"
            count = 0
        filled_chunks.append(f"{column} ({method}: {count})")

    dropped = list(info.get("dropped", []))

    parts = [f"policy={policy}", limit_text]
    if overrides:
        overrides_text = ", ".join(
            f"{col}:{val}" for col, val in sorted(overrides.items())
        )
        parts.append(f"overrides={overrides_text}")
    if filled_chunks:
        parts.append("filled=" + ", ".join(sorted(filled_chunks)))
    if dropped:
        parts.append("dropped=" + ", ".join(sorted(dropped)))
    return "; ".join(parts)


def _normalize_delta_days(delta_days: pd.Series) -> pd.Series:
    cleaned = delta_days.replace([np.inf, -np.inf], np.nan).dropna()
    return cleaned.astype(float)


def classify_frequency(
    index: pd.DatetimeIndex,
    *,
    max_gap_limit: Optional[int] = None,
) -> Dict[str, Any]:
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

        # Auto-fix invalid dates before parsing
        if auto_fix_dates:
            working, corrections = _auto_fix_invalid_dates(working, date_col)
            if corrections:
                for corr in corrections:
                    if corr["action"] == "fixed":
                        logger.info(
                            "Auto-corrected invalid date at row %d: %r → %r",
                            corr["row"],
                            corr["original"],
                            corr["corrected"],
                        )
                    else:
                        logger.warning(
                            "Dropped row %d with unfixable date: %r",
                            corr["row"],
                            corr["original"],
                        )

        try:
            parsed = pd.to_datetime(working[date_col], errors="coerce")
        except (TypeError, ValueError) as exc:
            sample_values = working[date_col].astype(str).tolist()
            preview = ", ".join(sample_values[:5])
            if len(sample_values) > 5:
                preview += " …"
            issues = [
                "Found dates that could not be parsed. "
                f"Examples: {preview or 'n/a'}."
            ]
            raise MarketDataValidationError(_format_issues(issues), issues) from exc
        if parsed.isna().any():
            bad_values = working.loc[parsed.isna(), date_col].astype(str).tolist()
            preview = ", ".join(bad_values[:5])
            if len(bad_values) > 5:
                preview += " …"
            issues = ["Found dates that could not be parsed. " f"Examples: {preview}."]
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
        logger.info("Data not in ascending date order; auto-sorting by date index.")
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
    max_gap_limit: Optional[int] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    info = classify_frequency(index, max_gap_limit=max_gap_limit)
    return info["canonical"], info["label"], info


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
) -> ValidatedMarketData:
    """Validate market data according to the ingest contract."""

    frame = _resolve_datetime_index(data, source=source)
    issues = _check_monotonic_index(frame.index)
    if issues:
        raise MarketDataValidationError(_format_issues(issues), issues)

    numeric_frame, numeric_issues = _coerce_numeric(frame)
    if numeric_issues:
        raise MarketDataValidationError(_format_issues(numeric_issues), numeric_issues)

    policy_frame, policy_info = apply_missing_policy(
        numeric_frame, missing_policy, limit=missing_limit
    )

    if policy_frame.empty:
        dropped = policy_info.get("dropped", [])
        detail = f" (dropped columns: {', '.join(dropped)})" if dropped else ""
        issues = [
            "Missing-data policy removed every column. "
            "Adjust the policy or limits to retain at least one series." + detail
        ]
        raise MarketDataValidationError(_format_issues(issues), issues)

    limit_candidates = [
        value
        for value in policy_info.get("limit_map", {}).values()
        if value is not None
    ]
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
        missing_policy=policy_info.get("policy", _DEFAULT_MISSING_POLICY),
        missing_policy_limit=policy_info.get("limit"),
        missing_policy_overrides={
            column: value
            for column, value in policy_info.get("policy_map", {}).items()
            if value != policy_info.get("policy", _DEFAULT_MISSING_POLICY)
        },
        missing_policy_limits=policy_info.get("limit_map", {}),
        missing_policy_filled=policy_info.get("filled", {}),
        missing_policy_dropped=list(policy_info.get("dropped", [])),
        missing_policy_summary=_summarise_missing_policy(policy_info),
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
            "metadata": metadata,
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
                column: (
                    details.model_dump()
                    if hasattr(details, "model_dump")
                    else dict(details)
                )
                for column, details in metadata.missing_policy_filled.items()
            },
            "missing_policy_dropped": list(metadata.missing_policy_dropped),
            "missing_policy_summary": metadata.missing_policy_summary,
        }
    )
    return frame
