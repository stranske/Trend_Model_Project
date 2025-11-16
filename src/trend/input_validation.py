"""Shared CSV validation helpers for uploads and scripts.

The Streamlit application, CLI entry-points, and helper scripts previously
implemented their own lightweight guards around uploaded CSV files.  That led
to slightly different behaviour depending on which path loaded the data.  This
module provides a single ``validate_input`` function that ensures every caller
enforces the same baseline requirements before the heavier
``trend_analysis`` validators run.

Typical usage::

    from trend.input_validation import InputSchema, validate_input

    schema = InputSchema(date_column="Date", required_columns=("Date", "ret"))
    cleaned = validate_input(df, schema)

The function raises :class:`InputValidationError` with human friendly feedback
that includes the first offending row when an issue is detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "InputSchema",
    "InputValidationError",
    "validate_input",
]


class InputValidationError(ValueError):
    """Raised when a CSV upload fails structural validation."""

    def __init__(self, message: str, *, issues: Sequence[str] | None = None) -> None:
        formatted = message.strip()
        super().__init__(formatted)
        self.issues: list[str] = list(issues or [])
        self.user_message = formatted


@dataclass(frozen=True, slots=True)
class InputSchema:
    """Schema definition used by :func:`validate_input`.

    Parameters
    ----------
    date_column:
        Name of the timestamp column in the uploaded file.
    required_columns:
        Columns that must be present.  Each entry is compared case-insensitively
        against the CSV header.
    non_nullable:
        Columns that must not contain missing values.  When ``None`` the set is
        derived from ``required_columns`` and always includes ``date_column``.
    """

    date_column: str = "date"
    required_columns: tuple[str, ...] = ("date", "ticker", "ret")
    non_nullable: tuple[str, ...] | None = None


def _normalise(name: str) -> str:
    return str(name).strip().casefold()


def _column_lookup(columns: Iterable[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for col in columns:
        lookup[_normalise(col)] = str(col)
    return lookup


def _first_true_position(mask: np.ndarray) -> int:
    hits = np.flatnonzero(mask)
    if hits.size == 0:  # pragma: no cover - defensive guard
        return -1
    return int(hits[0])


def _row_context(df: pd.DataFrame, position: int, date_column: str) -> str:
    row = df.iloc[position]
    context: list[str] = []
    if date_column in row.index:
        timestamp = row[date_column]
    elif isinstance(df.index, pd.DatetimeIndex):
        timestamp = df.index[position]
    else:
        timestamp = None
    if isinstance(timestamp, pd.Timestamp):
        context.append(f"{date_column}={timestamp.isoformat()}")
    elif timestamp is not None:
        context.append(f"{date_column}={timestamp!r}")

    for column in df.columns:
        if column == date_column:
            continue
        value = row[column]
        if pd.isna(value):
            display = "NaN"
        else:
            display = repr(value)
        context.append(f"{column}={display}")
        if len(context) >= 3:
            break
    return f" ({', '.join(context)})" if context else ""


def _check_monotonic(parsed: pd.Series, date_column: str) -> None:
    if len(parsed) < 2:
        return
    prev = parsed.iloc[0]
    for idx in range(1, len(parsed)):
        current = parsed.iloc[idx]
        if current < prev:
            raise InputValidationError(
                "Date column must be sorted in ascending order. "
                f"Row {idx + 1} contains {current.isoformat()} after {prev.isoformat()}."
            )
        prev = current


def validate_input(
    df: pd.DataFrame,
    schema: InputSchema | None = None,
    *,
    set_index: bool = True,
    drop_date_column: bool = True,
) -> pd.DataFrame:
    """Validate a raw CSV DataFrame and normalise the timestamp column.

    Parameters
    ----------
    df:
        Raw DataFrame parsed directly from CSV/Parquet uploads.
    schema:
        :class:`InputSchema` describing the expected structure.
    set_index:
        Whether to replace the index with the parsed datetime column.
    drop_date_column:
        When ``set_index`` is ``True``, controls whether the original date column
        is removed from the DataFrame (defaults to ``True``).
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("validate_input expects a pandas DataFrame")
    if df.empty:
        raise InputValidationError("Input dataset is empty. Provide at least one row.")

    schema = schema or InputSchema()
    working = df.copy()
    lookup = _column_lookup(working.columns)

    date_key = _normalise(schema.date_column)
    date_column = lookup.get(date_key)
    if date_column is None:
        raise InputValidationError(f"Missing required column '{schema.date_column}'.")

    resolved_required: dict[str, str] = {}
    for required in schema.required_columns:
        actual = lookup.get(_normalise(required))
        if actual is None:
            raise InputValidationError(f"Missing required column '{required}'.")
        resolved_required[_normalise(required)] = actual
    resolved_required.setdefault(date_key, date_column)

    non_nullable = schema.non_nullable or schema.required_columns
    resolved_non_nullable: list[str] = []
    for column in non_nullable:
        actual = lookup.get(_normalise(column))
        if actual is None:
            raise InputValidationError(f"Missing required column '{column}'.")
        resolved_non_nullable.append(actual)
    if date_column not in resolved_non_nullable:
        resolved_non_nullable.append(date_column)

    raw_dates = working[date_column]
    parsed = pd.to_datetime(raw_dates, utc=True, errors="coerce")
    invalid_mask = parsed.isna()
    if invalid_mask.any():
        pos = _first_true_position(invalid_mask.to_numpy())
        bad_value = raw_dates.iloc[pos]
        raise InputValidationError(
            f"Unable to parse '{schema.date_column}' at row {pos + 1}: {bad_value!r}."
        )

    _check_monotonic(parsed, date_column)
    duplicates = parsed.duplicated()
    if duplicates.any():
        pos = _first_true_position(duplicates.to_numpy())
        timestamp = parsed.iloc[pos]
        raise InputValidationError(
            "Duplicate timestamps detected. "
            f"Row {pos + 1} repeats {timestamp.isoformat()}."
        )

    working[date_column] = parsed
    for column in resolved_non_nullable:
        mask = working[column].isna()
        if mask.any():
            pos = _first_true_position(mask.to_numpy())
            context = _row_context(working, pos, date_column)
            raise InputValidationError(
                f"Column '{column}' contains missing values at row {pos + 1}{context}."
            )

    if set_index:
        working = working.set_index(date_column, drop=drop_date_column)
        working.index.name = date_column

    return working

