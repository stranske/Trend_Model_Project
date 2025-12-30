"""Date correction helpers for handling common date parsing issues.

This module provides utilities to detect and correct invalid dates that follow
recognizable patterns but have invalid day values (e.g., November 31 → November 30).
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass
class DateCorrection:
    """A proposed correction for an invalid date."""

    row_index: int
    original_value: str
    corrected_value: str
    explanation: str


@dataclass
class DateCorrectionResult:
    """Result of analyzing a date column for correctable issues."""

    corrections: list[DateCorrection]
    unfixable: list[tuple[int, str]]  # (row_index, original_value)
    trailing_empty_rows: list[int]  # Row indices of trailing empty/NaN rows to drop
    droppable_empty_rows: list[int]  # Row indices of non-trailing empty/NaN rows

    @property
    def has_corrections(self) -> bool:
        return len(self.corrections) > 0

    @property
    def has_unfixable(self) -> bool:
        return len(self.unfixable) > 0

    @property
    def has_trailing_empty(self) -> bool:
        return len(self.trailing_empty_rows) > 0

    @property
    def has_droppable_empty(self) -> bool:
        return len(self.droppable_empty_rows) > 0

    @property
    def total_droppable_rows(self) -> int:
        return len(self.trailing_empty_rows) + len(self.droppable_empty_rows)

    @property
    def all_fixable(self) -> bool:
        """True if all issues can be resolved (corrections + row drops)."""
        has_any_fix = self.has_corrections or self.has_trailing_empty or self.has_droppable_empty
        return has_any_fix and not self.has_unfixable


# Regex patterns for common date formats
_DATE_PATTERNS = [
    # MM/DD/YYYY or M/D/YYYY
    (
        re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$"),
        lambda m: (int(m.group(3)), int(m.group(1)), int(m.group(2))),
        lambda y, m, d: f"{m:02d}/{d:02d}/{y}",
    ),
    # MM-DD-YYYY or M-D-YYYY
    (
        re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$"),
        lambda m: (int(m.group(3)), int(m.group(1)), int(m.group(2))),
        lambda y, m, d: f"{m:02d}-{d:02d}-{y}",
    ),
    # YYYY-MM-DD
    (
        re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"),
        lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3))),
        lambda y, m, d: f"{y}-{m:02d}-{d:02d}",
    ),
    # YYYY/MM/DD
    (
        re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$"),
        lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3))),
        lambda y, m, d: f"{y}/{m:02d}/{d:02d}",
    ),
    # DD/MM/YYYY (European) - only try if day > 12 to avoid ambiguity
    (
        re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$"),
        lambda m: (int(m.group(3)), int(m.group(2)), int(m.group(1))),
        lambda y, m, d: f"{d:02d}/{m:02d}/{y}",
    ),
]


def _get_last_day_of_month(year: int, month: int) -> int:
    """Return the last valid day for the given month and year."""
    if month < 1 or month > 12:
        return 0
    return calendar.monthrange(year, month)[1]


def _try_correct_date(value: str) -> tuple[str, str] | None:
    """Try to correct an invalid date string.

    Returns (corrected_value, explanation) if fixable, None otherwise.
    """
    value = str(value).strip()

    for pattern, extractor, formatter in _DATE_PATTERNS:
        match = pattern.match(value)
        if not match:
            continue

        try:
            year, month, day = extractor(match)
        except (ValueError, IndexError):
            continue

        # Check if month is valid
        if month < 1 or month > 12:
            continue

        # Check if year is reasonable (1900-2100)
        if year < 1900 or year > 2100:
            continue

        last_day = _get_last_day_of_month(year, month)

        # If day is invalid but close to a valid value, suggest correction
        if day > last_day and day <= last_day + 3:
            # Day is slightly over the max - suggest last day of month
            corrected = formatter(year, month, last_day)
            month_name = calendar.month_name[month]
            explanation = (
                f"{month_name} {year} has {last_day} days; " f"corrected {day} → {last_day}"
            )
            return corrected, explanation

        if day < 1:
            # Day is 0 or negative - suggest first day
            corrected = formatter(year, month, 1)
            explanation = f"Invalid day {day}; corrected to 1"
            return corrected, explanation

    return None


def _is_empty_or_nan(value: str) -> bool:
    """Check if a value represents an empty or NaN date."""
    if pd.isna(value):
        return True
    s = str(value).strip().lower()
    return s in ("", "nan", "nat", "none", "null", "n/a", "na", "-", "--")


def _find_trailing_empty_rows(
    df: pd.DataFrame,
    date_column: str,
    failed_indices: list[int],
) -> list[int]:
    """Identify trailing rows with empty/NaN dates that can be dropped.

    Returns indices of contiguous empty rows at the END of the dataset.
    """
    if not failed_indices or len(df) == 0:
        return []

    # Get the last row index in the DataFrame
    last_df_idx = len(df) - 1

    # Find contiguous trailing empty rows starting from the end
    trailing = []
    expected_idx = last_df_idx

    # Check from the end of the DataFrame backwards
    for idx in range(last_df_idx, -1, -1):
        if idx != expected_idx:
            # Gap in indices - stop looking
            break

        val = df[date_column].iloc[idx]
        if _is_empty_or_nan(val):
            trailing.append(idx)
            expected_idx = idx - 1
        else:
            # Found a non-empty value - stop
            break

    return list(reversed(trailing))


def analyze_date_column(
    df: pd.DataFrame,
    date_column: str,
) -> DateCorrectionResult:
    """Analyze a date column and identify correctable issues.

    Parameters
    ----------
    df
        DataFrame containing the date column.
    date_column
        Name of the column containing date values.

    Returns
    -------
    DateCorrectionResult
        Contains lists of corrections, unfixable issues, and droppable rows.
    """
    corrections: list[DateCorrection] = []
    unfixable: list[tuple[int, str]] = []
    droppable_empty: list[int] = []

    # First try to parse all dates
    raw_dates = df[date_column].astype(str)
    parsed = pd.to_datetime(raw_dates, errors="coerce")

    # Find rows that failed to parse
    failed_mask = parsed.isna()
    failed_indices = df.index[failed_mask].tolist()

    # Identify trailing empty rows that can be dropped
    trailing_empty = _find_trailing_empty_rows(df, date_column, failed_indices)
    trailing_set = set(trailing_empty)

    for idx in failed_indices:
        # Skip trailing empty rows - they'll be dropped
        if idx in trailing_set:
            continue

        original = str(df[date_column].iloc[idx])

        # Check if it's an empty/NaN value (not at the end)
        if _is_empty_or_nan(original):
            # Empty rows in the middle can be dropped (with user approval)
            droppable_empty.append(idx)
            continue

        result = _try_correct_date(original)

        if result is not None:
            corrected, explanation = result
            corrections.append(
                DateCorrection(
                    row_index=idx,
                    original_value=original,
                    corrected_value=corrected,
                    explanation=explanation,
                )
            )
        else:
            unfixable.append((idx, original))

    return DateCorrectionResult(
        corrections=corrections,
        unfixable=unfixable,
        trailing_empty_rows=trailing_empty,
        droppable_empty_rows=droppable_empty,
    )


def apply_date_corrections(
    df: pd.DataFrame,
    date_column: str,
    corrections: Sequence[DateCorrection],
    drop_rows: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Apply date corrections to a DataFrame.

    Parameters
    ----------
    df
        Original DataFrame.
    date_column
        Name of the column containing date values.
    corrections
        List of corrections to apply.
    drop_rows
        Row indices to drop (empty/NaN date rows).

    Returns
    -------
    pd.DataFrame
        New DataFrame with corrected dates and dropped rows.
    """
    df = df.copy()

    # Drop empty rows first
    if drop_rows:
        df = df.drop(index=list(drop_rows), errors="ignore")

    # Apply date corrections
    for correction in corrections:
        if correction.row_index in df.index:
            df.loc[correction.row_index, date_column] = correction.corrected_value

    return df


def format_corrections_for_display(
    corrections: Sequence[DateCorrection],
    trailing_rows: Sequence[int] | None = None,
    droppable_rows: Sequence[int] | None = None,
    max_display: int = 10,
) -> str:
    """Format corrections for user display.

    Parameters
    ----------
    corrections
        List of corrections to format.
    trailing_rows
        Indices of trailing rows that will be dropped.
    droppable_rows
        Indices of non-trailing empty rows that will be dropped.
    max_display
        Maximum number of corrections to show in detail.

    Returns
    -------
    str
        Formatted string for display.
    """
    lines = []

    # Show date corrections
    if corrections:
        shown = list(corrections)[:max_display]
        for c in shown:
            lines.append(
                f"• Row {c.row_index + 1}: `{c.original_value}` → `{c.corrected_value}` "
                f"({c.explanation})"
            )
        if len(corrections) > max_display:
            lines.append(f"• ... and {len(corrections) - max_display} more date corrections")

    # Show droppable empty rows (non-trailing)
    if droppable_rows:
        if len(droppable_rows) == 1:
            lines.append(f"• Row {droppable_rows[0] + 1}: Empty/NaN date (will be removed)")
        elif len(droppable_rows) <= 5:
            row_nums = ", ".join(str(r + 1) for r in droppable_rows)
            lines.append(
                f"• Rows {row_nums}: {len(droppable_rows)} empty/NaN dates " "(will be removed)"
            )
        else:
            first_few = ", ".join(str(r + 1) for r in droppable_rows[:3])
            lines.append(
                f"• Rows {first_few}, ... : {len(droppable_rows)} empty/NaN dates "
                "(will be removed)"
            )

    # Show trailing row information
    if trailing_rows:
        if len(trailing_rows) == 1:
            lines.append(f"• Row {trailing_rows[0] + 1}: Empty date (will be removed)")
        else:
            first = trailing_rows[0] + 1
            last = trailing_rows[-1] + 1
            lines.append(
                f"• Rows {first}–{last}: {len(trailing_rows)} trailing empty rows "
                "(will be removed)"
            )

    if not lines:
        return "No corrections needed."

    return "\n".join(lines)
