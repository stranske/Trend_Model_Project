"""Date correction helpers for handling common date parsing issues."""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable, Sequence
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
    unfixable: list[tuple[int, str]]
    trailing_empty_rows: list[int]
    droppable_empty_rows: list[int]

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
        has_any_fix = (
            self.has_corrections or self.has_trailing_empty or self.has_droppable_empty
        )
        return has_any_fix and not self.has_unfixable


DateExtractor = Callable[[re.Match[str]], tuple[int, int, int]]
DateFormatter = Callable[[int, int, int], str]


_DATE_PATTERNS: list[tuple[re.Pattern[str], DateExtractor, DateFormatter]] = [
    (
        re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$"),
        lambda m: (int(m.group(3)), int(m.group(1)), int(m.group(2))),
        lambda y, m, d: f"{m:02d}/{d:02d}/{y}",
    ),
    (
        re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$"),
        lambda m: (int(m.group(3)), int(m.group(1)), int(m.group(2))),
        lambda y, m, d: f"{m:02d}-{d:02d}-{y}",
    ),
    (
        re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"),
        lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3))),
        lambda y, m, d: f"{y}-{m:02d}-{d:02d}",
    ),
    (
        re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$"),
        lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3))),
        lambda y, m, d: f"{y}/{m:02d}/{d:02d}",
    ),
    (
        re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$"),
        lambda m: (int(m.group(3)), int(m.group(2)), int(m.group(1))),
        lambda y, m, d: f"{d:02d}/{m:02d}/{y}",
    ),
]


def _get_last_day_of_month(year: int, month: int) -> int:
    if month < 1 or month > 12:
        return 0
    return calendar.monthrange(year, month)[1]


def _try_correct_date(value: str) -> tuple[str, str] | None:
    value = str(value).strip()

    for pattern, extractor, formatter in _DATE_PATTERNS:
        match = pattern.match(value)
        if not match:
            continue

        try:
            year, month, day = extractor(match)
        except (ValueError, IndexError):
            continue

        if month < 1 or month > 12:
            continue

        if year < 1900 or year > 2100:
            continue

        last_day = _get_last_day_of_month(year, month)

        if day > last_day and day <= last_day + 3:
            corrected = formatter(year, month, last_day)
            month_name = calendar.month_name[month]
            explanation = (
                f"{month_name} {year} has {last_day} days; corrected {day} → {last_day}"
            )
            return corrected, explanation

        if day < 1:
            corrected = formatter(year, month, 1)
            explanation = f"Invalid day {day}; corrected to 1"
            return corrected, explanation

    return None


def _is_empty_or_nan(value: str) -> bool:
    if pd.isna(value):
        return True
    s = str(value).strip().lower()
    return s in ("", "nan", "nat", "none", "null", "n/a", "na", "-", "--")


def _find_trailing_empty_rows(
    df: pd.DataFrame,
    date_column: str,
    failed_indices: list[int],
) -> list[int]:
    if not failed_indices or len(df) == 0:
        return []

    last_df_idx = len(df) - 1

    trailing = []
    expected_idx = last_df_idx

    for idx in range(last_df_idx, -1, -1):
        if idx != expected_idx:
            break

        val = df[date_column].iloc[idx]
        if _is_empty_or_nan(val):
            trailing.append(idx)
            expected_idx = idx - 1
        else:
            break

    return list(reversed(trailing))


def analyze_date_column(
    df: pd.DataFrame,
    date_column: str,
) -> DateCorrectionResult:
    corrections: list[DateCorrection] = []
    unfixable: list[tuple[int, str]] = []
    droppable_empty: list[int] = []

    raw_dates = df[date_column].astype(str)
    parsed = pd.to_datetime(raw_dates, errors="coerce")

    failed_mask = parsed.isna()
    failed_indices = df.index[failed_mask].tolist()

    trailing_empty = _find_trailing_empty_rows(df, date_column, failed_indices)
    trailing_set = set(trailing_empty)

    for idx in failed_indices:
        if idx in trailing_set:
            continue

        original = str(df[date_column].iloc[idx])

        if _is_empty_or_nan(original):
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
    df = df.copy()

    if drop_rows:
        df = df.drop(index=list(drop_rows), errors="ignore")

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
    lines = []

    if corrections:
        shown = list(corrections)[:max_display]
        for c in shown:
            lines.append(
                f"• Row {c.row_index + 1}: `{c.original_value}` → `{c.corrected_value}` "
                f"({c.explanation})"
            )
        if len(corrections) > max_display:
            lines.append(
                f"• ... and {len(corrections) - max_display} more date corrections"
            )

    if droppable_rows:
        if len(droppable_rows) == 1:
            lines.append(
                f"• Row {droppable_rows[0] + 1}: Empty/NaN date (will be removed)"
            )
        elif len(droppable_rows) <= 5:
            row_nums = ", ".join(str(r + 1) for r in droppable_rows)
            lines.append(
                f"• Rows {row_nums}: {len(droppable_rows)} empty/NaN dates (will be removed)"
            )
        else:
            first_few = ", ".join(str(r + 1) for r in droppable_rows[:3])
            lines.append(
                f"• Rows {first_few}, ... : {len(droppable_rows)} empty/NaN dates "
                "(will be removed)"
            )

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
