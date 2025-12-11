"""Light-weight CSV validation helpers for the Streamlit upload flow."""

from __future__ import annotations

import io
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import IO, Sequence

import pandas as pd

from streamlit_app.components.data_schema import (
    apply_original_headers,
    extract_headers_from_bytes,
)
from streamlit_app.components.date_correction import (
    DateCorrection,
    analyze_date_column,
)

logger = logging.getLogger(__name__)

_SAMPLE_PREVIEW = (
    """Date,Firm_A,Firm_B\n2020-01-31,0.012,-0.004\n2020-02-29,-0.003,0.011\n"""
)


@dataclass
class DateCorrectionNeeded:
    """Raised when dates can be corrected with user approval."""

    corrections: list[DateCorrection]
    unfixable: list[tuple[int, str]]
    trailing_empty_rows: list[int]
    droppable_empty_rows: list[int]
    raw_data: bytes
    original_name: str
    date_column: str

    @property
    def all_rows_to_drop(self) -> list[int]:
        """All row indices that will be dropped."""
        return self.trailing_empty_rows + self.droppable_empty_rows


class CSVValidationError(ValueError):
    """Raised when a user upload fails Streamlit-specific CSV validation."""

    def __init__(
        self,
        message: str,
        *,
        issues: Sequence[str] | None = None,
        sample_preview: str | None = None,
        date_correction: DateCorrectionNeeded | None = None,
    ) -> None:
        formatted = message.strip()
        super().__init__(formatted)
        self.user_message = formatted
        self.issues = list(issues or [])
        self.sample_preview = sample_preview
        self.date_correction = date_correction


def _buffer_from_upload(file: bytes | IO[bytes]) -> io.BytesIO:
    if isinstance(file, bytes):
        data = file
        name = "upload.csv"
    elif hasattr(file, "read"):
        handle = file
        try:
            current = handle.tell()
        except Exception:
            current = None
        handle.seek(0)
        data = handle.read()
        if current is not None:
            handle.seek(current)
        name = getattr(handle, "name", "upload.csv")
    else:
        raise TypeError("file must be bytes or a binary file-like object")

    buffer = io.BytesIO(data)
    if isinstance(name, str) and name:
        buffer.name = name
    return buffer


_FORMULA_PREFIXES = ("=", "+", "-", "@")


def _strip_formula_prefix(name: str) -> str:
    text = str(name).strip()
    while text.startswith(_FORMULA_PREFIXES):
        text = text[1:]
    return text


def _safe_column_name(name: str) -> str:
    cleaned = str(name).strip()
    stripped = _strip_formula_prefix(cleaned)
    if stripped != cleaned:
        return stripped or "column"
    return cleaned or "column"


def _normalise(name: str) -> str:
    return _strip_formula_prefix(str(name)).casefold()


def validate_uploaded_csv(
    file: bytes | IO[bytes],
    required_columns: Sequence[str],
    max_rows: int,
) -> None:
    """Parse ``file`` and raise :class:`CSVValidationError` on structural issues."""

    try:
        buffer = _buffer_from_upload(file)
        raw = buffer.getvalue()
        is_excel = buffer.name.lower().endswith((".xlsx", ".xls"))
        headers = extract_headers_from_bytes(raw, is_excel=is_excel)
        data_buffer = io.BytesIO(raw)
        data_buffer.name = buffer.name
        try:
            if is_excel:
                df = pd.read_excel(data_buffer)
            else:
                df = pd.read_csv(data_buffer)
        except Exception as exc:  # pragma: no cover - pandas already tested
            raise CSVValidationError(
                "We could not read the uploaded file. Confirm it's a flat table with a "
                "header row and try again.",
                issues=[str(exc)],
                sample_preview=_SAMPLE_PREVIEW,
            ) from exc
        apply_original_headers(df, headers)

        original_headers = [re.sub(r"\.\d+$", "", str(col)) for col in df.columns]
        safe_columns = [_safe_column_name(col) for col in original_headers]
        duplicates = {col for col, count in Counter(safe_columns).items() if count > 1}
        if duplicates:
            dup_display = ", ".join(sorted(duplicates))
            raise CSVValidationError(
                "Column names must be unique.",
                issues=(
                    [f"Duplicate headers detected: {dup_display}"]
                    if dup_display
                    else None
                ),
                sample_preview=_SAMPLE_PREVIEW,
            )
        df = df.copy()
        df.columns = safe_columns

        if df.empty:
            raise CSVValidationError(
                "The uploaded dataset is empty. Provide at least one row of returns.",
                issues=["Detected 0 rows of data."],
                sample_preview=_SAMPLE_PREVIEW,
            )

        if max_rows > 0 and len(df.index) > max_rows:
            raise CSVValidationError(
                "The upload is too large for the browser session.",
                issues=[
                    f"Detected {len(df.index):,} rows; the current limit is {max_rows:,} rows."
                ],
                sample_preview=_SAMPLE_PREVIEW,
            )

        lookup = {_normalise(col): str(col) for col in df.columns}
        missing = [col for col in required_columns if _normalise(col) not in lookup]
        if missing:
            raise CSVValidationError(
                "The uploaded file is missing required columns.",
                issues=[f"Missing columns: {', '.join(missing)}"],
                sample_preview=_SAMPLE_PREVIEW,
            )

        date_key = _normalise(required_columns[0]) if required_columns else None
        date_column = lookup.get(date_key) if date_key is not None else None
        if date_column is None:
            required = required_columns[0] if required_columns else "Date"
            raise CSVValidationError(
                "Unable to locate the date column. Ensure the first column is named 'Date'.",
                issues=[f"Missing required column: {required}"],
                sample_preview=_SAMPLE_PREVIEW,
            )

        try:
            parsed = pd.to_datetime(df[date_column], errors="coerce")
        except Exception as exc:
            raise CSVValidationError(
                "Date column could not be parsed. Confirm it contains ISO formatted dates.",
                issues=[str(exc)],
                sample_preview=_SAMPLE_PREVIEW,
            ) from exc

        if parsed.isna().any():
            # Check if the invalid dates can be corrected
            correction_result = analyze_date_column(df, date_column)

            if correction_result.all_fixable:
                # All issues can be resolved (corrections + row drops)
                correction_info = DateCorrectionNeeded(
                    corrections=correction_result.corrections,
                    unfixable=correction_result.unfixable,
                    trailing_empty_rows=correction_result.trailing_empty_rows,
                    droppable_empty_rows=correction_result.droppable_empty_rows,
                    raw_data=raw,
                    original_name=buffer.name,
                    date_column=date_column,
                )

                # Build descriptive issue list
                issues = []
                if correction_result.corrections:
                    issues.append(
                        f"{len(correction_result.corrections)} date(s) need correction "
                        "(e.g., November 31 → November 30)."
                    )
                total_drops = correction_result.total_droppable_rows
                if total_drops > 0:
                    issues.append(
                        f"{total_drops} row(s) with empty/NaN dates will be removed."
                    )

                raise CSVValidationError(
                    "Some dates have issues that can be automatically corrected.",
                    issues=issues,
                    sample_preview=_SAMPLE_PREVIEW,
                    date_correction=correction_info,
                )
            elif (
                correction_result.has_corrections
                or correction_result.has_trailing_empty
                or correction_result.has_droppable_empty
            ):
                # Some can be fixed, some cannot
                correction_info = DateCorrectionNeeded(
                    corrections=correction_result.corrections,
                    unfixable=correction_result.unfixable,
                    trailing_empty_rows=correction_result.trailing_empty_rows,
                    droppable_empty_rows=correction_result.droppable_empty_rows,
                    raw_data=raw,
                    original_name=buffer.name,
                    date_column=date_column,
                )
                first_unfixable = correction_result.unfixable[0]
                fixable_count = len(correction_result.corrections) + len(
                    correction_result.trailing_empty_rows
                )
                raise CSVValidationError(
                    "Some dates have issues. Some can be corrected, but others cannot be parsed.",
                    issues=[
                        f"{fixable_count} issue(s) can be auto-fixed, "
                        f"but {len(correction_result.unfixable)} cannot be parsed.",
                        f"Row {first_unfixable[0] + 1} contains '{first_unfixable[1]}' "
                        "which cannot be interpreted as a date.",
                    ],
                    sample_preview=_SAMPLE_PREVIEW,
                    date_correction=correction_info,
                )
            else:
                # No corrections possible - original error
                first_bad = int(parsed[parsed.isna()].index[0])
                example = df[date_column].iloc[first_bad]
                raise CSVValidationError(
                    "Some rows have invalid dates.",
                    issues=[
                        f"Row {first_bad + 1} contains '{example}' which cannot be parsed as a date."
                    ],
                    sample_preview=_SAMPLE_PREVIEW,
                )

        duplicates = parsed.duplicated()
        if duplicates.any():
            idx = int(duplicates[duplicates].index[0])
            stamp = parsed.iloc[idx]
            formatted = (
                stamp.strftime("%Y-%m-%d") if not pd.isna(stamp) else "(invalid)"
            )
            raise CSVValidationError(
                "Dates must be unique.",
                issues=[f"Row {idx + 1} repeats the date {formatted}."],
                sample_preview=_SAMPLE_PREVIEW,
            )

        logger.debug(
            "CSV upload validated: %s rows × %s columns", len(df.index), len(df.columns)
        )
    except CSVValidationError as err:
        logger.exception("CSV upload failed validation: %s", err)
        raise
