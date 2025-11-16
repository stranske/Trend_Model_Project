"""Light-weight CSV validation helpers for the Streamlit upload flow."""

from __future__ import annotations

import io
import logging
from typing import IO, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_SAMPLE_PREVIEW = """Date,Firm_A,Firm_B\n2020-01-31,0.012,-0.004\n2020-02-29,-0.003,0.011\n"""


class CSVValidationError(ValueError):
    """Raised when a user upload fails Streamlit-specific CSV validation."""

    def __init__(
        self,
        message: str,
        *,
        issues: Sequence[str] | None = None,
        sample_preview: str | None = None,
    ) -> None:
        formatted = message.strip()
        super().__init__(formatted)
        self.user_message = formatted
        self.issues = list(issues or [])
        self.sample_preview = sample_preview


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


def _normalise(name: str) -> str:
    return str(name).strip().casefold()


def validate_uploaded_csv(
    file: bytes | IO[bytes],
    required_columns: Sequence[str],
    max_rows: int,
) -> None:
    """Parse ``file`` and raise :class:`CSVValidationError` on structural issues."""

    try:
        buffer = _buffer_from_upload(file)
        try:
            if buffer.name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(buffer)
            else:
                df = pd.read_csv(buffer)
        except Exception as exc:  # pragma: no cover - pandas already tested
            raise CSVValidationError(
                "We could not read the uploaded file. Confirm it's a flat table with a "
                "header row and try again.",
                issues=[str(exc)],
                sample_preview=_SAMPLE_PREVIEW,
            ) from exc

        if df.empty:
            raise CSVValidationError(
                "The uploaded dataset is empty. Provide at least one row of returns.",
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
            raise CSVValidationError(
                "Unable to locate the date column. Ensure the first column is named 'Date'.",
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
            formatted = stamp.strftime("%Y-%m-%d") if not pd.isna(stamp) else "(invalid)"
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

