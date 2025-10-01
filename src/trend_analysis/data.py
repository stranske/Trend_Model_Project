import logging
import stat
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from .io.market_data import (
    MarketDataValidationError,
    ValidatedMarketData,
    validate_market_data,
)

logger = logging.getLogger(__name__)

ValidationErrorMode = Literal["raise", "log"]


def _finalise_validated_frame(
    validated: ValidatedMarketData, *, include_date_column: bool
) -> pd.DataFrame:
    base_frame = validated.frame
    if include_date_column:
        result = base_frame.reset_index()
    else:
        result = base_frame.copy()

    attrs = dict(base_frame.attrs)
    attrs.setdefault("market_data", {})
    attrs["market_data"]["metadata"] = validated.metadata
    attrs["market_data_mode"] = validated.metadata.mode.value
    attrs["market_data_frequency"] = validated.metadata.frequency
    attrs["market_data_frequency_label"] = validated.metadata.frequency_label
    attrs["market_data_columns"] = list(validated.metadata.columns)
    attrs["market_data_rows"] = validated.metadata.rows
    attrs["market_data_date_range"] = validated.metadata.date_range
    result.attrs = attrs
    return result


def _normalise_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.columns:
        if col == "Date":
            continue
        series = cleaned[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        coerced = series.astype(str).str.strip()
        has_percent = coerced.str.contains("%", na=False).any()
        coerced = coerced.str.replace(",", "", regex=False)
        coerced = coerced.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        coerced = coerced.str.replace("%", "", regex=False)
        numeric = pd.to_numeric(coerced, errors="coerce")
        if has_percent:
            numeric = numeric * 0.01
        if numeric.notna().any():
            cleaned[col] = numeric
    return cleaned


def _validate_payload(
    payload: pd.DataFrame,
    *,
    origin: str,
    errors: ValidationErrorMode,
    include_date_column: bool,
) -> Optional[pd.DataFrame]:
    payload = _normalise_numeric_strings(payload)
    try:
        validated = validate_market_data(payload, source=origin)
    except MarketDataValidationError as exc:
        if errors == "raise":
            raise
        msg_lower = exc.user_message.lower()
        message = exc.user_message
        if "could not be parsed" in msg_lower or "unable to parse" in msg_lower:
            message = f"{exc.user_message}\nUnable to parse Date values in {origin}"
        logger.error("Validation failed (%s): %s", origin, message)
        return None
    return _finalise_validated_frame(validated, include_date_column=include_date_column)


def _is_readable(mode: int) -> bool:
    """Check if a file mode indicates the file is readable.

    Parameters
    ----------
    mode : int
        File mode obtained from stat.st_mode

    Returns
    -------
    bool
        True if the file has read permissions for user, group, or others;
        False if no read permissions are available.
    """
    return (mode & (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)) != 0


def load_csv(
    path: str,
    *,
    errors: ValidationErrorMode = "log",
    include_date_column: bool = True,
    **_legacy_kwargs: object,
) -> Optional[pd.DataFrame]:
    """Load and validate a CSV expecting a ``Date`` column."""

    p = Path(path)
    try:
        if not p.exists():
            raise FileNotFoundError(path)
        if p.is_dir():
            raise IsADirectoryError(path)
        mode = p.stat().st_mode
        if not _is_readable(mode):
            if errors == "raise":
                raise PermissionError(f"Permission denied accessing file: {path}")
            logger.error(f"Permission denied accessing file: {path}")
            return None

        raw = pd.read_csv(str(p))
        return _validate_payload(
            raw,
            origin=str(p),
            errors=errors,
            include_date_column=include_date_column,
        )
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        if errors == "raise":
            raise
        logger.error(str(exc))
    except MarketDataValidationError as exc:
        if errors == "raise":
            raise
        msg_lower = exc.user_message.lower()
        message = exc.user_message
        if "could not be parsed" in msg_lower or "unable to parse" in msg_lower:
            message = f"{exc.user_message}\nUnable to parse Date values in {path}"
        logger.error("Validation failed (%s): %s", path, message)
    except Exception as exc:  # pragma: no cover - defensive guard
        if errors == "raise":
            raise
        logger.error("Unexpected error loading %s: %s", path, exc)
    return None


def load_parquet(
    path: str,
    *,
    errors: ValidationErrorMode = "log",
    include_date_column: bool = True,
    **_legacy_kwargs: object,
) -> Optional[pd.DataFrame]:
    """Load and validate a Parquet file containing market data."""

    p = Path(path)
    try:
        if not p.exists():
            raise FileNotFoundError(path)
        if p.is_dir():
            raise IsADirectoryError(path)
        mode = p.stat().st_mode
        if not _is_readable(mode):
            raise PermissionError(f"Permission denied accessing file: {path}")

        raw = pd.read_parquet(str(p))
        return _validate_payload(
            raw,
            origin=str(p),
            errors=errors,
            include_date_column=include_date_column,
        )
    except (
        FileNotFoundError,
        PermissionError,
        IsADirectoryError,
        pd.errors.EmptyDataError,
    ) as exc:
        if errors == "raise":
            raise
        logger.error(str(exc))
    except MarketDataValidationError as exc:
        if errors == "raise":
            raise
        msg_lower = exc.user_message.lower()
        message = exc.user_message
        if "could not be parsed" in msg_lower or "unable to parse" in msg_lower:
            message = f"{exc.user_message}\nUnable to parse Date values in {path}"
        logger.error("Validation failed (%s): %s", path, message)
    except Exception as exc:  # pragma: no cover - defensive guard
        if errors == "raise":
            raise
        logger.error("Unexpected error loading %s: %s", path, exc)
    return None


def validate_dataframe(
    df: pd.DataFrame,
    *,
    errors: ValidationErrorMode = "log",
    include_date_column: bool = True,
    origin: str = "dataframe",
) -> Optional[pd.DataFrame]:
    """Validate an in-memory DataFrame against the market data contract."""

    return _validate_payload(
        df,
        origin=origin,
        errors=errors,
        include_date_column=include_date_column,
    )


def identify_risk_free_fund(df: pd.DataFrame) -> Optional[str]:
    """Return the column with the lowest standard deviation.

    Columns named 'Date' or non-numeric dtypes are ignored. ``None`` is
    returned when no suitable columns are found.
    """

    num_cols = [c for c in df.select_dtypes("number").columns if c != "Date"]
    if not num_cols:
        return None
    rf = df[num_cols].std(skipna=True).idxmin()
    logger.info("Risk-free column: %s", rf)
    return str(rf)


def ensure_datetime(df: pd.DataFrame, column: str = "Date") -> pd.DataFrame:
    """Coerce ``column`` to datetime if needed.

    Treats malformed dates as validation errors rather than silently
    converting them to NaT values.
    """
    if column in df.columns and not is_datetime64_any_dtype(df[column]):
        try:
            df[column] = pd.to_datetime(df[column], format="%m/%d/%y")
        except Exception:
            # Try generic parsing, but detect malformed dates
            parsed_dates = pd.to_datetime(df[column], errors="coerce")
            if parsed_dates.isna().any():
                # Count malformed dates for better error reporting
                malformed_count = parsed_dates.isna().sum()
                malformed_mask = parsed_dates.isna()
                malformed_values = df.loc[malformed_mask, column].tolist()

                preview_vals = malformed_values[:5]
                preview_tail = "..." if len(malformed_values) > 5 else ""
                logger.error(
                    (
                        f"Found {malformed_count} malformed date(s) in column '{column}' "
                        f"that cannot be parsed: {preview_vals}{preview_tail}"
                    )
                )
                # Raise an exception to prevent malformed dates from being
                # processed as expired dates or other incorrect handling
                raise ValueError(
                    (
                        f"Malformed dates found in column '{column}'. "
                        "These should be treated as validation errors, not expiration failures."
                    )
                )
            df[column] = parsed_dates
    return df


__all__ = [
    "load_csv",
    "load_parquet",
    "validate_dataframe",
    "identify_risk_free_fund",
    "ensure_datetime",
]
