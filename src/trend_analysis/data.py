import logging
import stat
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

logger = logging.getLogger(__name__)


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


def load_csv(path: str) -> Optional[pd.DataFrame]:
    """Load a CSV expecting a 'Date' column.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame or None
        The loaded DataFrame if successful, otherwise ``None``.
    """
    p = Path(path)
    try:
        if not p.exists():
            raise FileNotFoundError(path)
        if p.is_dir():
            raise IsADirectoryError(path)
        mode = p.stat().st_mode
        if not _is_readable(mode):
            logger.error(f"Permission denied accessing file: {path}")
            return None

        df = pd.read_csv(str(p))
        if "Date" in df.columns:
            try:
                df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
            except ValueError:
                # Try generic parsing, but detect malformed dates
                parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
                if parsed_dates.isnull().any():
                    # Distinguish between null/empty dates and malformed string dates
                    malformed_mask = parsed_dates.isnull()
                    malformed_values = df.loc[malformed_mask, "Date"].tolist()
                    
                    # Check if these are null/empty dates (empty strings, NaN) vs malformed strings
                    null_dates = [v for v in malformed_values if v == '' or pd.isna(v)]
                    malformed_strings = [v for v in malformed_values if v != '' and not pd.isna(v)]
                    
                    if malformed_strings:
                        # Strict handling: reject entire file for malformed string dates
                        malformed_count = len(malformed_strings)
                        logger.error(
                            f"Validation failed ({path}): {malformed_count} malformed date(s) that cannot be parsed: {malformed_strings[:5]}{'...' if len(malformed_strings) > 5 else ''}"
                        )
                        return None
                    elif null_dates:
                        # Graceful handling: filter out null/empty dates but continue
                        null_count = len(null_dates)
                        logger.warning(
                            f"Found {null_count} null/empty date(s): {null_dates[:5]}{'...' if len(null_dates) > 5 else ''}. Removing these rows from the dataset."
                        )
                        # Filter out rows with null dates
                        valid_mask = ~malformed_mask
                        df = df.loc[valid_mask].copy()
                        parsed_dates = parsed_dates.loc[valid_mask]
                        
                        # If no valid dates remain, then return None
                        if len(df) == 0:
                            logger.error(f"No valid date rows remaining in {path} after filtering null dates")
                            return None
                        
                df["Date"] = parsed_dates
        # Coerce non-Date columns to numeric when they look like strings
        # (e.g., "0.56%", "1,234", or parentheses for negatives).
        for col in df.columns:
            if col == "Date":
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                s = df[col].astype(str).str.strip()
                # Detect if this column contains percentage values
                has_percent = s.str.contains("%", na=False).any()
                # Normalize common formats: remove commas, convert (x) to -x, drop %
                s = s.str.replace(",", "", regex=False)
                s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
                s = s.str.replace("%", "", regex=False)
                s = pd.to_numeric(s, errors="coerce")
                if has_percent:
                    # Use pandas-aware operation to satisfy type checkers
                    s = getattr(s, "multiply")(0.01)
                # If conversion produced some numbers, adopt it
                if pd.api.types.is_numeric_dtype(s):
                    df[col] = s
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return None
    except PermissionError:
        logger.error(f"Permission denied accessing file: {path}")
        return None
    except IsADirectoryError:
        logger.error(f"Path is a directory, not a file: {path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"No data in file: {path}")
        return None
    except pd.errors.ParserError as exc:
        logger.error(f"Parsing error in {path}: {exc}")
        return None

    if "Date" not in df.columns:
        logger.error(f"Validation failed ({path}): missing 'Date' column")
        return None

    if df["Date"].isnull().any():
        logger.warning(f"Null values found in 'Date' column of {path}")

    return df


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

                logger.error(
                    f"Found {malformed_count} malformed date(s) in column '{column}' that cannot be parsed: {malformed_values[:5]}{'...' if len(malformed_values) > 5 else ''}"
                )
                # Raise an exception to prevent malformed dates from being
                # processed as expired dates or other incorrect handling
                raise ValueError(
                    f"Malformed dates found in column '{column}'. "
                    "These should be treated as validation errors, not expiration failures."
                )
            df[column] = parsed_dates
    return df


__all__ = ["load_csv", "identify_risk_free_fund", "ensure_datetime"]
