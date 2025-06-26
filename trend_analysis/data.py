import logging
from typing import Optional
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

logger = logging.getLogger(__name__)


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
    try:
        df = pd.read_csv(path)
        if "Date" in df.columns:
            try:
                df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
            except ValueError:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                if df["Date"].isnull().any():
                    logger.warning(
                        "Could not parse all dates in %s using mm/dd/yy format",
                        path,
                    )
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
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
    """Coerce ``column`` to datetime if needed."""
    if column in df.columns and not is_datetime64_any_dtype(df[column]):
        try:
            df[column] = pd.to_datetime(df[column], format="%m/%d/%y")
        except Exception:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


__all__ = ["load_csv", "identify_risk_free_fund", "ensure_datetime"]
