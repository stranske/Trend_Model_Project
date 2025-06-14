import logging
from typing import Optional
import pandas as pd

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
        df = pd.read_csv(path, parse_dates=["Date"])
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"No data in file: {path}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error in {path}: {e}")
        return None
    except ValueError as e:
        # Raised when parse_dates references a missing column
        logger.error(f"Missing 'Date' column in {path}")
        return None

    if "Date" not in df.columns:
        logger.error(f"Validation failed ({path}): missing 'Date' column")
        return None

    if df["Date"].isnull().any():
        logger.warning(f"Null values found in 'Date' column of {path}")

    return df
