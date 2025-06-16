"""I/O helpers for the trend analysis project."""

from typing import Tuple, List


def pct(values: Tuple[float, float, float, float, float]) -> List[float]:
    """Convert select tuple values to percentages.

    Multiplies the first, second and fifth elements by 100 leaving the
    others unchanged.

    Parameters
    ----------
    values : Tuple[float, float, float, float, float]
        Input numeric values.

    Returns
    -------
    List[float]
        Values with percentage conversion applied.
    """

    return [values[0] * 100, values[1] * 100, values[2], values[3], values[4] * 100]
