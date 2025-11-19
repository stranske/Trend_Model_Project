"""Utility helpers for enforcing execution-lag alignment."""

from __future__ import annotations

from typing import TypeVar

import pandas as pd

PandasLike = TypeVar("PandasLike", pd.Series, pd.DataFrame)


def shift_by_execution_lag(obj: PandasLike, lag: int = 1) -> PandasLike:
    """Return ``obj`` shifted forward by ``lag`` periods to honour execution lag.

    Parameters
    ----------
    obj:
        Pandas Series or DataFrame containing the signal/weight history.
    lag:
        Number of periods between signal computation and execution. Must be a
        positive integer.  A lag of ``1`` corresponds to the common
        "compute-at-close, trade-next-bar" convention.

    Returns
    -------
    Pandas Series or DataFrame matching the input type with all values shifted
    forward by ``lag`` rows.  The original ``attrs`` mapping is preserved so
    downstream consumers keep any metadata they rely on.
    """

    if lag < 0:
        raise ValueError("execution lag must be non-negative")
    if not isinstance(obj, (pd.Series, pd.DataFrame)):
        raise TypeError(
            "shift_by_execution_lag expects a pandas Series or DataFrame"
        )

    if lag == 0:
        return obj.copy()

    shifted = obj.shift(lag)
    if obj.attrs:
        shifted.attrs = dict(obj.attrs)
    shifted.attrs["execution_lag"] = lag
    return shifted


__all__ = ["shift_by_execution_lag"]
