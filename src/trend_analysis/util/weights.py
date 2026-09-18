"""Helpers for handling portfolio weight mappings."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def normalize_weights(
    weights: Mapping[str, float] | pd.Series | None,
    *,
    percent_tolerance: float = 1e-2,
    fraction_tolerance: float = 1e-6,
) -> dict[str, float]:
    """Return weights as fractions.

    Percent-like and fraction-like detection uses the absolute *net* total
    (``abs(sum(weights))``). Percent-like inputs (≈ 100) are divided by 100;
    fraction-like inputs (≈ 1) are returned unchanged. For any other non-zero
    total, weights are normalised by dividing each value by the gross absolute
    sum (``sum(abs(weights))``) so mixed-sign ambiguous inputs scale
    consistently.
    """
    if weights is None:
        return {}

    if isinstance(weights, pd.Series):
        series = weights.astype(float).copy()
    elif isinstance(weights, Mapping):
        series = pd.Series({str(k): float(v) for k, v in weights.items()}, dtype=float)
    else:
        return {}

    if series.empty:
        return {}

    series = series.fillna(0.0)
    total = float(series.sum())
    total_abs = abs(total)

    if total_abs and np.isclose(total_abs, 100.0, rtol=0.0, atol=percent_tolerance):
        series = series / 100.0
    elif total_abs and np.isclose(total_abs, 1.0, rtol=0.0, atol=fraction_tolerance):
        series = series
    elif total_abs:
        gross_abs = float(series.abs().sum())
        if gross_abs:
            series = series / gross_abs

    return {str(k): float(v) for k, v in series.items()}
