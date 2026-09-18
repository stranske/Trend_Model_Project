"""Helpers for handling portfolio weight mappings."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _divide_by_gross_abs(series: pd.Series) -> pd.Series | None:
    """Divide series by sum(abs(weights)) using scaled arithmetic to avoid overflow."""
    values = series.to_numpy(dtype=float)
    abs_vals = np.abs(values)
    max_abs = float(np.max(abs_vals)) if len(abs_vals) else 0.0
    if max_abs == 0.0:
        return None
    scaled_abs_sum = float(np.sum(abs_vals / max_abs))
    if not np.isfinite(scaled_abs_sum) or scaled_abs_sum == 0.0:
        return None
    # series / gross_abs = (series / max_abs) / scaled_abs_sum — avoids overflow
    return series / max_abs / scaled_abs_sum


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
    consistently. Non-finite inputs are rejected and return an empty mapping.
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

    values = series.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        return {}

    series = series.fillna(0.0)
    total = float(series.sum())
    total_is_finite = np.isfinite(total)
    total_abs = abs(total) if total_is_finite else 0.0

    if (
        total_is_finite
        and total_abs
        and np.isclose(total_abs, 100.0, rtol=0.0, atol=percent_tolerance)
    ):
        series = series / 100.0
    elif (
        total_is_finite
        and total_abs
        and np.isclose(total_abs, 1.0, rtol=0.0, atol=fraction_tolerance)
    ):
        series = series
    elif total_is_finite and total_abs:
        normalized = _divide_by_gross_abs(series)
        if normalized is None:
            return {}
        series = normalized
    elif not total_is_finite:
        # Large finite inputs can overflow net sum while remaining valid per-value.
        normalized = _divide_by_gross_abs(series)
        if normalized is None:
            return {}
        series = normalized

    return {str(k): float(v) for k, v in series.items()}
