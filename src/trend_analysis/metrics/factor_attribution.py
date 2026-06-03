"""Returns-based factor attribution helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def factor_exposures(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    add_intercept: bool = True,
) -> pd.DataFrame:
    """Estimate per-manager factor exposures with ordinary least squares.

    Rows are aligned by index, then any row containing a missing manager return
    or factor value is dropped before fitting each manager column independently.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a pandas DataFrame")
    if returns.empty:
        raise ValueError("returns must contain at least one manager column")
    if factors.empty:
        raise ValueError("factors must contain at least one factor column")

    joined = returns.join(factors, how="inner", lsuffix="__return")
    aligned = joined.dropna(axis=0, how="any")
    clean_returns = aligned.loc[:, returns.columns]
    clean_factors = aligned.loc[:, factors.columns]

    min_observations = len(factors.columns) + 2
    if len(aligned) < min_observations:
        raise ValueError(
            "insufficient observations after alignment: "
            f"need at least {min_observations}, got {len(aligned)}"
        )

    factor_matrix = clean_factors.to_numpy(dtype=float)
    if add_intercept:
        design = np.column_stack([np.ones(len(clean_factors), dtype=float), factor_matrix])
    else:
        design = factor_matrix

    rows: list[dict[str, float]] = []
    for manager in clean_returns.columns:
        y = clean_returns[manager].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        if add_intercept:
            alpha = float(coefficients[0])
            betas = coefficients[1:]
        else:
            alpha = 0.0
            betas = coefficients

        fitted = design @ coefficients
        residual = y - fitted
        ss_residual = float(np.dot(residual, residual))
        centered = y - float(np.mean(y))
        ss_total = float(np.dot(centered, centered))
        if np.isclose(ss_total, 0.0):
            r_squared = 1.0 if np.isclose(ss_residual, 0.0) else 0.0
        else:
            r_squared = 1.0 - (ss_residual / ss_total)

        row = {factor: float(beta) for factor, beta in zip(clean_factors.columns, betas)}
        row["alpha"] = alpha
        row["r_squared"] = float(r_squared)
        rows.append(row)

    columns = list(clean_factors.columns) + ["alpha", "r_squared"]
    return pd.DataFrame(rows, index=clean_returns.columns, columns=columns)


__all__ = ["factor_exposures"]
