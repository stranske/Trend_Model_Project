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

    Rows are aligned by index, then any row with a missing manager return or
    factor value is dropped before fitting every manager on the same sample.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a pandas DataFrame")
    if len(returns.columns) == 0:
        raise ValueError("returns must contain at least one manager column")
    if len(factors.columns) == 0:
        raise ValueError("factors must contain at least one factor column")

    min_observations = len(factors.columns) + 2
    aligned_index = returns.index.intersection(factors.index)
    aligned_returns = returns.loc[aligned_index]
    aligned_factors = factors.loc[aligned_index]
    valid_rows = aligned_returns.notna().all(axis=1) & aligned_factors.notna().all(
        axis=1
    )
    clean_returns_all = aligned_returns.loc[valid_rows]
    clean_factors = aligned_factors.loc[valid_rows]

    rows: list[dict[str, float]] = []
    for manager in clean_returns_all.columns:
        clean_returns = clean_returns_all.loc[:, manager]
        if len(clean_factors) < min_observations:
            raise ValueError(
                f"insufficient observations after alignment for {manager}: "
                f"need at least {min_observations}, got {len(clean_factors)}"
            )

        factor_matrix = clean_factors.to_numpy(dtype=float)
        if add_intercept:
            design = np.column_stack(
                [np.ones(len(clean_factors), dtype=float), factor_matrix]
            )
        else:
            design = factor_matrix

        y = clean_returns.to_numpy(dtype=float)
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

    columns = list(aligned_factors.columns) + ["alpha", "r_squared"]
    return pd.DataFrame(rows, index=clean_returns_all.columns, columns=columns)


__all__ = ["factor_exposures"]
