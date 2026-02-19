from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..plugins import WeightEngine, weight_engine_registry

logger = logging.getLogger(__name__)


@weight_engine_registry.register("risk_parity")
@weight_engine_registry.register("vol_inverse")
class RiskParity(WeightEngine):
    """Simple inverse-volatility risk parity weighting with robustness
    checks."""

    def weight(self, cov: pd.DataFrame) -> pd.Series:
        if cov.empty:
            return pd.Series(dtype=float)
        if not cov.index.equals(cov.columns):
            raise ValueError("Covariance matrix must be square with matching labels")

        # Extract diagonal elements (variances)
        variances = np.diag(cov.values)

        # Check for non-positive variances
        if np.any(variances <= 0):
            logger.warning("Non-positive variances detected in covariance matrix")
            # Set minimum variance to prevent division by zero
            min_var = np.max(variances) * 1e-8
            variances = np.maximum(variances, min_var)

        std = np.sqrt(variances)
        inv = 1.0 / std

        # Check for numerical issues
        if np.any(~np.isfinite(inv)):
            logger.warning("Non-finite inverse volatilities detected, using equal weights")
            n = len(cov)
            return pd.Series(np.ones(n) / n, index=cov.index, dtype=float)

        w = inv / inv.sum()

        logger.debug(
            f"Computed risk parity weights with condition number: {np.linalg.cond(cov.values):.2e}"
        )

        return pd.Series(w, index=cov.index, dtype=float)
