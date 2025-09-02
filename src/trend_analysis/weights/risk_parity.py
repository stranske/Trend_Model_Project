from __future__ import annotations

import numpy as np
import pandas as pd

from ..plugins import WeightEngine, weight_engine_registry


@weight_engine_registry.register("risk_parity")
@weight_engine_registry.register("vol_inverse")
class RiskParity(WeightEngine):
    """Simple inverse-volatility risk parity weighting."""

    def weight(self, cov: pd.DataFrame) -> pd.Series:
        if cov.empty:
            return pd.Series(dtype=float)
        if not cov.index.equals(cov.columns):
            raise ValueError("Covariance matrix must be square with matching labels")
        std = np.sqrt(np.diag(cov))
        inv = 1.0 / std
        w = inv / inv.sum()
        return pd.Series(w, index=cov.index, dtype=float)
