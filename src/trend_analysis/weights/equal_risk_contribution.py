from __future__ import annotations

import numpy as np
import pandas as pd

from ..plugins import WeightEngine, weight_engine_registry


@weight_engine_registry.register("erc")
class EqualRiskContribution(WeightEngine):
    """Equal risk contribution weighting via iterative scaling."""

    def __init__(self, *, max_iter: int = 1000, tol: float = 1e-8) -> None:
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def weight(self, cov: pd.DataFrame) -> pd.Series:
        if cov.empty:
            return pd.Series(dtype=float)
        if not cov.index.equals(cov.columns):
            raise ValueError("Covariance matrix must be square with matching labels")
        n = len(cov)
        w = np.repeat(1.0 / n, n)
        cov_mat = cov.values
        for _ in range(self.max_iter):
            mrc = cov_mat @ w
            rc = w * mrc
            port_var = w @ mrc
            target = port_var / n
            if np.max(np.abs(rc - target)) < self.tol:
                break
            w *= target / rc
            w = np.clip(w, 0, None)
            w /= w.sum()
        return pd.Series(w, index=cov.index)
