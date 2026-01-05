from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..constants import NUMERICAL_TOLERANCE_HIGH
from ..plugins import WeightEngine, weight_engine_registry

logger = logging.getLogger(__name__)


@weight_engine_registry.register("erc")
class EqualRiskContribution(WeightEngine):
    """Equal risk contribution weighting via iterative scaling with robustness
    checks."""

    def __init__(self, *, max_iter: int = 1000, tol: float = 1e-8) -> None:
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def weight(self, cov: pd.DataFrame) -> pd.Series:
        if cov.empty:
            return pd.Series(dtype=float)
        if not cov.index.equals(cov.columns):
            raise ValueError("Covariance matrix must be square with matching labels")

        n = len(cov)
        cov_mat = cov.values

        # Check condition number
        condition_num = np.linalg.cond(cov_mat)
        logger.debug(f"ERC input covariance condition number: {condition_num:.2e}")

        if condition_num > 1e12:
            logger.warning(
                f"Ill-conditioned covariance matrix in ERC (condition: {condition_num:.2e}), adding regularization"
            )
            # Add small diagonal loading
            regularization = np.trace(cov_mat) / n * 1e-6
            cov_mat = cov_mat + regularization * np.eye(n)

        # Check for non-positive definite matrix
        try:
            eigenvals = np.linalg.eigvals(cov_mat)
            if np.any(eigenvals <= 0):
                logger.warning("Non-positive definite covariance matrix detected in ERC")
                # Apply more aggressive regularization
                min_eigenval = np.abs(np.min(eigenvals))
                regularization = min_eigenval + np.trace(cov_mat) / n * 1e-4
                cov_mat = cov_mat + regularization * np.eye(n)
        except np.linalg.LinAlgError:
            logger.error("Failed to compute eigenvalues in ERC, using equal weights")
            return pd.Series(np.ones(n) / n, index=cov.index)

        # Initialize weights
        w = np.repeat(1.0 / n, n)

        converged = False
        for iteration in range(self.max_iter):
            try:
                mrc = cov_mat @ w
                rc = w * mrc
                port_var = w @ mrc

                if port_var <= 0:
                    logger.warning("Non-positive portfolio variance in ERC, using equal weights")
                    return pd.Series(np.ones(n) / n, index=cov.index)

                target = port_var / n

                # Check convergence
                max_deviation = np.max(np.abs(rc - target))
                if max_deviation < self.tol:
                    converged = True
                    logger.debug(f"ERC converged after {iteration + 1} iterations")
                    break

                # Guard against division by zero in rc
                safe_rc = np.where(rc == 0, NUMERICAL_TOLERANCE_HIGH, rc)
                w *= target / safe_rc
                w = np.clip(w, 0, None)

                # Ensure weights sum to 1
                weight_sum = w.sum()
                if weight_sum == 0:
                    logger.warning("Zero sum weights in ERC iteration, resetting")
                    w = np.repeat(1.0 / n, n)
                else:
                    w /= weight_sum

            except (np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError) as e:
                logger.warning(f"Numerical error in ERC iteration {iteration}: {e}")
                # Reset to equal weights
                w = np.repeat(1.0 / n, n)
                break

        if not converged:
            logger.warning(f"ERC did not converge after {self.max_iter} iterations")

        return pd.Series(w, index=cov.index)
