from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from .._typing import FloatArray
from ..plugins import WeightEngine, weight_engine_registry

logger = logging.getLogger(__name__)


def _cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    cov_values = cov.to_numpy(dtype=float, copy=True)
    std: FloatArray = np.sqrt(np.diag(cov_values))

    # Check for zero standard deviations
    if np.any(std == 0):
        logger.warning("Zero standard deviations detected in correlation calculation")
        warnings.warn(
            "Zero standard deviations detected in correlation calculation",
            RuntimeWarning,
            stacklevel=2,
        )
        std = np.maximum(std, np.max(std) * 1e-8)

    # Construct the outer product as a DataFrame to preserve types for mypy
    denom = pd.DataFrame(np.outer(std, std), index=cov.index, columns=cov.columns)
    corr_df: pd.DataFrame = pd.DataFrame(cov_values, index=cov.index, columns=cov.columns) / denom
    corr_values = corr_df.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(corr_values, 1.0)
    corr_df = pd.DataFrame(corr_values, index=cov.index, columns=cov.columns)
    return corr_df


@weight_engine_registry.register("hrp")
class HierarchicalRiskParity(WeightEngine):
    """Hierarchical risk parity weighting with enhanced robustness."""

    def __init__(self) -> None:
        self.diagnostics: dict[str, object] = {}

    def weight(self, cov: pd.DataFrame) -> pd.Series:
        if cov.empty:
            return pd.Series(dtype=float)
        if not cov.index.equals(cov.columns):
            raise ValueError("Covariance matrix must be square with matching labels")

        # Check condition number
        condition_num = np.linalg.cond(cov.values)
        logger.debug(f"HRP input covariance condition number: {condition_num:.2e}")

        try:
            corr = _cov_to_corr(cov)

            # Check for invalid correlations
            corr_values = corr.to_numpy(dtype=float, copy=True)
            if np.any(~np.isfinite(corr_values)):
                logger.warning("Non-finite correlations detected in HRP calculation")
                warnings.warn(
                    "Non-finite correlations detected in HRP calculation",
                    RuntimeWarning,
                    stacklevel=2,
                )
                # Fall back to diagonal correlation matrix
                corr = pd.DataFrame(np.eye(len(cov)), index=cov.index, columns=cov.columns)
                corr_values = corr.to_numpy(dtype=float, copy=True)

            # Compute distance matrix as numpy array for typing clarity
            dist_arr: FloatArray = np.sqrt(0.5 * (1.0 - corr_values))

            # Ensure distance matrix is valid
            if np.any(~np.isfinite(dist_arr)) or np.any(dist_arr < 0):
                logger.warning("Invalid distance matrix in HRP, using equal weights")
                n = len(cov)
                return pd.Series(np.ones(n) / n, index=cov.index)

            condensed: FloatArray = squareform(dist_arr, checks=False)
            link = linkage(condensed, method="single")
            sort_ix = corr.index[leaves_list(link)]
            cov_sorted = cov.loc[sort_ix, sort_ix]
            w = pd.Series(1.0, index=sort_ix)
            clusters = [list(cov_sorted.index)]

            while clusters:
                new_clusters: list[list[str]] = []
                for cluster in clusters:
                    if len(cluster) <= 1:
                        continue
                    split = len(cluster) // 2
                    left = cluster[:split]
                    right = cluster[split:]
                    cov_left = cov_sorted.loc[left, left]
                    cov_right = cov_sorted.loc[right, right]

                    # Robust computation of cluster variances
                    try:
                        inv_left = 1 / np.diag(cov_left)
                        inv_left /= inv_left.sum()
                        inv_right = 1 / np.diag(cov_right)
                        inv_right /= inv_right.sum()
                        var_left = inv_left @ cov_left.to_numpy(dtype=float, copy=True) @ inv_left
                        var_right = (
                            inv_right @ cov_right.to_numpy(dtype=float, copy=True) @ inv_right
                        )

                        # Avoid division by zero
                        total_var = var_left + var_right
                        if total_var == 0:
                            alpha = 0.5
                        else:
                            alpha = 1 - var_left / total_var
                    except (ZeroDivisionError, np.linalg.LinAlgError):
                        logger.warning(
                            "Numerical issues in HRP cluster allocation, using equal split"
                        )
                        alpha = 0.5

                    w[left] *= alpha
                    w[right] *= 1 - alpha
                    new_clusters.extend([left, right])
                clusters = new_clusters

            w = w.reindex(cov.index).fillna(0.0)

            # Final normalization and validation
            if w.sum() == 0:
                logger.warning("Zero sum weights in HRP, using equal weights")
                n = len(cov)
                return pd.Series(np.ones(n) / n, index=cov.index)

            w /= w.sum()
            logger.debug("Successfully computed HRP weights")
            return w

        except Exception as exc:
            # Keep the original exception attached to the log record.  The
            # equal-weight fallback remains safe, but callers and operators
            # can now distinguish a degraded HRP result from normal output.
            self.diagnostics = {
                "fallback_used": True,
                "fallback_reason": "hrp_computation_error",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
            logger.exception("HRP computation failed; falling back to equal weights")
            n = len(cov)
            return pd.Series(np.ones(n) / n, index=cov.index)
