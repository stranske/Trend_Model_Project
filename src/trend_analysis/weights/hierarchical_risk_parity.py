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
    std: FloatArray = np.sqrt(np.diag(cov))

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
    corr_df: pd.DataFrame = cov / denom
    np.fill_diagonal(corr_df.values, 1.0)
    return corr_df


def _equal_weight_series(index: pd.Index) -> pd.Series:
    n = len(index)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series(np.ones(n) / n, index=index)


def _warn_and_equal(index: pd.Index, message: str) -> pd.Series:
    logger.warning(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    return _equal_weight_series(index)


@weight_engine_registry.register("hrp")
class HierarchicalRiskParity(WeightEngine):
    """Hierarchical risk parity weighting with enhanced robustness."""

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

            if np.any(~np.isfinite(corr.values)):
                return _warn_and_equal(
                    cov.index,
                    "Non-finite correlations detected in HRP calculation; using equal weights",
                )

            # Reject correlations outside of [-1, 1] before constructing distances.
            if np.any(np.abs(corr.values) > 1.0 + 1e-9):
                return _warn_and_equal(
                    cov.index,
                    "Correlation coefficients outside [-1, 1] detected; using equal weights",
                )

            dist_arr: FloatArray = np.sqrt(
                np.clip(0.5 * (1.0 - corr.values), 0.0, None)
            )

            if np.any(~np.isfinite(dist_arr)):
                return _warn_and_equal(
                    cov.index,
                    "Invalid distance matrix in HRP; using equal weights",
                )

            try:
                condensed: FloatArray = squareform(dist_arr, checks=False)
            except Exception:
                return _warn_and_equal(
                    cov.index,
                    "Failed to convert correlation matrix to distances; using equal weights",
                )

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
                        var_left = inv_left @ cov_left.values @ inv_left
                        var_right = inv_right @ cov_right.values @ inv_right

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
                return _warn_and_equal(
                    cov.index, "Zero sum weights in HRP; using equal weights"
                )

            w /= w.sum()
            logger.debug("Successfully computed HRP weights")
            return w

        except Exception as e:
            logger.error(f"HRP computation failed: {e}, falling back to equal weights")
            return _warn_and_equal(
                cov.index, "HRP computation failed; using equal weights"
            )
