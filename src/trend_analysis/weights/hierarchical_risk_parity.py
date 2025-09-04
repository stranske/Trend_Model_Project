from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from ..plugins import WeightEngine, weight_engine_registry


def _cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    std: np.ndarray = np.sqrt(np.diag(cov))
    # Construct the outer product as a DataFrame to preserve types for mypy
    denom = pd.DataFrame(np.outer(std, std), index=cov.index, columns=cov.columns)
    corr_df: pd.DataFrame = cov / denom
    np.fill_diagonal(corr_df.values, 1.0)
    return corr_df


@weight_engine_registry.register("hrp")
class HierarchicalRiskParity(WeightEngine):
    """Hierarchical risk parity weighting."""

    def weight(self, cov: pd.DataFrame) -> pd.Series:
        if cov.empty:
            return pd.Series(dtype=float)
        if not cov.index.equals(cov.columns):
            raise ValueError("Covariance matrix must be square with matching labels")
        corr = _cov_to_corr(cov)
        # Compute distance matrix as numpy array for typing clarity
        dist_arr: np.ndarray = np.sqrt(0.5 * (1.0 - corr.values))
        condensed: np.ndarray = squareform(dist_arr, checks=False)
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
                inv_left = 1 / np.diag(cov_left)
                inv_left /= inv_left.sum()
                inv_right = 1 / np.diag(cov_right)
                inv_right /= inv_right.sum()
                var_left = inv_left @ cov_left.values @ inv_left
                var_right = inv_right @ cov_right.values @ inv_right
                alpha = 1 - var_left / (var_left + var_right)
                w[left] *= alpha
                w[right] *= 1 - alpha
                new_clusters.extend([left, right])
            clusters = new_clusters
        w = w.reindex(cov.index).fillna(0.0)
        w /= w.sum()
        return w
