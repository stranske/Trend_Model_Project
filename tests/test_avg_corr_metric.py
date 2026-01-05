import numpy as np
import pandas as pd

from trend_analysis.core.rank_selection import (
    RiskStatsConfig,
    compute_metric_series_with_cache,
)
from trend_analysis.perf.cache import CovCache


def _rand_df(rows=40, cols=6, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(scale=0.01, size=(rows, cols))
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(cols)])


def test_avg_corr_metric_cache_equivalence():
    df = _rand_df()
    stats_cfg = RiskStatsConfig()
    # Without cache
    s_no = compute_metric_series_with_cache(df, "AvgCorr", stats_cfg, enable_cache=False)
    # With cache
    cache = CovCache()
    s_cache = compute_metric_series_with_cache(
        df, "AvgCorr", stats_cfg, cov_cache=cache, enable_cache=True
    )
    pd.testing.assert_series_equal(s_no, s_cache)
    # Basic sanity: values between -1 and 1
    assert (s_no.abs() <= 1 + 1e-12).all()
