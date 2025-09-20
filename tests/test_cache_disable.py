import numpy as np
import pandas as pd

from trend_analysis.core.rank_selection import (
    RiskStatsConfig,
    compute_metric_series_with_cache,
)
from trend_analysis.perf.cache import CovCache


def test_cache_disabled_path():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(25, 3)), columns=["A", "B", "C"])
    stats_cfg = RiskStatsConfig(periods_per_year=252, risk_free=0.0)
    cache = CovCache()
    # First call with cache disabled
    s1 = compute_metric_series_with_cache(
        df,
        "__COV_VAR__",
        stats_cfg,
        cov_cache=cache,
        window_start="2020-01",
        window_end="2020-12",
        enable_cache=False,
    )
    assert len(cache._store) == 0  # internal detail acceptable for targeted test
    # Second call with cache enabled
    s2 = compute_metric_series_with_cache(
        df,
        "__COV_VAR__",
        stats_cfg,
        cov_cache=cache,
        window_start="2020-01",
        window_end="2020-12",
        enable_cache=True,
    )
    assert len(cache._store) == 1
    pd.testing.assert_series_equal(s1, s2)
