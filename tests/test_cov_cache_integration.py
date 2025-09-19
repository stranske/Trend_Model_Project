import numpy as np
import pandas as pd

from trend_analysis.core.rank_selection import (
    RiskStatsConfig, compute_metric_series_with_cache)
from trend_analysis.perf.cache import CovCache


def test_compute_metric_series_with_cache_cov_var_hits_cache():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.standard_normal((12, 5)) * 0.01, columns=list("ABCDE"))
    cfg = RiskStatsConfig(periods_per_year=12, risk_free=0.0)
    cache = CovCache()
    # First call -> miss
    s1 = compute_metric_series_with_cache(
        df,
        "__COV_VAR__",
        cfg,
        cov_cache=cache,
        window_start="2025-01",
        window_end="2025-12",
    )
    assert len(cache) == 1
    # Second call same window/universe -> hit (cache length unchanged, identical values)
    s2 = compute_metric_series_with_cache(
        df,
        "__COV_VAR__",
        cfg,
        cov_cache=cache,
        window_start="2025-01",
        window_end="2025-12",
    )
    assert len(cache) == 1
    assert s1.equals(s2)
