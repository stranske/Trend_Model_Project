import numpy as np
import pandas as pd

from trend_analysis.perf.cache import CovCache, compute_cov_payload


def make_df(n=10, m=4, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, m)) * 0.01
    cols = [f"F{i}" for i in range(m)]
    return pd.DataFrame(data, columns=cols)


def test_covcache_get_or_compute_hits():
    df = make_df()
    cache = CovCache()
    key = cache.make_key("2020-01", "2020-12", df.columns)
    # First compute -> miss
    payload1 = cache.get_or_compute(key, lambda: compute_cov_payload(df))
    assert len(cache) == 1
    # Second compute -> hit (object identity preserved)
    payload2 = cache.get_or_compute(key, lambda: compute_cov_payload(df))
    assert payload1 is payload2
    assert payload1.cov.shape == (len(df.columns), len(df.columns))


def test_covcache_different_universe_keys():
    df = make_df()
    cache = CovCache()
    key1 = cache.make_key("2020-01", "2020-12", df.columns)
    key2 = cache.make_key("2020-01", "2020-12", list(df.columns)[1:])
    cache.get_or_compute(key1, lambda: compute_cov_payload(df))
    cache.get_or_compute(key2, lambda: compute_cov_payload(df.iloc[:, 1:]))
    assert len(cache) == 2
