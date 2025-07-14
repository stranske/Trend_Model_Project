import pandas as pd
from trend_analysis.multi_period.replacer import Rebalancer


def test_rebalancer_noop():
    cfg = {"triggers": {"sigma1": {"sigma": 1, "periods": 2}}}
    rb = Rebalancer(cfg)
    weights = pd.Series([0.5, 0.5], index=["A", "B"])
    sf = pd.DataFrame({"A": [0.1], "B": [0.2]})
    out = rb.apply_triggers(weights, sf)
    pd.testing.assert_series_equal(out, weights)
