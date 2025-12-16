import numpy as np
import pandas as pd

from trend_analysis.constants import NUMERICAL_TOLERANCE_MEDIUM
from trend_analysis.multi_period.replacer import Rebalancer

CFG = {
    "multi_period": {
        "min_funds": 1,
        "max_funds": 3,
        "triggers": {"sigma1": {"sigma": 1, "periods": 2}},
        "weight_curve": {"anchors": [[0, 1.0], [100, 1.0]]},
    },
    "random_seed": 0,
}


def make_score(z_a, z_b, z_c):
    return pd.DataFrame(
        {
            "zscore": [z_a, z_b, z_c],
            "rank": [1, 2, 3],
        },
        index=["A", "B", "C"],
    )


def test_removal_after_consecutive_lows():
    reb = Rebalancer(CFG)
    prev = pd.Series({"A": 0.5, "B": 0.5})
    sf = make_score(0.0, -1.1, 0.0)
    w1 = reb.apply_triggers(prev, sf)
    assert set(w1.index) == {"A", "B"}
    sf = make_score(0.0, -1.2, 0.0)
    w2 = reb.apply_triggers(w1, sf)
    assert set(w2.index) == {"A"}


def test_addition_on_positive_zscore():
    cfg = CFG
    reb = Rebalancer(cfg)
    prev = pd.Series({"A": 1.0})
    sf = make_score(0.1, -0.5, 1.5)
    out = reb.apply_triggers(prev, sf)
    assert set(out.index) == {"A", "C"}


def test_weights_normalised():
    reb = Rebalancer(CFG)
    prev = pd.Series({"A": 0.5, "B": 0.5})
    sf = make_score(0.0, -0.5, 1.5)
    out = reb.apply_triggers(prev, sf)
    assert np.isclose(out.sum(), 1.0, atol=NUMERICAL_TOLERANCE_MEDIUM)


def test_root_level_exit_thresholds_respected():
    cfg = {
        "portfolio": {
            # Legacy placement (outside portfolio.threshold_hold)
            "z_exit_soft": -0.1,
            "soft_strikes": 1,
            "constraints": {"max_funds": 3},
        }
    }
    reb = Rebalancer(cfg)
    prev = pd.Series({"A": 1.0})
    sf = pd.DataFrame({"zscore": [-0.2], "rank": [1]}, index=["A"])
    out = reb.apply_triggers(prev, sf)
    assert out.empty
