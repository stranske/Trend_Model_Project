from __future__ import annotations

import pandas as pd

from trend_analysis.constants import NUMERICAL_TOLERANCE_MEDIUM
from trend_analysis.multi_period.engine import run_schedule
from trend_analysis.multi_period.replacer import Rebalancer
from trend_analysis.selector import RankSelector
from trend_analysis.weighting import EqualWeight


def _mk_sf(z_by_name: dict[str, float]) -> pd.DataFrame:
    # Provide both a metric column and a zscore column; Rebalancer reads zscore
    df = pd.DataFrame({"Sharpe": z_by_name, "zscore": z_by_name}).astype(float)
    df.index.name = "name"
    return df


def test_rebalancer_thresholds_and_strikes():
    # Config overrides: make thresholds explicit
    cfg = {
        "portfolio": {
            "threshold_hold": {
                "z_exit_soft": -1.0,
                "soft_strikes": 2,
                "weighting": "equal",
            }
        }
    }

    # Two consecutive periods with same z-scores
    frames = {
        "2025-06-30": _mk_sf({"A": -1.2, "B": 0.0, "C": 1.2}),
        "2025-07-31": _mk_sf({"A": -1.2, "B": 0.1, "C": 1.1}),
    }

    selector = RankSelector(top_n=3, rank_column="Sharpe")
    weighting = EqualWeight()
    pf = run_schedule(
        frames,
        selector,
        weighting,
        rank_column="Sharpe",
        rebalancer=Rebalancer(cfg),
    )

    # After first period, A should still be present (1 strike), no drops yet
    w1 = pf.history[sorted(pf.history)[0]]
    assert set(w1.index) == {"A", "B", "C"}
    # After second period, A should be dropped (2 strikes), B and C remain
    w2 = pf.history[sorted(pf.history)[1]]
    assert set(w2.index) == {"B", "C"}
    # Equal weight for survivors
    assert (
        abs(w2.loc["B"] - 0.5) < NUMERICAL_TOLERANCE_MEDIUM
        and abs(w2.loc["C"] - 0.5) < NUMERICAL_TOLERANCE_MEDIUM
    )


def test_rebalancer_bayesian_weighting_option():
    # Use bayesian weighting on survivors
    cfg = {
        "portfolio": {
            "weighting": {"params": {"shrink_tau": 0.5}},
            "threshold_hold": {
                "z_exit_soft": -1.0,
                "soft_strikes": 2,
                "weighting": "score_prop_bayes",
            },
        }
    }

    frames = {
        "2025-06-30": _mk_sf({"A": -1.2, "B": 0.2, "C": 1.2}),
        "2025-07-31": _mk_sf({"A": -1.2, "B": 0.3, "C": 1.1}),
    }

    selector = RankSelector(top_n=3, rank_column="Sharpe")
    weighting = EqualWeight()
    pf = run_schedule(
        frames,
        selector,
        weighting,
        rank_column="Sharpe",
        rebalancer=Rebalancer(cfg),
    )

    # After second period, A should be dropped; B and C survive
    w2 = pf.history[sorted(pf.history)[1]]
    assert set(w2.index) == {"B", "C"}
    # Bayesian weighting should differ from equal when scores differ
    assert abs(w2.loc["B"] - w2.loc["C"]) > 1e-6
