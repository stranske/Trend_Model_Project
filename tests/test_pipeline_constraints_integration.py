import numpy as np
import pandas as pd

from trend_analysis.core.rank_selection import RiskStatsConfig
from trend_analysis.pipeline import _run_analysis


RUN_KWARGS = {"risk_free_column": "RF", "allow_risk_free_fallback": False}


def make_dummy_returns(n_months: int = 24) -> pd.DataFrame:
    dates = pd.date_range("2022-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(42)
    data = {
        "Date": dates,
        "FundA": rng.normal(0.01, 0.02, size=n_months),
        "FundB": rng.normal(0.008, 0.02, size=n_months),
        "FundC": rng.normal(0.011, 0.02, size=n_months),
        "FundD": rng.normal(0.009, 0.02, size=n_months),
        "RF": np.zeros(n_months),
    }
    return pd.DataFrame(data)


def test_pipeline_applies_cash_and_max_weight_constraints():
    df = make_dummy_returns()
    res = _run_analysis(
        df,
        in_start="2022-01",
        in_end="2022-12",
        out_start="2023-01",
        out_end="2023-12",
        target_vol=0.10,
        monthly_cost=0.0,
        selection_mode="all",
        stats_cfg=RiskStatsConfig(),
        constraints={
            "long_only": True,
            "max_weight": 0.4,
            "cash_weight": 0.1,
        },
        **RUN_KWARGS,
    )
    assert res is not None
    weights = res["fund_weights"]  # mapping fund->weight (excludes CASH synthetic line)
    assert isinstance(weights, dict)
    total = sum(weights.values())
    assert 0.99 <= total <= 1.01
    assert all(w <= 0.4 + 1e-9 for w in weights.values())
    # Ensure no CASH synthetic line leaked into fund weights
    assert "CASH" not in weights
