import numpy as np
import pandas as pd
from trend_portfolio_app.policy_engine import MetricSpec, PolicyConfig
from trend_portfolio_app.sim_runner import Simulator


def test_simulator_smoke():
    idx = pd.period_range(start="2019-01", end="2020-12", freq="M").to_timestamp(
        how="end"
    )
    df = pd.DataFrame(
        {
            "A": np.random.normal(0.01, 0.05, len(idx)),
            "B": np.random.normal(0.01, 0.05, len(idx)),
            "SPX": np.random.normal(0.005, 0.04, len(idx)),
        },
        index=idx,
    )
    df["Date"] = idx
    sim = Simulator(df, benchmark_col="SPX")
    policy = PolicyConfig(
        top_k=1, bottom_k=0, min_track_months=6, metrics=[MetricSpec("sharpe", 1.0)]
    )
    res = sim.run(
        start=idx[6], end=idx[-2], freq="ME", lookback_months=6, policy=policy
    )
    assert len(res.portfolio) > 0
