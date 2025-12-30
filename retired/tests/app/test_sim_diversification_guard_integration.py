import pandas as pd
from trend_portfolio_app.policy_engine import MetricSpec, PolicyConfig
from trend_portfolio_app.sim_runner import Simulator


def test_simulator_diversification_guard_integration():
    # Two funds per bucket; ensure we don't exceed cap per review
    dates = pd.period_range("2020-01", "2020-05", freq="M").to_timestamp(how="end")
    data = pd.DataFrame(
        {
            "A1": [0.05, 0.05, 0.05, 0.05, 0.05],
            "A2": [0.04, 0.04, 0.04, 0.04, 0.04],
            "B1": [0.03, 0.03, 0.03, 0.03, 0.03],
            "B2": [0.02, 0.02, 0.02, 0.02, 0.02],
        },
        index=dates,
    )
    data["Date"] = dates
    sim = Simulator(data)
    policy = PolicyConfig(
        top_k=3,
        bottom_k=0,
        cooldown_months=0,
        min_track_months=0,
        max_active=10,
        diversification_max_per_bucket=1,
        diversification_buckets={"A1": "A", "A2": "A", "B1": "B", "B2": "B"},
        metrics=[MetricSpec("sharpe", 1.0)],
    )
    res = sim.run(
        start=dates[1],
        end=dates[-2],
        freq="ME",
        lookback_months=2,
        policy=policy,
    )
    # Ensure sim executed and weights reflect per-bucket cap
    first_w = res.weights[res.dates[0]]
    # Only one from A-bucket and one from B-bucket in first allocation
    buckets = {"A1": "A", "A2": "A", "B1": "B", "B2": "B"}
    seen = {}
    for name in first_w.index:
        b = buckets.get(name, name)
        seen[b] = seen.get(b, 0) + 1
    assert all(v <= 1 for v in seen.values())
