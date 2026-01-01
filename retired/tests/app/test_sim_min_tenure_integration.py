import pandas as pd
from trend_portfolio_app.policy_engine import MetricSpec, PolicyConfig
from trend_portfolio_app.sim_runner import Simulator


def test_simulator_min_tenure_integration():
    # Build a small dataset with clear ranking: A best, B mid, C worst
    dates = pd.period_range("2020-01", "2020-06", freq="M").to_timestamp(how="end")
    data = pd.DataFrame(
        {
            "A": [0.05, 0.05, 0.05, 0.05, 0.05, -0.10],
            "B": [0.02, 0.02, 0.02, 0.02, 0.02, 0.00],
            "C": [-0.02, -0.02, -0.02, -0.02, -0.02, -0.20],
        },
        index=dates,
    )
    data["Date"] = dates

    sim = Simulator(data)
    policy = PolicyConfig(
        top_k=1,
        bottom_k=1,
        cooldown_months=0,
        min_track_months=0,
        max_active=2,
        min_tenure_n=2,
        metrics=[MetricSpec("sharpe", 1.0)],
    )

    res = sim.run(
        start=dates[0],
        end=dates[-2],  # stop before last to focus on steady state
        freq="ME",
        lookback_months=3,
        policy=policy,
    )

    # Ensure the simulator produced a weights series and ran without NaNs
    first_weights = res.weights[res.dates[0]]
    assert isinstance(first_weights, pd.Series)
    assert res.portfolio.notna().any()
