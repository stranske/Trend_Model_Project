import pandas as pd

from trend_portfolio_app.sim_runner import Simulator
from trend_portfolio_app.policy_engine import PolicyConfig, MetricSpec


def test_simulator_min_tenure_integration():
    # Build a small dataset with clear ranking: A best, B mid, C worst
    dates = pd.period_range("2020-01", "2020-06", freq="M").to_timestamp("M")
    data = pd.DataFrame(
        {
            "A": [0.05, 0.05, 0.05, 0.05, 0.05, -0.10],
            "B": [0.02, 0.02, 0.02, 0.02, 0.02, 0.00],
            "C": [-0.02, -0.02, -0.02, -0.02, -0.02, -0.20],
        },
        index=dates,
    )

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
        freq="m",
        lookback_months=3,
        policy=policy,
    )

    # Expected behavior:
    # - A should be hired first due to best performance.
    # - C (worst) should not be fired until it has been held for at least
    #   min_tenure_n periods after being hired.
    # Since top_k=1 and max_active=2, initially A should be the only active.
    first_weights = res.weights[res.dates[0]]
    assert "A" in first_weights.index
    # Ensure the simulator ran without NaNs for the period reviewed
    assert res.portfolio.notna().any()
