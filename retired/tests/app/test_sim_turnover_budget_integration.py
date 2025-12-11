import pandas as pd

from trend_portfolio_app.policy_engine import MetricSpec, PolicyConfig
from trend_portfolio_app.sim_runner import Simulator


def test_simulator_turnover_budget_integration():
    # Construct a simple panel where we would, without budget, fire worst and hire best two.
    dates = pd.period_range("2020-01", "2020-05", freq="M").to_timestamp(how="end")
    data = pd.DataFrame(
        {
            "A": [0.05, 0.05, 0.05, 0.05, 0.05],
            "B": [0.03, 0.03, 0.03, 0.03, 0.03],
            "C": [0.00, 0.00, 0.00, 0.00, 0.00],
            "D": [-0.02, -0.02, -0.02, -0.02, -0.02],
        },
        index=dates,
    )
    data["Date"] = dates
    sim = Simulator(data)

    policy = PolicyConfig(
        top_k=2,
        bottom_k=2,
        cooldown_months=0,
        min_track_months=0,
        max_active=2,
        turnover_budget_max_changes=1,  # only allow one change per review
        metrics=[MetricSpec("sharpe", 1.0)],
    )

    res = sim.run(
        start=dates[2],
        end=dates[-2],
        freq="ME",
        lookback_months=2,
        policy=policy,
    )

    # Ensure sim executed and budget limited churn doesn't break flow.
    assert res.portfolio.notna().any()
