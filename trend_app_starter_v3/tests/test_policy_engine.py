import pandas as pd
from trend_portfolio_app.policy_engine import (
    PolicyConfig,
    MetricSpec,
    decide_hires_fires,
    CooldownBook,
)


def test_decide_hires_fires_basic():
    sf = pd.DataFrame(
        {
            "sharpe": [2.0, 1.0, -1.0],
            "vol": [0.10, 0.15, 0.20],
        },
        index=["M1", "M2", "M3"],
    )
    directions = {"sharpe": +1, "vol": -1}
    policy = PolicyConfig(
        top_k=1,
        bottom_k=1,
        cooldown_months=2,
        min_track_months=0,
        max_active=10,
        metrics=[MetricSpec("sharpe", 1.0), MetricSpec("vol", 1.0)],
    )
    cd = CooldownBook()
    elig = {"M1": 24, "M2": 24, "M3": 24}
    decisions = decide_hires_fires(
        pd.Timestamp("2020-12-31"),
        sf,
        current=["M2"],
        policy=policy,
        directions=directions,
        cooldowns=cd,
        eligible_since=elig,
    )
    assert len(decisions["hire"]) == 1
