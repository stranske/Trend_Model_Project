import pandas as pd

from streamlit_app.components.policy_engine import (
    CooldownBook,
    MetricSpec,
    PolicyConfig,
    decide_hires_fires,
)


def test_turnover_budget_limits_changes():
    # Scores rank: A > B > C > D. Current holds C and D, bottom_k=2 => C,D slated to fire.
    sf = pd.DataFrame(
        {
            "sharpe": [4.0, 3.0, 2.0, 1.0],
            "vol": [0.10, 0.12, 0.15, 0.20],
        },
        index=["A", "B", "C", "D"],
    )
    directions = {"sharpe": +1, "vol": -1}
    policy = PolicyConfig(
        top_k=2,
        bottom_k=2,
        cooldown_months=0,
        min_track_months=0,
        max_active=4,
        turnover_budget_max_changes=2,  # cap total moves to 2
        metrics=[MetricSpec("sharpe", 1.0), MetricSpec("vol", 1.0)],
    )
    cd = CooldownBook()
    elig = {m: 24 for m in sf.index}

    decisions = decide_hires_fires(
        pd.Timestamp("2020-12-31"),
        sf,
        current=["C", "D"],
        policy=policy,
        directions=directions,
        cooldowns=cd,
        eligible_since=elig,
        tenure=None,
    )

    # With a budget of 2 moves total, ensure only 2 combined hires/fires are returned.
    assert len(decisions["hire"]) + len(decisions["fire"]) == 2

    # Priority order by score gap should prefer hiring A, then B over firing (since A,B have
    # highest scores; fires are prioritized by most negative score, but hires outrank due to higher prio).
    hired = [m for m, _ in decisions["hire"]]
    assert "A" in hired
