import pandas as pd

from streamlit_app.components.policy_engine import (
    CooldownBook,
    MetricSpec,
    PolicyConfig,
    decide_hires_fires,
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


def test_min_tenure_blocks_bottom_fire():
    sf = pd.DataFrame(
        {
            "sharpe": [2.0, 1.0, -1.0],
            "vol": [0.10, 0.15, 0.20],
        },
        index=["A", "B", "C"],
    )
    directions = {"sharpe": +1, "vol": -1}
    policy = PolicyConfig(
        top_k=0,
        bottom_k=1,
        cooldown_months=0,
        min_track_months=0,
        max_active=10,
        min_tenure_n=3,
        metrics=[MetricSpec("sharpe", 1.0), MetricSpec("vol", 1.0)],
    )
    cd = CooldownBook()
    elig = {"A": 24, "B": 24, "C": 24}
    # Current holds C (worst), but tenure only 2 < 3, so should NOT fire yet
    t = {"A": 0, "B": 0, "C": 2}
    decisions = decide_hires_fires(
        pd.Timestamp("2020-12-31"),
        sf,
        current=["C"],
        policy=policy,
        directions=directions,
        cooldowns=cd,
        eligible_since=elig,
        tenure=t,
    )
    assert decisions["fire"] == []
    # Once tenure reaches threshold, firing is allowed
    t["C"] = 3
    decisions2 = decide_hires_fires(
        pd.Timestamp("2021-01-31"),
        sf,
        current=["C"],
        policy=policy,
        directions=directions,
        cooldowns=cd,
        eligible_since=elig,
        tenure=t,
    )
    assert decisions2["fire"] == [("C", "bottom_k")]
