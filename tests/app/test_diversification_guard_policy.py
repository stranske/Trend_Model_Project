import pandas as pd

from streamlit_app.components.policy_engine import (
    CooldownBook,
    MetricSpec,
    PolicyConfig,
    decide_hires_fires,
)


def test_diversification_guard_limits_per_bucket():
    # A1,A2 share bucket 'A'; B1,B2 share bucket 'B'; scores descending
    idx = ["A1", "A2", "B1", "B2"]
    sf = pd.DataFrame(
        {
            "sharpe": [4.0, 3.0, 2.0, 1.0],
        },
        index=idx,
    )
    directions = {"sharpe": +1}
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
    cd = CooldownBook()
    elig = dict.fromkeys(sf.index, 24)

    decisions = decide_hires_fires(
        pd.Timestamp("2020-12-31"),
        sf,
        current=[],
        policy=policy,
        directions=directions,
        cooldowns=cd,
        eligible_since=elig,
        tenure=None,
    )
    hired = [m for m, _ in decisions["hire"]]
    # Expect one from each bucket: A1, B1 (and then none further due to cap)
    assert "A1" in hired and "B1" in hired
    assert len(hired) == 2
