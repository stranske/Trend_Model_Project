import pandas as pd

from trend_portfolio_app.policy_engine import (
    CooldownBook,
    MetricSpec,
    PolicyConfig,
    decide_hires_fires,
    zscore,
)


def test_policy_config_dict_cooldown_and_zscore():
    policy = PolicyConfig(metrics=[MetricSpec("m", 1.0)])
    cfg = policy.dict()
    assert cfg["top_k"] == policy.top_k

    cb = CooldownBook()
    cb.set("a", 1)
    cb.tick()
    assert not cb.in_cooldown("a")

    s = pd.Series([1.0, 1.0, 1.0])
    assert zscore(s).eq(0.0).all()


def test_policy_engine_allow_add_ci_level_and_diversification_break():
    # Two candidates so loop breaks after reaching top_k
    score_frame = pd.DataFrame({"m": [1.0, 2.0]}, index=["a", "b"])
    policy = PolicyConfig(
        top_k=1,
        diversification_max_per_bucket=10,
        metrics=[MetricSpec("m", 1.0)],
    )
    directions = {"m": 1}
    decisions = decide_hires_fires(
        pd.Timestamp("2020-01-31"),
        score_frame,
        current=[],
        policy=policy,
        directions=directions,
        cooldowns=CooldownBook(),
        eligible_since={"a": 24, "b": 24},
    )
    # Only first candidate hired due to top_k limit -> break executed
    assert decisions["hire"] == [("b", "top_k")]

    # Negative score with ci_level>0 should be rejected
    score_frame_neg = pd.DataFrame({"m": [-1.0, 1.0]}, index=["c", "d"])
    policy_neg = PolicyConfig(
        top_k=2,
        ci_level=0.95,
        metrics=[MetricSpec("m", 1.0)],
        add_rules=["confidence_interval"],
    )
    decisions_neg = decide_hires_fires(
        pd.Timestamp("2020-01-31"),
        score_frame_neg,
        current=[],
        policy=policy_neg,
        directions=directions,
        cooldowns=CooldownBook(),
        eligible_since={"c": 24, "d": 24},
    )
    assert ("c", "top_k") not in decisions_neg["hire"]
