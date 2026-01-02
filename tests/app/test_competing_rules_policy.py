import pandas as pd

from streamlit_app.components.policy_engine import (
    CooldownBook,
    MetricSpec,
    PolicyConfig,
    decide_hires_fires,
)


def test_competing_rules_sticky_add_and_drop():
    sf = pd.DataFrame({"sharpe": [3.0, 2.0, -1.0]}, index=["A", "B", "C"]).sort_index()
    directions = {"sharpe": +1}
    policy = PolicyConfig(
        top_k=2,
        bottom_k=1,
        cooldown_months=0,
        min_track_months=0,
        max_active=10,
        add_rules=["sticky_rank_window", "threshold_hold"],
        drop_rules=["sticky_rank_window", "threshold_hold"],
        sticky_add_x=2,
        sticky_drop_y=2,
        metrics=[MetricSpec("sharpe", 1.0)],
    )
    cd = CooldownBook()
    elig = {m: 24 for m in sf.index}
    rule_state = {
        "add_streak": {"A": 1, "B": 1},  # not enough yet to add
        "drop_streak": {"C": 1},  # not enough yet to drop
    }
    d = pd.Timestamp("2020-12-31")
    decisions = decide_hires_fires(
        d,
        sf,
        current=["C"],
        policy=policy,
        directions=directions,
        cooldowns=cd,
        eligible_since=elig,
        tenure=None,
        rule_state=rule_state,
    )
    assert decisions["hire"] == []  # sticky add gate blocks
    assert decisions["fire"] == []  # sticky drop gate blocks

    # Once streaks reach threshold, actions proceed
    rule_state["add_streak"] = {"A": 2, "B": 2}
    rule_state["drop_streak"] = {"C": 2}
    decisions2 = decide_hires_fires(
        d,
        sf,
        current=["C"],
        policy=policy,
        directions=directions,
        cooldowns=cd,
        eligible_since=elig,
        tenure=None,
        rule_state=rule_state,
    )
    assert len(decisions2["hire"]) > 0
    assert decisions2["fire"] == [("C", "bottom_k")]


def test_confidence_interval_add_rule_reporting_only():
    sf = pd.DataFrame({"sharpe": [1.0, -0.1]}, index=["A", "B"]).sort_index()
    directions = {"sharpe": +1}
    policy = PolicyConfig(
        top_k=1,
        bottom_k=0,
        cooldown_months=0,
        min_track_months=0,
        max_active=10,
        add_rules=["confidence_interval", "threshold_hold"],
        ci_level=0.90,  # reporting-only; should not gate hires
        metrics=[MetricSpec("sharpe", 1.0)],
    )
    cd = CooldownBook()
    elig = {m: 24 for m in sf.index}

    decisions = decide_hires_fires(
        pd.Timestamp("2020-12-31"),
        sf,
        current=[],
        policy=policy,
        directions=directions,
        cooldowns=cd,
        eligible_since=elig,
        tenure=None,
        rule_state={},
    )
    hired = [m for m, _ in decisions["hire"]]
    assert hired == ["A"]
