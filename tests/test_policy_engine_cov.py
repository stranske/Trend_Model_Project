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


def test_decide_hires_fires_diversification_and_turnover(monkeypatch):
    """Bucket caps and turnover limits should constrain hires/fires."""

    score_frame = pd.DataFrame(
        {"m": [3.0, 2.5, 0.5, 1.5]}, index=["a", "b", "c", "d"]
    )
    policy = PolicyConfig(
        top_k=3,
        max_active=3,
        bottom_k=1,
        metrics=[MetricSpec("m", 1.0)],
        diversification_max_per_bucket=1,
        diversification_buckets={"a": "g1", "b": "g1", "c": "g2", "d": "g2"},
        turnover_budget_max_changes=1,
        min_track_months=0,
        min_tenure_n=0,
    )
    directions = {"m": 1}
    eligible_since = {k: 12 for k in score_frame.index}
    tenure = {"a": 3, "c": 3}

    def fake_rank_scores(sf, metric_weights, metric_directions):
        return pd.Series({"a": 2.0, "b": 1.2, "c": -0.5, "d": 1.0}, index=sf.index)

    monkeypatch.setattr(
        "trend_portfolio_app.policy_engine.rank_scores", fake_rank_scores
    )

    decisions = decide_hires_fires(
        pd.Timestamp("2020-01-31"),
        score_frame,
        current=["a", "c"],
        policy=policy,
        directions=directions,
        cooldowns=CooldownBook(),
        eligible_since=eligible_since,
        tenure=tenure,
    )

    hires = decisions["hire"]
    fires = decisions["fire"]

    assert hires == [("d", "top_k")]
    assert fires == []
    assert all(h[0] != "b" for h in hires), "Bucket cap should skip second g1 fund"


def test_decide_hires_fires_turnover_budget_prioritises(monkeypatch):
    """Turnover limits should prioritise hires/fires based on scores."""

    score_frame = pd.DataFrame({"m": [3.0, 2.0, 0.5]}, index=["a", "b", "c"])
    policy = PolicyConfig(
        top_k=2,
        bottom_k=1,
        min_track_months=0,
        turnover_budget_max_changes=2,
        metrics=[MetricSpec("m", 1.0)],
    )
    directions = {"m": 1}
    eligible_since = {k: 12 for k in score_frame.index}
    tenure = {"a": 5}

    def fake_rank_scores(sf, metric_weights, metric_directions):
        return pd.Series({"a": -1.0, "b": 2.0, "c": 1.5}, index=sf.index)

    monkeypatch.setattr(
        "trend_portfolio_app.policy_engine.rank_scores", fake_rank_scores
    )

    decisions = decide_hires_fires(
        pd.Timestamp("2020-01-31"),
        score_frame,
        current=["a"],
        policy=policy,
        directions=directions,
        cooldowns=CooldownBook(),
        eligible_since=eligible_since,
        tenure=tenure,
    )

    # Turnover cap of 2 should keep the top two moves (both hires) and drop the fire
    assert len(decisions["hire"]) == 2
    assert decisions["fire"] == []
    assert {m for m, _ in decisions["hire"]} == {"b", "c"}


def test_decide_hires_fires_turnover_budget_trims_mixed(monkeypatch):
    score_frame = pd.DataFrame({"m": [3.0, 2.0, -1.0]}, index=["a", "b", "c"])
    policy = PolicyConfig(
        top_k=2,
        bottom_k=1,
        max_active=2,
        min_track_months=0,
        turnover_budget_max_changes=1,
        diversification_max_per_bucket=1,
        diversification_buckets={"a": "g1", "b": "g1", "c": "g2"},
        metrics=[MetricSpec("m", 1.0)],
    )
    directions = {"m": 1}
    eligible_since = {k: 12 for k in score_frame.index}
    tenure = {"c": 5}

    def fake_rank_scores(sf, metric_weights, metric_directions):  # noqa: ARG001
        return pd.Series({"a": 1.0, "b": 2.0, "c": -1.0}, index=sf.index)

    monkeypatch.setattr(
        "trend_portfolio_app.policy_engine.rank_scores", fake_rank_scores
    )

    decisions = decide_hires_fires(
        pd.Timestamp("2020-01-31"),
        score_frame,
        current=["c"],
        policy=policy,
        directions=directions,
        cooldowns=CooldownBook(),
        eligible_since=eligible_since,
        tenure=tenure,
    )

    # With a turnover budget of one move the higher scored hire should be kept
    assert decisions["hire"] == [("b", "top_k")]
    assert decisions["fire"] == []
