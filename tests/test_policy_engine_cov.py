import numpy as np
import pandas as pd

from streamlit_app.components.policy_engine import (
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

    score_frame = pd.DataFrame({"m": [3.0, 2.5, 0.5, 1.5]}, index=["a", "b", "c", "d"])
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
    eligible_since = dict.fromkeys(score_frame.index, 12)
    tenure = {"a": 3, "c": 3}

    def fake_rank_scores(sf, metric_weights, metric_directions):
        return pd.Series({"a": 2.0, "b": 1.2, "c": -0.5, "d": 1.0}, index=sf.index)

    monkeypatch.setattr("streamlit_app.components.policy_engine.rank_scores", fake_rank_scores)

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
    eligible_since = dict.fromkeys(score_frame.index, 12)
    tenure = {"a": 5}

    def fake_rank_scores(sf, metric_weights, metric_directions):
        return pd.Series({"a": -1.0, "b": 2.0, "c": 1.5}, index=sf.index)

    monkeypatch.setattr("streamlit_app.components.policy_engine.rank_scores", fake_rank_scores)

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


def test_decide_hires_fires_turnover_budget_mixed_moves(monkeypatch):
    """Turnover gating should compare hires and fires and keep the best
    move."""

    score_frame = pd.DataFrame({"m": [4.0, 3.0, -2.0]}, index=["hire1", "hire2", "drop"])
    policy = PolicyConfig(
        top_k=2,
        bottom_k=1,
        turnover_budget_max_changes=1,
        min_track_months=0,
        metrics=[MetricSpec("m", 1.0)],
    )

    def fake_rank_scores(sf, metric_weights, metric_directions):  # noqa: ARG001
        return pd.Series({"hire1": 2.0, "hire2": 1.0, "drop": -1.5}, index=sf.index)

    monkeypatch.setattr("streamlit_app.components.policy_engine.rank_scores", fake_rank_scores)

    decisions = decide_hires_fires(
        pd.Timestamp("2020-01-31"),
        score_frame,
        current=["drop"],
        policy=policy,
        directions={"m": 1},
        cooldowns=CooldownBook(),
        eligible_since=dict.fromkeys(score_frame.index, 12),
        tenure={"drop": 5},
    )

    assert decisions["hire"] == [("hire1", "top_k")]
    assert decisions["fire"] == []


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
    eligible_since = dict.fromkeys(score_frame.index, 12)
    tenure = {"c": 5}

    def fake_rank_scores(sf, metric_weights, metric_directions):  # noqa: ARG001
        return pd.Series({"a": 1.0, "b": 2.0, "c": -1.0}, index=sf.index)

    monkeypatch.setattr("streamlit_app.components.policy_engine.rank_scores", fake_rank_scores)

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


def test_decide_hires_fires_turnover_budget_mixes_hires_and_fires(monkeypatch):
    score_frame = pd.DataFrame({"m": [4.0, 3.0, 1.0]}, index=["a", "b", "c"])
    policy = PolicyConfig(
        top_k=2,
        bottom_k=1,
        max_active=2,
        turnover_budget_max_changes=1,
        min_track_months=0,
        metrics=[MetricSpec("m", 1.0)],
    )
    directions = {"m": 1}
    eligible_since = dict.fromkeys(score_frame.index, 12)
    tenure = {"a": 2}

    def fake_rank_scores(sf, metric_weights, metric_directions):  # noqa: ARG001
        return pd.Series({"a": -0.5, "b": 1.5, "c": 1.0}, index=sf.index)

    monkeypatch.setattr("streamlit_app.components.policy_engine.rank_scores", fake_rank_scores)

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

    assert len(decisions["hire"]) == 1
    assert decisions["hire"][0][0] in {"b", "c"}
    assert decisions["fire"] == []


def test_decide_hires_fires_bucket_skip_and_nan_priorities(monkeypatch):
    score_frame = pd.DataFrame({"m": [3.0, 2.5, 1.5, -0.5]}, index=["a", "b", "c", "d"])
    policy = PolicyConfig(
        top_k=2,
        bottom_k=1,
        max_active=3,
        min_track_months=0,
        diversification_max_per_bucket=1,
        diversification_buckets={"a": "g1", "b": "g1", "c": "g2", "d": "g3"},
        turnover_budget_max_changes=1,
        metrics=[MetricSpec("m", 1.0)],
    )

    eligible = dict.fromkeys(score_frame.index, 12)
    tenure = {"a": 5, "d": 5}

    def fake_rank_scores(sf, metric_weights, metric_directions):  # noqa: ARG001
        return pd.Series({"a": 5.0, "b": 4.0, "c": 3.0, "d": np.nan}, index=sf.index)

    monkeypatch.setattr("streamlit_app.components.policy_engine.rank_scores", fake_rank_scores)

    decisions = decide_hires_fires(
        pd.Timestamp("2020-01-31"),
        score_frame,
        current=["a", "d"],
        policy=policy,
        directions={"m": 1},
        cooldowns=CooldownBook(),
        eligible_since=eligible,
        tenure=tenure,
    )

    hires = decisions["hire"]
    fires = decisions["fire"]

    # Bucket guard should skip hiring "b" (same bucket as "a") while allowing "c"
    assert ("c", "top_k") in hires
    assert all(name != "b" for name, _ in hires)
    # NaN priority for the fired manager should drop it from turnover-constrained moves
    assert fires == []


def test_decide_hires_fires_unknown_bucket_defaults_to_name(monkeypatch):
    score_frame = pd.DataFrame({"m": [2.0, 3.5]}, index=["known", "mystery"])
    policy = PolicyConfig(
        top_k=1,
        max_active=2,
        min_track_months=0,
        diversification_max_per_bucket=1,
        diversification_buckets={"known": "bucket"},
        metrics=[MetricSpec("m", 1.0)],
    )

    def fake_rank_scores(sf, metric_weights, metric_directions):  # noqa: ARG001
        return pd.Series({"known": 0.5, "mystery": 2.0}, index=sf.index)

    monkeypatch.setattr("streamlit_app.components.policy_engine.rank_scores", fake_rank_scores)

    decisions = decide_hires_fires(
        pd.Timestamp("2020-01-31"),
        score_frame,
        current=["known"],
        policy=policy,
        directions={"m": 1},
        cooldowns=CooldownBook(),
        eligible_since=dict.fromkeys(score_frame.index, 12),
    )

    assert decisions["hire"] == [("mystery", "top_k")]
