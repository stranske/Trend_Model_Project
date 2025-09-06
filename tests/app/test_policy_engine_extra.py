import pandas as pd
from trend_portfolio_app.policy_engine import (
    CooldownBook,
    MetricSpec,
    PolicyConfig,
    decide_hires_fires,
    zscore,
)


def test_policy_config_dict_includes_metrics():
    cfg = PolicyConfig(metrics=[MetricSpec("m", 2.0)])
    out = cfg.dict()
    assert out["metrics"][0]["name"] == "m"
    assert out["metrics"][0]["weight"] == 2.0


def test_cooldownbook_tick_and_remove():
    book = CooldownBook()
    book.set("A", 1)
    assert book.in_cooldown("A")
    book.tick()
    assert not book.in_cooldown("A")


def test_zscore_zero_variance_returns_zero_series():
    s = pd.Series([1.0, 1.0, 1.0])
    out = zscore(s)
    assert (out == 0).all()


def test_decide_hires_fires_confidence_interval_blocks_negative():
    sf = pd.DataFrame({"m": [-1.0, 1.0]}, index=["M1", "M2"])
    policy = PolicyConfig(
        top_k=1,
        min_track_months=0,
        metrics=[MetricSpec("m", 1.0)],
        add_rules=["confidence_interval"],
        ci_level=0.95,
    )
    directions = {"m": 1}
    cd = CooldownBook()
    decisions = decide_hires_fires(
        pd.Timestamp("2020-01-31"), sf, [], policy, directions, cd, {"M1": 24, "M2": 24}
    )
    assert all(m != "M1" for m, _ in decisions["hire"])


def test_decide_hires_fires_breaks_after_topk():
    sf = pd.DataFrame({"m": [3.0, 2.0]}, index=["A", "B"])
    policy = PolicyConfig(
        top_k=1,
        diversification_max_per_bucket=2,
        diversification_buckets={"A": "a", "B": "b"},
        min_track_months=0,
        metrics=[MetricSpec("m", 1.0)],
    )
    directions = {"m": 1}
    cd = CooldownBook()
    decisions = decide_hires_fires(
        pd.Timestamp("2020-01-31"), sf, [], policy, directions, cd, {"A": 24, "B": 24}
    )
    assert len(decisions["hire"]) == 1
