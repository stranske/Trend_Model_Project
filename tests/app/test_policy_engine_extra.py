import pandas as pd

from streamlit_app.components.policy_engine import (
    CooldownBook,
    MetricSpec,
    PolicyConfig,
    decide_hires_fires,
    zscore,
)


def test_policy_config_dict():
    cfg = PolicyConfig(metrics=[MetricSpec("m1")])
    data = cfg.dict()
    assert data["top_k"] == 10
    assert data["metrics"][0]["name"] == "m1"


def test_cooldown_tick_and_set():
    book = CooldownBook()
    book.set("a", 1)
    book.tick()
    assert not book.in_cooldown("a")


def test_zscore_zero_variance():
    s = pd.Series([1.0, 1.0, 1.0])
    out = zscore(s)
    assert (out == 0).all()


def test_confidence_interval_gate():
    sf = pd.DataFrame({"m1": [-1.0, 2.0]}, index=["a", "b"])
    policy = PolicyConfig(
        top_k=2,
        ci_level=0.95,
        add_rules=["confidence_interval"],
        metrics=[MetricSpec("m1")],
    )
    decisions = decide_hires_fires(
        pd.Timestamp("2020-01-31"),
        sf,
        [],
        policy,
        {"m1": 1},
        CooldownBook(),
        {"a": 100, "b": 100},
    )
    assert ("a", "top_k") not in decisions["hire"]


def test_diversification_break():
    sf = pd.DataFrame({"m1": [1.0, 2.0]}, index=["a", "b"])
    policy = PolicyConfig(
        top_k=0,
        diversification_max_per_bucket=1,
        diversification_buckets={"a": "x", "b": "y"},
        metrics=[MetricSpec("m1")],
    )
    decisions = decide_hires_fires(
        pd.Timestamp("2020-01-31"),
        sf,
        [],
        policy,
        {"m1": 1},
        CooldownBook(),
        {"a": 100, "b": 100},
    )
    assert decisions["hire"] == []
