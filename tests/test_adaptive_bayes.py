import pandas as pd
from numpy.testing import assert_allclose
from trend_analysis.weighting import AdaptiveBayesWeighting, ScorePropSimple


def make_scores(a=1.0, b=0.0, c=0.0):
    return pd.Series({"A": a, "B": b, "C": c})


def test_drift_toward_winner():
    w = AdaptiveBayesWeighting(max_w=None)
    df = pd.DataFrame(index=["A", "B", "C"])
    w1 = w.weight(df)["weight"]
    w.update(make_scores(2.0, 0.0, -1.0), 30)
    w2 = w.weight(df)["weight"]
    w.update(make_scores(2.0, 0.0, -1.0), 30)
    w3 = w.weight(df)["weight"]
    assert w3["A"] > w2["A"] > w1["A"]


def test_sum_to_one():
    w = AdaptiveBayesWeighting(max_w=None)
    df = pd.DataFrame(index=["A", "B"])
    weights = w.weight(df)
    assert_allclose(weights["weight"].sum(), 1.0, rtol=1e-12)


def test_clip_respects_max_w():
    w = AdaptiveBayesWeighting(max_w=0.6)
    df = pd.DataFrame(index=["A", "B"])
    w.update(pd.Series({"A": 10.0, "B": -5.0}), 30)
    out = w.weight(df)
    assert out["weight"].max() <= 0.6 + 1e-9


def test_half_life_zero_equals_simple():
    df = pd.DataFrame({"Sharpe": [1.0, 2.0]}, index=["A", "B"])
    scores = df["Sharpe"]
    w = AdaptiveBayesWeighting(half_life=0, max_w=None)
    w.update(scores, 30)
    out = w.weight(df)["weight"]
    exp = ScorePropSimple("Sharpe").weight(df)["weight"]
    assert_allclose(out.values, exp.values)


def test_state_roundtrip():
    w = AdaptiveBayesWeighting(max_w=None)
    df = pd.DataFrame(index=["A", "B"])
    w.update(pd.Series({"A": 1.0, "B": 0.5}), 30)
    state = w.get_state()
    w2 = AdaptiveBayesWeighting(max_w=None)
    w2.set_state(state)
    assert_allclose(w.weight(df)["weight"].values, w2.weight(df)["weight"].values)
