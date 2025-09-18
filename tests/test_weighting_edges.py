import numpy as np
import pandas as pd
import pytest

from trend_analysis.weighting import (
    AdaptiveBayesWeighting,
    BaseWeighting,
    EqualWeight,
    ScorePropBayesian,
    ScorePropSimple,
)


def make_df() -> pd.DataFrame:
    return pd.DataFrame({"Sharpe": [1.0, -0.5, 0.0]}, index=["A", "B", "C"])


def test_equal_weight_empty():
    df = pd.DataFrame(columns=["Sharpe"])
    out = EqualWeight().weight(df)
    assert out.empty


def test_base_weighting_is_abstract():
    class Dummy(BaseWeighting):
        pass

    dummy = Dummy()
    with pytest.raises(NotImplementedError):
        dummy.weight(pd.DataFrame())


def test_scoreprop_simple_missing_column():
    df = make_df()
    with pytest.raises(KeyError):
        ScorePropSimple(column="Foo").weight(df)


def test_scoreprop_simple_zero_sum_fallback():
    df = pd.DataFrame({"Sharpe": [-1.0, -2.0]}, index=["A", "B"])
    out = ScorePropSimple().weight(df)
    assert pytest.approx(out.loc["A", "weight"]) == 0.5
    assert pytest.approx(out["weight"].sum()) == 1.0


def test_scoreprop_bayesian_missing_column():
    df = make_df()
    with pytest.raises(KeyError):
        ScorePropBayesian(column="Foo").weight(df)


def test_scoreprop_bayesian_zero_sum_fallback():
    df = pd.DataFrame({"Sharpe": [-1.0, -2.0]}, index=["A", "B"])
    out = ScorePropBayesian().weight(df)
    assert pytest.approx(out["weight"].sum()) == 1.0
    assert pytest.approx(out.loc["A", "weight"]) == 0.5


def test_adaptive_bayes_prior_mean_length_mismatch():
    weighting = AdaptiveBayesWeighting(prior_mean=np.array([0.75, 0.25]))
    df = pd.DataFrame(index=["A", "B", "C"])
    with pytest.raises(ValueError):
        weighting.weight(df)


def test_adaptive_bayes_weight_caps_and_state_roundtrip():
    weighting = AdaptiveBayesWeighting(max_w=0.4)
    state = {
        "mean": {"A": 0.8, "B": 0.1, "C": 0.1},
        "tau": {"A": 1.0, "B": 1.0, "C": 1.0},
    }
    weighting.set_state(state)
    df = pd.DataFrame(index=["A", "B", "C"])
    out = weighting.weight(df)
    assert list(out.index) == ["A", "B", "C"]
    # Weight cap redistributes the deficit to funds with room below the cap.
    assert pytest.approx(out.loc["A", "weight"], rel=1e-6) == 0.4
    assert pytest.approx(out.loc["B", "weight"], rel=1e-6) == 0.3
    assert pytest.approx(out["weight"].sum(), rel=1e-6) == 1.0
    # Round-trip state serialization keeps the most recent posterior.
    new_state = weighting.get_state()
    assert new_state["mean"]["A"] == pytest.approx(state["mean"]["A"])


def test_adaptive_bayes_cap_renormalise_when_no_room():
    weighting = AdaptiveBayesWeighting(max_w=0.3)
    weighting.set_state(
        {
            "mean": {"A": 2.0, "B": 2.0, "C": 2.0},
            "tau": {"A": 1.0, "B": 1.0, "C": 1.0},
        }
    )
    df = pd.DataFrame(index=["A", "B", "C"])
    out = weighting.weight(df)
    assert pytest.approx(out["weight"].sum(), rel=1e-6) == 1.0
    # With every asset at the cap, the implementation renormalises weights so
    # the portfolio still sums to one.
    assert out["weight"].max() > 0.3
    assert pytest.approx(out["weight"].max(), rel=1e-6) == pytest.approx(1 / 3)


def test_adaptive_bayes_adds_new_assets():
    weighting = AdaptiveBayesWeighting(max_w=None)
    weighting.update(pd.Series({"A": 1.0, "B": 0.5}), days=30)
    # Introduce a new asset "C" not seen during the update
    df = pd.DataFrame(index=["A", "B", "C"])
    out = weighting.weight(df)
    assert "C" in out.index
    # Newly added asset receives a finite non-negative weight
    assert out.loc["C", "weight"] >= 0.0


def test_adaptive_bayes_update_without_half_life_decay():
    weighting = AdaptiveBayesWeighting(half_life=0, obs_sigma=1.0, prior_tau=2.0)
    scores = pd.Series({"A": 0.5, "B": 1.5})
    weighting.update(scores, days=30)
    assert weighting.mean is not None
    assert weighting.tau is not None
    # With zero half-life the prior precision is reset before updating,
    # so the posterior precision equals the observational precision (=1.0).
    assert pytest.approx(weighting.tau.loc["A"], rel=1e-6) == pytest.approx(1.0)
    assert pytest.approx(weighting.mean.loc["B"], rel=1e-6) == 1.5
