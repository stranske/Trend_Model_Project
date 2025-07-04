import pandas as pd
import pytest
from trend_analysis.weighting import EqualWeight, ScorePropSimple, ScorePropBayesian


def make_df() -> pd.DataFrame:
    return pd.DataFrame({"Sharpe": [1.0, -0.5, 0.0]}, index=["A", "B", "C"])


def test_equal_weight_empty():
    df = pd.DataFrame(columns=["Sharpe"])
    out = EqualWeight().weight(df)
    assert out.empty


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
