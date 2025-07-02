import pandas as pd
import numpy as np
from trend_analysis.pipeline import run_analysis, Stats, calc_portfolio_returns
from trend_analysis.metrics import (
    annual_return,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    information_ratio
)


def make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    data = {
        "Date": dates,
        "RF": 0.0,
        "A": [0.02, 0.03, -0.01, 0.04, 0.02, 0.01],
        "B": [0.01, 0.02, -0.02, 0.03, 0.02, 0.0],
    }
    return pd.DataFrame(data)


def test_metrics_roundtrip():
    df = make_df()
    series = df["A"]
    rf = df["RF"]
    r = annualize_return(series)
    v = annualize_volatility(series)
    s = sharpe_ratio(series, rf)
    so = sortino_ratio(series, rf)
    mdd = max_drawdown(series)
    port = calc_portfolio_returns(np.array([0.5, 0.5]), df[["A", "B"]])
    assert isinstance(r, float) and isinstance(v, float)
    assert isinstance(s, float) and isinstance(so, float)
    assert isinstance(mdd, float)
    assert port.shape[0] == series.shape[0]


def test_run_analysis_basic():
    df = make_df()
    res = run_analysis(df, "2020-01", "2020-03", "2020-04", "2020-06", 0.1, 0.0)
    assert res is not None
    assert set(res["selected_funds"]) == {"A", "B"}
    assert "in_sample_stats" in res
    assert isinstance(res["in_ew_stats"], Stats)


def test_run_analysis_random_selection():
    df = make_df()
    res = run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        0.1,
        0.0,
        selection_mode="random",
        random_n=1,
        seed=1,
    )
    assert len(res["selected_funds"]) == 1
