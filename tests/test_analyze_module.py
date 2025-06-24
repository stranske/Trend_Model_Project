import pandas as pd
import numpy as np
from trend_analysis import analyze


def _make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="M")
    return pd.DataFrame({
        "Date": dates,
        "A": [np.nan, 0.02, 0.03, 0.03, 0.02, 0.01],
        "B": [0.02, 0.01, 0.02, 0.01, 0.03, 0.02],
        "RF": 0.0,
    })


def test_run_analysis_drop_and_weights():
    df = _make_df()
    res = analyze.run_analysis(
        df,
        ["A", "B"],
        None,
        None,
        "RF",
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
    )
    assert res["selected_funds"] == ["B"]
    assert res["fund_weights"]["B"] == 1.0


def test_run_analysis_custom_weights():
    df = _make_df()
    res = analyze.run_analysis(
        df,
        ["A", "B"],
        None,
        {"A": 0.2, "B": 0.8},
        "RF",
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
    )
    assert res["fund_weights"] == {"A": 0.2, "B": 0.8}


