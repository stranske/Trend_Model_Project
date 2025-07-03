import pandas as pd
import numpy as np
from trend_analysis import pipeline


def _make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "A": [np.nan, 0.02, 0.03, 0.03, 0.02, 0.01],
            "B": [0.02, 0.01, 0.02, 0.01, 0.03, 0.02],
            "RF": 0.0,
        }
    )


def test_run_analysis_drop_and_weights():
    df = _make_df()
    res = pipeline.run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        1.0,
        0.0,
    )
    assert res["selected_funds"] == ["B"]
    assert res["fund_weights"]["B"] == 1.0


def test_run_analysis_custom_weights():
    df = _make_df()
    res = pipeline.run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        1.0,
        0.0,
        custom_weights={"A": 20, "B": 80},
    )
    assert res["fund_weights"] == {"B": 0.8}
