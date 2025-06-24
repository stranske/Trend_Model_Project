import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from trend_analysis.analyze import run_analysis


def make_df():
    dates = pd.date_range("2020-01-01", periods=4, freq="M")
    return pd.DataFrame({
        "Date": dates,
        "A": [0.1, 0.2, 0.3, 0.4],
        "B": [float("nan"), 0.2, 0.1, 0.2],
    })


def test_run_analysis_drops_and_weights():
    df = make_df()
    res = run_analysis(
        df,
        selected=["A", "B"],
        w_vec=None,
        w_dict=None,
        rf_col="A",
        in_start="2020-01",
        in_end="2020-02",
        out_start="2020-03",
        out_end="2020-04",
    )
    assert res["selected_funds"] == ["A"]
    assert res["dropped"] == ["B"]
    assert res["fund_weights"] == {"A": 1.0}
    assert res["in_sample_mean"]["A"] == 0.1
    assert res["out_sample_mean"]["A"] == 0.3


def test_run_analysis_custom_weights():
    df = make_df()
    res = run_analysis(
        df,
        selected=["A"],
        w_vec=None,
        w_dict={"A": 0.7},
        rf_col="A",
        in_start="2020-01",
        in_end="2020-02",
        out_start="2020-03",
        out_end="2020-04",
    )
    assert res["selected_funds"] == ["A"]
    assert res["fund_weights"] == {"A": 0.7}
