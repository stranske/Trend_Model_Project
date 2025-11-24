import pandas as pd

from trend_analysis.pipeline import run_analysis


def make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "A": 0.01,
            "SPX": 0.02,
        }
    )


def test_indices_promoted_to_benchmarks():
    df = make_df()
    res = run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        1.0,
        0.0,
        indices_list=["SPX"],
        risk_free_column="RF",
    )
    assert "SPX" in res["benchmark_ir"]


def test_missing_benchmark_column_ignored():
    df = make_df()
    res = run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        1.0,
        0.0,
        benchmarks={"foo": "FOO"},
        risk_free_column="RF",
    )
    assert "foo" not in res["benchmark_ir"]


def test_indices_already_in_benchmarks():
    df = make_df()
    res = run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        1.0,
        0.0,
        indices_list=["SPX"],
        benchmarks={"SPX": "SPX"},
        risk_free_column="RF",
    )
    assert list(res["benchmark_ir"].keys()) == ["SPX"]
