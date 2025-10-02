import numpy as np
import pandas as pd
import pytest

from trend_analysis.core.rank_selection import RiskStatsConfig
from trend_analysis.metrics import (
    annualize_return,
    annualize_volatility,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from trend_analysis.pipeline import Stats, calc_portfolio_returns, run_analysis


def make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    data = {
        "Date": dates,
        "RF": 0.0,
        "A": [0.02, 0.03, -0.01, 0.04, 0.02, 0.01],
        "B": [0.01, 0.02, -0.02, 0.03, 0.02, 0.0],
    }
    return pd.DataFrame(data)


def make_daily_df(monthly_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | pd.Timestamp]] = []
    for _, row in monthly_df.iterrows():
        period = row["Date"].to_period("M")
        days = pd.date_range(period.to_timestamp(), period.to_timestamp("M"), freq="B")
        count = len(days)
        for day in days:
            record: dict[str, float | pd.Timestamp] = {"Date": day}
            for column in monthly_df.columns:
                if column == "Date":
                    continue
                value = row[column]
                if pd.isna(value):
                    record[column] = float("nan")
                    continue
                if column == "RF":
                    record[column] = float(value) / count
                else:
                    record[column] = float((1.0 + value) ** (1.0 / count) - 1.0)
            records.append(record)
    return pd.DataFrame(records)


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


def test_run_analysis_returns_score_frame():
    df = make_df()
    res = run_analysis(df, "2020-01", "2020-03", "2020-04", "2020-06", 0.1, 0.0)
    sf = res.get("score_frame")
    assert isinstance(sf, pd.DataFrame)
    cfg = RiskStatsConfig()
    assert sf.columns.tolist() == cfg.metrics_to_run
    assert sf.attrs["insample_len"] == 3


def test_run_analysis_normalises_daily_inputs():
    monthly_df = make_df()
    daily_df = make_daily_df(monthly_df)

    res_monthly = run_analysis(
        monthly_df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        0.1,
        0.0,
    )
    res_daily = run_analysis(
        daily_df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        0.1,
        0.0,
        missing_policy="drop",
    )

    assert res_monthly["input_frequency"]["code"] == "M"
    assert res_daily["input_frequency"]["code"] == "D"
    assert set(res_monthly["selected_funds"]) == set(res_daily["selected_funds"])

    for key in ("in_sample_stats", "out_sample_stats"):
            for fund, monthly_stats in res_monthly[key].items():
                daily_stats = res_daily[key][fund]
                for attr in ("cagr", "vol", "sharpe", "sortino", "information_ratio", "max_drawdown"):
                    expected = getattr(monthly_stats, attr)
                    actual = getattr(daily_stats, attr)
                    if pd.isna(expected) and pd.isna(actual):
                        continue
                    assert actual == pytest.approx(expected, rel=1e-6)


def test_run_analysis_missing_policy_summary_zero_fill():
    df = make_df()
    df.loc[2, "A"] = pd.NA

    res_drop = run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        0.1,
        0.0,
        missing_policy="drop",
    )
    res_zero = run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        0.1,
        0.0,
        missing_policy="zero",
    )

    assert res_drop["missing_data_policy"]["policy"] == "drop"
    assert "A" not in res_drop["selected_funds"]

    zero_policy = res_zero["missing_data_policy"]
    assert zero_policy["policy"] == "zero"
    assert zero_policy["total_filled"] >= 1
    assert "A" in res_zero["selected_funds"]
