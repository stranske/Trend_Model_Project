import numpy as np
import pandas as pd

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


def _make_daily_equivalent_df(monthly_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    for record in monthly_df.to_dict("records"):
        month_end = record["Date"]
        start = pd.Timestamp(month_end).replace(day=1)
        rng = pd.date_range(start, month_end, freq="D")
        for day in rng[:-1]:
            rows.append({"Date": day, "RF": record["RF"], "A": 0.0, "B": 0.0})
        rows.append(
            {
                "Date": rng[-1],
                "RF": record["RF"],
                "A": record["A"],
                "B": record["B"],
            }
        )
    return pd.DataFrame(rows).sort_values("Date")


def test_run_analysis_detects_daily_frequency():
    daily_df = _make_daily_equivalent_df(make_df())
    res = run_analysis(
        daily_df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        0.1,
        0.0,
    )
    assert res is not None
    preprocess = res.get("preprocessing", {})
    assert preprocess.get("input_frequency") == "D"
    assert preprocess.get("resampled_to_monthly") is True
    summary = res.get("preprocessing_summary")
    assert isinstance(summary, str) and "Cadence" in summary


def test_run_analysis_missing_policy_zero_keeps_asset():
    df = make_df()
    df.loc[1, "B"] = np.nan

    dropped = run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        0.1,
        0.0,
    )
    assert dropped is not None
    assert set(dropped["selected_funds"]) == {"A"}

    filled = run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        0.1,
        0.0,
        missing_policy={"default": "zero"},
    )
    assert filled is not None
    assert set(filled["selected_funds"]) == {"A", "B"}
    missing_meta = filled.get("preprocessing", {}).get("missing", {})
    assert missing_meta.get("dropped") == []


def test_run_analysis_includes_regime_breakdown() -> None:
    df = make_df()
    df["SPX"] = [0.04, 0.03, -0.02, -0.01, 0.025, 0.03]
    res = run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        0.1,
        0.0,
        indices_list=["SPX"],
        regime_cfg={
            "enabled": True,
            "proxy": "SPX",
            "lookback": 2,
            "smoothing": 1,
            "min_observations": 1,
        },
    )
    assert res is not None
    table = res.get("performance_by_regime")
    assert isinstance(table, pd.DataFrame)
    assert ("User", "Risk-On") in table.columns
