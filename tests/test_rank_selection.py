import pandas as pd
from trend_analysis.core import rank_selection as rs
from trend_analysis.pipeline import run_analysis


def make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "A": [0.02, 0.03, -0.01, 0.04, 0.02, 0.01],
            "B": [0.01, 0.02, -0.02, 0.03, 0.02, 0.0],
        }
    )


def test_rank_select_funds_top_n():
    df = make_df()
    in_df = df.loc[df.index[:3], ["A", "B"]]
    cfg = rs.RiskStatsConfig(risk_free=0.0)
    selected = rs.rank_select_funds(
        in_df, cfg, inclusion_approach="top_n", n=1, score_by="AnnualReturn"
    )
    assert selected == ["A"]


def test_rank_transform_sorts_best_first():
    df = make_df()
    in_df = df.loc[df.index[:3], ["A", "B"]]
    cfg = rs.RiskStatsConfig(risk_free=0.0)
    selected = rs.rank_select_funds(
        in_df,
        cfg,
        inclusion_approach="top_n",
        n=1,
        score_by="AnnualReturn",
        transform="rank",
    )
    assert selected == ["A"]


def test_metric_alias_handled():
    df = make_df()
    in_df = df.loc[df.index[:3], ["A", "B"]]
    cfg = rs.RiskStatsConfig(risk_free=0.0)
    selected = rs.rank_select_funds(
        in_df,
        cfg,
        inclusion_approach="top_n",
        n=1,
        score_by="annual_return",
    )
    assert selected == ["A"]


def test_run_analysis_rank_mode():
    df = make_df()
    res = run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        0.1,
        0.0,
        selection_mode="rank",
        rank_kwargs={"inclusion_approach": "top_n", "n": 1, "score_by": "AnnualReturn"},
    )
    assert res is not None
    assert res["selected_funds"] == ["A"]
