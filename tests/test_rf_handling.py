import pandas as pd


def _mini_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="M")
    # Two simple funds with different vol, plus an explicit RF series
    return pd.DataFrame(
        {
            "Date": dates,
            "A": [0.02, 0.01, -0.01, 0.015, 0.0, 0.01],
            "B": [0.005, 0.0, 0.002, 0.004, 0.0, 0.003],
            "RF": [0.001] * 6,
        }
    )


def test_pipeline_respects_lowest_vol_exclusion_as_rf():
    df = _mini_df()
    from trend_analysis import pipeline
    from trend_analysis.core.rank_selection import RiskStatsConfig

    res = pipeline._run_analysis(  # type: ignore[attr-defined]
        df,
        in_start="2020-01",
        in_end="2020-03",
        out_start="2020-04",
        out_end="2020-06",
        target_vol=0.10,
        monthly_cost=0.0,
        selection_mode="all",
        stats_cfg=RiskStatsConfig(),
    )
    assert res is not None
    funds = set(res["selected_funds"])  # type: ignore[index]
    # Pipeline picks the lowest-vol column as RF and excludes it from funds
    assert "RF" not in funds


def test_pipeline_constant_rf_via_stats_cfg_executes():
    df = _mini_df().drop(columns=["RF"])  # no RF column available
    from trend_analysis import pipeline
    from trend_analysis.core.rank_selection import RiskStatsConfig

    # Ensure setting risk_free in stats_cfg executes and returns metrics
    res = pipeline._run_analysis(  # type: ignore[attr-defined]
        df,
        in_start="2020-01",
        in_end="2020-03",
        out_start="2020-04",
        out_end="2020-06",
        target_vol=0.10,
        monthly_cost=0.0,
        selection_mode="all",
        stats_cfg=RiskStatsConfig(risk_free=0.0),
    )
    assert res is not None
    # Out-of-sample stats should exist for non-RF funds.
    out_stats = res.get("out_sample_stats")  # type: ignore[assignment]
    assert isinstance(out_stats, dict)
    # Determine lowest-vol col as RF and ensure it's excluded from out_stats
    ret_cols = [c for c in df.columns if c != "Date"]
    rf_col = min(ret_cols, key=lambda c: df[c].std())
    assert rf_col not in out_stats
    assert len(out_stats) >= 1


def test_identify_risk_free_name_heuristics():
    # Named RF candidate should be preferred over other numerics
    dates = pd.date_range("2021-01-31", periods=4, freq="M")
    df = pd.DataFrame(
        {
            "Date": dates,
            "A": [0.01, 0.02, -0.01, 0.015],
            "RiskFree": [0.002, 0.002, 0.002, 0.002],
            "B": [0.001, 0.001, 0.0, 0.001],
        }
    )
    from trend_analysis.data import identify_risk_free_fund

    assert identify_risk_free_fund(df) == "RiskFree"


def test_identify_risk_free_name_aliases_case_insensitive():
    dates = pd.date_range("2021-01-31", periods=3, freq="M")
    df = pd.DataFrame(
        {
            "Date": dates,
            "t-Bill": [0.001] * 3,
            "A": [0.01, -0.005, 0.02],
        }
    )
    from trend_analysis.data import identify_risk_free_fund

    assert identify_risk_free_fund(df) == "t-Bill"
