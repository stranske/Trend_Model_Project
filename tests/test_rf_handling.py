import pandas as pd
import pytest

from trend_analysis.metrics import sharpe_ratio


def _mini_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
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
        allow_risk_free_fallback=True,
    )
    assert res is not None
    funds = set(res["selected_funds"])  # type: ignore[index]
    # Pipeline picks the lowest-vol column as RF and excludes it from funds
    assert "RF" not in funds
    assert res.get("risk_free_column") == "RF"
    assert res.get("risk_free_source") == "fallback"


def test_risk_free_fallback_uses_in_sample_window_alignment():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            # Only populated after the in-sample window; should not be picked as RF
            "FutureRF": [pd.NA, pd.NA, pd.NA, 0.0005, 0.0005, 0.0005],
            "Stable": [0.001] * 6,
            "Fund": [0.02, 0.01, -0.005, 0.003, 0.002, 0.004],
        }
    )
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
        allow_risk_free_fallback=True,
    )

    assert res is not None
    assert res.get("risk_free_column") == "Stable"
    assert res.get("risk_free_source") == "fallback"
    assert set(res.get("selected_funds", [])) == {"Fund"}


def test_risk_free_fallback_requires_window_coverage():
    dates = pd.date_range("2021-02-28", periods=6, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            # Too sparse to qualify even though volatility would be minimal
            "SparseRF": [0.0002, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
            "CoveredRF": [0.0003, 0.0002, 0.00025, 0.0003, 0.0002, 0.00028],
            "Fund": [0.02, 0.01, -0.01, 0.015, 0.0, 0.012],
        }
    )
    from trend_analysis import pipeline
    from trend_analysis.core.rank_selection import RiskStatsConfig

    res = pipeline._run_analysis(  # type: ignore[attr-defined]
        df,
        in_start="2021-02",
        in_end="2021-04",
        out_start="2021-05",
        out_end="2021-07",
        target_vol=0.10,
        monthly_cost=0.0,
        selection_mode="all",
        stats_cfg=RiskStatsConfig(),
        allow_risk_free_fallback=True,
    )

    assert res is not None
    assert res.get("risk_free_column") == "CoveredRF"
    assert res.get("risk_free_source") == "fallback"
    assert set(res.get("selected_funds", [])) == {"Fund"}


def test_risk_free_fallback_rejects_all_sparse_series():
    dates = pd.date_range("2021-02-28", periods=6, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "SparseRF": [0.0002, pd.NA, pd.NA, pd.NA, pd.NA, 0.0002],
            "AlsoSparse": [pd.NA, pd.NA, pd.NA, 0.0003, pd.NA, pd.NA],
        }
    )
    from trend_analysis import pipeline
    from trend_analysis.core.rank_selection import RiskStatsConfig

    with pytest.raises(ValueError, match="coverage requirement"):
        pipeline._run_analysis(  # type: ignore[attr-defined]
            df,
            in_start="2021-02",
            in_end="2021-04",
            out_start="2021-05",
            out_end="2021-07",
            target_vol=0.10,
            monthly_cost=0.0,
            selection_mode="all",
            stats_cfg=RiskStatsConfig(),
            allow_risk_free_fallback=True,
        )


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
        allow_risk_free_fallback=True,
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


def test_pipeline_requires_configured_risk_free_column():
    df = _mini_df().drop(columns=["RF"])
    from trend_analysis import pipeline
    from trend_analysis.core.rank_selection import RiskStatsConfig

    with pytest.raises(ValueError, match="Configured risk-free column 'RF'"):
        pipeline._run_analysis(  # type: ignore[attr-defined]
            df,
            in_start="2020-01",
            in_end="2020-03",
            out_start="2020-04",
            out_end="2020-06",
            target_vol=0.10,
            monthly_cost=0.0,
            selection_mode="all",
            stats_cfg=RiskStatsConfig(),
            risk_free_column="RF",
            allow_risk_free_fallback=False,
        )


def test_pipeline_requires_flag_for_fallback_when_missing_rf():
    df = _mini_df().drop(columns=["RF"])
    from trend_analysis import pipeline
    from trend_analysis.core.rank_selection import RiskStatsConfig

    with pytest.raises(ValueError, match="allow_risk_free_fallback"):
        pipeline._run_analysis(  # type: ignore[attr-defined]
            df,
            in_start="2020-01",
            in_end="2020-03",
            out_start="2020-04",
            out_end="2020-06",
            target_vol=0.10,
            monthly_cost=0.0,
            selection_mode="all",
            stats_cfg=RiskStatsConfig(),
            allow_risk_free_fallback=False,
        )


def test_pipeline_implicit_fallback_disabled_by_default():
    df = _mini_df().drop(columns=["RF"])
    from trend_analysis import pipeline
    from trend_analysis.core.rank_selection import RiskStatsConfig

    kwargs = dict(
        df=df,
        in_start="2020-01",
        in_end="2020-03",
        out_start="2020-04",
        out_end="2020-06",
        target_vol=0.10,
        monthly_cost=0.0,
        selection_mode="all",
        stats_cfg=RiskStatsConfig(),
    )
    with pytest.raises(ValueError, match="allow_risk_free_fallback"):
        pipeline._run_analysis(  # type: ignore[attr-defined]
            **kwargs,
        )


def test_identify_risk_free_name_heuristics():
    # Named RF candidate should be preferred over other numerics
    dates = pd.date_range("2021-01-31", periods=4, freq="ME")
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
    dates = pd.date_range("2021-01-31", periods=3, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "t-Bill": [0.001] * 3,
            "A": [0.01, -0.005, 0.02],
        }
    )
    from trend_analysis.data import identify_risk_free_fund

    assert identify_risk_free_fund(df) == "t-Bill"


def test_pipeline_uses_configured_series_for_metrics():
    df = _mini_df().rename(columns={"RF": "Cash"})
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
        risk_free_column="Cash",
        allow_risk_free_fallback=False,
    )
    assert res is not None
    assert res.get("risk_free_column") == "Cash"
    assert res.get("risk_free_source") == "configured"
    out_slice = df.set_index("Date").loc["2020-04-30":"2020-06-30"]
    expected_sharpe = sharpe_ratio(
        res["out_sample_scaled"]["A"], risk_free=out_slice["Cash"]
    )
    assert res["out_sample_stats"]["A"].sharpe == pytest.approx(expected_sharpe)
