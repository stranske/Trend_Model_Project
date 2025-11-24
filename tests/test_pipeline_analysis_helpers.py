import pandas as pd

from trend_analysis.pipeline import (
    AnalysisResult,
    RiskStatsConfig,
    _preprocess_stage,
    _resolve_windows_stage,
    _run_analysis,
    _select_assets_stage,
)


def _sample_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="M")
    return pd.DataFrame(
        {
            "Date": dates,
            "fund_a": [0.01, 0.0, 0.02, -0.01, 0.03, 0.0],
            "fund_b": [0.0, 0.01, -0.02, 0.01, 0.0, 0.02],
            "cash": [0.0] * 6,
        }
    )


def test_preprocess_stage_returns_diagnostics_on_failure():
    df = pd.DataFrame({"Date": [pd.NaT]})
    result, diag = _preprocess_stage(df)
    assert result is None
    assert diag.stage == "preprocessing"
    assert diag.status == "failed"


def test_resolve_windows_stage_handles_empty_slices():
    prep, _ = _preprocess_stage(_sample_df())
    windows, diag = _resolve_windows_stage(
        prep,
        in_start="2010-01",
        in_end="2010-02",
        out_start="2010-03",
        out_end="2010-04",
    )
    assert windows is None
    assert diag.status == "failed"
    assert diag.stage == "window_resolution"


def test_select_assets_stage_success_path():
    prep, _ = _preprocess_stage(_sample_df())
    windows, _ = _resolve_windows_stage(
        prep,
        in_start="2020-01",
        in_end="2020-03",
        out_start="2020-04",
        out_end="2020-06",
    )
    stats_cfg = RiskStatsConfig(metrics_to_run=["Sharpe"], risk_free=0.0)
    selection, diag = _select_assets_stage(
        prep,
        windows,
        selection_mode="all",
        random_n=2,
        custom_weights=None,
        rank_kwargs=None,
        manual_funds=None,
        indices_list=None,
        stats_cfg=stats_cfg,
        seed=1,
        risk_free_column=None,
        allow_risk_free_fallback=True,
    )
    assert selection is not None
    assert selection.fund_cols
    assert diag.status == "success"


def test_run_analysis_returns_result_object():
    df = _sample_df()
    result = _run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        1.0,
        0.0,
    )
    assert isinstance(result, AnalysisResult)
    assert result.success
    assert "out_sample_stats" in (result.payload or {})


def test_run_analysis_reports_failure_reason():
    df = pd.DataFrame({"Date": [pd.NaT]})
    result = _run_analysis(
        df,
        "2020-01",
        "2020-01",
        "2020-02",
        "2020-02",
        1.0,
        0.0,
    )
    assert isinstance(result, AnalysisResult)
    assert not result.success
    assert any(d.status == "failed" for d in result.diagnostics)
