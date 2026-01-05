import pandas as pd

from trend_analysis import pipeline
from trend_analysis.pipeline import PipelineResult, RiskStatsConfig
from trend_analysis.stages import portfolio as portfolio_stage
from trend_analysis.stages import preprocessing as preprocessing_stage
from trend_analysis.stages import selection as selection_stage


def _sample_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME", tz="UTC")
    frame = pd.DataFrame(
        {
            "Date": dates,
            "A": [0.01, 0.02, 0.015, 0.0, -0.01, 0.005],
            "B": [0.02, 0.01, 0.0, 0.005, -0.02, 0.01],
            "rf": [0.001] * 6,
        }
    )
    frame.attrs["calendar_settings"] = {"timezone": None}
    return frame


def test_stage_isolation_matches_pipeline_output() -> None:
    df = _sample_frame()
    stats_cfg = RiskStatsConfig(metrics_to_run=["Sharpe"], risk_free=0.0)

    pipeline_result = pipeline.run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        0.1,
        0.0,
        stats_cfg=stats_cfg,
        risk_free_column="rf",
        allow_risk_free_fallback=True,
    )

    preprocess = preprocessing_stage._prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=stats_cfg,
        periods_per_year_override=None,
        allow_risk_free_fallback=True,
    )
    assert not isinstance(preprocess, PipelineResult)

    window = preprocessing_stage._build_sample_windows(
        preprocess,
        in_start="2020-01",
        in_end="2020-03",
        out_start="2020-04",
        out_end="2020-06",
    )
    assert not isinstance(window, PipelineResult)

    selection = selection_stage._select_universe(
        preprocess,
        window,
        in_label="2020-01",
        in_end_label="2020-03",
        selection_mode="all",
        random_n=2,
        custom_weights=None,
        rank_kwargs=None,
        manual_funds=None,
        indices_list=None,
        seed=1,
        stats_cfg=stats_cfg,
        risk_free_column="rf",
        allow_risk_free_fallback=True,
    )
    assert not isinstance(selection, PipelineResult)

    computation = portfolio_stage._compute_weights_and_stats(
        preprocess,
        window,
        selection,
        target_vol=0.1,
        monthly_cost=0.0,
        custom_weights=None,
        weighting_scheme=None,
        constraints=None,
        risk_window=None,
        previous_weights=None,
        lambda_tc=None,
        max_turnover=None,
        signal_spec=None,
        weight_policy=None,
        warmup=0,
        min_floor=0.0,
        stats_cfg=stats_cfg,
        weight_engine_params=None,
    )

    manual = portfolio_stage._assemble_analysis_output(
        preprocess,
        window,
        selection,
        computation,
        benchmarks=None,
        regime_cfg=None,
        target_vol=0.1,
        monthly_cost=0.0,
        min_floor=0.0,
    )

    expected = pipeline_result.unwrap()
    actual = manual.unwrap()

    assert expected["selected_funds"] == actual["selected_funds"]
    assert expected["risk_free_column"] == actual["risk_free_column"]
    assert expected["fund_weights"] == actual["fund_weights"]
    assert expected["out_sample_stats"] == actual["out_sample_stats"]
    pd.testing.assert_frame_equal(expected["out_sample_scaled"], actual["out_sample_scaled"])
