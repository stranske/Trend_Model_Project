import pandas as pd
import pytest

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


def test_rank_selection_uses_window_cadence_without_mutating_caller_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _sample_frame()
    caller_cfg = RiskStatsConfig(periods_per_year=12, risk_free=0.0)
    preprocess = preprocessing_stage._prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=caller_cfg,
        periods_per_year_override=52,
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

    observed_periods: list[int] = []

    def _rank_stub(_: pd.DataFrame, stats_cfg: RiskStatsConfig, **__: object) -> list[str]:
        observed_periods.append(stats_cfg.periods_per_year)
        return ["A"]

    def _score_stub(_: pd.DataFrame, *args: object, **kwargs: object) -> pd.DataFrame:
        stats_cfg = kwargs["stats_cfg"]
        assert isinstance(stats_cfg, RiskStatsConfig)
        observed_periods.append(stats_cfg.periods_per_year)
        return pd.DataFrame({"Sharpe": [1.0]}, index=["A"])

    monkeypatch.setattr(selection_stage, "rank_select_funds", _rank_stub)
    monkeypatch.setattr(selection_stage, "single_period_run", _score_stub)
    selection = selection_stage._select_universe(
        preprocess,
        window,
        in_label="2020-01",
        in_end_label="2020-03",
        selection_mode="rank",
        random_n=1,
        custom_weights=None,
        rank_kwargs=None,
        manual_funds=None,
        indices_list=None,
        seed=1,
        stats_cfg=caller_cfg,
        risk_free_column="rf",
        allow_risk_free_fallback=True,
    )

    assert not isinstance(selection, PipelineResult)
    assert observed_periods == [52, 52]
    assert caller_cfg.periods_per_year == 12


def test_run_analysis_propagates_window_cadence_to_portfolios_and_benchmark() -> None:
    """The public entrypoint keeps every published metric on the window cadence."""
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=8, freq="ME", tz="UTC"),
            "A": [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.04, -0.015],
            "B": [-0.01, 0.015, -0.005, 0.025, -0.02, 0.01, -0.01, 0.02],
            "Bench": [0.005, -0.01, 0.02, -0.005, 0.01, -0.02, 0.025, -0.01],
            "rf": [0.0] * 8,
        }
    )
    frame.attrs["calendar_settings"] = {"timezone": None}
    common = dict(
        stats_cfg=RiskStatsConfig(risk_free=0.0),
        risk_free_column="rf",
        allow_risk_free_fallback=True,
        benchmarks={"bench": "Bench"},
    )

    monthly = pipeline.run_analysis(
        frame,
        "2020-01",
        "2020-04",
        "2020-05",
        "2020-08",
        None,
        0.0,
        periods_per_year=12,
        **common,
    ).unwrap()
    weekly = pipeline.run_analysis(
        frame,
        "2020-01",
        "2020-04",
        "2020-05",
        "2020-08",
        None,
        0.0,
        periods_per_year=52,
        **common,
    ).unwrap()

    assert weekly["out_sample_stats"]["A"].vol > monthly["out_sample_stats"]["A"].vol
    assert weekly["out_sample_stats"]["A"].sortino > monthly["out_sample_stats"]["A"].sortino
    assert (
        weekly["out_sample_stats"]["A"].information_ratio
        > monthly["out_sample_stats"]["A"].information_ratio
    )
    assert weekly["out_sample_stats"]["A"].sharpe > monthly["out_sample_stats"]["A"].sharpe
    assert weekly["out_ew_stats"].vol > monthly["out_ew_stats"].vol
    assert weekly["out_user_stats"].vol > monthly["out_user_stats"].vol
    assert (
        weekly["benchmark_stats"]["bench"]["out_sample"].vol
        > monthly["benchmark_stats"]["bench"]["out_sample"].vol
    )
    assert weekly["benchmark_ir"]["bench"]["A"] > monthly["benchmark_ir"]["bench"]["A"]
