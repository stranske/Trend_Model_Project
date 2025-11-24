import pandas as pd
import pytest

from trend_analysis.pipeline import (
    PipelineReasonCode,
    _assemble_analysis_output,
    _build_sample_windows,
    _compute_weights_and_stats,
    _prepare_preprocess_stage,
    _select_universe,
)
from trend_analysis.pipeline import RiskStatsConfig


def _make_simple_frame() -> pd.DataFrame:
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


def test_prepare_preprocess_stage_rejects_missing_dates() -> None:
    df = pd.DataFrame({"Date": [pd.NaT, pd.NaT], "A": [1, 2]})

    result = _prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=RiskStatsConfig(risk_free=0.0),
        periods_per_year_override=None,
    )

    assert isinstance(result.diagnostic.reason_code, str)
    assert result.diagnostic.reason_code == PipelineReasonCode.NO_VALID_DATES.value


def test_prepare_preprocess_stage_returns_payload() -> None:
    df = _make_simple_frame()

    stage = _prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=1,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=RiskStatsConfig(risk_free=0.0),
        periods_per_year_override=None,
    )

    assert stage.freq_summary.code == "M"
    assert stage.preprocess_info["summary"].startswith("Cadence")
    assert stage.warmup == 1


def test_build_sample_windows_handles_empty_slice() -> None:
    df = _make_simple_frame()
    stats_cfg = RiskStatsConfig(metrics_to_run=["Return"], risk_free=0.0)
    preprocess = _prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=stats_cfg,
        periods_per_year_override=None,
    )

    window = _build_sample_windows(
        preprocess,
        in_start="2019-01-01",
        in_end="2019-12-31",
        out_start="2020-12-31",
        out_end="2020-12-31",
    )

    assert isinstance(window.diagnostic.reason_code, str)
    assert window.diagnostic.reason_code == PipelineReasonCode.SAMPLE_WINDOW_EMPTY.value


def test_select_universe_reports_missing_funds() -> None:
    df = _make_simple_frame()
    preprocess = _prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=RiskStatsConfig(risk_free=0.0),
        periods_per_year_override=None,
    )
    window = _build_sample_windows(
        preprocess,
        in_start="2020-01-31",
        in_end="2020-03-31",
        out_start="2020-04-30",
        out_end="2020-06-30",
    )

    selection = _select_universe(
        preprocess,
        window,
        in_label="2020-01-31",
        in_end_label="2020-03-31",
        selection_mode="manual",
        random_n=1,
        custom_weights=None,
        rank_kwargs=None,
        manual_funds=[],
        indices_list=None,
        seed=1,
        stats_cfg=RiskStatsConfig(risk_free=0.0),
        risk_free_column="rf",
        allow_risk_free_fallback=True,
    )

    assert selection.diagnostic.reason_code == PipelineReasonCode.NO_FUNDS_SELECTED.value


def test_compute_weights_and_stats_produces_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _make_simple_frame()
    stats_cfg = RiskStatsConfig(metrics_to_run=["Sharpe"], risk_free=0.0)

    def _fake_single_period_run(df: pd.DataFrame, *_: object, **__: object) -> pd.DataFrame:
        cols = [c for c in df.columns if c != "Date"]
        return pd.DataFrame({"Sharpe": [0.1] * len(cols)}, index=cols)

    monkeypatch.setattr("trend_analysis.pipeline.single_period_run", _fake_single_period_run)
    preprocess = _prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=stats_cfg,
        periods_per_year_override=None,
    )
    window = _build_sample_windows(
        preprocess,
        in_start="2020-01-31",
        in_end="2020-03-31",
        out_start="2020-04-30",
        out_end="2020-06-30",
    )
    selection = _select_universe(
        preprocess,
        window,
        in_label="2020-01-31",
        in_end_label="2020-03-31",
        selection_mode="all",
        random_n=1,
        custom_weights=None,
        rank_kwargs=None,
        manual_funds=None,
        indices_list=None,
        seed=1,
        stats_cfg=stats_cfg,
        risk_free_column="rf",
        allow_risk_free_fallback=True,
    )
    computation = _compute_weights_and_stats(
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
    )

    assert set(computation.in_stats.keys()) == set(selection.fund_cols)
    assert computation.risk_diagnostics.scale_factors.shape[0] == len(selection.fund_cols)


def test_assemble_analysis_output_wraps_success(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _make_simple_frame()
    stats_cfg = RiskStatsConfig(metrics_to_run=["Sharpe"], risk_free=0.0)

    def _fake_single_period_run(df: pd.DataFrame, *_: object, **__: object) -> pd.DataFrame:
        cols = [c for c in df.columns if c != "Date"]
        return pd.DataFrame({"Sharpe": [0.1] * len(cols)}, index=cols)

    monkeypatch.setattr("trend_analysis.pipeline.single_period_run", _fake_single_period_run)
    preprocess = _prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=stats_cfg,
        periods_per_year_override=None,
    )
    window = _build_sample_windows(
        preprocess,
        in_start="2020-01-31",
        in_end="2020-03-31",
        out_start="2020-04-30",
        out_end="2020-06-30",
    )
    selection = _select_universe(
        preprocess,
        window,
        in_label="2020-01-31",
        in_end_label="2020-03-31",
        selection_mode="all",
        random_n=1,
        custom_weights=None,
        rank_kwargs=None,
        manual_funds=None,
        indices_list=None,
        seed=1,
        stats_cfg=stats_cfg,
        risk_free_column="rf",
        allow_risk_free_fallback=True,
    )
    computation = _compute_weights_and_stats(
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
    )

    result = _assemble_analysis_output(
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

    assert result.diagnostic is None
    assert result.value is not None
    assert set(result.value["selected_funds"]) == set(selection.fund_cols)

