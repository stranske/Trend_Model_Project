import warnings

import pandas as pd
import pytest

from trend_analysis.pipeline import (
    PipelineReasonCode,
    PipelineResult,
    RiskStatsConfig,
    _assemble_analysis_output,
    _build_sample_windows,
    _compute_weights_and_stats,
    _prepare_preprocess_stage,
    _select_universe,
    _WindowStage,
)


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
        allow_risk_free_fallback=None,
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
        allow_risk_free_fallback=None,
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
        allow_risk_free_fallback=None,
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


def test_build_sample_windows_preserves_tz_without_warnings() -> None:
    df = _make_simple_frame()
    preprocess = _prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=RiskStatsConfig(risk_free=0.0),
        periods_per_year_override=None,
        allow_risk_free_fallback=None,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        window = _build_sample_windows(
            preprocess,
            in_start="2020-01-31",
            in_end="2020-03-31",
            out_start="2020-04-30",
            out_end="2020-06-30",
        )

    assert isinstance(window, _WindowStage)
    assert window.in_start == pd.Timestamp("2020-01-31")
    assert window.out_end == pd.Timestamp("2020-06-30")
    assert window.in_df.index.tz is None
    assert window.out_df.index.tz is None


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
        allow_risk_free_fallback=None,
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

    assert (
        selection.diagnostic.reason_code == PipelineReasonCode.NO_FUNDS_SELECTED.value
    )


def test_select_universe_rejects_unknown_indices() -> None:
    df = _make_simple_frame()
    preprocess = _prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=RiskStatsConfig(risk_free=0.0),
        periods_per_year_override=None,
        allow_risk_free_fallback=None,
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
        indices_list=["Missing"],
        seed=1,
        stats_cfg=RiskStatsConfig(risk_free=0.0),
        risk_free_column="rf",
        allow_risk_free_fallback=True,
    )

    assert isinstance(selection, PipelineResult)
    assert selection.diagnostic is not None
    assert selection.diagnostic.reason_code == PipelineReasonCode.INDICES_ABSENT.value
    assert selection.diagnostic.context == {
        "requested_indices": ["Missing"],
        "missing_indices": ["Missing"],
        "available_columns": ["A", "B", "rf"],
    }


def test_compute_weights_and_stats_produces_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _make_simple_frame()
    stats_cfg = RiskStatsConfig(metrics_to_run=["Sharpe"], risk_free=0.0)

    def _fake_single_period_run(
        df: pd.DataFrame, *_: object, **__: object
    ) -> pd.DataFrame:
        cols = [c for c in df.columns if c != "Date"]
        return pd.DataFrame({"Sharpe": [0.1] * len(cols)}, index=cols)

    monkeypatch.setattr(
        "trend_analysis.pipeline.single_period_run", _fake_single_period_run
    )
    preprocess = _prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=stats_cfg,
        periods_per_year_override=None,
        allow_risk_free_fallback=None,
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
    assert computation.risk_diagnostics.scale_factors.shape[0] == len(
        selection.fund_cols
    )


def test_compute_weights_scopes_signal_inputs_to_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _make_simple_frame()
    extra = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-07-31", tz="UTC")],
            "A": [0.02],
            "B": [0.01],
            "rf": [0.001],
        }
    )
    df = pd.concat([base, extra], ignore_index=True)
    df.attrs["calendar_settings"] = {"timezone": None}
    stats_cfg = RiskStatsConfig(metrics_to_run=["Sharpe"], risk_free=0.0)

    def _fake_single_period_run(
        df: pd.DataFrame, *_: object, **__: object
    ) -> pd.DataFrame:
        cols = [c for c in df.columns if c != "Date"]
        return pd.DataFrame({"Sharpe": [0.1] * len(cols)}, index=cols)

    monkeypatch.setattr(
        "trend_analysis.pipeline.single_period_run", _fake_single_period_run
    )

    observed: dict[str, pd.Timestamp] = {}

    def _fake_compute_trend_signals(
        df: pd.DataFrame, *_: object, **__: object
    ) -> pd.DataFrame:
        observed["min"] = df.index.min()
        observed["max"] = df.index.max()
        return pd.DataFrame(0.0, index=df.index, columns=df.columns)

    monkeypatch.setattr(
        "trend_analysis.pipeline.compute_trend_signals", _fake_compute_trend_signals
    )

    preprocess = _prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=stats_cfg,
        periods_per_year_override=None,
        allow_risk_free_fallback=None,
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

    _compute_weights_and_stats(
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

    assert observed["min"] == window.in_df.index.min()
    assert observed["max"] == window.out_df.index.max()


def test_compute_weights_rejects_out_of_window_signal_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _make_simple_frame()
    stats_cfg = RiskStatsConfig(metrics_to_run=["Sharpe"], risk_free=0.0)

    def _fake_single_period_run(
        df: pd.DataFrame, *_: object, **__: object
    ) -> pd.DataFrame:
        cols = [c for c in df.columns if c != "Date"]
        return pd.DataFrame({"Sharpe": [0.1] * len(cols)}, index=cols)

    monkeypatch.setattr(
        "trend_analysis.pipeline.single_period_run", _fake_single_period_run
    )
    preprocess = _prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=stats_cfg,
        periods_per_year_override=None,
        allow_risk_free_fallback=None,
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

    future_row = window.out_df.iloc[[-1]].copy()
    future_row.index = [window.out_end + pd.offsets.MonthEnd()]
    tampered_out = pd.concat([window.out_df, future_row]).sort_index()
    tampered_window = _WindowStage(
        in_df=window.in_df,
        out_df=tampered_out,
        in_start=window.in_start,
        in_end=window.in_end,
        out_start=window.out_start,
        out_end=window.out_end,
        periods_per_year=window.periods_per_year,
        date_col=window.date_col,
    )

    with pytest.raises(ValueError, match="active analysis window"):
        _compute_weights_and_stats(
            preprocess,
            tampered_window,
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


def test_assemble_analysis_output_wraps_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _make_simple_frame()
    stats_cfg = RiskStatsConfig(metrics_to_run=["Sharpe"], risk_free=0.0)

    def _fake_single_period_run(
        df: pd.DataFrame, *_: object, **__: object
    ) -> pd.DataFrame:
        cols = [c for c in df.columns if c != "Date"]
        return pd.DataFrame({"Sharpe": [0.1] * len(cols)}, index=cols)

    monkeypatch.setattr(
        "trend_analysis.pipeline.single_period_run", _fake_single_period_run
    )
    preprocess = _prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=stats_cfg,
        periods_per_year_override=None,
        allow_risk_free_fallback=None,
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
