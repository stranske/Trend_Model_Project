"""preprocessing.winsorise must change processed returns when enabled."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trend_analysis.config import Config
from trend_analysis.core.rank_selection import RiskStatsConfig, clear_window_metric_cache
from trend_analysis.multi_period import run as run_mp
from trend_analysis.pipeline_helpers import _attach_calendar_settings
from trend_analysis.pipeline_runner import _run_analysis_with_diagnostics
from trend_analysis.diagnostics import PipelineResult
from trend_analysis.stages.preprocessing import (
    _PreprocessStage,
    _WindowStage,
    _build_sample_windows,
    _prepare_preprocess_stage,
    _resolve_winsorise_settings,
)

OUTLIER = 5.0  # +500% monthly return
_UNSET = object()


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(7)
    fund_a = rng.normal(0.01, 0.02, len(dates))
    fund_a[-1] = OUTLIER  # extreme outlier on the tip
    fund_b = rng.normal(0.005, 0.015, len(dates))
    return pd.DataFrame({"Date": dates, "FundA": fund_a, "FundB": fund_b})


def _run(
    enabled: bool,
    limits: list[float] | object = _UNSET,
) -> _PreprocessStage:
    winsorise: dict[str, object] = {"enabled": enabled}
    if limits is not _UNSET:
        winsorise["limits"] = limits
    cfg = {
        "data": {"frequency": "M", "timezone": "UTC"},
        "preprocessing": {"winsorise": winsorise},
    }
    df = _frame()
    _attach_calendar_settings(df, cfg)
    stage = _prepare_preprocess_stage(
        df,
        floor_vol=None,
        warmup_periods=0,
        missing_policy=None,
        missing_limit=None,
        stats_cfg=RiskStatsConfig(risk_free=0.0),
        periods_per_year_override=None,
        allow_risk_free_fallback=None,
    )
    assert isinstance(stage, _PreprocessStage)
    return stage


def test_winsorise_limits_clip_outlier_returns() -> None:
    off = _run(enabled=False)
    on = _run(enabled=True, limits=[0.05, 0.95])

    tip_off = float(off.df["FundA"].iloc[-1])
    tip_on = float(on.df["FundA"].iloc[-1])

    assert tip_off == pytest.approx(OUTLIER)
    assert tip_on < tip_off
    expected_cap = float(off.df["FundA"].quantile(0.95))
    assert tip_on == pytest.approx(expected_cap)
    assert not on.df["FundA"].equals(off.df["FundA"])
    # Every column is clipped to its own quantile band.
    for col in ("FundA", "FundB"):
        lo, hi = off.df[col].quantile([0.05, 0.95])
        assert on.df[col].min() >= lo - 1e-12
        assert on.df[col].max() <= hi + 1e-12
    assert on.preprocess_info["winsorise"]["enabled"] is True
    assert on.preprocess_info["winsorise"]["clipped_cells"] > 0
    assert off.preprocess_info["winsorise"]["enabled"] is False
    assert off.preprocess_info["winsorise"]["clipped_cells"] == 0


def test_omitted_limits_default_quantiles_clip_outliers() -> None:
    resolved = _resolve_winsorise_settings({"winsorise": {"enabled": True}})
    assert resolved == {"enabled": True, "limits": (0.01, 0.99)}

    off = _run(enabled=False)
    on = _run(enabled=True)
    assert float(on.df["FundA"].iloc[-1]) < float(off.df["FundA"].iloc[-1])
    cap = float(off.df["FundA"].quantile(0.99))
    assert float(on.df["FundA"].iloc[-1]) == pytest.approx(cap)


def test_shipped_default_limits_still_clip_tip_outlier() -> None:
    off = _run(enabled=False)
    on = _run(enabled=True, limits=[0.01, 0.99])
    assert float(on.df["FundA"].iloc[-1]) < float(off.df["FundA"].iloc[-1])


def _causal_frame(*, out_may_return: float) -> pd.DataFrame:
    dates = pd.date_range("2019-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(11)
    body = rng.normal(0.01, 0.02, len(dates))
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": body,
            "FundB": rng.normal(0.005, 0.015, len(dates)),
        }
    )
    df.loc[df["Date"] == pd.Timestamp("2020-05-31"), "FundA"] = out_may_return
    return df


def test_out_of_sample_change_does_not_alter_in_sample_window() -> None:
    cfg = {
        "data": {"frequency": "M", "timezone": "UTC", "allow_risk_free_fallback": True},
        "preprocessing": {"winsorise": {"enabled": True, "limits": [0.05, 0.95]}},
    }

    def _in_sample_col(frame: pd.DataFrame) -> pd.Series:
        stage = _prepare_preprocess_stage(
            frame,
            floor_vol=None,
            warmup_periods=0,
            missing_policy=None,
            missing_limit=None,
            stats_cfg=RiskStatsConfig(risk_free=0.0),
            periods_per_year_override=None,
            allow_risk_free_fallback=True,
        )
        assert isinstance(stage, _PreprocessStage)
        window = _build_sample_windows(
            stage,
            in_start="2019-01",
            in_end="2020-04",
            out_start="2020-05",
            out_end="2020-06",
        )
        assert not isinstance(window, PipelineResult)
        assert isinstance(window, _WindowStage)
        return window.in_df["FundA"].copy()

    mild = _causal_frame(out_may_return=0.04)
    extreme = _causal_frame(out_may_return=5.0)
    _attach_calendar_settings(mild, cfg)
    _attach_calendar_settings(extreme, cfg)

    in_mild = _in_sample_col(mild)
    in_extreme = _in_sample_col(extreme)
    pd.testing.assert_series_equal(in_mild, in_extreme)


def test_pipeline_causal_invariance_via_run_analysis_with_diagnostics() -> None:
    cfg = {
        "data": {"frequency": "M", "timezone": "UTC", "allow_risk_free_fallback": True},
        "preprocessing": {"winsorise": {"enabled": True, "limits": [0.05, 0.95]}},
    }

    def _run_diag(frame: pd.DataFrame) -> pd.DataFrame:
        _attach_calendar_settings(frame, cfg)
        result = _run_analysis_with_diagnostics(
            frame,
            in_start="2019-01",
            in_end="2020-04",
            out_start="2020-05",
            out_end="2020-06",
            target_vol=None,
            monthly_cost=0.0,
            stats_cfg=RiskStatsConfig(risk_free=0.0),
            allow_risk_free_fallback=True,
            manual_funds=["FundA"],
            selection_mode="manual",
        )
        assert isinstance(result, PipelineResult) and result.value is not None
        scaled = result.value.get("in_sample_scaled")
        assert isinstance(scaled, pd.DataFrame)
        return scaled

    mild = _causal_frame(out_may_return=0.04)
    extreme = _causal_frame(out_may_return=5.0)
    s_mild = _run_diag(mild)
    s_extreme = _run_diag(extreme)
    if "FundA" in s_mild.columns and "FundA" in s_extreme.columns:
        pd.testing.assert_series_equal(
            s_mild["FundA"],
            s_extreme["FundA"],
            check_names=False,
        )


def _rank_causal_frame(*, out_return: float) -> pd.DataFrame:
    dates = pd.date_range("2019-01-31", periods=26, freq="ME")
    rng = np.random.default_rng(23)
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": rng.normal(0.012, 0.02, len(dates)),
            "FundB": rng.normal(0.010, 0.02, len(dates)),
            "FundC": rng.normal(0.008, 0.02, len(dates)),
            "FundD": rng.normal(0.006, 0.02, len(dates)),
        }
    )
    # Only the out-of-sample rows differ between scenarios.
    out_mask = df["Date"] >= pd.Timestamp("2021-01-31")
    df.loc[out_mask, ["FundA", "FundB", "FundC", "FundD"]] = out_return
    return df


def test_rank_selection_scores_and_weights_ignore_out_of_sample_returns() -> None:
    cfg = {
        "data": {"frequency": "M", "timezone": "UTC", "allow_risk_free_fallback": True},
        "preprocessing": {"winsorise": {"enabled": True, "limits": [0.1, 0.9]}},
    }

    def _run_rank(frame: pd.DataFrame) -> dict:
        clear_window_metric_cache()
        _attach_calendar_settings(frame, cfg)
        result = _run_analysis_with_diagnostics(
            frame,
            in_start="2019-01",
            in_end="2020-12",
            out_start="2021-01",
            out_end="2021-02",
            target_vol=None,
            monthly_cost=0.0,
            stats_cfg=RiskStatsConfig(risk_free=0.0),
            allow_risk_free_fallback=True,
            selection_mode="rank",
            rank_kwargs={"inclusion_approach": "top_n", "n": 2, "score_by": "Sharpe"},
            weighting_scheme="score_prop",
        )
        assert isinstance(result, PipelineResult) and result.value is not None
        return result.value

    mild = _run_rank(_rank_causal_frame(out_return=0.01))
    extreme = _run_rank(_rank_causal_frame(out_return=-0.5))
    clear_window_metric_cache()

    pd.testing.assert_frame_equal(mild["score_frame"], extreme["score_frame"])
    assert mild["selected_funds"] == extreme["selected_funds"]
    assert mild["fund_weights"] == pytest.approx(extreme["fund_weights"])
    pd.testing.assert_frame_equal(mild["in_sample_scaled"], extreme["in_sample_scaled"])


def _threshold_hold_config(*, winsorise_enabled: bool) -> Config:
    cfg_data = yaml.safe_load(Path("config/defaults.yml").read_text())
    cfg_data.setdefault("data", {})["allow_risk_free_fallback"] = True
    cfg_data["preprocessing"] = {
        "winsorise": {"enabled": winsorise_enabled, "limits": [0.05, 0.95]},
    }
    cfg_data["multi_period"] = {
        "frequency": "M",
        "in_sample_len": 3,
        "out_sample_len": 1,
        "start": "2020-01",
        "end": "2020-04",
    }
    portfolio = cfg_data.setdefault("portfolio", {})
    portfolio["policy"] = "threshold_hold"
    portfolio["max_turnover"] = 1.0
    portfolio.setdefault("threshold_hold", {}).update({"target_n": 2, "metric": "Sharpe"})
    portfolio.setdefault("constraints", {}).update(
        {"max_funds": 2, "min_weight": 0.05, "max_weight": 0.95}
    )
    portfolio["weighting"] = {"name": "adaptive_bayes", "params": {}}
    return Config(**cfg_data)


def _threshold_hold_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "Alpha": [OUTLIER, 0.02, 0.021, 0.02],
            "Beta": [0.01, 0.011, 0.012, 0.013],
            "Gamma": [0.015, 0.016, 0.017, 0.018],
        }
    )


def test_threshold_hold_winsorise_changes_scores_weights_and_metrics() -> None:
    df = _threshold_hold_df()
    off = run_mp(_threshold_hold_config(winsorise_enabled=False), df=df.copy())
    on = run_mp(_threshold_hold_config(winsorise_enabled=True), df=df.copy())
    assert off and on
    # The selection score frame is computed from ``_valid_universe`` output before the
    # pipeline call, so it only changes if threshold-hold inputs are clipped.
    sf_off = off[0]["selection_score_frame"]
    sf_on = on[0]["selection_score_frame"]
    assert "Alpha" in sf_off.index and "Alpha" in sf_on.index
    assert not sf_off.loc["Alpha"].equals(sf_on.loc["Alpha"])
    assert (
        off[0]["fund_weights"] != on[0]["fund_weights"]
        or off[0]["in_sample_stats"] != on[0]["in_sample_stats"]
    )


@pytest.mark.parametrize("limits", [[0.9, 0.1], [0.5], [-0.1, 0.9], ["a", "b"]])
def test_invalid_limits_fail_loudly(limits: list) -> None:
    with pytest.raises(ValueError, match="winsorise.limits"):
        _resolve_winsorise_settings({"winsorise": {"enabled": True, "limits": limits}})


def test_disabled_ignores_limits() -> None:
    assert _resolve_winsorise_settings({"winsorise": {"enabled": False, "limits": [0.9, 0.1]}}) == {
        "enabled": False,
        "limits": None,
    }
