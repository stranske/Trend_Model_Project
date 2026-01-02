from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from trend_analysis import pipeline
from trend_analysis.core import rank_selection as rank_selection_mod
from trend_analysis.core.rank_selection import RiskStatsConfig
from trend_analysis.engine import optimizer as optimizer_mod


def _base_returns_frame() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-29",
            "2020-03-31",
            "2020-04-30",
        ]
    )
    return pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.02, np.nan, 0.03, 0.01],
            "FundB": [0.015, 0.016, 0.014, 0.013],
            "Benchmark": [0.005, 0.004, 0.006, 0.005],
            "RF": [0.001, 0.001, 0.001, 0.001],
        }
    )


def _clean_returns_frame() -> pd.DataFrame:
    frame = _base_returns_frame().copy()
    frame.loc[:, "FundA"] = [0.02, 0.021, 0.022, 0.023]
    return frame


def _regime_returns_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.02, 0.021, 0.019, 0.02, 0.021, 0.02],
            "FundB": [0.015, 0.014, 0.016, 0.015, 0.014, 0.015],
            "FundC": [0.025, 0.026, 0.024, 0.025, 0.026, 0.025],
            "FundD": [0.01, 0.011, 0.009, 0.01, 0.011, 0.01],
            "SPX": [-0.02, -0.03, -0.01, 0.02, 0.03, 0.01],
            "ACWI": [0.03, 0.02, 0.01, 0.0, 0.01, 0.02],
            "RF": [0.001] * 6,
        }
    )


RUN_KWARGS = {"risk_free_column": "RF", "allow_risk_free_fallback": False}


def test_single_period_run_injects_avg_corr_metric() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "FundA": [0.02, 0.01, 0.015],
            "FundB": [0.01, 0.012, 0.011],
        }
    )

    class RiskStatsConfigWithExtraMetrics(RiskStatsConfig):
        def __init__(self):
            super().__init__()
            self.extra_metrics = ["AvgCorr"]

    stats_cfg = RiskStatsConfigWithExtraMetrics()

    score_frame = pipeline.single_period_run(
        df, "2020-01", "2020-03", stats_cfg=stats_cfg
    )

    assert "AvgCorr" in score_frame.columns
    # AvgCorr column should contain finite values for the analysed funds.
    assert score_frame["AvgCorr"].notna().all()


def test_single_period_run_surfaces_avg_corr_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "FundA": [0.02, 0.01, 0.015],
            "FundB": [0.01, 0.012, 0.011],
        }
    )

    class StatsCfg(RiskStatsConfig):
        def __init__(self) -> None:
            super().__init__()
            self.extra_metrics = ["AvgCorr"]

    stats_cfg = StatsCfg()

    def boom(*_args, **_kwargs):
        raise RuntimeError("avg corr failure")

    monkeypatch.setattr(rank_selection_mod, "compute_metric_series_with_cache", boom)

    with pytest.raises(RuntimeError, match="AvgCorr"):
        pipeline.single_period_run(df, "2020-01", "2020-03", stats_cfg=stats_cfg)


def test_run_analysis_na_tolerant_filtering_preserves_funds() -> None:
    df = _base_returns_frame()
    stats_cfg = RiskStatsConfig()
    setattr(
        stats_cfg,
        "na_as_zero_cfg",
        {
            "enabled": True,
            "max_missing_per_window": 1,
            "max_consecutive_gap": 1,
        },
    )

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        **RUN_KWARGS,
    )

    assert result is not None
    assert "FundA" in result["selected_funds"]


def test_run_analysis_na_tolerant_filtering_drops_excessive_gaps() -> None:
    df = _base_returns_frame()
    stats_cfg = RiskStatsConfig()
    setattr(
        stats_cfg,
        "na_as_zero_cfg",
        {
            "enabled": True,
            "max_missing_per_window": 0,
            "max_consecutive_gap": 0,
        },
    )

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        **RUN_KWARGS,
    )

    assert result is None or "FundA" not in result.get("selected_funds", [])


def test_regime_enabled_changes_selection_count() -> None:
    df = _regime_returns_frame()
    rank_kwargs = {
        "inclusion_approach": "top_n",
        "n": 3,
        "score_by": "Sharpe",
        "transform": "raw",
    }
    base = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=1.0,
        monthly_cost=0.0,
        selection_mode="rank",
        rank_kwargs=rank_kwargs,
        regime_cfg={"enabled": False, "proxy": "SPX"},
        **RUN_KWARGS,
    )
    assert base is not None
    enabled = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=1.0,
        monthly_cost=0.0,
        selection_mode="rank",
        rank_kwargs=rank_kwargs,
        regime_cfg={"enabled": True, "proxy": "SPX"},
        **RUN_KWARGS,
    )
    assert enabled is not None
    assert len(base["selected_funds"]) == 3
    assert len(enabled["selected_funds"]) == 2


def test_regime_proxy_changes_selection_count() -> None:
    df = _regime_returns_frame()
    rank_kwargs = {
        "inclusion_approach": "top_n",
        "n": 3,
        "score_by": "Sharpe",
        "transform": "raw",
    }
    spx = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=1.0,
        monthly_cost=0.0,
        selection_mode="rank",
        rank_kwargs=rank_kwargs,
        regime_cfg={"enabled": True, "proxy": "SPX"},
        **RUN_KWARGS,
    )
    acwi = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=1.0,
        monthly_cost=0.0,
        selection_mode="rank",
        rank_kwargs=rank_kwargs,
        regime_cfg={"enabled": True, "proxy": "ACWI"},
        **RUN_KWARGS,
    )
    assert spx is not None
    assert acwi is not None
    assert len(spx["selected_funds"]) != len(acwi["selected_funds"])


def test_max_active_positions_caps_weights() -> None:
    df = _regime_returns_frame()
    rank_kwargs = {
        "inclusion_approach": "top_n",
        "n": 4,
        "score_by": "Sharpe",
        "transform": "raw",
    }
    constraints = {
        "long_only": True,
        "max_weight": 1.0,
        "max_active_positions": 2,
    }
    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=1.0,
        monthly_cost=0.0,
        selection_mode="rank",
        rank_kwargs=rank_kwargs,
        constraints=constraints,
        **RUN_KWARGS,
    )
    assert result is not None
    assert len(result["selected_funds"]) == 4
    weights = result["risk_diagnostics"]["final_weights"]
    active = weights[weights > 0]
    assert len(active) <= 2


def test_regime_proxy_resolves_benchmark_alias() -> None:
    df = _regime_returns_frame()
    rank_kwargs = {
        "inclusion_approach": "top_n",
        "n": 3,
        "score_by": "Sharpe",
        "transform": "raw",
    }
    baseline = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=1.0,
        monthly_cost=0.0,
        selection_mode="rank",
        rank_kwargs=rank_kwargs,
        benchmarks={"spx": "SPX"},
        regime_cfg={"enabled": False, "proxy": "spx"},
        **RUN_KWARGS,
    )
    enabled = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=1.0,
        monthly_cost=0.0,
        selection_mode="rank",
        rank_kwargs=rank_kwargs,
        benchmarks={"spx": "SPX"},
        regime_cfg={
            "enabled": True,
            "proxy": "spx",
            "lookback": 1,
            "smoothing": 1,
            "threshold": 0.0,
            "neutral_band": 0.0,
            "min_observations": 1,
        },
        **RUN_KWARGS,
    )
    assert baseline is not None
    assert enabled is not None
    assert len(baseline["selected_funds"]) == 3
    assert len(enabled["selected_funds"]) == 2


def test_regime_enabled_scales_top_pct_selection_count() -> None:
    df = _regime_returns_frame()
    rank_kwargs = {
        "inclusion_approach": "top_pct",
        "pct": 0.5,
        "score_by": "Sharpe",
        "transform": "raw",
    }
    base = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=1.0,
        monthly_cost=0.0,
        selection_mode="rank",
        rank_kwargs=rank_kwargs,
        regime_cfg={"enabled": False, "proxy": "SPX"},
        **RUN_KWARGS,
    )
    enabled = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=1.0,
        monthly_cost=0.0,
        selection_mode="rank",
        rank_kwargs=rank_kwargs,
        regime_cfg={"enabled": True, "proxy": "SPX"},
        **RUN_KWARGS,
    )
    assert base is not None
    assert enabled is not None
    assert len(base["selected_funds"]) == 3
    assert len(enabled["selected_funds"]) == 2


def test_regime_enabled_scales_random_selection_count() -> None:
    df = _regime_returns_frame()
    base = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=1.0,
        monthly_cost=0.0,
        selection_mode="random",
        random_n=4,
        regime_cfg={"enabled": False, "proxy": "SPX"},
        **RUN_KWARGS,
    )
    enabled = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=1.0,
        monthly_cost=0.0,
        selection_mode="random",
        random_n=4,
        regime_cfg={"enabled": True, "proxy": "SPX"},
        **RUN_KWARGS,
    )
    assert base is not None
    assert enabled is not None
    assert len(base["selected_funds"]) == 4
    assert len(enabled["selected_funds"]) == 2


# Default multiplier from config/defaults.yml regime.risk_off_target_vol_multiplier
RISK_OFF_TARGET_VOL_MULTIPLIER = 0.5


def test_regime_enabled_scales_target_vol_in_all_mode() -> None:
    df = _regime_returns_frame()
    base = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=1.0,
        monthly_cost=0.0,
        selection_mode="all",
        indices_list=["SPX", "ACWI"],
        regime_cfg={"enabled": False, "proxy": "SPX"},
        **RUN_KWARGS,
    )
    enabled = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        target_vol=1.0,
        monthly_cost=0.0,
        selection_mode="all",
        indices_list=["SPX", "ACWI"],
        regime_cfg={"enabled": True, "proxy": "SPX"},
        **RUN_KWARGS,
    )
    assert base is not None
    assert enabled is not None
    assert base["selected_funds"] == enabled["selected_funds"]
    base_scale = base["risk_diagnostics"]["scale_factors"]
    enabled_scale = enabled["risk_diagnostics"]["scale_factors"].reindex(
        base_scale.index
    )
    mask = base_scale > 0
    ratio = enabled_scale[mask] / base_scale[mask]
    assert np.allclose(ratio, RISK_OFF_TARGET_VOL_MULTIPLIER, rtol=1e-3, atol=1e-3)


def test_run_analysis_avg_corr_metrics_populate_stats() -> None:
    df = _clean_returns_frame()
    stats_cfg = RiskStatsConfig()
    stats_cfg.metrics_to_run = list(stats_cfg.metrics_to_run) + ["AvgCorr"]
    setattr(stats_cfg, "extra_metrics", ["AvgCorr"])

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        **RUN_KWARGS,
    )

    assert result is not None
    in_stats = result["in_sample_stats"]["FundA"]
    out_stats = result["out_sample_stats"]["FundA"]

    assert in_stats.is_avg_corr is not None
    assert in_stats.os_avg_corr is None
    assert out_stats.os_avg_corr is not None
    assert out_stats.is_avg_corr is None


def test_run_analysis_skips_avg_corr_for_single_fund() -> None:
    df = _clean_returns_frame()[["Date", "FundA", "Benchmark", "RF"]].copy()
    stats_cfg = RiskStatsConfig()
    stats_cfg.metrics_to_run = list(stats_cfg.metrics_to_run) + ["AvgCorr"]
    setattr(stats_cfg, "extra_metrics", ["AvgCorr"])

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        **RUN_KWARGS,
    )

    assert result is not None
    in_stats = result["in_sample_stats"]["FundA"]
    out_stats = result["out_sample_stats"]["FundA"]
    assert in_stats.is_avg_corr is None
    assert out_stats.os_avg_corr is None


def test_run_analysis_does_not_duplicate_existing_avg_corr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _clean_returns_frame()
    stats_cfg = RiskStatsConfig()
    stats_cfg.metrics_to_run = list(stats_cfg.metrics_to_run) + ["AvgCorr", "AvgCorr"]

    def fake_single_period_run(*args, **kwargs) -> pd.DataFrame:
        frame = pd.DataFrame(
            {"Sharpe": [1.0, 0.5], "AvgCorr": [0.1, 0.2]},
            index=["FundA", "FundB"],
        )
        frame.attrs["insample_len"] = 2
        frame.attrs["period"] = ("2020-01", "2020-02")
        return frame

    monkeypatch.setattr(pipeline, "single_period_run", fake_single_period_run)

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        **RUN_KWARGS,
    )

    assert result is not None
    score_frame = result["score_frame"]
    assert list(score_frame.columns).count("AvgCorr") == 1
    pd.testing.assert_index_equal(score_frame.index, pd.Index(["FundA", "FundB"]))


def test_run_analysis_avg_corr_corr_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _clean_returns_frame()
    stats_cfg = RiskStatsConfig()
    stats_cfg.metrics_to_run = list(stats_cfg.metrics_to_run) + ["AvgCorr"]
    setattr(stats_cfg, "extra_metrics", ["AvgCorr"])

    original_corr = pd.DataFrame.corr

    def flaky_corr(self, *args, **kwargs):  # type: ignore[override]
        caller = inspect.stack()[1]
        if caller.filename.endswith("pipeline.py"):
            raise RuntimeError("corr failure")
        return original_corr(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "corr", flaky_corr)

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        **RUN_KWARGS,
    )

    assert result is not None
    in_stats = result["in_sample_stats"]["FundA"]
    out_stats = result["out_sample_stats"]["FundA"]
    assert in_stats.is_avg_corr is None
    assert out_stats.os_avg_corr is None


def test_run_analysis_constraint_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _clean_returns_frame()
    stats_cfg = RiskStatsConfig()

    def boom(*_args, **_kwargs):
        raise RuntimeError("constraint failure")

    monkeypatch.setattr(optimizer_mod, "apply_constraints", boom)

    custom_weights = {"FundA": 60.0, "FundB": 40.0}
    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        custom_weights=custom_weights,
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        constraints={"max_weight": 0.5},
        **RUN_KWARGS,
    )

    assert result is not None
    weights = result["fund_weights"]
    assert weights["FundA"] == pytest.approx(0.6)
    assert weights["FundB"] == pytest.approx(0.4)


def test_run_analysis_applies_constraints_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _clean_returns_frame()
    stats_cfg = RiskStatsConfig()

    captured_inputs: list[pd.Series] = []

    def succeed(weights: pd.Series, constraints: dict[str, object]) -> pd.Series:
        captured_inputs.append(weights.copy())
        # Return skewed weights that should propagate to the result payload.
        return pd.Series({"FundA": 0.7, "FundB": 0.3}, dtype=float)

    monkeypatch.setattr(optimizer_mod, "apply_constraints", succeed)

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        custom_weights={"FundA": 55.0, "FundB": 45.0},
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        constraints={"max_weight": 0.8},
        **RUN_KWARGS,
    )

    assert result is not None
    weights = result["fund_weights"]
    assert weights["FundA"] == pytest.approx(0.7)
    assert weights["FundB"] == pytest.approx(0.3)
    assert captured_inputs and list(captured_inputs[0].index) == ["FundA", "FundB"]


def test_run_analysis_constraint_violation_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _clean_returns_frame()
    stats_cfg = RiskStatsConfig()

    def raise_violation(*_args, **_kwargs):
        raise optimizer_mod.ConstraintViolation("no feasible weights")

    monkeypatch.setattr(optimizer_mod, "apply_constraints", raise_violation)

    custom_weights = {"FundA": 70.0, "FundB": 30.0}
    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        custom_weights=custom_weights,
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        constraints={"max_weight": 0.4},
        **RUN_KWARGS,
    )

    assert result is not None
    weights = result["fund_weights"]
    assert weights["FundA"] == pytest.approx(0.7)
    assert weights["FundB"] == pytest.approx(0.3)


def test_run_analysis_constraint_missing_groups_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _clean_returns_frame()
    stats_cfg = RiskStatsConfig()

    def raise_key_error(*_args, **_kwargs):
        raise KeyError("Missing group mapping for assets: FundA")

    monkeypatch.setattr(optimizer_mod, "apply_constraints", raise_key_error)

    custom_weights = {"FundA": 55.0, "FundB": 45.0}
    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        custom_weights=custom_weights,
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        constraints={"group_caps": {"Tech": 0.6}},
        **RUN_KWARGS,
    )

    assert result is not None
    weights = result["fund_weights"]
    assert weights["FundA"] == pytest.approx(0.55)
    assert weights["FundB"] == pytest.approx(0.45)


def test_run_analysis_benchmark_ir_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _clean_returns_frame()
    stats_cfg = RiskStatsConfig()

    original_calc = pipeline.calc_portfolio_returns

    def tagging_calc(weights: np.ndarray, returns_df: pd.DataFrame) -> pd.Series:
        series = original_calc(weights, returns_df)
        if np.allclose(weights, np.repeat(1.0 / len(weights), len(weights))):
            series.attrs["portfolio_role"] = "equal_weight"
        else:
            series.attrs["portfolio_role"] = "user_weight"
        return series

    original_ir = pipeline.information_ratio

    def flaky_information_ratio(a, b):
        if isinstance(a, pd.Series) and a.attrs.get("portfolio_role") == "equal_weight":
            raise RuntimeError("portfolio IR failure")
        return original_ir(a, b)

    monkeypatch.setattr(pipeline, "calc_portfolio_returns", tagging_calc)
    monkeypatch.setattr(pipeline, "information_ratio", flaky_information_ratio)

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        custom_weights={"FundA": 55.0, "FundB": 45.0},
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        **RUN_KWARGS,
    )

    assert result is not None
    ir_payload = result["benchmark_ir"].get("SPX", {})
    # Fallback path should skip enriching the portfolio-level IR keys.
    assert "equal_weight" not in ir_payload
    assert "user_weight" not in ir_payload


def test_run_analysis_benchmark_ir_handles_scalar_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _clean_returns_frame()
    stats_cfg = RiskStatsConfig()

    def scalar_information_ratio(*_args, **_kwargs):
        return 0.42

    monkeypatch.setattr(pipeline, "information_ratio", scalar_information_ratio)

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        custom_weights={"FundA": 55.0, "FundB": 45.0},
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        **RUN_KWARGS,
    )

    assert result is not None
    ir_payload = result["benchmark_ir"].get("SPX", {})
    assert ir_payload["FundA"] == pytest.approx(0.42)
    assert ir_payload["equal_weight"] == pytest.approx(0.42)
    assert ir_payload["user_weight"] == pytest.approx(0.42)


def test_run_analysis_benchmark_ir_populates_portfolio_entries() -> None:
    df = _clean_returns_frame()
    stats_cfg = RiskStatsConfig()

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        custom_weights={"FundA": 60.0, "FundB": 40.0},
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        **RUN_KWARGS,
    )

    payload = result["benchmark_ir"].get("SPX")
    assert payload is not None
    assert {"equal_weight", "user_weight"}.issubset(payload)
    assert np.isfinite(payload["equal_weight"])
    assert np.isfinite(payload["user_weight"])


def test_run_analysis_benchmark_ir_handles_scalar_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _clean_returns_frame()
    stats_cfg = RiskStatsConfig()

    calls: list[str] = []

    original_ir = pipeline.information_ratio

    def scalar_information_ratio(a, b):
        calls.append(type(a).__name__)
        if isinstance(a, pd.DataFrame):
            return 0.42
        return original_ir(a, b)

    monkeypatch.setattr(pipeline, "information_ratio", scalar_information_ratio)

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        custom_weights={"FundA": 60.0, "FundB": 40.0},
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        **RUN_KWARGS,
    )

    assert result is not None
    ir_payload = result["benchmark_ir"].get("SPX", {})
    # Scalar response should yield a mapping entry for at least the first selected fund.
    assert "FundA" in ir_payload
    assert "FundB" not in ir_payload
    # Ensure follow-up portfolio enrichment ran (calls include Series for eq/user paths).
    assert "Series" in calls


def test_run_analysis_constraints_missing_groups_fallbacks() -> None:
    df = _clean_returns_frame()
    stats_cfg = RiskStatsConfig()

    custom_weights = {"FundA": 70.0, "FundB": 30.0}

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        custom_weights=custom_weights,
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        constraints={"group_caps": {"GroupA": 0.5}},
        **RUN_KWARGS,
    )

    assert result is not None
    weights = result["fund_weights"]
    # Missing group mapping triggers the fallback branch so original weights survive.
    assert weights["FundA"] == pytest.approx(0.7)
    assert weights["FundB"] == pytest.approx(0.3)


def test_run_analysis_benchmark_ir_non_numeric_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _clean_returns_frame()
    stats_cfg = RiskStatsConfig()

    original_ir = pipeline.information_ratio
    original_calc = pipeline.calc_portfolio_returns

    def tagging_calc(weights: np.ndarray, returns_df: pd.DataFrame) -> pd.Series:
        series = original_calc(weights, returns_df)
        role = "equal_weight"
        if not np.allclose(weights, np.repeat(1.0 / len(weights), len(weights))):
            role = "user_weight"
        series.attrs["portfolio_role"] = role
        return series

    def fake_information_ratio(a, b):
        if isinstance(a, pd.DataFrame):
            return pd.Series({"FundA": 0.4, "FundB": 0.2})
        if isinstance(a, pd.Series) and a.attrs.get("portfolio_role"):
            return {"role": a.attrs["portfolio_role"]}
        return original_ir(a, b)

    monkeypatch.setattr(pipeline, "calc_portfolio_returns", tagging_calc)
    monkeypatch.setattr(pipeline, "information_ratio", fake_information_ratio)

    result = pipeline._run_analysis(
        df,
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        target_vol=1.0,
        monthly_cost=0.0,
        stats_cfg=stats_cfg,
        custom_weights={"FundA": 60.0, "FundB": 40.0},
        indices_list=["Benchmark"],
        benchmarks={"SPX": "Benchmark"},
        **RUN_KWARGS,
    )

    assert result is not None
    ir_payload = result["benchmark_ir"].get("SPX", {})
    assert "FundA" in ir_payload and "FundB" in ir_payload
    # Non-numeric portfolio IR enrichment should yield NaN placeholders
    assert np.isnan(ir_payload.get("equal_weight"))
    assert np.isnan(ir_payload.get("user_weight"))
