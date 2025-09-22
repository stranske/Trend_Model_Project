from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.core import rank_selection as rank_selection_mod
from trend_analysis.core.rank_selection import RiskStatsConfig
from trend_analysis.engine import optimizer as optimizer_mod
from trend_analysis import pipeline


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


def test_single_period_run_swallows_avg_corr_failure(
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

    score_frame = pipeline.single_period_run(
        df, "2020-01", "2020-03", stats_cfg=stats_cfg
    )

    # Failure of the optional metric should not prevent the base metrics from materialising.
    assert set(score_frame.columns) == set(stats_cfg.metrics_to_run)


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
    )

    assert result is None or "FundA" not in (result or {}).get("selected_funds", [])


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
    )

    assert result is not None
    in_stats = result["in_sample_stats"]["FundA"]
    out_stats = result["out_sample_stats"]["FundA"]

    assert in_stats.is_avg_corr is not None
    assert in_stats.os_avg_corr is None
    assert out_stats.os_avg_corr is not None
    assert out_stats.is_avg_corr is None


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
    )

    assert result is not None
    weights = result["fund_weights"]
    assert weights["FundA"] == pytest.approx(0.7)
    assert weights["FundB"] == pytest.approx(0.3)
    assert captured_inputs and list(captured_inputs[0].index) == ["FundA", "FundB"]


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
    )

    assert result is not None
    ir_payload = result["benchmark_ir"].get("SPX", {})
    # Fallback path should skip enriching the portfolio-level IR keys.
    assert "equal_weight" not in ir_payload
    assert "user_weight" not in ir_payload


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
    )

    assert result is not None
    ir_payload = result["benchmark_ir"].get("SPX", {})
    # Scalar response should yield a mapping entry for at least the first selected fund.
    assert "FundA" in ir_payload
    assert "FundB" not in ir_payload
    # Ensure follow-up portfolio enrichment ran (calls include Series for eq/user paths).
    assert "Series" in calls
