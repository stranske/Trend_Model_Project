import numpy as np
import pandas as pd
import pytest
from trend_analysis import pipeline
from trend_analysis.core.rank_selection import (
    compute_metric_series_with_cache,
    RiskStatsConfig,
)
from trend_analysis.export import summary_frame_from_result
from trend_analysis.perf.cache import CovCache


def _rand_df(rows=40, cols=6, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(scale=0.01, size=(rows, cols))
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(cols)])


def test_avg_corr_metric_cache_equivalence():
    df = _rand_df()
    stats_cfg = RiskStatsConfig()
    # Without cache
    s_no = compute_metric_series_with_cache(
        df, "AvgCorr", stats_cfg, enable_cache=False
    )
    # With cache
    cache = CovCache()
    s_cache = compute_metric_series_with_cache(
        df, "AvgCorr", stats_cfg, cov_cache=cache, enable_cache=True
    )
    pd.testing.assert_series_equal(s_no, s_cache)
    # Basic sanity: values between -1 and 1
    assert (s_no.abs() <= 1 + 1e-12).all()


def test_run_analysis_attaches_avg_corr_to_stats():
    dates = pd.date_range("2020-01-31", periods=6, freq="M")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": np.linspace(0.0, 0.0005, len(dates)),
            "FundA": np.linspace(0.01, 0.015, len(dates)),
            "FundB": np.linspace(0.008, 0.012, len(dates)),
        }
    )
    cfg = RiskStatsConfig()
    cfg.extra_metrics = ["AvgCorr"]

    result = pipeline.run_analysis(
        df,
        "2020-01",
        "2020-03",
        "2020-04",
        "2020-06",
        0.1,
        0.0,
        stats_cfg=cfg,
    )
    assert result is not None

    in_stats = result["in_sample_stats"]
    out_stats = result["out_sample_stats"]
    assert in_stats["FundA"].avg_corr is not None
    assert out_stats["FundA"].avg_corr is not None

    in_df = df.set_index("Date").loc["2020-01-31":"2020-03-31", ["FundA", "FundB"]]
    out_df = df.set_index("Date").loc["2020-04-30":"2020-06-30", ["FundA", "FundB"]]
    expected_in = in_df.corr().loc["FundA", "FundB"]
    expected_out = out_df.corr().loc["FundA", "FundB"]
    assert in_stats["FundA"].avg_corr == pytest.approx(expected_in, abs=1e-12)
    assert out_stats["FundA"].avg_corr == pytest.approx(expected_out, abs=1e-12)

    summary = summary_frame_from_result(result)
    assert {"IS AvgCorr", "OS AvgCorr"}.issubset(summary.columns)
    fund_row = summary.loc[summary["Name"] == "FundA"].iloc[0]
    assert fund_row["IS AvgCorr"] == pytest.approx(expected_in, abs=1e-12)
    assert fund_row["OS AvgCorr"] == pytest.approx(expected_out, abs=1e-12)
