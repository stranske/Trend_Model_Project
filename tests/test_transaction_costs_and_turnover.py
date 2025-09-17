from pathlib import Path

import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]

from trend_analysis.config import Config
from trend_analysis.export import (combined_summary_result,
                                   execution_metrics_frame,
                                   summary_frame_from_result)
from trend_analysis.multi_period import run as run_mp


def make_df():
    dates = pd.date_range("1990-01-31", periods=6, freq="ME")
    return pd.DataFrame({"Date": dates, "A": 0.01, "B": 0.02, "C": 0.0})


def make_cfg():
    cfg_data = yaml.safe_load(Path("config/defaults.yml").read_text())
    cfg_data["multi_period"] = {
        "frequency": "M",
        "in_sample_len": 2,
        "out_sample_len": 1,
        "start": "1990-01",
        "end": "1990-06",
    }
    # Enable simple non-zero controls
    cfg_data["portfolio"]["transaction_cost_bps"] = 10
    cfg_data["portfolio"]["max_turnover"] = 0.5
    # Use threshold-hold policy so execution metrics are produced in results
    cfg_data["portfolio"]["policy"] = "threshold_hold"
    return Config(**cfg_data)


def test_period_results_include_turnover_and_cost():
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    assert results, "no results returned"
    # Each period result should carry execution metrics with correct values
    # Turnover is 0.5 in the first period because the portfolio is fully reallocated (max_turnover=0.5),
    # and 0.0 in subsequent periods due to the threshold-hold policy (no further rebalancing).
    expected_turnover = [0.5, 0.0, 0.0, 0.0]
    # Transaction cost is calculated as turnover * transaction_cost_rate (10 bps = 0.0010),
    # so 0.5 * 0.0010 = 0.0005 in the first period, and 0.0 thereafter.
    expected_cost = [0.0005, 0.0, 0.0, 0.0]
    for res, to_exp, cost_exp in zip(results, expected_turnover, expected_cost):
        assert "turnover" in res
        assert "transaction_cost" in res
        assert isinstance(res["turnover"], float)
        assert isinstance(res["transaction_cost"], float)
        assert res["turnover"] == pytest.approx(to_exp)
        assert res["transaction_cost"] == pytest.approx(cost_exp)

    # Export frame should reflect the same execution metrics
    frame = execution_metrics_frame(results)
    assert frame.shape[0] == len(results)
    assert frame["Turnover"].tolist() == pytest.approx(expected_turnover)
    assert frame["Transaction Cost"].tolist() == pytest.approx(expected_cost)


def test_export_summary_schema_unchanged_and_metrics_available_elsewhere():
    df = make_df()
    cfg = make_cfg()
    results = run_mp(cfg, df)
    # Summary frame keeps legacy 14+bench columns (no extra execution columns)
    period_df = summary_frame_from_result(results[0])
    assert "OS Turnover" not in period_df.columns
    assert "OS Cost" not in period_df.columns
    # Combined summary frame also keeps same schema
    summary = combined_summary_result(results)
    comb_df = summary_frame_from_result(summary)
    assert "OS Turnover" not in comb_df.columns
    assert "OS Cost" not in comb_df.columns
