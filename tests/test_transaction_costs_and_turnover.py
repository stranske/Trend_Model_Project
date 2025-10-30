from pathlib import Path

import pandas as pd
import yaml

from trend_analysis.config import Config
from trend_analysis.export import combined_summary_result, summary_frame_from_result
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
    # Each period result should carry execution metrics
    for res in results:
        assert "turnover" in res
        assert "transaction_cost" in res
        assert isinstance(res["turnover"], float)
        assert isinstance(res["transaction_cost"], float)


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
