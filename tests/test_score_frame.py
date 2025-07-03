import pandas as pd
from pathlib import Path
from trend_analysis.pipeline import single_period_run
from trend_analysis.core.rank_selection import RiskStatsConfig


def make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "A": [0.02, 0.03, -0.01, 0.04, 0.02, 0.01],
            "B": [0.01, 0.02, -0.02, 0.03, 0.02, 0.0],
        }
    )


def test_single_period_run_golden():
    df = make_df()[["Date", "A", "B"]]
    sf = single_period_run(df, "2020-01", "2020-03")
    expected = pd.read_csv(Path("tests/score_frame_golden.csv"), index_col=0)
    pd.testing.assert_frame_equal(sf, expected)
    assert sf.attrs["insample_len"] == 3


def test_column_order_respects_config():
    df = make_df()[["Date", "A", "B"]]
    cfg = RiskStatsConfig(metrics_to_run=["Volatility", "AnnualReturn"])
    sf = single_period_run(df, "2020-01", "2020-03", stats_cfg=cfg)
    assert sf.columns.tolist() == cfg.metrics_to_run
