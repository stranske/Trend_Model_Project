from __future__ import annotations

from typing import Any, cast

import pandas as pd

from trend_analysis.core.rank_selection import RiskStatsConfig
from trend_analysis.pipeline import single_period_run


def test_single_period_run_avg_corr_metadata() -> None:
    dataset = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2022-01-31",
                    "2022-02-28",
                    "2022-03-31",
                    "2022-04-30",
                ]
            ),
            "FundOne": [0.01, 0.03, 0.02, 0.05],
            "FundTwo": [0.04, 0.01, 0.03, 0.02],
            "FundThree": [0.02, 0.02, 0.01, 0.04],
        }
    )
    baseline_cfg = RiskStatsConfig(metrics_to_run=["Sharpe"])
    baseline_frame = single_period_run(
        dataset,
        "2022-01",
        "2022-04",
        stats_cfg=baseline_cfg,
    )
    assert baseline_frame.attrs["insample_len"] == 4
    assert baseline_frame.attrs["period"] == ("2022-01", "2022-04")

    stats_cfg = RiskStatsConfig(metrics_to_run=["Sharpe", "Sortino"])
    cast_stats = cast(Any, stats_cfg)
    cast_stats.extra_metrics = ["AvgCorr"]
    score_frame = single_period_run(
        dataset,
        "2022-01",
        "2022-04",
        stats_cfg=stats_cfg,
    )
    assert "AvgCorr" in score_frame.columns
    assert score_frame["AvgCorr"].notna().all()
    assert score_frame.filter(items=["Sharpe", "Sortino"]).shape[1] == 2
