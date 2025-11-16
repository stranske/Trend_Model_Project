import numpy as np
import pandas as pd

from trend_analysis.core.rank_selection import RiskStatsConfig
from trend_analysis import pipeline


def test_pipeline_run_applies_calendar_alignment():
    dates = pd.date_range("2024-01-01", "2024-02-29", freq="D")
    df = pd.DataFrame({"Date": dates})
    df["FundA"] = np.linspace(0.001, 0.01, len(dates))
    df["FundB"] = np.linspace(0.02, 0.005, len(dates))
    df["RF"] = 0.0001
    df.attrs["calendar_settings"] = {
        "frequency": "D",
        "timezone": "US/Eastern",
        "holiday_calendar": "simple",
    }

    result = pipeline._run_analysis(
        df,
        "2024-01",
        "2024-01",
        "2024-02",
        "2024-02",
        target_vol=0.1,
        monthly_cost=0.0,
        stats_cfg=RiskStatsConfig(risk_free=0.0),
    )
    assert result is not None

    alignment = result["preprocessing"]["calendar_alignment"]
    assert alignment["weekend_rows_dropped"] > 0
    assert alignment["holiday_rows_dropped"] > 0
    assert (result["in_sample_scaled"].index.dayofweek < 5).all()
