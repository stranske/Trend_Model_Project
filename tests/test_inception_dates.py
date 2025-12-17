from __future__ import annotations

import pandas as pd

from trend_analysis.data import compute_inception_dates


def test_compute_inception_dates_detects_first_nonzero() -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
            "FundB": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    inception = compute_inception_dates(df)
    assert inception["FundA"] == pd.Timestamp("2020-03-31")
    assert inception["FundB"] is None
