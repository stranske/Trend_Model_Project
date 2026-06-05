from __future__ import annotations

import math

import pandas as pd

from scripts.run_real_model import _align_risk_free_to_portfolio
from trend_analysis.metrics import sharpe_ratio


def test_align_risk_free_reindexes_missing_portfolio_dates_without_nan() -> None:
    portfolio = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
    )
    rf_series = pd.Series(
        [0.001, 0.002],
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
        name="Risk-Free Rate",
    )

    aligned = _align_risk_free_to_portfolio(rf_series, portfolio.index)

    assert isinstance(aligned, pd.Series)
    pd.testing.assert_index_equal(aligned.index, portfolio.index)
    assert aligned.tolist() == [0.001, 0.002, 0.0]
    assert not aligned.isna().any()
    assert math.isfinite(sharpe_ratio(portfolio, aligned))
