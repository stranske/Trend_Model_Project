import pandas as pd
import pytest

from trend_analysis.metrics import summary


def test_summary_table_snapshot() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.015],
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )
    weights = {
        returns.index[0]: pd.Series({"A": 0.6, "B": 0.4}),
        returns.index[1]: pd.Series({"A": 0.7, "B": 0.3}),
        returns.index[2]: pd.Series({"A": 0.5, "B": 0.5}),
    }
    bench = pd.Series(0.0, index=returns.index)

    out = summary.summary_table(
        returns,
        weights,
        benchmark=bench,
        periods_per_year=12,
        transaction_cost_bps=25,
    )

    assert out.loc["CAGR", "value"] == pytest.approx(0.012639995599)
    assert out.loc["vol", "value"] == pytest.approx(0.065368187982)
    assert out.loc["max_drawdown", "value"] == pytest.approx(0.0205)
    assert out.loc["information_ratio", "value"] == pytest.approx(0.214171456060)
    assert out.loc["sharpe", "value"] == pytest.approx(0.193366161568)
    assert out.loc["turnover", "value"] == pytest.approx(0.2)
    assert out.loc["cost_impact", "value"] == pytest.approx(0.0005)
    assert out.loc["hit_rate", "value"] == pytest.approx(2 / 3)
