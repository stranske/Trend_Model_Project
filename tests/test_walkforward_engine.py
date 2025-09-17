import numpy as np
import pandas as pd
import pytest

from trend_analysis.engine.walkforward import walk_forward


def test_walkforward_split_counts_and_regime_aggregation():
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    df = pd.DataFrame({"Date": dates, "metric": np.arange(1, 13)})
    regimes = pd.Series(["A"] * 6 + ["B"] * 6, index=dates)

    res = walk_forward(
        df,
        train_size=3,
        test_size=2,
        step_size=2,
        metric_cols=["metric"],
        regimes=regimes,
    )

    assert len(res.splits) == 4

    # expected OOS indices (0-based): 3-4,5-6,7-8,9-10
    expected_idx = [3, 4, 5, 6, 7, 8, 9, 10]
    expected_oos_mean = df.iloc[expected_idx]["metric"].mean()
    assert res.oos["metric"] == expected_oos_mean

    # regime means
    assert res.by_regime.loc["A", "metric"] == 5
    assert res.by_regime.loc["B", "metric"] == 9

    # full period mean for completeness
    assert res.full["metric"] == df["metric"].mean()


def test_walk_forward_requires_datetime_index():
    df = pd.DataFrame({"metric": [1, 2, 3]})
    with pytest.raises(ValueError, match="DatetimeIndex"):
        walk_forward(df, train_size=2, test_size=1, step_size=1)


def test_walk_forward_empty_splits_and_regimes():
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    df = pd.DataFrame({"Date": dates, "metric": [1.0, 2.0]})
    regimes = pd.Series(["R", "R"], index=dates)

    res = walk_forward(
        df,
        train_size=3,
        test_size=2,
        step_size=1,
        regimes=regimes,
    )

    assert res.splits == []
    assert res.oos.isna().all()
    assert res.by_regime.empty
