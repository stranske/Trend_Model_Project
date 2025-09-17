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

    expected_idx = [3, 4, 5, 6, 7, 8, 9, 10]
    expected_oos_mean = df.iloc[expected_idx]["metric"].mean()
    assert res.oos.loc["mean", "metric"] == expected_oos_mean

    expected_oos_ir = pytest.approx(10.6066017178, rel=1e-9)
    assert res.oos.loc["information_ratio", "metric"] == expected_oos_ir

    # regime aggregates
    assert res.by_regime.loc["A", ("metric", "mean")] == 5
    assert res.by_regime.loc["B", ("metric", "mean")] == 9
    assert res.by_regime.loc["A", ("metric", "information_ratio")] == pytest.approx(
        17.3205080757
    )
    assert res.by_regime.loc["B", ("metric", "information_ratio")] == pytest.approx(
        19.7180120701
    )

    # per-window table includes train/test boundaries and window statistics
    assert ("window", "train_start") in res.oos_windows.columns
    assert ("metric", "mean") in res.oos_windows.columns
    first_window = res.oos_windows.loc[1]
    assert first_window["window", "test_start"] == pd.Timestamp("2020-04-30")
    assert first_window["window", "test_end"] == pd.Timestamp("2020-05-31")
    assert first_window["metric", "mean"] == pytest.approx(4.5)
    assert first_window["metric", "information_ratio"] == pytest.approx(22.0454076850)

    # full-period summary retains mean + IR
    assert res.full.loc["mean", "metric"] == df["metric"].mean()
    assert res.full.loc["information_ratio", "metric"] == pytest.approx(6.2449979984)


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
    assert res.oos.loc["mean"].isna().all()
    assert res.oos.loc["information_ratio"].isna().all()
    assert res.oos_windows.empty
    assert res.by_regime.empty
