import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from trend_analysis.backtesting import harness
from trend_analysis.backtesting.harness import BacktestResult, CostModel, run_backtest


def _sample_returns(periods: int = 8) -> pd.DataFrame:
    index = pd.date_range("2021-01-31", periods=periods, freq="ME")
    data = {
        "AAA": np.linspace(0.01, 0.08, periods),
        "BBB": np.linspace(-0.02, 0.03, periods),
    }
    return pd.DataFrame(data, index=index)


def test_backtest_result_summary_and_json_round_trip():
    index = pd.date_range("2020-01-01", periods=3, freq="ME")
    result = BacktestResult(
        returns=pd.Series([0.01, np.nan, 0.02], index=index),
        equity_curve=pd.Series([1.0, 1.01, 1.0202], index=index),
        weights=pd.DataFrame(
            [[0.6, 0.4], [0.5, 0.5], [0.0, 1.0]], index=index, columns=["AAA", "BBB"]
        ),
        turnover=pd.Series([0.1, 0.2, 0.0], index=index),
        per_period_turnover=pd.Series([0.1, 0.2, 0.0], index=index),
        transaction_costs=pd.Series([0.0005, 0.0010, 0.0], index=index),
        rolling_sharpe=pd.Series([np.nan, 0.3, 0.4], index=index),
        drawdown=pd.Series([0.0, -0.01, -0.02], index=index),
        metrics={"sharpe": np.float64(1.234), "volatility": np.float32(0.5)},
        calendar=index,
        window_mode="rolling",
        window_size=2,
        training_windows={
            index[0]: (index[0], index[1]),
            index[2]: (index[1], index[2]),
        },
        cost_model=CostModel(),
    )

    summary = result.summary()
    assert summary["window_mode"] == "rolling"
    assert summary["window_size"] == 2
    assert summary["calendar"] == [ts.isoformat() for ts in index]
    assert summary["metrics"] == {
        "sharpe": pytest.approx(1.234),
        "volatility": pytest.approx(0.5),
    }
    assert summary["returns"] == {
        index[0].isoformat(): pytest.approx(0.01),
        index[2].isoformat(): pytest.approx(0.02),
    }
    # weights drop zero values and emit ISO keys
    weights = summary["weights"]
    assert index[0].isoformat() in weights and "AAA" in weights[index[0].isoformat()]
    assert index[2].isoformat() in weights and weights[index[2].isoformat()] == {
        "BBB": pytest.approx(1.0)
    }
    # training windows render nested ISO mappings
    assert summary["training_windows"][index[0].isoformat()] == {
        "start": index[0].isoformat(),
        "end": index[1].isoformat(),
    }

    encoded = result.to_json(indent=2)
    decoded = json.loads(encoded)
    assert decoded == summary


def test_run_backtest_with_dynamic_strategy_and_transaction_costs():
    returns = _sample_returns()

    def strategy(window: pd.DataFrame):
        if len(window) % 2 == 0:
            # Ensure Series inputs are handled
            return pd.Series({"AAA": 0.7, "BBB": 0.3})
        # Exercise mapping branch
        return {"AAA": 0.55, "BBB": 0.45}

    result = run_backtest(
        returns,
        strategy,
        rebalance_freq="M",
        window_size=4,
        transaction_cost_bps=25,
        initial_weights={"AAA": 0.2, "CCC": 0.8},
        min_trade=0.0,
    )

    assert isinstance(result, BacktestResult)
    assert list(result.calendar) == sorted(result.training_windows)
    assert result.window_mode == "rolling"
    assert result.turnover.index.equals(result.transaction_costs.index)
    assert result.weights.columns.tolist() == ["AAA", "BBB"]
    # At least one realised return should be computed after transaction costs
    realised = result.returns.dropna()
    assert realised.size >= 1
    assert (result.equity_curve >= 0).all()
    # Rolling Sharpe uses active mask and should include finite values once enough data accrues
    assert result.rolling_sharpe.notna().any()
    # Summary generation exercises remaining helpers
    summary = result.summary()
    assert summary["metrics"]["final_value"] == pytest.approx(
        result.equity_curve.iloc[-1]
    )


def test_run_backtest_handles_duplicate_rebalance_dates():
    index = pd.to_datetime(
        [
            "2021-01-31",
            "2021-01-31",
            "2021-02-28",
            "2021-02-28",
            "2021-03-31",
            "2021-03-31",
        ]
    )
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.01, 0.02, 0.02, 0.03, 0.03],
            "BBB": [0.0, 0.0, -0.01, -0.01, 0.02, 0.02],
        },
        index=index,
    )

    result = run_backtest(
        returns,
        lambda window: pd.Series({"AAA": 0.5, "BBB": 0.5}),
        rebalance_freq="M",
        window_size=2,
        transaction_cost_bps=0.0,
        min_trade=0.0,
    )

    realised = result.returns.dropna()
    assert realised.index.is_monotonic_increasing
    assert realised.size >= 1


def test_run_backtest_expanding_mode_and_error_conditions(monkeypatch):
    returns = _sample_returns(5)

    expanding = run_backtest(
        returns,
        lambda window: pd.Series({col: 1.0 / len(window.columns) for col in window}),
        rebalance_freq="Q",
        window_size=2,
        window_mode="expanding",
        rolling_sharpe_window=1,
        transaction_cost_bps=0.0,
        min_trade=0.0,
    )
    assert expanding.window_mode == "expanding"
    assert expanding.training_windows

    # Validation failures
    with pytest.raises(ValueError, match="window_size"):
        run_backtest(
            returns,
            lambda df: {"AAA": 1.0},
            rebalance_freq="M",
            window_size=0,
            transaction_cost_bps=0.0,
            min_trade=0.0,
        )
    with pytest.raises(ValueError, match="window_mode"):
        run_backtest(
            returns,
            lambda df: {"AAA": 1.0},
            rebalance_freq="M",
            window_size=2,
            window_mode="invalid",
            transaction_cost_bps=0.0,
            min_trade=0.0,
        )
    with pytest.raises(ValueError, match="transaction_cost_bps"):
        run_backtest(
            returns,
            lambda df: {"AAA": 1.0},
            rebalance_freq="M",
            window_size=2,
            transaction_cost_bps=-1,
            min_trade=0.0,
        )

    empty_df = returns.iloc[0:0]
    with pytest.raises(ValueError, match="numeric columns"):
        run_backtest(
            empty_df,
            lambda df: {"AAA": 1.0},
            rebalance_freq="M",
            window_size=2,
            transaction_cost_bps=0.0,
            min_trade=0.0,
        )

    original_prepare = harness._prepare_returns

    def empty_prepare(df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - helper
        return pd.DataFrame(columns=df.columns)

    monkeypatch.setattr(harness, "_prepare_returns", empty_prepare)
    with pytest.raises(ValueError, match="at least one row"):
        run_backtest(
            returns,
            lambda df: {"AAA": 1.0},
            rebalance_freq="M",
            window_size=2,
            transaction_cost_bps=0.0,
            min_trade=0.0,
        )
    harness._prepare_returns = original_prepare

    def empty_calendar(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([], name="rebalance_date")

    original_calendar = harness._rebalance_calendar
    monkeypatch.setattr(harness, "_rebalance_calendar", empty_calendar)
    with pytest.raises(ValueError, match="rebalance calendar"):
        run_backtest(
            returns,
            lambda df: {"AAA": 1.0},
            rebalance_freq="M",
            window_size=2,
            transaction_cost_bps=0.0,
            min_trade=0.0,
        )
    harness._rebalance_calendar = original_calendar
    with pytest.raises(ValueError, match="window_size too large"):
        run_backtest(
            returns,
            lambda df: {"AAA": 1.0},
            rebalance_freq="M",
            window_size=50,
            transaction_cost_bps=0.0,
            min_trade=0.0,
        )


def test_prepare_returns_and_frequency_helpers():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2021-01-31", "2021-02-28"]),
            "AAA": [0.1, 0.2],
            "BBB": [0.0, 0.1],
        }
    )
    prepared = harness._prepare_returns(df)
    assert prepared.index.is_monotonic_increasing
    assert prepared.columns.tolist() == ["AAA", "BBB"]

    with pytest.raises(ValueError):
        harness._prepare_returns(pd.DataFrame({"AAA": ["bad"], "BBB": ["also bad"]}))

    calendar = harness._rebalance_calendar(prepared.index, "m")
    assert list(calendar) == list(prepared.index)
    quarter = harness._rebalance_calendar(prepared.index, "2q")
    assert quarter[-1] == prepared.index[-1]

    assert harness._normalise_frequency(" m ") == "ME"
    assert harness._normalise_frequency("ME") == "ME"
    assert harness._normalise_frequency("Q") == "QE"
    assert harness._normalise_frequency("2y") == "2YE"
    assert harness._normalise_frequency("custom") == "custom"


def test_infer_periods_per_year_and_weight_helpers(monkeypatch):
    idx_daily = pd.date_range("2022-01-01", periods=10, freq="B")
    idx_weekly = pd.date_range("2022-01-02", periods=10, freq="W")
    idx_monthly = pd.date_range("2022-01-31", periods=10, freq="ME")
    idx_quarter = pd.date_range("2022-03-31", periods=6, freq="QE")
    idx_short = pd.DatetimeIndex([pd.Timestamp("2022-01-01")])
    idx_negative = pd.DatetimeIndex(
        [pd.Timestamp("2022-01-02"), pd.Timestamp("2022-01-01")]
    )

    assert harness._infer_periods_per_year(idx_short) == 1
    assert harness._infer_periods_per_year(idx_negative) == 1
    assert harness._infer_periods_per_year(idx_daily) == 252
    assert harness._infer_periods_per_year(idx_weekly) == 52
    assert harness._infer_periods_per_year(idx_monthly) == 12
    assert harness._infer_periods_per_year(idx_quarter) == 4
    idx_irregular = pd.to_datetime(["2022-01-01", "2022-01-21", "2022-02-10"])
    assert harness._infer_periods_per_year(idx_irregular) == 18

    original_diff = np.diff

    def fake_diff(values):  # pragma: no cover - helper
        return np.array([], dtype=np.int64)

    monkeypatch.setattr(np, "diff", fake_diff)
    assert harness._infer_periods_per_year(idx_daily) == 1
    np.diff = original_diff

    columns = ["AAA", "BBB"]
    base = harness._initial_weights(columns, None)
    assert base.tolist() == [0.0, 0.0]
    seeded = harness._initial_weights(columns, {"AAA": 0.4, "CCC": 0.6})
    assert seeded.loc["AAA"] == pytest.approx(0.4)
    assert seeded.loc["BBB"] == 0.0

    normalised_series = harness._normalise_weights(
        pd.Series({"AAA": 0.7, "BBB": 0.3}), columns
    )
    assert normalised_series.sum() == pytest.approx(1.0)
    normalised_mapping = harness._normalise_weights({"AAA": 0.5}, columns)
    assert normalised_mapping.tolist() == [0.5, 0.0]


def test_metric_and_serialisation_utilities():
    series = pd.Series(
        [1.0, 1.2, 1.1], index=pd.date_range("2022-01-01", periods=3, freq="D")
    )
    drawdown = harness._compute_drawdown(series)
    assert drawdown.iloc[0] == 0.0
    assert drawdown.iloc[2] == pytest.approx(1.1 / 1.2 - 1, rel=1e-6)

    returns = pd.Series(
        [0.01, -0.02, 0.03, np.nan],
        index=series.index.insert(3, series.index[-1] + timedelta(days=1)),
    )
    sharpe = harness._rolling_sharpe(returns, periods_per_year=252, window=1)
    expected_sharpe = (
        returns.rolling(window=2).mean().iloc[1]
        / returns.rolling(window=2).std(ddof=0).iloc[1]
        * np.sqrt(252)
    )
    assert sharpe.iloc[1] == pytest.approx(expected_sharpe)

    metrics = harness._compute_metrics(
        returns.fillna(0.0), series, drawdown, 252, returns.notna()
    )
    assert {"cagr", "volatility", "sharpe", "final_value"} <= metrics.keys()

    empty_dict = harness._series_to_dict(pd.Series(dtype=float))
    assert empty_dict == {}
    populated_dict = harness._series_to_dict(returns)
    assert populated_dict  # at least one item present

    weights = pd.DataFrame(
        {
            pd.Timestamp("2022-01-01"): [0.6, 0.4],
            pd.Timestamp("2022-02-01"): [0.0, 0.0],
            pd.Timestamp("2022-03-01"): [0.0, 1.0],
        },
        index=["AAA", "BBB"],
    ).T
    weights_dict = harness._weights_to_dict(weights)
    assert list(weights_dict) == ["2022-01-01T00:00:00", "2022-03-01T00:00:00"]

    ts = pd.Timestamp("2022-01-05")
    td = pd.Timedelta(days=1)
    assert harness._json_default(ts) == ts.isoformat()
    assert harness._json_default(td) == td.isoformat()
    assert harness._json_default(np.float32(1.23)) == pytest.approx(1.23)
    assert harness._json_default(np.int64(5)) == 5.0
    with pytest.raises(TypeError):
        harness._json_default(object())

    assert harness._to_float(None) != harness._to_float(1.23)
    assert np.isnan(harness._to_float(None))
    assert harness._to_float(np.float32(2.5)) == pytest.approx(2.5)


def test_compute_metrics_handles_empty_returns():
    returns = pd.Series(
        [0.01, -0.02], index=pd.date_range("2022-01-01", periods=2, freq="D")
    )
    equity = pd.Series([1.0, 0.98], index=returns.index)
    drawdown = pd.Series([-0.0, -0.02], index=returns.index)
    mask = pd.Series([False, False], index=returns.index)

    metrics = harness._compute_metrics(
        returns, equity, drawdown, periods_per_year=0, active_mask=mask
    )
    assert np.isnan(metrics["cagr"])
    assert metrics["final_value"] == pytest.approx(0.98)


def test_weights_to_dict_handles_empty_frames():
    empty = pd.DataFrame(columns=["AAA", "BBB"])
    assert harness._weights_to_dict(empty) == {}
