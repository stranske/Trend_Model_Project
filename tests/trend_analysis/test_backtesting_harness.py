from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from trend_analysis.backtesting import harness as h


@pytest.fixture()
def sample_calendar() -> pd.DatetimeIndex:
    return pd.date_range("2021-01-31", periods=4, freq="ME")


@pytest.fixture()
def sample_backtest_result(sample_calendar: pd.DatetimeIndex) -> h.BacktestResult:
    returns = pd.Series([0.01, -0.005, 0.002, 0.003], index=sample_calendar)
    equity = (1 + returns.fillna(0)).cumprod()
    weights = pd.DataFrame(
        [[0.6, 0.4], [0.55, 0.45], [0.5, 0.5], [0.0, 1.0]],
        index=sample_calendar,
        columns=["FundA", "FundB"],
    )
    turnover = pd.Series([0.1, 0.05, 0.0, np.nan], index=sample_calendar)
    tx_costs = pd.Series([0.0001, 0.0002, 0.0, np.nan], index=sample_calendar)
    rolling_sharpe = pd.Series([np.nan, 0.8, 0.7, 0.9], index=sample_calendar)
    drawdown = equity / equity.cummax() - 1
    metrics = {
        "cagr": 0.12,
        "volatility": 0.09,
        "max_drawdown": -0.05,
        "sharpe": 1.2,
    }
    training_windows = {
        sample_calendar[0]: (
            sample_calendar[0] - timedelta(days=60),
            sample_calendar[0],
        ),
        sample_calendar[1]: (
            sample_calendar[1] - timedelta(days=60),
            sample_calendar[1],
        ),
    }

    return h.BacktestResult(
        returns=returns,
        equity_curve=equity,
        weights=weights,
        turnover=turnover,
        transaction_costs=tx_costs,
        rolling_sharpe=rolling_sharpe,
        drawdown=drawdown,
        metrics=metrics,
        calendar=sample_calendar,
        window_mode="rolling",
        window_size=3,
        training_windows=training_windows,
    )


def test_backtest_result_summary_and_json(
    sample_backtest_result: h.BacktestResult,
) -> None:
    summary = sample_backtest_result.summary()

    assert summary["window_mode"] == "rolling"
    assert summary["window_size"] == 3
    assert len(summary["calendar"]) == 4
    assert summary["metrics"]["cagr"] == pytest.approx(0.12)
    assert summary["turnover"]["2021-02-28T00:00:00"] == pytest.approx(0.05)
    assert summary["weights"]["2021-03-31T00:00:00"]["FundA"] == pytest.approx(0.5)

    json_blob = sample_backtest_result.to_json()
    parsed = json.loads(json_blob)
    assert parsed["training_windows"]
    # Ensure explicit zeros round-trip instead of being silently dropped.
    assert parsed["turnover"]["2021-03-31T00:00:00"] == pytest.approx(0.0)


def test_run_backtest_covers_transaction_costs_and_calendar() -> None:
    dates = pd.date_range("2022-01-01", periods=8, freq="D")
    returns = pd.DataFrame(
        {
            "Date": dates,
            "FundA": np.linspace(0.001, 0.004, len(dates)),
            "FundB": np.linspace(-0.002, 0.001, len(dates)),
        }
    )

    def alternating_strategy(frame: pd.DataFrame) -> dict[str, float]:
        weight = 0.7 if frame.index[-1].day % 2 else 0.3
        return {"FundA": weight, "FundB": 1 - weight}

    result = h.run_backtest(
        returns,
        alternating_strategy,
        rebalance_freq="2W",
        window_size=3,
        window_mode="rolling",
        transaction_cost_bps=15,
        rolling_sharpe_window=2,
        initial_weights={"FundB": 1.0},
    )

    assert isinstance(result, h.BacktestResult)
    assert result.weights.shape[1] == 2
    assert not result.turnover.empty
    assert result.transaction_costs.index.equals(result.turnover.index)
    # Rolling Sharpe shares the return index even when insufficient history exists.
    assert result.rolling_sharpe.index.equals(result.returns.index)
    assert all(result.calendar >= pd.Timestamp("2022-01-01"))


def test_run_backtest_expanding_mode_handles_date_column_only() -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    returns = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.02, -0.01, 0.015, 0.0, 0.01],
            "FundB": [0.0, 0.01, -0.005, 0.002, 0.003, -0.002],
        }
    )

    def equal_weight(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(1 / len(frame.columns), index=frame.columns)

    result = h.run_backtest(
        returns,
        equal_weight,
        rebalance_freq="M",
        window_size=2,
        window_mode="expanding",
        transaction_cost_bps=0,
    )

    assert set(result.metrics) >= {"cagr", "sharpe"}
    assert result.window_mode == "expanding"
    assert not result.weights.empty


@pytest.mark.parametrize(
    "kwargs, expected_exception, message",
    [
        ({"window_size": 0}, ValueError, "window_size"),
        ({"window_mode": "invalid"}, ValueError, "window_mode"),
        ({"transaction_cost_bps": -1}, ValueError, "transaction_cost_bps"),
    ],
)
def test_run_backtest_input_validation(
    kwargs: dict[str, object], expected_exception: type[Exception], message: str
) -> None:
    df = pd.DataFrame(
        {"Date": pd.date_range("2021-01-01", periods=3), "FundA": [0.0, 0.1, -0.1]}
    )

    call_kwargs = {"rebalance_freq": "ME", "window_size": 2, "window_mode": "rolling"}
    call_kwargs.update(kwargs)

    with pytest.raises(expected_exception) as excinfo:
        h.run_backtest(df, lambda _: {"FundA": 1.0}, **call_kwargs)

    assert message in str(excinfo.value)


def test_run_backtest_requires_enough_history_for_window() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=3),
            "FundA": [0.01, 0.02, 0.03],
        }
    )

    def strategy(frame: pd.DataFrame) -> dict[str, float]:
        return {"FundA": 1.0}

    with pytest.raises(ValueError, match="window_size too large"):
        h.run_backtest(
            df,
            strategy,
            rebalance_freq="M",
            window_size=10,
            window_mode="rolling",
        )


def test_run_backtest_errors_on_empty_prepared_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(
        {"Date": pd.date_range("2022-01-01", periods=3), "FundA": [0.1, 0.2, 0.3]}
    )

    monkeypatch.setattr(h, "_prepare_returns", lambda _: pd.DataFrame())

    with pytest.raises(ValueError, match="at least one row"):
        h.run_backtest(
            df,
            lambda _: {"FundA": 1.0},
            rebalance_freq="M",
            window_size=2,
            window_mode="rolling",
        )


def test_run_backtest_errors_when_calendar_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(
        {"Date": pd.date_range("2022-01-01", periods=4, freq="D"), "FundA": 0.01}
    )

    monkeypatch.setattr(h, "_rebalance_calendar", lambda *_: pd.DatetimeIndex([]))

    with pytest.raises(ValueError, match="rebalance calendar"):
        h.run_backtest(
            df,
            lambda _: {"FundA": 1.0},
            rebalance_freq="M",
            window_size=2,
            window_mode="rolling",
        )


def test_run_backtest_handles_duplicate_index_and_pending_cost() -> None:
    dates = pd.to_datetime(
        [
            "2021-01-31",
            "2021-01-31",
            "2021-02-28",
            "2021-02-28",
            "2021-03-31",
        ]
    )
    returns = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.002, 0.003, 0.004, 0.005],
            "FundB": [0.0, 0.001, 0.002, 0.003, 0.004],
        }
    )

    def equal_weight(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(0.5, index=frame.columns)

    result = h.run_backtest(
        returns,
        equal_weight,
        rebalance_freq="M",
        window_size=2,
        window_mode="rolling",
        transaction_cost_bps=50,
    )

    prepared = returns.set_index("Date").astype(float)
    expected_raw = float(np.dot(prepared.iloc[2], [0.5, 0.5]))
    expected = expected_raw - 0.005

    # First non-null portfolio return reflects a one-off cost deduction.
    first_realised = result.returns.dropna().iloc[0]
    assert first_realised == pytest.approx(expected)
    # Duplicate dates still yield a single set of stored weights per rebalance.
    assert list(result.weights.index) == sorted(result.weights.index)


def test_helpers_cover_frequency_conversion_and_json_default(
    sample_calendar: pd.DatetimeIndex,
) -> None:
    # Normalisation preserves suffix when already aligned and substitutes when required.
    assert h._normalise_frequency("3M") == "3ME"
    assert h._normalise_frequency("Q") == "QE"
    assert h._normalise_frequency("  1y ") == "1YE"

    inferred = h._infer_periods_per_year(sample_calendar)
    assert inferred == 12

    weights = pd.DataFrame(
        [[0.0, 0.5], [np.nan, 0.4]],
        index=sample_calendar[:2],
        columns=["A", "B"],
    )
    weights_dict = h._weights_to_dict(weights)
    assert list(weights_dict) == [
        sample_calendar[0].isoformat(),
        sample_calendar[1].isoformat(),
    ]

    series = pd.Series([1.0, np.nan], index=sample_calendar[:2])
    assert h._series_to_dict(series) == {sample_calendar[0].isoformat(): 1.0}

    assert h._json_default(pd.Timestamp("2021-01-01")) == "2021-01-01T00:00:00"
    assert (
        h._json_default(pd.Timestamp("2021-01-01") - pd.Timestamp("2020-12-31"))
        == "P1DT0H0M0S"
    )
    assert h._json_default(np.float64(1.23)) == pytest.approx(1.23)

    with pytest.raises(TypeError):
        h._json_default(object())


def test_initial_and_normalised_weights_round_trip() -> None:
    columns = ["X", "Y"]
    init = h._initial_weights(columns, {"X": 0.2})
    assert list(init.index) == columns
    assert init.loc["X"] == pytest.approx(0.2)
    assert init.loc["Y"] == 0.0

    normalised = h._normalise_weights({"X": 0.3}, columns)
    assert normalised.loc["X"] == pytest.approx(0.3)
    assert normalised.loc["Y"] == 0.0


def test_run_backtest_errors_for_empty_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=1, freq="D"),
            "FundA": [0.01],
        }
    )

    monkeypatch.setattr(h, "_prepare_returns", lambda _: pd.DataFrame())

    with pytest.raises(ValueError, match="at least one row"):
        h.run_backtest(
            df,
            lambda _: {"FundA": 1.0},
            rebalance_freq="D",
            window_size=1,
        )


def test_run_backtest_requires_calendar_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2024-01-01 12:00", periods=3, freq="D")
    returns = pd.DataFrame({"Date": dates, "FundA": [0.01, 0.0, -0.02]})

    monkeypatch.setattr(
        h,
        "_rebalance_calendar",
        lambda _index, _freq: pd.DatetimeIndex([], name="rebalance_date"),
    )

    with pytest.raises(ValueError, match="rebalance calendar produced no dates"):
        h.run_backtest(
            returns,
            lambda _: {"FundA": 1.0},
            rebalance_freq="M",
            window_size=1,
        )


def test_run_backtest_handles_duplicate_index_and_pending_costs() -> None:
    dates = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-01",
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
        ]
    )
    returns = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.02, 0.03, 0.04, 0.05],
            "FundB": [0.0, 0.0, 0.001, 0.002, 0.003],
        }
    )

    result = h.run_backtest(
        returns,
        lambda frame: pd.Series({"FundA": 1.0, "FundB": 0.0}, index=frame.columns),
        rebalance_freq="D",
        window_size=2,
        transaction_cost_bps=20,
    )

    # Transaction costs should be deducted exactly once per rebalance window.
    first_cost = result.transaction_costs.iloc[0]
    assert first_cost > 0
    first_return = result.returns.dropna().iloc[0]
    assert first_return == pytest.approx(0.03 - first_cost)
    # Duplicate index entries should still populate weights for each rebalance date.
    assert len(result.weights) == 3


def test_prepare_returns_requires_datetime_index() -> None:
    df = pd.DataFrame({"FundA": [0.01, 0.02]}, index=[0, 1])

    with pytest.raises(ValueError, match="DatetimeIndex"):
        h._prepare_returns(df)


def test_prepare_returns_requires_numeric_columns() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "Label": ["a", "b"],
        }
    )

    with pytest.raises(ValueError, match="numeric columns"):
        h._prepare_returns(df)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("3ME", "3ME"),
        ("bm", "bm"),
    ],
)
def test_normalise_frequency_preserves_valid_suffixes(raw: str, expected: str) -> None:
    assert h._normalise_frequency(raw) == expected


def test_infer_periods_per_year_branches() -> None:
    assert h._infer_periods_per_year(pd.DatetimeIndex([])) == 1
    duplicate = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    assert h._infer_periods_per_year(duplicate) == 1
    daily = pd.date_range("2024-01-01", periods=10, freq="B")
    assert h._infer_periods_per_year(daily) == 252
    weekly = pd.date_range("2024-01-05", periods=10, freq="W-FRI")
    assert h._infer_periods_per_year(weekly) == 52
    monthly = pd.date_range("2024-01-31", periods=10, freq="ME")
    assert h._infer_periods_per_year(monthly) == 12
    quarterly = pd.date_range("2024-03-31", periods=6, freq="QE")
    assert h._infer_periods_per_year(quarterly) == 4


def test_rolling_sharpe_enforces_minimum_window() -> None:
    returns = pd.Series(
        [0.0, 0.01, -0.02, 0.03], index=pd.date_range("2024-01-01", periods=4, freq="D")
    )
    sharpe = h._rolling_sharpe(returns, periods_per_year=12, window=1)

    assert sharpe.index.equals(returns.index)
    assert np.isnan(sharpe.iloc[0])


def test_series_and_weights_to_dict_handle_empty_inputs() -> None:
    empty_series = pd.Series(dtype=float)
    assert h._series_to_dict(empty_series) == {}

    weights = pd.DataFrame(
        [[0.0, 0.0], [np.nan, np.nan]],
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
        columns=["A", "B"],
    )
    assert h._weights_to_dict(weights) == {}
