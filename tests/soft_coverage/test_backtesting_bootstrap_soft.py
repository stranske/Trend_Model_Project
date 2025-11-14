"""Soft coverage for bootstrap utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataclasses import replace

from trend_analysis.backtesting.bootstrap import (
    BacktestResult,
    _bootstrap_paths,
    _init_rng,
    _validate_inputs,
    bootstrap_equity,
)


def make_backtest_result() -> BacktestResult:
    dates = pd.date_range("2024-01-31", periods=5, freq="M")
    returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01], index=dates)
    equity = (1 + returns).cumprod()
    weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=dates)
    turnover = pd.Series(np.linspace(0.0, 0.2, len(dates)), index=dates)
    costs = pd.Series(0.001, index=dates)
    rolling = pd.Series(0.5, index=dates)
    drawdown = pd.Series([0.0, -0.01, -0.02, -0.01, 0.0], index=dates)
    metrics = {"sharpe": 1.2}
    training_windows = {dates[-1]: (dates[0], dates[-2])}
    return BacktestResult(
        returns=returns,
        equity_curve=equity,
        weights=weights,
        turnover=turnover,
        transaction_costs=costs,
        rolling_sharpe=rolling,
        drawdown=drawdown,
        metrics=metrics,
        calendar=dates,
        window_mode="rolling",
        window_size=3,
        training_windows=training_windows,
    )


def test_init_rng_accepts_seed_and_generator() -> None:
    seed_rng = _init_rng(123)
    assert isinstance(seed_rng, np.random.Generator)

    existing = np.random.default_rng(42)
    assert _init_rng(existing) is existing


def test_validate_inputs_enforces_types_and_lengths() -> None:
    result = make_backtest_result()
    validated = _validate_inputs(result, n=10, block=2)
    assert isinstance(validated, pd.Series)

    with pytest.raises(ValueError):
        _validate_inputs(result, n=0, block=2)
    with pytest.raises(ValueError):
        _validate_inputs(result, n=1, block=0)
    with pytest.raises(TypeError):
        invalid = replace(result, returns=result.returns.to_frame())  # type: ignore[arg-type]
        _validate_inputs(invalid, n=1, block=1)
    with pytest.raises(ValueError):
        empty_returns = replace(result, returns=pd.Series(dtype=float))
        _validate_inputs(empty_returns, n=1, block=1)


def test_bootstrap_paths_wraps_indices() -> None:
    returns = make_backtest_result().returns
    rng = np.random.default_rng(0)
    paths = _bootstrap_paths(returns, n_paths=3, block=2, rng=rng)
    assert paths.shape == (3, len(returns))


def test_bootstrap_equity_generates_bands(monkeypatch: pytest.MonkeyPatch) -> None:
    result = make_backtest_result()
    rng = np.random.default_rng(0)
    monkeypatch.setattr("numpy.random.default_rng", lambda seed=None: rng)

    band = bootstrap_equity(result, n=5, block=2, random_state=0)
    assert list(band.columns) == ["p05", "median", "p95"]
    assert band.index.equals(result.equity_curve.index)

