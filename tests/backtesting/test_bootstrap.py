from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.backtesting import BacktestResult, bootstrap_equity


def _make_result(returns: pd.Series) -> BacktestResult:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0 if not equity.empty else equity
    return BacktestResult(
        returns=returns,
        equity_curve=equity,
        weights=pd.DataFrame(dtype=float),
        turnover=pd.Series(dtype=float),
        transaction_costs=pd.Series(dtype=float),
        rolling_sharpe=pd.Series(dtype=float),
        drawdown=drawdown,
        metrics={},
        calendar=pd.DatetimeIndex([]),
        window_mode="rolling",
        window_size=1,
        training_windows={},
    )


def test_bootstrap_equity_constant_returns_aligns_with_realised():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    returns = pd.Series([np.nan, 0.01, 0.01, 0.01, 0.01, 0.01], index=idx)
    result = _make_result(returns)

    band = bootstrap_equity(result, n=25, block=3, random_state=123)

    # Pre-live rows should be NaN so overlays align with the realised curve.
    assert band.loc[idx[0]].isna().all()

    realised_curve = result.equity_curve
    active_mask = returns.notna()
    for col in ["p05", "median", "p95"]:
        np.testing.assert_allclose(band.loc[active_mask, col], realised_curve.loc[active_mask])


def test_bootstrap_equity_invalid_inputs():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    returns = pd.Series([np.nan, np.nan, np.nan], index=idx)
    result = _make_result(returns)

    with pytest.raises(ValueError):
        bootstrap_equity(result)
    with pytest.raises(ValueError):
        bootstrap_equity(result, n=0)
    with pytest.raises(ValueError):
        bootstrap_equity(_make_result(pd.Series([0.1], index=[idx[0]])), block=0)


def test_bootstrap_equity_handles_long_blocks():
    idx = pd.date_range("2021-01-31", periods=4, freq="ME")
    returns = pd.Series([0.02, -0.01, 0.015, -0.005], index=idx)
    result = _make_result(returns)

    band = bootstrap_equity(result, n=50, block=10, random_state=0)

    assert list(band.index) == list(result.equity_curve.index)
    assert band[["p05", "median", "p95"]].notna().all(axis=None)

    realised = result.equity_curve
    assert (realised <= band["p95"] + 1e-12).all()
    assert (realised >= band["p05"] - 1e-12).all()


def test_bootstrap_equity_deterministic_seed():
    idx = pd.date_range("2022-01-31", periods=5, freq="ME")
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01], index=idx)
    result = _make_result(returns)

    first = bootstrap_equity(result, n=100, block=2, random_state=42)
    second = bootstrap_equity(result, n=100, block=2, random_state=42)
    third = bootstrap_equity(result, n=100, block=2, random_state=7)

    assert first.equals(second)
    assert not first.equals(third)
