from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trend_analysis.backtesting import BacktestResult, bootstrap_equity
from trend_analysis.backtesting.bootstrap import _init_rng, _validate_inputs


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


def test_init_rng_reuses_generators_and_validate_inputs_types() -> None:
    rng = np.random.default_rng(123)
    assert _init_rng(rng) is rng

    base = _make_result(pd.Series([0.01, 0.02], index=pd.date_range("2020-01-31", periods=2, freq="ME")))
    with pytest.raises(TypeError, match="returns must be a pandas Series"):
        _validate_inputs(replace(base, returns=pd.DataFrame(base.returns)), n=1, block=1)
    with pytest.raises(TypeError, match="equity_curve must be a pandas Series"):
        _validate_inputs(
            replace(base, equity_curve=base.equity_curve.to_frame()), n=1, block=1
        )


def test_bootstrap_equity_sanitises_non_finite_alignment() -> None:
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    returns = pd.Series([0.05, 0.0, 0.01], index=idx)
    result = _make_result(returns)
    faulty_equity = result.equity_curve.copy()
    faulty_equity.iloc[0] = np.inf
    result = replace(result, equity_curve=faulty_equity)

    band = bootstrap_equity(result, n=10, block=2, random_state=np.random.default_rng(7))
    assert np.isfinite(band.to_numpy(dtype=float)).all()


def test_bootstrap_equity_handles_zero_denominator_and_extreme_base_value() -> None:
    idx = pd.date_range("2025-01-31", periods=3, freq="ME")
    returns = pd.Series([-1.0, 0.02, 0.01], index=idx)
    base_result = _make_result(returns)

    zero_band = bootstrap_equity(base_result, n=5, block=1, random_state=0)
    assert np.isfinite(zero_band.iloc[1:].to_numpy(dtype=float)).all()

    near_zero_returns = pd.Series([-0.999999, 0.02, 0.01], index=idx)
    near_zero_result = _make_result(near_zero_returns)
    amplified_equity = near_zero_result.equity_curve.copy()
    amplified_equity.iloc[0] = np.float64(1e308)
    near_zero_result = replace(near_zero_result, equity_curve=amplified_equity)

    extreme_band = bootstrap_equity(
        near_zero_result,
        n=5,
        block=1,
        random_state=np.random.default_rng(9),
    )
    assert np.isfinite(extreme_band.to_numpy(dtype=float)).all()


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
        np.testing.assert_allclose(
            band.loc[active_mask, col], realised_curve.loc[active_mask]
        )


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


def test_bootstrap_equity_respects_equity_scale():
    idx = pd.date_range("2023-01-31", periods=6, freq="ME")
    returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005, 0.012], index=idx)
    base_result = _make_result(returns)

    scaled_equity = base_result.equity_curve * 250.0
    scaled_result = replace(base_result, equity_curve=scaled_equity)

    band_base = bootstrap_equity(base_result, n=200, block=3, random_state=0)
    band_scaled = bootstrap_equity(scaled_result, n=200, block=3, random_state=0)

    active_mask = returns.notna()
    pd.testing.assert_frame_equal(
        band_scaled.loc[active_mask].div(250.0),
        band_base.loc[active_mask],
    )
