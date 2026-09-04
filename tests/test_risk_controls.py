"""Regression coverage for the allocation-stage risk controls."""

from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.risk import RiskWindow, compute_constrained_weights, realised_volatility


def test_low_target_volatility_leaves_cash_exposure() -> None:
    """A low target must cap exposure instead of silently remaining fully invested."""

    returns = pd.DataFrame(
        {
            "A": [0.01, -0.01, 0.02, -0.02, 0.01, -0.01],
            "B": [0.01, -0.01, 0.02, -0.02, 0.01, -0.01],
        },
        index=pd.date_range("2024-01-31", periods=6, freq="ME"),
    )

    weights, diagnostics = compute_constrained_weights(
        {"A": 0.5, "B": 0.5},
        returns,
        window=RiskWindow(length=6),
        target_vol=0.02,
        periods_per_year=12,
        floor_vol=None,
        long_only=True,
        max_weight=None,
    )

    # Derive the exposure cap from these inputs rather than pinning an
    # implementation-specific literal. A 2% target must leave the balance in
    # cash instead of normalizing the asset weights back to 100% exposure.
    fully_invested_returns = returns.mul(pd.Series({"A": 0.5, "B": 0.5}), axis=1).sum(axis=1)
    fully_invested_volatility = realised_volatility(
        fully_invested_returns.to_frame("portfolio"),
        RiskWindow(length=6),
        periods_per_year=12,
    )["portfolio"].dropna().iloc[-1]
    expected_exposure = min(1.0, 0.02 / fully_invested_volatility)
    assert weights.sum() == pytest.approx(expected_exposure)
    assert diagnostics.portfolio_volatility.dropna().iloc[-1] == pytest.approx(0.02)
