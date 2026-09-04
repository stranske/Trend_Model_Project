"""Regression coverage for the allocation-stage risk controls."""

from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.risk import RiskWindow, compute_constrained_weights


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

    # The fully-invested portfolio is about 4.9% annualized, so the 2% target
    # must leave the balance in cash rather than normalizing the asset weights
    # back to 100% exposure.
    assert weights.sum() == pytest.approx(0.4082482904638631)
    assert diagnostics.portfolio_volatility.dropna().iloc[-1] == pytest.approx(0.02)
