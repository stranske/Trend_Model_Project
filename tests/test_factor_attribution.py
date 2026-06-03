from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.metrics.factor_attribution import factor_exposures


def test_recovers_planted_betas() -> None:
    index = pd.date_range("2025-01-31", periods=8, freq="ME")
    factors = pd.DataFrame(
        {
            "equity": [0.01, -0.02, 0.03, 0.00, 0.04, -0.01, 0.02, -0.03],
            "trend": [0.03, 0.01, -0.02, 0.04, -0.01, 0.02, -0.03, 0.00],
        },
        index=index,
    )
    alpha = 0.001
    manager = 0.6 * factors["equity"] - 0.2 * factors["trend"] + alpha
    returns = pd.DataFrame({"manager_a": manager}, index=index)

    exposures = factor_exposures(returns, factors)

    assert exposures.loc["manager_a", "equity"] == pytest.approx(0.6, abs=1e-6)
    assert exposures.loc["manager_a", "trend"] == pytest.approx(-0.2, abs=1e-6)
    assert exposures.loc["manager_a", "alpha"] == pytest.approx(alpha, abs=1e-6)
    assert exposures.loc["manager_a", "r_squared"] >= 1 - 1e-6


def test_raises_on_insufficient_observations() -> None:
    index = pd.RangeIndex(3)
    returns = pd.DataFrame({"manager_a": [0.01, 0.02, 0.03]}, index=index)
    factors = pd.DataFrame(
        {
            "equity": [0.01, 0.02, 0.03],
            "trend": [0.00, 0.01, 0.02],
        },
        index=index,
    )

    with pytest.raises(ValueError, match="insufficient observations after alignment"):
        factor_exposures(returns, factors)


def test_aligns_and_drops_nan_rows() -> None:
    returns = pd.DataFrame(
        {"manager_a": [0.001, 0.006, np.nan, 0.004, 0.007, 0.002]},
        index=pd.RangeIndex(6),
    )
    factors = pd.DataFrame(
        {
            "equity": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
            "trend": [0.01, 0.00, 0.03, 0.02, 0.05, 0.04],
        },
        index=pd.RangeIndex(1, 7),
    )

    exposures = factor_exposures(returns, factors)

    assert list(exposures.columns) == ["equity", "trend", "alpha", "r_squared"]
    assert list(exposures.index) == ["manager_a"]
