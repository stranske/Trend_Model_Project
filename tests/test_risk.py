from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import trend_analysis.risk as risk
from trend_analysis.engine.optimizer import ConstraintViolation
from trend_analysis.risk import (
    RiskWindow,
    compute_constrained_weights,
    realised_volatility,
)


def test_realised_volatility_returns_annualised_series() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.015, 0.005],
            "B": [0.005, 0.007, -0.01, 0.012],
        },
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )
    window = RiskWindow(length=3, decay="simple")

    vol = realised_volatility(returns, window, periods_per_year=12)

    assert list(vol.columns) == ["A", "B"]
    assert len(vol) == 4
    # Last value should reflect the most recent window and be positive
    assert vol.iloc[-1]["A"] > 0
    assert vol.iloc[-1]["B"] > 0


def test_compute_constrained_weights_scales_and_constrains() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.005, 0.004, 0.011],
            "B": [-0.004, 0.006, 0.007, -0.003, 0.005],
            "C": [0.002, -0.001, 0.003, 0.004, 0.002],
        },
        index=pd.date_range("2021-01-31", periods=5, freq="ME"),
    )
    base = {"A": 0.5, "B": 0.3, "C": 0.2}

    weights, diagnostics = compute_constrained_weights(
        base,
        returns,
        window=RiskWindow(length=3, decay="simple"),
        target_vol=0.15,
        periods_per_year=12,
        floor_vol=0.02,
        long_only=True,
        max_weight=0.6,
        previous_weights=None,
        max_turnover=None,
    )

    assert abs(weights.sum() - 1.0) < 1e-9
    assert (weights >= 0).all()
    assert diagnostics.turnover_value >= 0
    assert not diagnostics.asset_volatility.empty
    assert diagnostics.scale_factors.loc["A"] >= 0


def test_compute_constrained_weights_respects_turnover_cap() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.015, 0.005],
            "B": [0.005, 0.007, -0.01, 0.012],
        },
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )
    base = {"A": 0.7, "B": 0.3}
    prev = {"A": 0.2, "B": 0.8}

    weights, diagnostics = compute_constrained_weights(
        base,
        returns,
        window=RiskWindow(length=2, decay="simple"),
        target_vol=0.1,
        periods_per_year=12,
        floor_vol=0.01,
        long_only=True,
        max_weight=0.8,
        previous_weights=prev,
        max_turnover=0.2,
    )

    assert diagnostics.turnover_value <= 0.2000001
    assert abs(weights.sum() - 1.0) < 1e-9


def test_compute_constrained_weights_lambda_penalty_reduces_turnover() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.02, 0.015, -0.01, 0.025],
            "B": [-0.01, 0.01, 0.02, -0.005],
        },
        index=pd.date_range("2021-06-30", periods=4, freq="ME"),
    )
    base = {"A": 0.8, "B": 0.2}
    prev = {"A": 0.1, "B": 0.9}

    _, diag_zero = compute_constrained_weights(
        base,
        returns,
        window=RiskWindow(length=2, decay="simple"),
        target_vol=0.12,
        periods_per_year=12,
        floor_vol=0.01,
        long_only=True,
        max_weight=0.9,
        previous_weights=prev,
        lambda_tc=0.0,
        max_turnover=None,
    )
    _, diag_high = compute_constrained_weights(
        base,
        returns,
        window=RiskWindow(length=2, decay="simple"),
        target_vol=0.12,
        periods_per_year=12,
        floor_vol=0.01,
        long_only=True,
        max_weight=0.9,
        previous_weights=prev,
        lambda_tc=0.8,
        max_turnover=None,
    )

    assert diag_high.turnover_value < diag_zero.turnover_value


def test_compute_constrained_weights_respects_max_weight_with_vol_adjust() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.001, 0.002, 0.0015, 0.001, 0.0018],
            "B": [0.05, -0.04, 0.06, -0.05, 0.04],
            "C": [0.03, -0.02, 0.025, -0.03, 0.02],
        },
        index=pd.date_range("2023-01-31", periods=5, freq="ME"),
    )
    base = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}

    weights, diagnostics = compute_constrained_weights(
        base,
        returns,
        window=RiskWindow(length=3, decay="simple"),
        target_vol=0.2,
        periods_per_year=12,
        floor_vol=0.01,
        long_only=True,
        max_weight=0.35,
        previous_weights=None,
        max_turnover=None,
    )

    assert abs(weights.sum() - 1.0) < 1e-9
    assert weights.max() <= 0.35 + 1e-9
    assert diagnostics.scale_factors.loc["A"] > 1.0


def test_compute_constrained_weights_caps_after_large_vol_scaling() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.0005, 0.0007, 0.0006, 0.0008, 0.0007],
            "B": [0.06, -0.05, 0.07, -0.06, 0.05],
            "C": [0.04, -0.03, 0.035, -0.04, 0.03],
        },
        index=pd.date_range("2023-06-30", periods=5, freq="ME"),
    )
    base = {"A": 0.4, "B": 0.3, "C": 0.3}

    weights, diagnostics = compute_constrained_weights(
        base,
        returns,
        window=RiskWindow(length=3, decay="simple"),
        target_vol=0.6,
        periods_per_year=12,
        floor_vol=0.01,
        long_only=True,
        max_weight=0.35,
        previous_weights=None,
        max_turnover=None,
    )

    assert abs(weights.sum() - 1.0) < 1e-9
    assert weights.max() <= 0.35 + 1e-9
    assert diagnostics.scale_factors.loc["A"] > 1.0


def test_periods_per_year_from_code_defaults() -> None:
    assert risk.periods_per_year_from_code(None) == 12.0
    assert risk.periods_per_year_from_code("W") == 52.0
    assert risk.periods_per_year_from_code("unknown") == 12.0


def test_realised_volatility_validates_inputs() -> None:
    with pytest.raises(ValueError):
        risk.realised_volatility(pd.DataFrame(), risk.RiskWindow(length=3))
    with pytest.raises(ValueError):
        risk.realised_volatility(pd.DataFrame({"A": [0.1]}), risk.RiskWindow(length=0))
    with pytest.raises(ValueError):
        risk.realised_volatility(
            pd.DataFrame({"A": [0.1, -0.2]}),
            risk.RiskWindow(length=2, decay="ewma", ewma_lambda=1.5),
        )


def test_normalise_handles_zero_sum() -> None:
    series = pd.Series({"A": 0.0, "B": 0.0})
    normalised = risk._normalise(series)
    assert normalised.equals(series)


def test_enforce_turnover_cap_with_previous() -> None:
    target = pd.Series({"A": 0.7, "B": 0.3})
    prev = pd.Series({"A": 0.1, "B": 0.9})
    adjusted, turnover = risk._enforce_turnover_cap(target, prev, max_turnover=0.4)
    assert abs(adjusted.sum() - 1.0) < 1e-12
    assert turnover <= 0.4 + 1e-6


def test_compute_constrained_weights_validates_inputs() -> None:
    with pytest.raises(ValueError):
        risk.compute_constrained_weights(
            {},
            pd.DataFrame(),
            window=risk.RiskWindow(2),
            target_vol=0.1,
            periods_per_year=12,
            floor_vol=None,
            long_only=True,
            max_weight=None,
        )
    with pytest.raises(ValueError):
        risk.compute_constrained_weights(
            {},
            pd.DataFrame({"A": [0.1]}),
            window=risk.RiskWindow(2),
            target_vol=0.1,
            periods_per_year=12,
            floor_vol=None,
            long_only=True,
            max_weight=None,
        )

    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.02, 0.01]})
    base = {"A": 0.0, "B": 0.0}
    with pytest.raises(ConstraintViolation):
        risk.compute_constrained_weights(
            base,
            returns,
            window=risk.RiskWindow(1),
            target_vol=0.1,
            periods_per_year=12,
            floor_vol=None,
            long_only=True,
            max_weight=None,
        )


def test_compute_constrained_weights_uses_dummy_turnover_index() -> None:
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.0, 0.01]})
    base = {"A": 0.6, "B": 0.4}
    weights, diagnostics = risk.compute_constrained_weights(
        base,
        returns,
        window=risk.RiskWindow(2),
        target_vol=0.1,
        periods_per_year=12,
        floor_vol=None,
        long_only=True,
        max_weight=None,
        previous_weights={"A": 0.5},
        max_turnover=None,
    )
    assert "rebalance" in diagnostics.turnover.index.name


def test_ensure_series_accepts_series_and_sorts() -> None:
    weights = pd.Series({"B": 0.3, "A": 0.7}, dtype=float)

    result = risk._ensure_series(weights)

    assert list(result.index) == ["A", "B"]
    assert result.dtype == float


def test_realised_volatility_supports_ewma_decay() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.02, -0.01, 0.015],
            "B": [0.01, 0.005, -0.02],
        },
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )
    window = RiskWindow(length=2, decay="ewma", ewma_lambda=0.8)

    vol = risk.realised_volatility(returns, window, periods_per_year=12.0)

    alpha = 1.0 - window.ewma_lambda

    def _ewma_std(values: np.ndarray) -> float:
        series = pd.Series(values, dtype=float)
        return float(series.ewm(alpha=alpha, adjust=False).std(bias=False).iloc[-1])

    expected = (
        returns.astype(float)
        .rolling(window=window.length, min_periods=1)
        .apply(_ewma_std, raw=True)
        .mul(np.sqrt(12.0))
    )
    pdt.assert_frame_equal(vol, expected)
