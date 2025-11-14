"""Soft coverage for signal generation and risk utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.risk import (
    RiskDiagnostics,
    RiskWindow,
    _enforce_turnover_cap,
    _normalise,
    _scale_factors,
    compute_constrained_weights,
    periods_per_year_from_code,
    realised_volatility,
)
from trend_analysis.signals import (
    TrendSpec,
    _as_float_frame,
    _zscore_rows,
    compute_trend_signals,
)


def test_trend_spec_validation_guards_inputs() -> None:
    with pytest.raises(ValueError):
        TrendSpec(window=0)
    with pytest.raises(ValueError):
        TrendSpec(min_periods=0)
    with pytest.raises(ValueError):
        TrendSpec(lag=0)
    with pytest.raises(ValueError):
        TrendSpec(vol_target=-1.0)


def test_as_float_frame_and_zscore_rows() -> None:
    frame = pd.DataFrame({"A": ["1", 2], "B": ["3", 4]})
    converted = _as_float_frame(frame)
    assert converted.dtypes.tolist() == [float, float]

    zscored = _zscore_rows(converted)
    assert np.isclose(zscored.iloc[0].sum(), 0.0)
    assert np.isclose(zscored.iloc[1].sum(), 0.0)


def test_compute_trend_signals_supports_vol_adjust_and_zscore() -> None:
    returns = pd.DataFrame(
        {
            "AssetA": [0.02, 0.01, -0.01, 0.03, 0.02, 0.01],
            "AssetB": [0.01, 0.02, -0.02, 0.02, 0.01, 0.00],
        },
        index=pd.date_range("2024-01-31", periods=6, freq="M"),
    )
    spec = TrendSpec(
        window=3, min_periods=2, vol_adjust=True, vol_target=0.5, zscore=True
    )
    signal = compute_trend_signals(returns, spec)

    spec_payload = signal.attrs["spec"]
    assert spec_payload["window"] == 3
    assert spec_payload["zscore"] is True
    assert not signal.isna().any().any()


def test_compute_trend_signals_rejects_empty_frames() -> None:
    with pytest.raises(ValueError):
        compute_trend_signals(pd.DataFrame(), TrendSpec())


def test_periods_per_year_from_code_defaults() -> None:
    assert periods_per_year_from_code(None) == 12.0
    assert periods_per_year_from_code("w") == 52.0
    assert periods_per_year_from_code("unknown") == 12.0


def test_realised_volatility_supports_simple_and_ewma() -> None:
    returns = pd.DataFrame(
        {
            "AssetA": [0.01, 0.02, 0.03, -0.01, 0.00],
            "AssetB": [0.00, 0.01, 0.02, 0.01, -0.02],
        },
        index=pd.date_range("2024-01-31", periods=5, freq="M"),
    )
    window = RiskWindow(length=3)
    simple = realised_volatility(returns, window)
    assert simple.shape == returns.shape

    ewma = realised_volatility(
        returns, RiskWindow(length=3, decay="ewma", ewma_lambda=0.8)
    )
    assert not ewma.iloc[-1].isna().any()


def test_realised_volatility_validates_inputs() -> None:
    with pytest.raises(ValueError):
        realised_volatility(pd.DataFrame(), RiskWindow(length=3))
    with pytest.raises(ValueError):
        realised_volatility(pd.DataFrame({"A": [0.0]}), RiskWindow(length=0))
    with pytest.raises(ValueError):
        realised_volatility(
            pd.DataFrame({"A": [0.0]}),
            RiskWindow(length=2, decay="ewma", ewma_lambda=1.5),
        )


def test_scale_factors_and_normalise_behaviour() -> None:
    latest = pd.Series({"A": 0.2, "B": 0.5})
    factors = _scale_factors(latest, 1.0, floor_vol=0.1)
    assert factors.loc["A"] > factors.loc["B"]

    weights = pd.Series({"A": 0.3, "B": 0.7})
    normalised = _normalise(weights)
    assert np.isclose(normalised.sum(), 1.0)


def test_enforce_turnover_cap_scales_excess_turnover() -> None:
    target = pd.Series({"A": 0.6, "B": 0.4})
    prev = pd.Series({"A": 0.2, "B": 0.8})
    adjusted, turnover = _enforce_turnover_cap(target, prev, max_turnover=0.5)
    assert np.isclose(turnover, 0.5)
    assert np.isclose(adjusted.sum(), prev.sum())


def test_compute_constrained_weights_integration() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03, 0.02, 0.01],
            "B": [0.02, 0.01, -0.01, 0.00, 0.03],
        },
        index=pd.date_range("2024-01-31", periods=5, freq="M"),
    )
    base_weights = {"A": 0.6, "B": 0.4}

    final_weights, diagnostics = compute_constrained_weights(
        base_weights,
        returns,
        window=RiskWindow(length=3),
        target_vol=0.15,
        periods_per_year=12.0,
        floor_vol=0.05,
        long_only=True,
        max_weight=0.8,
        previous_weights={"A": 0.5, "B": 0.5},
        max_turnover=0.5,
        group_caps={"grp": 1.0},
        groups={"A": "grp", "B": "grp"},
    )

    assert isinstance(diagnostics, RiskDiagnostics)
    assert np.isclose(final_weights.sum(), 1.0)
    assert diagnostics.asset_volatility.shape[1] == 2
    assert diagnostics.turnover_value <= 0.5 + 1e-9
