"""Regression tests for A15: volatility-mode regime threshold validation.

In volatility mode the regime signal is ``threshold - volatility`` and realized
volatility is non-negative, so a non-positive ``regime.threshold`` collapses the
classification to all Risk-Off. ``normalise_settings`` must reject that config;
``rolling_return`` mode (the default) is unaffected.
"""

import pytest

from trend_analysis.regimes import normalise_settings


def test_volatility_regime_rejects_zero_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        normalise_settings({"method": "volatility", "threshold": 0.0})


def test_volatility_regime_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        normalise_settings({"method": "volatility", "threshold": -0.1})


def test_volatility_alias_rejects_zero_threshold() -> None:
    # "vol" / "std" normalise to the volatility method and must validate too.
    for alias in ("vol", "std"):
        with pytest.raises(ValueError, match="threshold must be positive"):
            normalise_settings({"method": alias, "threshold": 0.0})


def test_rolling_return_regime_allows_zero_threshold() -> None:
    settings = normalise_settings({"method": "rolling_return", "threshold": 0.0})

    assert settings.method == "rolling_return"
    assert settings.threshold == 0.0


def test_volatility_regime_accepts_positive_threshold() -> None:
    settings = normalise_settings({"method": "volatility", "threshold": 0.05})

    assert settings.method == "volatility"
    assert settings.threshold == 0.05
