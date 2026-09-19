import pandas as pd
import pytest

from trend_analysis.regimes import compute_regimes, normalise_settings


def test_annualise_volatility_keeps_regime_classification_invariant() -> None:
    proxy = pd.Series(
        [0.0, 0.0, 0.012, 0.0, 0.04, -0.04, 0.001, -0.001],
        index=pd.date_range("2024-01-31", periods=8, freq="ME"),
    )
    base_config = {
        "enabled": True,
        "method": "volatility",
        "lookback": 2,
        "smoothing": 1,
        "threshold": 0.01,
        "neutral_band": 0.0,
        "cache": False,
    }

    non_annualised = compute_regimes(
        proxy,
        normalise_settings({**base_config, "annualise_volatility": False}),
        freq="M",
        periods_per_year=12,
    )
    annualised = compute_regimes(
        proxy,
        normalise_settings({**base_config, "annualise_volatility": True}),
        freq="M",
        periods_per_year=12,
    )

    assert {"Risk-On", "Risk-Off"} <= set(non_annualised)
    pd.testing.assert_series_equal(annualised, non_annualised)


def test_annualise_volatility_scales_neutral_band_with_signal() -> None:
    proxy = pd.Series(
        [0.0, 0.017, 0.017, 0.0, 0.04, -0.04, 0.001, -0.001],
        index=pd.date_range("2024-01-31", periods=8, freq="ME"),
    )
    base_config = {
        "enabled": True,
        "method": "volatility",
        "lookback": 2,
        "smoothing": 1,
        "threshold": 0.01,
        "neutral_band": 0.002,
        "default_label": "Neutral",
        "cache": False,
    }

    non_annualised = compute_regimes(
        proxy,
        normalise_settings({**base_config, "annualise_volatility": False}),
        freq="M",
        periods_per_year=12,
    )
    annualised = compute_regimes(
        proxy,
        normalise_settings({**base_config, "annualise_volatility": True}),
        freq="M",
        periods_per_year=12,
    )

    assert "Neutral" in set(non_annualised)
    pd.testing.assert_series_equal(annualised, non_annualised)


@pytest.mark.parametrize("field", ["enabled", "cache", "annualise_volatility"])
@pytest.mark.parametrize("flag_value", ["false", "true", "0", "no"])
def test_regime_control_values_reject_string_boolean_flags(field: str, flag_value: str) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        normalise_settings({field: flag_value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threshold", 10**1000),
        ("neutral_band", 10**1000),
        ("threshold", float("nan")),
        ("threshold", float("inf")),
        ("neutral_band", float("nan")),
        ("neutral_band", float("-inf")),
    ],
)
def test_regime_control_values_reject_non_finite_numeric_controls(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="must be a finite number"):
        normalise_settings({field: value})


def test_regime_control_values_reject_non_finite_and_string_boolean() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        normalise_settings({"enabled": "false", "cache": "false"})
    with pytest.raises(ValueError, match="must be a finite number"):
        normalise_settings({"neutral_band": float("nan")})
