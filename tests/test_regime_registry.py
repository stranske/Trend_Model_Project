from __future__ import annotations

import pandas as pd

from trend_analysis.regimes import (
    RegimeModel,
    RegimeSettings,
    _compute_regime_series,
    compute_regimes,
    normalise_settings,
    regime_registry,
)


def _proxy() -> pd.Series:
    return pd.Series(
        [0.01, 0.02, -0.03, 0.01, 0.04, -0.02],
        index=pd.date_range("2024-01-31", periods=6, freq="ME"),
        name="SPX",
    )


def test_default_binary_regime_unchanged() -> None:
    settings = RegimeSettings(
        enabled=True,
        lookback=2,
        smoothing=1,
        threshold=0.0,
        neutral_band=0.0,
        cache=False,
    )

    expected = _compute_regime_series(
        _proxy(),
        settings,
        freq="ME",
        periods_per_year=12,
    )

    routed = compute_regimes(_proxy(), settings, freq="ME", periods_per_year=12)

    assert routed.equals(expected)


def test_registry_dispatches_named_model() -> None:
    @regime_registry.register("toy_all_weather")
    class ToyAllWeatherRegimeModel(RegimeModel):
        def classify(
            self,
            proxy: pd.Series,
            settings: RegimeSettings,
            *,
            freq: str,
            periods_per_year: float | None,
        ) -> pd.Series:
            del settings, freq, periods_per_year
            return pd.Series("All-Weather", index=proxy.index, dtype="string")

    settings = RegimeSettings(enabled=True, model="toy_all_weather", cache=False)

    routed = compute_regimes(_proxy(), settings, freq="ME", periods_per_year=12)

    assert set(routed.dropna()) == {"All-Weather"}


def test_normalise_settings_accepts_regime_model_name() -> None:
    settings = normalise_settings({"enabled": True, "model": "binary_threshold"})

    assert settings.model == "binary_threshold"
    assert "binary_threshold" in regime_registry.available()
