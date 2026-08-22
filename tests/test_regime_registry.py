from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.regimes import (
    RegimeModel,
    RegimeSettings,
    _compute_regime_series,
    compute_regimes,
    normalise_settings,
    regime_registry,
    resolve_regime_model,
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


def test_late_registered_plugin_survives_config_normalization() -> None:
    model_name = "late_bound_regime_contract"

    settings = normalise_settings({"enabled": True, "model": model_name, "cache": False})
    assert settings.model == model_name

    @regime_registry.register(model_name)
    class LateBoundRegimeModel(RegimeModel):
        def classify(
            self,
            proxy: pd.Series,
            settings: RegimeSettings,
            *,
            freq: str,
            periods_per_year: float | None,
        ) -> pd.Series:
            del settings, freq, periods_per_year
            return pd.Series("Late-Bound", index=proxy.index, dtype="string")

    routed = compute_regimes(_proxy(), settings, freq="ME", periods_per_year=12)

    assert set(routed.dropna()) == {"Late-Bound"}


def test_unknown_regime_model_lists_available_at_live_boundary() -> None:
    settings = RegimeSettings(enabled=True, model="missing_regime_model", cache=False)

    with pytest.raises(ValueError, match="Unknown regime model 'missing_regime_model'") as exc:
        compute_regimes(_proxy(), settings, freq="ME", periods_per_year=12)

    message = str(exc.value)
    assert "Available models:" in message
    assert "binary_threshold" in message


def test_resolve_regime_model_uses_live_registry_diagnostic() -> None:
    with pytest.raises(ValueError, match="Available models:"):
        resolve_regime_model("not_a_real_regime_model")
