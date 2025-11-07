import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trend_analysis import regimes


def test_normalise_settings_coercion_and_defaults() -> None:
    config: dict[str, Any] = {
        "enabled": "yes",
        "proxy": "  SPY  ",
        "method": "VOL",
        "lookback": "10",
        "smoothing": "0",  # coerced to minimum of 1
        "threshold": "0.25",
        "neutral_band": "-0.05",  # absolute value applied
        "min_observations": "-3",  # coerced to minimum of 1
        "risk_on_label": "  On  ",
        "risk_off_label": "  Off  ",
        "default_label": "  ",  # falls back to risk_on_label
        "cache": False,
        "annualise_volatility": False,
    }

    settings = regimes.normalise_settings(config)

    assert settings.enabled is True
    assert settings.proxy == "SPY"
    assert settings.method == "volatility"
    assert settings.lookback == 10
    assert settings.smoothing == 1
    assert math.isclose(settings.threshold, 0.25)
    assert math.isclose(settings.neutral_band, 0.05)
    assert settings.min_obs == 1
    assert settings.risk_on_label == "On"
    assert settings.risk_off_label == "Off"
    # Default label falls back to the cleaned risk-on label
    assert settings.default_label == "On"
    assert settings.cache is False
    assert settings.annualise_volatility is False


def test_compute_regimes_disabled_returns_empty() -> None:
    series = pd.Series(
        [0.01, 0.02], index=pd.date_range("2024-01-01", periods=2, freq="D")
    )
    disabled_settings = regimes.RegimeSettings(enabled=False)

    result = regimes.compute_regimes(series, disabled_settings, freq="D")

    assert result.empty
    assert result.dtype == "string"


def test_compute_regime_series_volatility_uses_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    proxy = pd.Series([0.01, -0.015, 0.02, -0.01, 0.005, 0.0], index=dates)

    settings = regimes.RegimeSettings(
        method="volatility",
        lookback=3,
        smoothing=2,
        threshold=0.02,
        neutral_band=0.0,
        cache=True,
        risk_on_label="Bull",
        risk_off_label="Bear",
        default_label="Flat",
    )

    class DummyCache:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, str, str]] = []

        def get_or_compute(
            self, dataset_hash: str, window: int, freq: str, method_tag: str, compute_fn
        ):
            self.calls.append((dataset_hash, window, freq, method_tag))
            result = compute_fn()
            assert isinstance(result, pd.Series)
            return result

    dummy_cache = DummyCache()
    monkeypatch.setattr(regimes, "compute_dataset_hash", lambda objs: "hash123")
    monkeypatch.setattr(regimes, "get_cache", lambda: dummy_cache)

    labels = regimes._compute_regime_series(
        proxy, settings, freq="D", periods_per_year=None
    )

    assert len(dummy_cache.calls) == 1
    dataset_hash, window, freq_code, method_tag = dummy_cache.calls[0]
    assert dataset_hash == "hash123"
    assert window == 3
    assert freq_code == "D"
    assert method_tag.startswith("regime_volatility_thr0.020000_")
    assert set(labels.unique()) <= {"Bull", "Bear", "Flat"}
    # Ensure the neutral band split has been applied across the index
    assert labels.index.equals(proxy.dropna().index)


def test_compute_regime_series_return_method_neutral_band() -> None:
    dates = pd.date_range("2024-02-01", periods=5, freq="ME")
    proxy = pd.Series([0.02, -0.01, 0.015, -0.02, 0.01], index=dates)

    settings = regimes.RegimeSettings(
        method="rolling_return",
        lookback=2,
        smoothing=1,
        threshold=0.0,
        neutral_band=0.01,
        cache=False,
        risk_on_label="Up",
        risk_off_label="Down",
        default_label="Flat",
    )

    labels = regimes._compute_regime_series(
        proxy, settings, freq="M", periods_per_year=12
    )

    assert not labels.empty
    assert set(labels.unique()) <= {"Up", "Down", "Flat"}
    # Neutral band > 0 should allow the default label to appear when signal is small
    assert "Flat" in set(labels.unique())


def test_aggregate_performance_by_regime_metrics_and_notes() -> None:
    dates = pd.date_range("2023-01-31", periods=4, freq="ME")
    regimes_series = pd.Series(
        ["Risk-On", "Risk-Off", "Risk-On", "Risk-Off"], index=dates, dtype="string"
    )
    returns = pd.Series([0.02, -0.01, 0.015, 0.0], index=dates)
    risk_free = pd.Series([0.001, 0.001, 0.001, 0.001], index=dates)

    settings = regimes.RegimeSettings(enabled=True, min_obs=2)

    table, notes = regimes.aggregate_performance_by_regime(
        {"Core": returns},
        risk_free,
        regimes_series,
        settings,
        periods_per_year=12,
    )

    assert list(table.columns.names) == ["portfolio", "regime"]
    assert table.loc["Observations", ("Core", "Risk-On")] == 2
    assert table.loc["Observations", ("Core", "All")] == 4
    assert not np.isnan(table.loc["CAGR", ("Core", "All")])
    assert not notes


def test_aggregate_performance_by_regime_insufficient_observations_note() -> None:
    dates = pd.date_range("2023-01-31", periods=3, freq="ME")
    regimes_series = pd.Series(
        ["Risk-On", "Risk-Off", "Risk-On"], index=dates, dtype="string"
    )
    returns = pd.Series([0.01, -0.02, 0.03], index=dates)

    settings = regimes.RegimeSettings(enabled=True, min_obs=5)

    table, notes = regimes.aggregate_performance_by_regime(
        {"Alpha": returns},
        0.0,
        regimes_series,
        settings,
        periods_per_year=12,
    )

    assert table.loc["Observations", ("Alpha", "Risk-On")] == 2
    assert math.isnan(table.loc["CAGR", ("Alpha", "Risk-On")])
    assert len(notes) == 3
    assert all("fewer than" in note for note in notes)


def test_build_regime_payload_missing_proxy() -> None:
    data = pd.DataFrame(
        {"Date": pd.date_range("2024-01-01", periods=3, freq="D"), "Other": [1, 2, 3]}
    )
    config: dict[str, Any] = {"enabled": True}

    payload = regimes.build_regime_payload(
        data=data,
        out_index=pd.DatetimeIndex([]),
        returns_map={"Core": pd.Series(dtype=float)},
        risk_free=0.0,
        config=config,
        freq_code="D",
        periods_per_year=252,
    )

    assert payload["notes"] == [
        "Regime proxy column not specified; skipping regime analysis."
    ]
    assert payload["labels"].empty


def test_build_regime_payload_generates_summary_and_notes() -> None:
    dates = pd.date_range("2024-03-01", periods=6, freq="D")
    data = pd.DataFrame(
        {"Date": dates, "Proxy": [0.01, -0.02, 0.015, -0.005, 0.02, -0.01]}
    )
    out_index = pd.date_range("2024-03-01", periods=7, freq="D")
    returns = pd.Series([0.03, -0.01, 0.02, 0.015, -0.005, 0.01], index=dates)

    config: dict[str, Any] = {
        "enabled": True,
        "proxy": "Proxy",
        "method": "rolling_return",
        "lookback": 2,
        "smoothing": 1,
        "threshold": 0.0,
        "neutral_band": 0.0,
        "min_observations": 10,
        "cache": False,
    }

    payload = regimes.build_regime_payload(
        data=data,
        out_index=out_index,
        returns_map={"Growth": returns},
        risk_free=0.0,
        config=config,
        freq_code="D",
        periods_per_year=252,
    )

    assert "settings" in payload and payload["settings"]["enabled"] is True
    assert not payload["labels"].empty
    assert payload["out_labels"].index.equals(out_index)
    assert payload["table"].columns.get_level_values("portfolio")[0] == "Growth"
    assert payload["summary"]
    # Notes are deduplicated and populated from insufficient observation handling
    assert payload["notes"]
