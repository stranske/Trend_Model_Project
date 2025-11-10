from __future__ import annotations

"""Focused soft-coverage tests for :mod:`trend_analysis.regimes`."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

import trend_analysis.regimes as regimes


class DummyCache:
    """Capture invocations to the caching layer."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str, str]] = []

    def get_or_compute(
        self, dataset_hash: str, window: int, freq: str, tag: str, func: Any
    ) -> pd.Series:
        self.calls.append((dataset_hash, window, freq, tag))
        return func()


def _sample_index(periods: int = 8, freq: str = "M") -> pd.DatetimeIndex:
    return pd.date_range("2024-01-31", periods=periods, freq=freq)


def test_regime_setting_coercion_and_defaults() -> None:
    assert regimes._coerce_positive_int("5", 2) == 5
    assert regimes._coerce_positive_int(None, 7, minimum=3) == 7
    assert regimes._coerce_positive_int(-1, 4) == 1
    assert regimes._coerce_float("1.25", 0.5) == pytest.approx(1.25)
    assert regimes._coerce_float(object(), 2.0) == pytest.approx(2.0)

    cfg = {
        "enabled": "yes",
        "proxy": "  SPX  ",
        "method": "VOL",
        "lookback": "4",
        "smoothing": "2",
        "threshold": "0.1",
        "neutral_band": "0.05",
        "min_observations": "3",
        "risk_on_label": " On ",
        "risk_off_label": " Off ",
        "default_label": " ",
        "cache": False,
        "annualise_volatility": False,
    }
    settings = regimes.normalise_settings(cfg)
    assert settings.enabled is True
    assert settings.proxy == "SPX"
    assert settings.method == "volatility"
    assert settings.lookback == 4
    assert settings.smoothing == 2
    assert settings.neutral_band == pytest.approx(0.05)
    assert settings.risk_on_label == "On"
    assert settings.risk_off_label == "Off"
    assert settings.default_label == "On"
    assert settings.cache is False
    assert settings.annualise_volatility is False

    assert regimes._default_periods_per_year("A") == 1.0
    assert regimes._default_periods_per_year("Q") == 4.0
    assert regimes._default_periods_per_year("M") == 12.0
    assert regimes._default_periods_per_year("daily") == 252.0
    assert regimes._default_periods_per_year("unknown") == 252.0


def test_normalise_settings_handles_none_and_blank_labels() -> None:
    default_settings = regimes.normalise_settings(None)
    assert isinstance(default_settings, regimes.RegimeSettings)
    assert default_settings.default_label == "Risk-On"

    cfg = {
        "risk_on_label": " ",
        "risk_off_label": "  ",
        "default_label": "",
    }
    normalised = regimes.normalise_settings(cfg)
    assert normalised.risk_on_label == "Risk-On"
    assert normalised.risk_off_label == "Risk-Off"
    assert normalised.default_label == "Risk-On"


def test_normalise_settings_risk_off_whitespace() -> None:
    settings = regimes.normalise_settings({"risk_off_label": " \t "})
    assert settings.risk_off_label == "Risk-Off"


def test_compute_regime_series_handles_empty_inputs() -> None:
    settings = regimes.RegimeSettings(enabled=True)
    empty_result = regimes._compute_regime_series(
        pd.Series(dtype=float), settings, freq="M", periods_per_year=12
    )
    assert empty_result.empty

    nan_proxy = pd.Series([np.nan, np.nan], index=_sample_index(2))
    nan_result = regimes._compute_regime_series(
        nan_proxy, settings, freq="M", periods_per_year=12
    )
    assert nan_result.empty


def test_compute_regime_series_volatility_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = _sample_index(6)
    proxy = pd.Series([0.0, 0.01, 0.015, 0.2, 0.18, 0.22], index=dates)
    settings = regimes.RegimeSettings(
        enabled=True,
        method="volatility",
        lookback=2,
        smoothing=1,
        threshold=0.12,
        neutral_band=0.0,
        cache=True,
        annualise_volatility=True,
    )

    cache = DummyCache()
    monkeypatch.setattr(regimes, "compute_dataset_hash", lambda seq: "hash")
    monkeypatch.setattr(regimes, "get_cache", lambda: cache)

    labels = regimes._compute_regime_series(
        proxy, settings, freq="M", periods_per_year=6.0
    )

    assert labels.dtype == "string"
    assert set(labels.dropna().unique()) >= {settings.risk_on_label, settings.risk_off_label}
    assert cache.calls and cache.calls[0][3].startswith("regime_volatility")


def test_compute_regime_series_return_with_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = _sample_index(5)
    proxy = pd.Series([0.02, 0.01, 0.0, -0.01, 0.03], index=dates)
    settings = regimes.RegimeSettings(
        enabled=True,
        method="rolling_return",
        lookback=2,
        smoothing=1,
        threshold=0.0,
        neutral_band=0.0,
        cache=True,
    )

    cache = DummyCache()
    monkeypatch.setattr(regimes, "compute_dataset_hash", lambda seq: "hash")
    monkeypatch.setattr(regimes, "get_cache", lambda: cache)

    labels = regimes._compute_regime_series(
        proxy, settings, freq="M", periods_per_year=None
    )

    assert not labels.empty
    assert cache.calls and cache.calls[0][3].startswith("regime_rolling_return")


def test_compute_regime_series_return_without_cache() -> None:
    dates = _sample_index(6, "B")
    proxy = pd.Series([0.02, 0.01, -0.03, 0.04, 0.02, -0.01], index=dates)
    settings = regimes.RegimeSettings(
        enabled=True,
        method="rolling_return",
        lookback=3,
        smoothing=2,
        threshold=0.0,
        neutral_band=0.01,
        cache=False,
    )

    labels = regimes._compute_regime_series(
        proxy, settings, freq="B", periods_per_year=None
    )

    assert labels.iloc[-1] in {settings.risk_on_label, settings.risk_off_label}
    assert settings.default_label in labels.values


def test_rolling_signal_helpers_cover_errors() -> None:
    series = pd.Series([0.01, -0.02, 0.03, 0.01])
    with pytest.raises(ValueError):
        regimes._rolling_return_signal(series, window=0, smoothing=1)
    rolling = regimes._rolling_return_signal(series, window=2, smoothing=2)
    assert -1.0 < rolling.iloc[-1] < 1.0
    assert not np.isnan(rolling.iloc[-1])

    with pytest.raises(ValueError):
        regimes._rolling_volatility_signal(
            series, window=0, smoothing=1, periods_per_year=1, annualise=True
        )
    vol = regimes._rolling_volatility_signal(
        series, window=2, smoothing=2, periods_per_year=4, annualise=True
    )
    assert vol.iloc[-1] > 0
    vol_no_annual = regimes._rolling_volatility_signal(
        series, window=2, smoothing=1, periods_per_year=None, annualise=False
    )
    assert vol_no_annual.iloc[-1] >= 0


def test_compute_regimes_respects_enabled_flag() -> None:
    series = pd.Series([0.01, 0.02, -0.01], index=_sample_index(3))
    disabled = regimes.RegimeSettings(enabled=False)
    enabled = regimes.RegimeSettings(enabled=True, lookback=2, smoothing=1, cache=False)

    assert regimes.compute_regimes(series, disabled, freq="M").empty
    result = regimes.compute_regimes(series, enabled, freq="M")
    assert not result.empty


def test_compute_regime_series_short_history_and_neutral_band() -> None:
    dates = _sample_index(3)
    proxy = pd.Series([0.05, 0.01, -0.02], index=dates)
    settings = regimes.RegimeSettings(
        enabled=True,
        method="rolling_return",
        lookback=5,
        smoothing=1,
        threshold=0.01,
        neutral_band=0.02,
        cache=False,
    )

    labels = regimes._compute_regime_series(proxy, settings, freq="M", periods_per_year=12)
    assert len(labels) == len(proxy)
    assert set(labels.unique()) <= {
        settings.default_label,
        settings.risk_on_label,
        settings.risk_off_label,
    }


def test_compute_regime_series_volatility_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = _sample_index(5)
    proxy = pd.Series([0.0, 0.02, 0.01, 0.03, 0.015], index=dates)
    settings = regimes.RegimeSettings(
        enabled=True,
        method="volatility",
        lookback=3,
        smoothing=1,
        threshold=0.2,
        neutral_band=0.0,
        cache=True,
        annualise_volatility=False,
    )
    cache = DummyCache()
    monkeypatch.setattr(regimes, "compute_dataset_hash", lambda seq: "vol-hash")
    monkeypatch.setattr(regimes, "get_cache", lambda: cache)

    labels = regimes._compute_regime_series(proxy, settings, freq="W", periods_per_year=None)
    assert not labels.empty
    assert cache.calls[0][3].endswith("_annual0_ppy52.000000")


def test_hit_rate_and_summary_variants() -> None:
    empty = pd.Series(dtype=float)
    assert np.isnan(regimes._format_hit_rate(empty))

    mixed = pd.Series([1, -1, 0, 2], dtype=float)
    assert regimes._format_hit_rate(mixed) == pytest.approx(0.5)

    settings = regimes.RegimeSettings(risk_on_label="On", risk_off_label="Off")
    risk_on = pd.Series({"CAGR": 0.12})
    risk_off = pd.Series({"CAGR": 0.04})
    summary = regimes._summarise_regime_outcome(settings, risk_on, risk_off)
    assert "outpacing" in summary

    summary_similar = regimes._summarise_regime_outcome(
        settings, pd.Series({"CAGR": 0.1004}), pd.Series({"CAGR": 0.1003})
    )
    assert "Performance was similar" in summary_similar

    summary_off = regimes._summarise_regime_outcome(
        settings, pd.Series({"CAGR": -0.02}), pd.Series({"CAGR": 0.05})
    )
    assert "outperformed" in summary_off

    off_only = pd.Series({"CAGR": 0.06})
    assert "insufficient" in regimes._summarise_regime_outcome(settings, None, off_only)
    on_only = pd.Series({"CAGR": 0.08})
    assert "insufficient" in regimes._summarise_regime_outcome(settings, on_only, None)


def test_hit_rate_handles_all_nan_values() -> None:
    nan_series = pd.Series([np.nan, np.nan], dtype=float)
    assert np.isnan(regimes._format_hit_rate(nan_series))


def test_summary_returns_none_when_no_cagr() -> None:
    settings = regimes.RegimeSettings()
    summary = regimes._summarise_regime_outcome(
        settings, pd.Series({"Other": 1.0}), pd.Series({"Else": 2.0})
    )
    assert summary is None


def test_aggregate_performance_and_build_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = _sample_index(6)
    returns = pd.Series([0.01, 0.0, -0.02, 0.03, 0.02, -0.01], index=dates)
    regimes_series = pd.Series(
        ["Risk-On", "Risk-On", "Risk-Off", "Risk-Off", "Risk-On", "Risk-Off"],
        index=dates,
        dtype="string",
    )
    risk_free = pd.Series(0.0, index=dates)
    settings = regimes.RegimeSettings(enabled=True, min_obs=2)

    monkeypatch.setattr(regimes, "annual_return", lambda series, periods_per_year: 0.2)
    monkeypatch.setattr(regimes, "sharpe_ratio", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(regimes, "max_drawdown", lambda series: -0.3)

    table, notes = regimes.aggregate_performance_by_regime(
        {"Strategy": returns}, risk_free, regimes_series, settings, periods_per_year=12
    )
    assert not table.empty
    assert table.loc["CAGR", ("Strategy", "Risk-On")] == pytest.approx(0.2)
    assert "observations" not in " ".join(notes).lower()

    extra_index = dates.append(pd.DatetimeIndex([dates[-1] + pd.offsets.MonthEnd()]))
    data = pd.DataFrame({"Date": dates, "Proxy": returns.values, "Other": returns.values})

    summary_table = table.copy()
    summary_table.loc["Observations", :] = 4.0
    summary_table.loc["CAGR", ("Strategy", "Risk-Off")] = -0.05
    summary_table.loc["CAGR", ("Strategy", "Risk-On")] = 0.10

    def fake_compute_regimes(*args: Any, **kwargs: Any) -> pd.Series:
        return regimes_series

    def fake_aggregate(
        returns_map: dict[str, pd.Series],
        rf: pd.Series,
        out_labels: pd.Series,
        cfg: regimes.RegimeSettings,
        *,
        periods_per_year: float,
    ) -> tuple[pd.DataFrame, list[str]]:
        return summary_table, ["Duplicate note", "Duplicate note"]

    monkeypatch.setattr(regimes, "compute_regimes", fake_compute_regimes)
    monkeypatch.setattr(regimes, "aggregate_performance_by_regime", fake_aggregate)

    payload = regimes.build_regime_payload(
        data=data,
        out_index=extra_index,
        returns_map={"Strategy": returns},
        risk_free=risk_free,
        config={"enabled": True, "proxy": "Proxy"},
        freq_code="M",
        periods_per_year=12,
    )

    assert payload["labels"].equals(regimes_series)
    assert payload["out_labels"].index.equals(extra_index)
    assert payload["summary"] is not None and "Risk-On" in payload["summary"]
    assert payload["notes"] == ["Duplicate note"]

    disabled = regimes.build_regime_payload(
        data=data,
        out_index=dates,
        returns_map={"Strategy": returns},
        risk_free=risk_free,
        config={"enabled": False},
        freq_code="M",
        periods_per_year=12,
    )
    assert disabled["notes"] == ["Regime analysis disabled in configuration."]

    missing_proxy = regimes.build_regime_payload(
        data=data,
        out_index=dates,
        returns_map={"Strategy": returns},
        risk_free=risk_free,
        config={"enabled": True},
        freq_code="M",
        periods_per_year=12,
    )
    assert missing_proxy["notes"] == [
        "Regime proxy column not specified; skipping regime analysis."
    ]

    data_without_proxy = pd.DataFrame({"Date": dates, "Other": returns.values})
    missing_column = regimes.build_regime_payload(
        data=data_without_proxy,
        out_index=dates,
        returns_map={"Strategy": returns},
        risk_free=risk_free,
        config={"enabled": True, "proxy": "Proxy"},
        freq_code="M",
        periods_per_year=12,
    )
    assert "Proxy column 'Proxy' not found" in missing_column["notes"][0]


def test_aggregate_performance_handles_disabled_and_empty() -> None:
    disabled_settings = regimes.RegimeSettings(enabled=False)
    table, notes = regimes.aggregate_performance_by_regime(
        {}, 0.0, pd.Series(dtype="string"), disabled_settings, periods_per_year=12
    )
    assert table.empty and notes == []

    active_settings = regimes.RegimeSettings(enabled=True)
    table, notes = regimes.aggregate_performance_by_regime(
        {"A": pd.Series([0.01, 0.02], index=_sample_index(2))},
        0.0,
        pd.Series(dtype="string"),
        active_settings,
        periods_per_year=12,
    )
    assert table.empty and "Regime labels were unavailable" in notes[0]


def test_aggregate_performance_notes_for_low_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = _sample_index(4)
    regimes_series = pd.Series(
        ["Risk-On", "Risk-Off", "Risk-On", "Risk-Off"], index=dates, dtype="string"
    )
    returns = pd.Series([0.01, -0.02, 0.03, -0.01], index=dates)
    settings = regimes.RegimeSettings(enabled=True, min_obs=3)

    monkeypatch.setattr(regimes, "annual_return", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(regimes, "sharpe_ratio", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(regimes, "max_drawdown", lambda *args, **kwargs: 0.0)

    table, notes = regimes.aggregate_performance_by_regime(
        {"Strategy": returns}, 0.0, regimes_series, settings, periods_per_year=12
    )
    assert table.loc["Observations", ("Strategy", "Risk-On")] == 2.0
    assert any("fewer than" in note for note in notes)


def test_aggregate_performance_all_period_note(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = _sample_index(2)
    regimes_series = pd.Series(["Risk-On", "Risk-Off"], index=dates, dtype="string")
    returns = pd.Series([0.01, -0.02], index=dates)
    settings = regimes.RegimeSettings(enabled=True, min_obs=3)

    monkeypatch.setattr(regimes, "annual_return", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(regimes, "sharpe_ratio", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(regimes, "max_drawdown", lambda *args, **kwargs: 0.0)

    table, notes = regimes.aggregate_performance_by_regime(
        {"Strategy": returns}, 0.0, regimes_series, settings, periods_per_year=12
    )
    assert any("All-period aggregate" in note for note in notes)


def test_build_regime_payload_handles_empty_regimes(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = _sample_index(2)
    data = pd.DataFrame({"Date": dates, "Proxy": [0.0, 0.0]})
    returns_map = {"Strategy": pd.Series([0.01, 0.02], index=dates)}
    risk_free = pd.Series(0.0, index=dates)

    monkeypatch.setattr(
        regimes, "compute_regimes", lambda *args, **kwargs: pd.Series(dtype="string")
    )

    payload = regimes.build_regime_payload(
        data=data,
        out_index=dates,
        returns_map=returns_map,
        risk_free=risk_free,
        config={"enabled": True, "proxy": "Proxy"},
        freq_code="M",
        periods_per_year=12,
    )
    assert "did not produce regime labels" in payload["notes"][0]


def test_build_regime_payload_gap_note(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = _sample_index(3)
    data = pd.DataFrame({"Date": dates, "Proxy": [0.0, 0.0, 0.0]})
    returns_map: dict[str, pd.Series] = {}
    risk_free = 0.0

    regimes_series = pd.Series([pd.NA, pd.NA, pd.NA], index=dates, dtype="string")

    monkeypatch.setattr(regimes, "compute_regimes", lambda *args, **kwargs: regimes_series)
    monkeypatch.setattr(
        regimes,
        "aggregate_performance_by_regime",
        lambda *args, **kwargs: (pd.DataFrame(), []),
    )

    payload = regimes.build_regime_payload(
        data=data,
        out_index=dates,
        returns_map=returns_map,
        risk_free=risk_free,
        config={"enabled": True, "proxy": "Proxy"},
        freq_code="M",
        periods_per_year=12,
    )
    assert "Regime labels contained gaps" in payload["notes"][0]


def test_build_regime_payload_missing_out_index_note(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = _sample_index(3)
    data = pd.DataFrame({"Date": dates, "Proxy": [0.01, 0.02, 0.03]})

    class ShortReindexSeries(pd.Series):
        @property
        def _constructor(self) -> type[pd.Series]:
            return ShortReindexSeries

        def reindex(self, index=None, *args: Any, **kwargs: Any) -> pd.Series:  # type: ignore[override]
            result = super().reindex(index, *args, **kwargs)
            if index is not None and len(index) > 1:
                return result.iloc[:-1]
            return result

    base = pd.Series(["Risk-On", "Risk-Off", "Risk-On"], index=dates, dtype="string")
    short_series = ShortReindexSeries(base)

    monkeypatch.setattr(regimes, "compute_regimes", lambda *args, **kwargs: short_series)
    monkeypatch.setattr(
        regimes,
        "aggregate_performance_by_regime",
        lambda *args, **kwargs: (pd.DataFrame(), []),
    )

    payload = regimes.build_regime_payload(
        data=data,
        out_index=dates,
        returns_map={},
        risk_free=0.0,
        config={"enabled": True, "proxy": "Proxy"},
        freq_code="M",
        periods_per_year=12,
    )
    assert any("Proxy series missing" in note for note in payload["notes"])


def test_build_regime_payload_summary_fallback_and_note_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = _sample_index(3)
    data = pd.DataFrame({"Date": dates, "Proxy": [0.0, 0.0, 0.0]})
    regimes_series = pd.Series(["Risk-On", "Risk-Off", "Risk-On"], index=dates, dtype="string")

    monkeypatch.setattr(regimes, "compute_regimes", lambda *args, **kwargs: regimes_series)
    monkeypatch.setattr(
        regimes,
        "aggregate_performance_by_regime",
        lambda *args, **kwargs: (pd.DataFrame(), ["Repeated", "Repeated"]),
    )

    payload = regimes.build_regime_payload(
        data=data,
        out_index=dates,
        returns_map={},
        risk_free=0.0,
        config={"enabled": True, "proxy": "Proxy"},
        freq_code="M",
        periods_per_year=12,
    )

    assert payload["summary"] == "Repeated"
    assert payload["notes"] == ["Repeated"]
