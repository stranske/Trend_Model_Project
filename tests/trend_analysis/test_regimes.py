"""Test suite targeting :mod:`trend_analysis.regimes`."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

import trend_analysis.regimes as regimes


def test_coerce_positive_int_handles_invalid_inputs() -> None:
    """The helper should coerce values to integers respecting the minimum."""

    assert regimes._coerce_positive_int("7", 3) == 7
    # ``None`` falls back to the default but not below the minimum bound.
    assert regimes._coerce_positive_int(None, 0, minimum=2) == 2
    # Negative numbers clamp at the provided minimum.
    assert regimes._coerce_positive_int(-4, 5) == 1


def test_coerce_float_falls_back_to_default() -> None:
    assert regimes._coerce_float("3.14", 1.0) == pytest.approx(3.14)
    # Non-convertible entries fall back to the provided default value.
    assert regimes._coerce_float(object(), 2.5) == pytest.approx(2.5)


def test_normalise_settings_defaults_when_none() -> None:
    settings = regimes.normalise_settings(None)
    assert isinstance(settings, regimes.RegimeSettings)
    assert settings.enabled is False


def test_normalise_settings_defaults_empty_labels() -> None:
    cfg = {"risk_on_label": "  ", "risk_off_label": "  ", "default_label": ""}
    settings = regimes.normalise_settings(cfg)
    assert settings.risk_on_label == "Risk-On"
    assert settings.risk_off_label == "Risk-Off"
    assert settings.default_label == "Risk-On"


def test_normalise_settings_interprets_user_mapping() -> None:
    cfg: Mapping[str, Any] = {
        "enabled": "yes",
        "proxy": " Proxy  ",
        "method": "VOL",
        "lookback": "21",
        "smoothing": 4.9,
        "threshold": "0.05",
        "neutral_band": "-0.2",  # magnitude is enforced positive
        "min_observations": "7",
        "cache": False,
        "annualise_volatility": False,
        "risk_on_label": "  Bull  ",
        "risk_off_label": "Bear  ",
        "default_label": "  ",  # empty coerces to risk-on label
    }

    settings = regimes.normalise_settings(cfg)

    assert settings.enabled is True
    assert settings.proxy == "Proxy"
    assert settings.method == "volatility"
    assert settings.lookback == 21
    assert settings.smoothing == 4
    assert settings.threshold == pytest.approx(0.05)
    assert settings.neutral_band == pytest.approx(0.2)
    assert settings.min_obs == 7
    assert settings.cache is False
    assert settings.annualise_volatility is False
    assert settings.risk_on_label == "Bull"
    assert settings.risk_off_label == "Bear"
    assert settings.default_label == "Bull"


def test_rolling_return_signal_supports_smoothing() -> None:
    series = pd.Series([0.05, -0.02, 0.01, 0.03, 0.04], index=pd.RangeIndex(5))

    with pytest.raises(ValueError):
        regimes._rolling_return_signal(series, window=0, smoothing=1)

    result = regimes._rolling_return_signal(series, window=3, smoothing=2)

    # Manual calculation: rolling product over 3 periods then 2-period mean.
    expected = (1.0 + series).rolling(3).apply(np.prod, raw=True) - 1.0
    expected = expected.rolling(2).mean()
    pd.testing.assert_series_equal(result, expected)


def test_rolling_volatility_signal_supports_annualisation() -> None:
    series = pd.Series([0.01, 0.02, -0.01, 0.00, 0.03], index=pd.RangeIndex(5))

    with pytest.raises(ValueError):
        regimes._rolling_volatility_signal(
            series, window=0, smoothing=1, periods_per_year=12, annualise=True
        )

    result = regimes._rolling_volatility_signal(
        series,
        window=3,
        smoothing=2,
        periods_per_year=12,
        annualise=True,
    )

    expected = series.rolling(3).std(ddof=0).rolling(2).mean() * np.sqrt(12)
    pd.testing.assert_series_equal(result, expected)


def test_rolling_volatility_signal_without_smoothing_or_annualise() -> None:
    series = pd.Series([0.01, 0.02, 0.03], index=pd.RangeIndex(3))
    result = regimes._rolling_volatility_signal(
        series, window=2, smoothing=1, periods_per_year=0.0, annualise=False
    )
    expected = series.rolling(2).std(ddof=0)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "freq, expected",
    [
        ("A", 1.0),
        ("Q", 4.0),
        ("M", 12.0),
        ("W", 52.0),
        ("B", 252.0),
        ("", 252.0),
    ],
)
def test_default_periods_per_year_mapping(freq: str, expected: float) -> None:
    assert regimes._default_periods_per_year(freq) == expected


def test_compute_regime_series_uses_cache_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    dates = pd.date_range("2022-01-01", periods=10, freq="B")
    proxy = pd.Series(np.linspace(-0.02, 0.05, len(dates)), index=dates)

    settings = regimes.RegimeSettings(
        enabled=True,
        method="volatility",
        lookback=3,
        smoothing=2,
        threshold=0.01,
        neutral_band=0.005,
        cache=True,
        annualise_volatility=True,
    )

    calls: list[tuple[str, int, str, str]] = []

    class DummyCache:
        def get_or_compute(
            self,
            dataset_hash: str,
            window: int,
            freq: str,
            method: str,
            compute_fn: Any,
        ) -> pd.Series:
            calls.append((dataset_hash, window, freq, method))
            return compute_fn()

    monkeypatch.setenv("TREND_ROLLING_CACHE", str(tmp_path))
    monkeypatch.setattr(regimes, "get_cache", lambda: DummyCache())

    result = regimes._compute_regime_series(
        proxy,
        settings,
        freq="B",
        periods_per_year=None,
    )

    assert not result.empty
    # Cache was invoked with the expected parameters.
    assert calls and calls[0][1] == settings.lookback
    assert "regime_volatility" in calls[0][3]


def test_compute_regime_series_handles_zero_periods() -> None:
    dates = pd.date_range("2022-01-01", periods=5, freq="B")
    proxy = pd.Series(np.linspace(0.01, 0.05, len(dates)), index=dates)
    settings = regimes.RegimeSettings(
        enabled=True,
        method="volatility",
        lookback=2,
        smoothing=1,
        cache=False,
    )
    labels = regimes._compute_regime_series(proxy, settings, freq="B", periods_per_year=0)
    assert isinstance(labels, pd.Series)


def test_compute_regime_series_handles_edge_cases() -> None:
    empty = pd.Series(dtype=float)
    settings = regimes.RegimeSettings(enabled=True)
    assert regimes._compute_regime_series(empty, settings, freq="D", periods_per_year=None).empty

    nan_series = pd.Series([np.nan, np.nan], index=pd.date_range("2022-01-01", periods=2))
    assert regimes._compute_regime_series(nan_series, settings, freq="D", periods_per_year=None).empty


def test_compute_regime_series_respects_neutral_band_and_history() -> None:
    dates = pd.date_range("2022-01-01", periods=4, freq="D")
    proxy = pd.Series([0.01, -0.02, 0.03, -0.01], index=dates)
    settings = regimes.RegimeSettings(
        enabled=True,
        lookback=5,
        smoothing=1,
        neutral_band=0.0,
        cache=False,
    )
    labels = regimes._compute_regime_series(proxy, settings, freq="D", periods_per_year=0)
    # Insufficient history returns the pre-filled labels.
    assert (labels == settings.default_label).all()


def test_compute_regimes_respects_enabled_flag() -> None:
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    proxy = pd.Series(np.linspace(0.0, 0.02, len(dates)), index=dates)

    disabled = regimes.RegimeSettings(enabled=False)
    assert regimes.compute_regimes(proxy, disabled, freq="D").empty

    enabled = regimes.RegimeSettings(enabled=True, lookback=2, smoothing=1, cache=False)
    result = regimes.compute_regimes(proxy, enabled, freq="D")
    assert isinstance(result, pd.Series)
    assert result.dtype == "string"


def test_format_hit_rate_handles_missing_data() -> None:
    assert np.isnan(regimes._format_hit_rate(pd.Series(dtype=float)))

    series = pd.Series([1.0, -1.0, 2.0, 0.0])
    hit_rate = regimes._format_hit_rate(series)
    assert hit_rate == pytest.approx(0.5)

    nan_series = pd.Series([np.nan, np.nan])
    assert np.isnan(regimes._format_hit_rate(nan_series))


def test_summarise_regime_outcome_covers_scenarios() -> None:
    settings = regimes.RegimeSettings(risk_on_label="On", risk_off_label="Off")

    # No data for either side
    assert regimes._summarise_regime_outcome(settings, None, None) is None

    off_only = pd.Series({"CAGR": 0.04})
    summary = regimes._summarise_regime_outcome(settings, None, off_only)
    assert "insufficient data" in summary

    on_only = pd.Series({"CAGR": 0.08})
    summary = regimes._summarise_regime_outcome(settings, on_only, None)
    assert "insufficient data" in summary

    both = (
        pd.Series({"CAGR": 0.12}),
        pd.Series({"CAGR": 0.05}),
    )
    summary = regimes._summarise_regime_outcome(settings, *both)
    assert "outpacing" in summary

    similar = (
        pd.Series({"CAGR": 0.10}),
        pd.Series({"CAGR": 0.1005}),
    )
    summary = regimes._summarise_regime_outcome(settings, *similar)
    assert "Performance was similar" in summary

    inverse = (
        pd.Series({"CAGR": 0.04}),
        pd.Series({"CAGR": 0.06}),
    )
    summary = regimes._summarise_regime_outcome(settings, *inverse)
    assert "outperformed" in summary


def test_aggregate_performance_by_regime_builds_metrics() -> None:
    dates = pd.date_range("2021-01-01", periods=8, freq="ME")
    regimes_series = pd.Series(
        ["Risk-On", "Risk-Off", "Risk-On", "Risk-Off", "Risk-On", "Risk-Off", "Risk-On", "Risk-Off"],
        index=dates,
        dtype="string",
    )
    returns_map = {
        "Alpha": pd.Series(np.linspace(0.01, 0.08, len(dates)), index=dates),
        "Beta": pd.Series([0.02, 0.01], index=dates[:2]),
    }
    risk_free = pd.Series(0.001, index=dates)

    settings = regimes.RegimeSettings(enabled=True, min_obs=2)

    table, notes = regimes.aggregate_performance_by_regime(
        returns_map,
        risk_free,
        regimes_series,
        settings,
        periods_per_year=12,
    )

    assert not table.empty
    # Verify multi-index columns and expected metrics are present.
    assert ("Alpha", "Risk-On") in table.columns
    assert ("Alpha", "All") in table.columns
    assert "CAGR" in table.index
    # Beta series has insufficient observations triggering a deduplicated note.
    assert any("fewer than" in note for note in notes)


def test_aggregate_performance_deduplicates_notes() -> None:
    dates = pd.date_range("2021-01-01", periods=3, freq="ME")
    regimes_series = pd.Series(["Risk-On", "Risk-Off", "Risk-On"], index=dates, dtype="string")
    returns_map = {
        "Alpha": pd.Series([0.01, 0.02], index=dates[:2]),
        "Beta": pd.Series([0.03, 0.04], index=dates[:2]),
    }
    settings = regimes.RegimeSettings(enabled=True, min_obs=5)
    table, notes = regimes.aggregate_performance_by_regime(
        returns_map,
        0.0,
        regimes_series,
        settings,
        periods_per_year=12,
    )
    assert not table.empty
    # Deduplicated notes should only include one entry per unique message.
    risk_on_notes = [note for note in notes if note.startswith("Risk-On")]
    assert len(risk_on_notes) == 1


def test_aggregate_performance_handles_disabled_and_missing_regimes() -> None:
    settings = regimes.RegimeSettings(enabled=False)
    table, notes = regimes.aggregate_performance_by_regime({}, 0.0, pd.Series(dtype="string"), settings, periods_per_year=12)
    assert table.empty and notes == []

    settings = regimes.RegimeSettings(enabled=True)
    table, notes = regimes.aggregate_performance_by_regime(
        {"Alpha": pd.Series([0.01, 0.02])}, 0.0, pd.Series(dtype="string"), settings, periods_per_year=12
    )
    assert table.empty
    assert "Regime labels were unavailable" in notes[0]


def test_aggregate_performance_sufficient_data_no_notes() -> None:
    dates = pd.date_range("2021-01-01", periods=12, freq="ME")
    regimes_series = pd.Series(["Risk-On", "Risk-Off"] * 6, index=dates, dtype="string")
    returns_map = {"Alpha": pd.Series(np.linspace(0.01, 0.12, len(dates)), index=dates)}
    settings = regimes.RegimeSettings(enabled=True, min_obs=2)
    table, notes = regimes.aggregate_performance_by_regime(
        returns_map,
        0.0,
        regimes_series,
        settings,
        periods_per_year=12,
    )
    assert not table.empty
    assert notes == []


def test_build_regime_payload_covers_branching(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-01", periods=10, freq="ME")
    data = pd.DataFrame({
        "Date": dates,
        "Proxy": np.linspace(-0.01, 0.03, len(dates)),
    })
    returns_series = pd.Series(np.linspace(0.0, 0.05, len(dates)), index=dates)
    returns_map = {"Portfolio": returns_series}

    disabled = regimes.build_regime_payload(
        data=data,
        out_index=dates,
        returns_map=returns_map,
        risk_free=0.0,
        config={"enabled": False},
        freq_code="M",
        periods_per_year=12,
    )
    assert "disabled" in disabled["notes"][0]

    missing_proxy = regimes.build_regime_payload(
        data=data,
        out_index=dates,
        returns_map=returns_map,
        risk_free=0.0,
        config={"enabled": True},
        freq_code="M",
        periods_per_year=12,
    )
    assert "not specified" in missing_proxy["notes"][0]

    wrong_proxy = regimes.build_regime_payload(
        data=data,
        out_index=dates,
        returns_map=returns_map,
        risk_free=0.0,
        config={"enabled": True, "proxy": "Missing"},
        freq_code="M",
        periods_per_year=12,
    )
    assert "not found" in wrong_proxy["notes"][0]

    config = {
        "enabled": True,
        "proxy": "Proxy",
        "lookback": 3,
        "smoothing": 1,
        "neutral_band": 0.0,
        "threshold": 0.0,
        "cache": False,
    }

    payload = regimes.build_regime_payload(
        data=data,
        out_index=dates,
        returns_map=returns_map,
        risk_free=0.0,
        config=config,
        freq_code="M",
        periods_per_year=12,
    )

    assert isinstance(payload["labels"], pd.Series)
    assert payload["labels"].dtype == "string"
    assert isinstance(payload["table"], pd.DataFrame)
    assert payload["notes"]  # includes any fill/caching notes as applicable
    # Summary either reports performance or relays a note fallback.
    assert isinstance(payload["summary"], str)

    # Regimes missing data path populates notes and summary fallback.
    nan_data = data.assign(Proxy=np.nan)
    payload_missing = regimes.build_regime_payload(
        data=nan_data,
        out_index=dates,
        returns_map=returns_map,
        risk_free=0.0,
        config=config,
        freq_code="M",
        periods_per_year=12,
    )
    assert "did not produce regime labels" in payload_missing["notes"][0]

def test_build_regime_payload_summary_fallback() -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="ME")
    data = pd.DataFrame({"Date": dates, "Proxy": [0.01, -0.02, 0.03, -0.01, 0.02, -0.03]})
    returns_map = {"Alpha": pd.Series([0.01, -0.01, 0.02, -0.02, 0.01, -0.01], index=dates)}
    config = {
        "enabled": True,
        "proxy": "Proxy",
        "lookback": 3,
        "smoothing": 1,
        "neutral_band": 0.001,
        "threshold": 0.5,
        "cache": False,
        "min_observations": 5,
    }
    payload = regimes.build_regime_payload(
        data=data,
        out_index=dates,
        returns_map=returns_map,
        risk_free=0.0,
        config=config,
        freq_code="M",
        periods_per_year=12,
    )
    assert isinstance(payload["summary"], str)
    assert payload["summary"] == payload["notes"][0]


def test_build_regime_payload_handles_gap_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="ME")
    data = pd.DataFrame({"Date": dates, "Proxy": np.linspace(-0.01, 0.02, len(dates))})
    returns_map = {"Portfolio": pd.Series(np.linspace(0.0, 0.03, len(dates)), index=dates)}
    config = {
        "enabled": True,
        "proxy": "Proxy",
        "lookback": 3,
        "smoothing": 1,
        "neutral_band": 0.0,
        "threshold": 0.0,
        "cache": False,
    }

    class GapSeries(pd.Series):
        @property
        def _constructor(self):  # type: ignore[override]
            return GapSeries

        def reindex(self, index=None, *args, **kwargs):  # type: ignore[override]
            base = super().reindex(index[:-1], *args, **kwargs)
            return GapSeries(base)

    gap_series = GapSeries([np.nan, np.nan], index=dates[:2], dtype=float)

    monkeypatch.setattr(regimes, "compute_regimes", lambda *args, **kwargs: gap_series)
    monkeypatch.setattr(
        regimes,
        "aggregate_performance_by_regime",
        lambda *args, **kwargs: (pd.DataFrame(), ["Duplicate note", "Duplicate note", "Extra"]),
    )

    payload = regimes.build_regime_payload(
        data=data,
        out_index=dates,
        returns_map=returns_map,
        risk_free=0.0,
        config=config,
        freq_code="M",
        periods_per_year=12,
    )

    assert any("Regime labels contained gaps" in note for note in payload["notes"])
    assert any("Proxy series missing" in note for note in payload["notes"])
    assert payload["notes"].count("Duplicate note") == 1


def test_build_regime_payload_generates_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="ME")
    data = pd.DataFrame({"Date": dates, "Proxy": np.linspace(0.0, 0.02, len(dates))})
    returns_map = {"P": pd.Series(np.linspace(0.0, 0.04, len(dates)), index=dates)}
    config = {
        "enabled": True,
        "proxy": "Proxy",
        "lookback": 2,
        "smoothing": 1,
        "neutral_band": 0.001,
        "threshold": 0.0,
        "cache": False,
    }

    table = pd.DataFrame(
        {
            ("P", "Risk-On"): pd.Series({"CAGR": 0.12, "Sharpe": 1.0, "Max Drawdown": -0.1, "Hit Rate": 0.6, "Observations": 3}),
            ("P", "Risk-Off"): pd.Series({"CAGR": 0.04, "Sharpe": 0.5, "Max Drawdown": -0.2, "Hit Rate": 0.4, "Observations": 3}),
        }
    )

    monkeypatch.setattr(
        regimes,
        "compute_regimes",
        lambda *args, **kwargs: pd.Series(["Risk-On"] * len(dates), index=dates, dtype="string"),
    )
    monkeypatch.setattr(
        regimes,
        "aggregate_performance_by_regime",
        lambda *args, **kwargs: (table, []),
    )

    payload = regimes.build_regime_payload(
        data=data,
        out_index=dates,
        returns_map=returns_map,
        risk_free=0.0,
        config=config,
        freq_code="M",
        periods_per_year=12,
    )

    assert "outpacing" in payload["summary"]


def test_build_regime_payload_no_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="ME")
    data = pd.DataFrame({"Date": dates, "Proxy": np.linspace(0.0, 0.03, len(dates))})
    returns_map = {"P": pd.Series(np.linspace(0.0, 0.03, len(dates)), index=dates)}
    config = {
        "enabled": True,
        "proxy": "Proxy",
        "lookback": 2,
        "smoothing": 1,
        "neutral_band": 0.001,
        "threshold": 0.0,
        "cache": False,
    }

    monkeypatch.setattr(
        regimes,
        "compute_regimes",
        lambda *args, **kwargs: pd.Series(["Risk-On"] * len(dates), index=dates, dtype="string"),
    )
    monkeypatch.setattr(
        regimes,
        "aggregate_performance_by_regime",
        lambda *args, **kwargs: (pd.DataFrame(), []),
    )

    payload = regimes.build_regime_payload(
        data=data,
        out_index=dates,
        returns_map=returns_map,
        risk_free=0.0,
        config=config,
        freq_code="M",
        periods_per_year=12,
    )

    assert payload["notes"] == []

