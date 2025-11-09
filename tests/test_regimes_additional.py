"""Additional coverage for ``trend_analysis.regimes`` utility helpers."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from trend_analysis.regimes import (
    RegimeSettings,
    _coerce_float,
    _coerce_positive_int,
    _compute_regime_series,
    _default_periods_per_year,
    _format_hit_rate,
    _rolling_return_signal,
    _rolling_volatility_signal,
    _summarise_regime_outcome,
    aggregate_performance_by_regime,
    build_regime_payload,
    compute_regimes,
    normalise_settings,
)


def test_regime_settings_to_dict_and_normalisation() -> None:
    raw_config: dict[str, Any] = {
        "enabled": True,
        "proxy": "  SPX  ",
        "method": "VOL",
        "lookback": "21",
        "smoothing": "0",
        "threshold": "0.25",
        "neutral_band": "-0.5",
        "min_observations": 0,
        "risk_on_label": "",
        "risk_off_label": "  ",
        "default_label": "",
        "cache": False,
        "annualise_volatility": False,
    }

    settings = normalise_settings(raw_config)
    # ``normalise_settings`` must coerce strings and guard empty labels.
    assert settings.proxy == "SPX"
    assert settings.method == "volatility"
    assert settings.lookback == 21
    # Smoothing coerced to minimum value of 1.
    assert settings.smoothing == 1
    assert settings.threshold == pytest.approx(0.25)
    # Neutral band is treated as absolute value.
    assert settings.neutral_band == pytest.approx(0.5)
    # Minimum observation count never drops below 1.
    assert settings.min_obs == 1
    # Empty labels fall back to defaults.
    assert settings.risk_on_label == "Risk-On"
    assert settings.risk_off_label == "Risk-Off"
    assert settings.default_label == "Risk-On"
    assert settings.cache is False
    assert settings.annualise_volatility is False
    # ``to_dict`` exercises the dataclass helper.
    as_dict = settings.to_dict()
    assert as_dict["method"] == "volatility"
    assert as_dict["proxy"] == "SPX"


def test_normalise_settings_default_instance() -> None:
    assert normalise_settings(None) == RegimeSettings()


@pytest.mark.parametrize(
    "value,default,minimum,expected",
    [
        (None, 5, 2, 5),
        ("10", 5, 2, 10),
        (-3, 1, 2, 2),
    ],
)
def test_coerce_positive_int_handles_invalid_inputs(
    value: Any, default: int, minimum: int, expected: int
) -> None:
    assert _coerce_positive_int(value, default, minimum=minimum) == expected


def test_coerce_float_falls_back_to_default() -> None:
    assert _coerce_float("3.5", 0.0) == pytest.approx(3.5)
    assert _coerce_float(object(), 1.2) == pytest.approx(1.2)


def test_rolling_signals_cover_validation_and_smoothing() -> None:
    series = pd.Series([0.01, 0.02, -0.01, 0.03])
    rolled = _rolling_return_signal(series, window=2, smoothing=2)
    # Smoothing ensures the final observation averages consecutive compounded returns.
    products = []
    for i in range(1, len(series)):
        window_vals = series.iloc[i - 1 : i + 1]
        products.append(np.prod(1 + window_vals.values) - 1.0)
    expected = np.mean(products[-2:])
    assert rolled.iloc[-1] == pytest.approx(expected)

    with pytest.raises(ValueError):
        _rolling_return_signal(series, window=0, smoothing=1)

    vol = _rolling_volatility_signal(
        series,
        window=2,
        smoothing=2,
        periods_per_year=12,
        annualise=True,
    )
    assert vol.iloc[-1] > 0
    expected_vol = series.rolling(2).std(ddof=0).rolling(2).mean().iloc[-1]
    assert np.isclose(vol.iloc[-1], expected_vol * np.sqrt(12))

    with pytest.raises(ValueError):
        _rolling_volatility_signal(
            series, window=0, smoothing=1, periods_per_year=None, annualise=False
        )


def test_default_periods_per_year_mappings() -> None:
    assert _default_periods_per_year("A") == 1.0
    assert _default_periods_per_year("q") == 4.0
    assert _default_periods_per_year("monthly") == 12.0
    assert _default_periods_per_year("weekly") == 52.0
    # Daily/business day codes collapse to trading-day approximation.
    assert _default_periods_per_year("bd") == 252.0
    assert _default_periods_per_year("x") == 252.0


def test_compute_regime_series_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2024-01-31", periods=6, freq="M")
    proxy = pd.Series([0.01, 0.02, -0.01, 0.015, 0.03, 0.025], index=dates)
    settings = RegimeSettings(
        enabled=True,
        proxy="Proxy",
        method="volatility",
        lookback=2,
        smoothing=1,
        threshold=0.0,
        neutral_band=0.0,
        cache=True,
        annualise_volatility=True,
    )

    captured: dict[str, Any] = {}

    def fake_hash(objs: list[pd.Series]) -> str:
        captured["hash_input"] = objs
        return "hash"

    class FakeCache:
        def get_or_compute(
            self, dataset_hash: str, window: int, freq: str, tag: str, func: Any
        ) -> pd.Series:
            captured["cache_args"] = (dataset_hash, window, freq, tag)
            result = func()
            return pd.Series("Cached", index=result.index, dtype="string")

    monkeypatch.setattr("trend_analysis.regimes.compute_dataset_hash", fake_hash)
    monkeypatch.setattr("trend_analysis.regimes.get_cache", lambda: FakeCache())

    labels = _compute_regime_series(proxy, settings, freq="M", periods_per_year=None)

    assert labels.iloc[-1] == "Cached"
    dataset_hash, window, freq, tag = captured["cache_args"]
    assert dataset_hash == "hash"
    assert window == 2
    assert freq == "M"
    assert "regime_volatility" in tag
    assert "annual1" in tag
    assert "ppy12" in tag  # Monthly cadence defaults to 12 periods per year.


def test_compute_regime_series_without_cache_returns_labels() -> None:
    dates = pd.date_range("2024-01-31", periods=6, freq="M")
    proxy = pd.Series([0.01, 0.02, -0.01, 0.03, 0.025, -0.02], index=dates)
    settings = RegimeSettings(
        enabled=True,
        proxy="Proxy",
        cache=False,
        neutral_band=0.0,
        lookback=3,
        smoothing=1,
    )

    labels = _compute_regime_series(proxy, settings, freq="M", periods_per_year=12)
    assert set(labels.unique()) <= {settings.risk_on_label, settings.risk_off_label}


def test_compute_regime_series_handles_empty_input() -> None:
    settings = RegimeSettings(enabled=True)
    empty_series = pd.Series(dtype=float)
    assert _compute_regime_series(
        empty_series, settings, freq="M", periods_per_year=12
    ).empty
    nan_series = pd.Series(
        [np.nan, np.nan], index=pd.date_range("2024-01-31", periods=2, freq="M")
    )
    assert _compute_regime_series(
        nan_series, settings, freq="M", periods_per_year=12
    ).empty


def test_compute_regimes_disabled_returns_empty() -> None:
    settings = RegimeSettings(enabled=False)
    proxy = pd.Series(
        [0.01, 0.02], index=pd.date_range("2024-01-31", periods=2, freq="M")
    )
    result = compute_regimes(proxy, settings, freq="M", periods_per_year=12)
    assert result.empty


def test_format_hit_rate_variants() -> None:
    assert np.isnan(_format_hit_rate(pd.Series(dtype=float)))
    assert np.isnan(_format_hit_rate(pd.Series([np.nan, np.nan])))
    series = pd.Series([0.0, 1.0, -1.0, 0.5])
    assert _format_hit_rate(series) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "risk_on,risk_off,expected",
    [
        (None, None, None),
        (pd.Series({"CAGR": np.nan}), pd.Series({"CAGR": 0.1}), "Risk-Off"),
        (pd.Series({"CAGR": 0.2}), pd.Series({"CAGR": np.nan}), "Risk-On"),
        (pd.Series({"CAGR": 0.105}), pd.Series({"CAGR": 0.1045}), "similar"),
        (pd.Series({"CAGR": 0.2}), pd.Series({"CAGR": 0.1}), "outpacing"),
        (pd.Series({"CAGR": 0.05}), pd.Series({"CAGR": 0.15}), "outperformed"),
    ],
)
def test_summarise_regime_outcome_branches(
    risk_on: pd.Series | None, risk_off: pd.Series | None, expected: str | None
) -> None:
    settings = RegimeSettings(risk_on_label="Risk-On", risk_off_label="Risk-Off")
    summary = _summarise_regime_outcome(settings, risk_on, risk_off)
    if expected is None:
        assert summary is None
    else:
        assert summary is not None and expected.lower() in summary.lower()


def test_aggregate_performance_by_regime_edge_cases() -> None:
    settings = RegimeSettings(enabled=False)
    empty_table, empty_notes = aggregate_performance_by_regime(
        returns_map={},
        risk_free=0.0,
        regimes=pd.Series(dtype="string"),
        settings=settings,
        periods_per_year=12,
    )
    assert empty_table.empty and empty_notes == []

    settings = RegimeSettings(enabled=True, min_obs=10)
    regimes = pd.Series(dtype="string")
    table, notes = aggregate_performance_by_regime(
        returns_map={
            "Fund": pd.Series(
                [0.01, 0.02], index=pd.date_range("2024-01-31", periods=2, freq="M")
            )
        },
        risk_free=0.0,
        regimes=regimes,
        settings=settings,
        periods_per_year=12,
    )
    assert table.empty
    assert "unavailable" in notes[0]

    regimes = pd.Series(
        ["Risk-On", "Risk-Off"],
        index=pd.date_range("2024-01-31", periods=2, freq="M"),
        dtype="string",
    )
    series = pd.Series([0.01, -0.02], index=regimes.index)
    table, notes = aggregate_performance_by_regime(
        returns_map={"Fund": series},
        risk_free=pd.Series(0.0, index=regimes.index),
        regimes=regimes,
        settings=settings,
        periods_per_year=12,
    )
    # Observations fall below ``min_obs`` and produce deduplicated warnings.
    assert table.loc["Observations"].notna().any()
    assert any("risk-on" in note.lower() for note in notes)
    assert any("risk-off" in note.lower() for note in notes)
    assert any("all-period aggregate" in note.lower() for note in notes)


def test_build_regime_payload_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2024-01-31", periods=3, freq="M")
    data = pd.DataFrame(
        {"Date": dates, "Proxy": [0.01, 0.02, -0.01], "Fund": [0.02, 0.01, 0.03]}
    )
    returns_map = {"Fund": data.set_index("Date")["Fund"]}

    # Disabled configuration short-circuits early.
    payload = build_regime_payload(
        data=data,
        out_index=pd.Index([]),
        returns_map=returns_map,
        risk_free=0.0,
        config={"enabled": False},
        freq_code="M",
        periods_per_year=12,
    )
    assert payload["notes"] == ["Regime analysis disabled in configuration."]

    # Missing proxy value generates a specific note.
    payload = build_regime_payload(
        data=data,
        out_index=pd.Index([]),
        returns_map=returns_map,
        risk_free=0.0,
        config={"enabled": True},
        freq_code="M",
        periods_per_year=12,
    )
    assert "proxy column not specified" in payload["notes"][0].lower()

    # Proxy not present in the frame yields a skip message.
    payload = build_regime_payload(
        data=data,
        out_index=pd.Index([]),
        returns_map=returns_map,
        risk_free=0.0,
        config={"enabled": True, "proxy": "Missing"},
        freq_code="M",
        periods_per_year=12,
    )
    assert "not found" in payload["notes"][0].lower()

    # Force ``compute_regimes`` to return empty to trigger fallback note.
    original_compute = compute_regimes
    monkeypatch.setattr(
        "trend_analysis.regimes.compute_regimes",
        lambda *_args, **_kwargs: pd.Series(dtype="string"),
    )
    payload = build_regime_payload(
        data=data,
        out_index=pd.Index([]),
        returns_map=returns_map,
        risk_free=0.0,
        config={"enabled": True, "proxy": "Proxy"},
        freq_code="M",
        periods_per_year=12,
    )
    assert "did not produce regime labels" in payload["notes"][0].lower()

    # Restore ``compute_regimes`` and exercise summary/notes logic with gaps.
    monkeypatch.setattr("trend_analysis.regimes.compute_regimes", original_compute)

    data_with_gap = data.copy()
    returns_map = {"Fund": data_with_gap.set_index("Date")["Fund"]}
    payload = build_regime_payload(
        data=data_with_gap,
        out_index=pd.DatetimeIndex(list(dates) + [dates[-1] + pd.offsets.MonthEnd()]),
        returns_map=returns_map,
        risk_free=0.0,
        config={
            "enabled": True,
            "proxy": "Proxy",
            "lookback": 1,
            "smoothing": 1,
            "threshold": 0,
            "min_observations": 1,
        },
        freq_code="M",
        periods_per_year=12,
    )
    assert payload["labels"].dtype == "string"
    if payload["notes"]:
        assert payload["summary"] in payload["notes"] or payload["summary"] is None


def test_build_regime_payload_handles_missing_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2024-01-31", periods=2, freq="M")
    data = pd.DataFrame({"Date": dates, "Proxy": [0.01, 0.02], "Fund": [0.02, 0.01]})
    returns_map = {"Fund": data.set_index("Date")["Fund"]}

    monkeypatch.setattr(
        "trend_analysis.regimes.compute_regimes",
        lambda *_args, **_kwargs: pd.Series(
            [pd.NA, pd.NA], index=dates, dtype="string"
        ),
    )

    payload = build_regime_payload(
        data=data,
        out_index=pd.DatetimeIndex(list(dates) + [dates[-1] + pd.offsets.MonthEnd()]),
        returns_map=returns_map,
        risk_free=0.0,
        config={
            "enabled": True,
            "proxy": "Proxy",
            "lookback": 1,
            "smoothing": 1,
            "threshold": 0,
            "min_observations": 1,
        },
        freq_code="M",
        periods_per_year=12,
    )

    assert any("forward/backward fill" in note.lower() for note in payload["notes"])


def test_build_regime_payload_generates_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2024-01-31", periods=6, freq="M")
    data = pd.DataFrame(
        {
            "Date": dates,
            "Proxy": [0.01, 0.02, -0.01, 0.03, -0.02, 0.025],
            "Fund": [0.02, 0.03, -0.01, 0.04, 0.02, 0.01],
        }
    )
    returns_map = {"Fund": data.set_index("Date")["Fund"]}

    monkeypatch.setattr(
        "trend_analysis.regimes.compute_regimes",
        lambda *_args, **_kwargs: pd.Series(
            ["Risk-On", "Risk-Off", "Risk-On", "Risk-Off", "Risk-On", "Risk-Off"],
            index=dates,
            dtype="string",
        ),
    )

    payload = build_regime_payload(
        data=data,
        out_index=pd.DatetimeIndex(dates),
        returns_map=returns_map,
        risk_free=0.0,
        config={
            "enabled": True,
            "proxy": "Proxy",
            "lookback": 1,
            "smoothing": 1,
            "threshold": 0,
            "min_observations": 1,
        },
        freq_code="M",
        periods_per_year=12,
    )

    assert not payload["table"].empty
    assert isinstance(payload["summary"], str)
    assert (
        "risk-on" in payload["summary"].lower()
        or "risk-off" in payload["summary"].lower()
    )


def test_compute_regime_series_volatility_tag_includes_periods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2024-01-31", periods=10, freq="M")
    proxy = pd.Series(np.linspace(0.01, 0.05, len(dates)), index=dates)
    settings = RegimeSettings(
        enabled=True,
        method="volatility",
        lookback=3,
        smoothing=1,
        cache=True,
        annualise_volatility=True,
    )

    class DummyCache:
        def __init__(self) -> None:
            self.tags: list[str] = []

        def get_or_compute(
            self,
            dataset_hash: str,
            window: int,
            freq: str,
            method_tag: str,
            compute: Callable[[], pd.Series],
        ) -> pd.Series:
            self.tags.append(method_tag)
            return compute()

    cache = DummyCache()

    monkeypatch.setattr("trend_analysis.regimes.get_cache", lambda: cache)
    monkeypatch.setattr(
        "trend_analysis.regimes.compute_dataset_hash",
        lambda payload: "hash",
    )

    labels = _compute_regime_series(
        proxy,
        settings,
        freq="M",
        periods_per_year=12,
    )

    assert not labels.empty
    assert any("ppy12.000000" in tag for tag in cache.tags)


def test_compute_regime_series_volatility_tag_skips_when_no_periods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2024-01-31", periods=8, freq="M")
    proxy = pd.Series(np.linspace(0.01, 0.04, len(dates)), index=dates)
    settings = RegimeSettings(
        enabled=True,
        method="volatility",
        lookback=3,
        smoothing=1,
        cache=True,
        annualise_volatility=True,
    )

    class DummyCache:
        def __init__(self) -> None:
            self.tags: list[str] = []

        def get_or_compute(
            self,
            dataset_hash: str,
            window: int,
            freq: str,
            method_tag: str,
            compute: Callable[[], pd.Series],
        ) -> pd.Series:
            self.tags.append(method_tag)
            return compute()

    cache = DummyCache()

    monkeypatch.setattr("trend_analysis.regimes.get_cache", lambda: cache)
    monkeypatch.setattr(
        "trend_analysis.regimes.compute_dataset_hash",
        lambda payload: "hash",
    )
    monkeypatch.setattr(
        "trend_analysis.regimes._default_periods_per_year",
        lambda _freq: 0.0,
    )

    labels = _compute_regime_series(
        proxy,
        settings,
        freq="Z",
        periods_per_year=None,
    )

    assert not labels.empty
    assert all("ppy" not in tag for tag in cache.tags)


def test_build_regime_payload_uses_notes_when_no_user_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2024-01-31", periods=6, freq="M")
    data = pd.DataFrame({"Date": dates, "Proxy": np.linspace(100, 120, len(dates))})

    regimes = pd.Series(["Risk-On"] * len(dates), index=dates, dtype="string")
    monkeypatch.setattr(
        "trend_analysis.regimes.compute_regimes",
        lambda *args, **kwargs: regimes,
    )

    columns = pd.MultiIndex.from_tuples(
        [("User", "All")], names=["portfolio", "regime"]
    )
    table = pd.DataFrame([[0.1], [0.2]], index=["CAGR", "Sharpe"], columns=columns)
    monkeypatch.setattr(
        "trend_analysis.regimes.aggregate_performance_by_regime",
        lambda *args, **kwargs: (table, ["Only aggregate data available."]),
    )

    payload = build_regime_payload(
        data=data,
        out_index=dates,
        returns_map={"User": pd.Series(0.01, index=dates)},
        risk_free=0.0,
        config={
            "enabled": True,
            "proxy": "Proxy",
            "method": "rolling_return",
        },
        freq_code="M",
        periods_per_year=12,
    )

    assert payload["summary"] == "Only aggregate data available."
    assert payload["notes"][0] == "Only aggregate data available."
