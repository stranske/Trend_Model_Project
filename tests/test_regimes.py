import pandas as pd
import pytest

from trend_analysis.regimes import (
    RegimeSettings,
    _coerce_float,
    _coerce_positive_int,
    _compute_regime_series,
    _default_periods_per_year,
    _rolling_volatility_signal,
    _summarise_regime_outcome,
    build_regime_payload,
    compute_regimes,
    normalise_settings,
)


def _sample_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    data = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.02, 0.03, -0.01, 0.01, 0.02, 0.015],
            "FundB": [0.01, 0.02, -0.015, 0.0, 0.018, 0.012],
            "RF": 0.0,
            "SPX": [0.05, 0.04, -0.03, -0.04, 0.02, 0.03],
        }
    )
    return data


def test_build_regime_payload_produces_labels_and_table() -> None:
    data = _sample_frame()
    index = pd.DatetimeIndex(data["Date"])
    returns_map = {
        "User": data.set_index("Date")["FundA"],
        "Equal-Weight": data.set_index("Date")["FundB"],
    }
    risk_free = pd.Series(0.0, index=index)
    payload = build_regime_payload(
        data=data,
        out_index=index,
        returns_map=returns_map,
        risk_free=risk_free,
        config={
            "enabled": True,
            "proxy": "SPX",
            "lookback": 2,
            "smoothing": 1,
            "threshold": 0.0,
            "min_observations": 1,
        },
        freq_code="M",
        periods_per_year=12,
    )
    table = payload["table"]
    assert not table.empty
    assert ("User", "Risk-On") in table.columns
    assert payload["labels"].dtype == "string"
    summary = payload["summary"]
    assert isinstance(summary, str)
    assert "CAGR" in summary


def test_build_regime_payload_notes_for_insufficient_observations() -> None:
    data = _sample_frame()
    index = pd.DatetimeIndex(data["Date"])
    returns_map = {"User": data.set_index("Date")["FundA"]}
    risk_free = pd.Series(0.0, index=index)
    payload = build_regime_payload(
        data=data,
        out_index=index,
        returns_map=returns_map,
        risk_free=risk_free,
        config={
            "enabled": True,
            "proxy": "SPX",
            "lookback": 3,
            "smoothing": 1,
            "min_observations": 10,
        },
        freq_code="M",
        periods_per_year=12,
    )
    notes = payload["notes"]
    assert any("fewer than" in note.lower() for note in notes)
    assert any("all-period aggregate" in note.lower() for note in notes)


def test_build_regime_payload_deduplicates_repeated_notes() -> None:
    data = _sample_frame()
    index = pd.DatetimeIndex(data["Date"])
    returns_map = {
        "User": data.set_index("Date")["FundA"],
        "Equal-Weight": data.set_index("Date")["FundB"],
    }
    risk_free = pd.Series(0.0, index=index)
    payload = build_regime_payload(
        data=data,
        out_index=index,
        returns_map=returns_map,
        risk_free=risk_free,
        config={
            "enabled": True,
            "proxy": "SPX",
            "lookback": 3,
            "smoothing": 1,
            "min_observations": 10,
        },
        freq_code="M",
        periods_per_year=12,
    )
    notes = payload["notes"]
    risk_on_notes = [note for note in notes if "risk-on regime" in note.lower()]
    risk_off_notes = [note for note in notes if "risk-off regime" in note.lower()]
    assert len(risk_on_notes) == 1
    assert len(risk_off_notes) == 1


def test_build_regime_payload_volatility_method() -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    proxy = [0.01, 0.015, 0.20, 0.18, 0.19, 0.21]
    data = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.02, 0.03, -0.01, 0.01, 0.02, 0.015],
            "FundB": [0.01, 0.02, -0.015, 0.0, 0.018, 0.012],
            "RF": 0.0,
            "SPX": proxy,
        }
    )
    index = pd.DatetimeIndex(data["Date"])
    returns_map = {"User": data.set_index("Date")["FundA"]}
    payload = build_regime_payload(
        data=data,
        out_index=index,
        returns_map=returns_map,
        risk_free=0.0,
        config={
            "enabled": True,
            "proxy": "SPX",
            "method": "volatility",
            "lookback": 2,
            "smoothing": 1,
            "threshold": 0.05,
            "annualise_volatility": False,
            "min_observations": 1,
        },
        freq_code="M",
        periods_per_year=12,
    )
    labels = payload["labels"].dropna()
    assert not labels.empty
    assert {"Risk-On", "Risk-Off"}.issuperset(set(labels.unique()))
    assert payload["table"].shape[1] > 0


def test_regime_summary_highlights_stronger_regime() -> None:
    dates = pd.date_range("2021-01-31", periods=6, freq="ME")
    proxy = [0.02, -0.03, 0.01, -0.02, 0.03, -0.01]
    data = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.04, 0.015, 0.05, 0.02, 0.045],
            "RF": 0.0,
            "SPX": proxy,
        }
    )
    index = pd.DatetimeIndex(data["Date"])
    returns_map = {"User": data.set_index("Date")["FundA"]}
    payload = build_regime_payload(
        data=data,
        out_index=index,
        returns_map=returns_map,
        risk_free=0.0,
        config={
            "enabled": True,
            "proxy": "SPX",
            "lookback": 1,
            "smoothing": 1,
            "threshold": 0.0,
            "min_observations": 1,
        },
        freq_code="M",
        periods_per_year=12,
    )
    summary = payload["summary"]
    assert isinstance(summary, str)
    assert "outperformed" in summary.lower()
    assert "risk-off" in summary.lower()


def test_regime_summary_identifies_similar_performance() -> None:
    settings = RegimeSettings(risk_on_label="Risk-On", risk_off_label="Risk-Off")
    risk_on = pd.Series({"CAGR": 0.105, "Sharpe": 1.1})
    risk_off = pd.Series({"CAGR": 0.1045, "Sharpe": 0.9})

    summary = _summarise_regime_outcome(settings, risk_on, risk_off)

    assert isinstance(summary, str)
    assert "similar" in summary.lower()
    assert "risk-on" in summary.lower()
    assert "risk-off" in summary.lower()


def test_normalise_settings_aliases_and_defaults() -> None:
    config = {
        "enabled": "yes",
        "proxy": "  SPX  ",
        "method": "Vol",
        "lookback": "15",
        "smoothing": "",
        "threshold": "0.25",
        "neutral_band": "-0.5",
        "min_observations": 0,
        "risk_on_label": "  ",
        "risk_off_label": "",
        "default_label": "",
        "cache": False,
        "annualise_volatility": False,
    }
    settings = normalise_settings(config)
    assert settings.enabled is True
    assert settings.proxy == "SPX"
    assert settings.method == "volatility"
    assert settings.lookback == 15
    # Blank smoothing should fall back to default of 3.
    assert settings.smoothing == 3
    assert settings.threshold == pytest.approx(0.25)
    # Neutral band coerced to absolute value.
    assert settings.neutral_band == pytest.approx(0.5)
    # Minimum observations coerced to minimum of 1.
    assert settings.min_obs == 1
    # Labels default to canonical names when empty.
    assert settings.risk_on_label == "Risk-On"
    assert settings.risk_off_label == "Risk-Off"
    assert settings.default_label == "Risk-On"
    assert settings.cache is False
    assert settings.annualise_volatility is False


def test_coerce_helpers_and_default_periods() -> None:
    assert _coerce_positive_int("7", 3) == 7
    assert _coerce_positive_int("not-int", 5) == 5
    assert _coerce_positive_int(-2, 5, minimum=2) == 2

    assert _coerce_float("1.5", 0.0) == pytest.approx(1.5)
    assert _coerce_float(None, 2.0) == pytest.approx(2.0)

    assert _default_periods_per_year("A") == 1.0
    assert _default_periods_per_year("Q") == 4.0
    assert _default_periods_per_year("M") == 12.0
    assert _default_periods_per_year("W") == 52.0
    assert _default_periods_per_year("B") == 252.0


def test_compute_regime_series_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2023-01-31", periods=6, freq="ME")
    series = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, 0.015], index=dates)
    settings = RegimeSettings(enabled=True, cache=True, lookback=2, smoothing=1)

    captured: dict[str, tuple] = {}

    class DummyCache:
        def get_or_compute(self, *args):
            captured["key"] = args
            return pd.Series("Risk-On", index=series.index, dtype="string")

    with monkeypatch.context() as mp:
        mp.setattr("trend_analysis.regimes.get_cache", lambda: DummyCache())
        mp.setattr("trend_analysis.regimes.compute_dataset_hash", lambda _: "hash")
        labels = _compute_regime_series(
            series,
            settings,
            freq="M",
            periods_per_year=None,
        )

    assert (labels == "Risk-On").all()
    assert captured["key"][0] == "hash"
    assert captured["key"][1] == settings.lookback
    assert captured["key"][2] == "M"
    assert "regime_rolling_return" in captured["key"][3]


def test_compute_regimes_disabled_returns_empty() -> None:
    series = pd.Series(
        [0.01, 0.02], index=pd.date_range("2023-01-31", periods=2, freq="M")
    )
    settings = RegimeSettings(enabled=False)
    result = compute_regimes(series, settings, freq="M")
    assert result.empty


def test_rolling_volatility_signal_validates_window() -> None:
    series = pd.Series(
        [0.01, 0.02], index=pd.date_range("2023-01-31", periods=2, freq="M")
    )
    with pytest.raises(ValueError, match="window must be positive"):
        _rolling_volatility_signal(
            series,
            window=0,
            smoothing=1,
            periods_per_year=12,
            annualise=True,
        )

    vol = _rolling_volatility_signal(
        series,
        window=1,
        smoothing=2,
        periods_per_year=12,
        annualise=True,
    )
    assert isinstance(vol, pd.Series)
