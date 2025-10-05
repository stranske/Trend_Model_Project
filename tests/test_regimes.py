import pandas as pd

from trend_analysis.regimes import (
    RegimeSettings,
    _summarise_regime_outcome,
    build_regime_payload,
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
