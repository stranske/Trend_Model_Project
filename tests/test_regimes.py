import pandas as pd

from trend_analysis.regimes import build_regime_payload


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
    assert payload["summary"]


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
