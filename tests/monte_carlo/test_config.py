from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.monte_carlo.config import resolve_risk_free_source


def _base_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "A": [0.02, 0.01, -0.01, 0.015, 0.0, 0.01],
            "B": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
            "RF": [0.0005] * 6,
        }
    )


def _config_base() -> dict[str, object]:
    return {
        "version": "0.1.0",
        "data": {
            "date_column": "Date",
            "frequency": "M",
            "risk_free_column": None,
            "allow_risk_free_fallback": False,
        },
        "metrics": {
            "rf_rate_annual": 0.0,
            "rf_override_enabled": False,
        },
        "benchmarks": {},
    }


def test_resolve_risk_free_override() -> None:
    returns = _base_returns()
    cfg = _config_base()
    cfg["metrics"] = {"rf_rate_annual": 0.12, "rf_override_enabled": True}

    res = resolve_risk_free_source(returns, cfg)

    expected = (1.0 + 0.12) ** (1.0 / 12.0) - 1.0
    assert res.source == "override"
    assert res.column is None
    assert isinstance(res.risk_free, float)
    assert res.risk_free == pytest.approx(expected)


def test_resolve_risk_free_configured_column() -> None:
    returns = _base_returns()
    cfg = _config_base()
    cfg["data"] = {
        "date_column": "Date",
        "frequency": "M",
        "risk_free_column": "RF",
        "allow_risk_free_fallback": True,
    }

    res = resolve_risk_free_source(returns, cfg)

    assert res.source == "configured"
    assert res.column == "RF"
    pd.testing.assert_series_equal(res.risk_free, returns["RF"])


def test_resolve_risk_free_fallback() -> None:
    returns = _base_returns().drop(columns=["RF"])
    cfg = _config_base()
    cfg["data"] = {
        "date_column": "Date",
        "frequency": "M",
        "risk_free_column": None,
        "allow_risk_free_fallback": True,
    }

    res = resolve_risk_free_source(returns, cfg)

    assert res.source == "fallback"
    assert res.column == "B"
    pd.testing.assert_series_equal(res.risk_free, returns["B"])
