from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.util.weights import normalize_weights


def test_normalize_weights_fraction_input() -> None:
    weights = {"FundA": 0.6, "FundB": 0.4}
    normalized = normalize_weights(weights)
    assert normalized["FundA"] == pytest.approx(0.6)
    assert normalized["FundB"] == pytest.approx(0.4)


def test_normalize_weights_percent_input() -> None:
    weights = pd.Series({"FundA": 60.0, "FundB": 40.0})
    normalized = normalize_weights(weights)
    assert normalized["FundA"] == pytest.approx(0.6)
    assert normalized["FundB"] == pytest.approx(0.4)


def test_normalize_weights_handles_rounding_error() -> None:
    weights_low = {"FundA": 50.0, "FundB": 49.999}
    weights_high = {"FundA": 50.0, "FundB": 50.001}

    normalized_low = normalize_weights(weights_low)
    normalized_high = normalize_weights(weights_high)

    assert normalized_low["FundA"] == pytest.approx(0.5)
    assert normalized_low["FundB"] == pytest.approx(0.49999)
    assert normalized_high["FundA"] == pytest.approx(0.5)
    assert normalized_high["FundB"] == pytest.approx(0.50001)


def test_normalize_weights_empty_or_none() -> None:
    assert normalize_weights(None) == {}
    assert normalize_weights({}) == {}
    empty_series = pd.Series(dtype=float)
    assert normalize_weights(empty_series) == {}


def test_normalize_weights_long_short() -> None:
    weights = {"FundA": 120.0, "FundB": -20.0}
    normalized = normalize_weights(weights)
    assert normalized["FundA"] == pytest.approx(1.2)
    assert normalized["FundB"] == pytest.approx(-0.2)
