from __future__ import annotations

import numpy as np
import pandas as pd

from trend_analysis.monte_carlo.models import (
    log_returns_to_prices,
    price_availability_mask,
    prices_to_log_returns,
    returns_availability_mask,
)


def test_price_availability_mask_flags_invalid_values() -> None:
    prices = pd.DataFrame(
        {
            "A": [100.0, 0.0, -5.0, np.nan, 110.0],
            "B": [50.0, 52.0, 53.0, np.inf, 54.0],
        }
    )

    mask = price_availability_mask(prices)

    assert mask["A"].tolist() == [True, False, False, False, True]
    assert mask["B"].tolist() == [True, True, True, False, True]


def test_returns_availability_mask_requires_adjacent_prices() -> None:
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, np.nan, 121.0],
            "B": [50.0, 50.0, 52.0, 52.0],
        }
    )

    price_mask = price_availability_mask(prices)
    returns_mask = returns_availability_mask(price_mask)

    assert returns_mask["A"].tolist() == [False, True, False, False]
    assert returns_mask["B"].tolist() == [False, True, True, True]


def test_prices_to_log_returns_and_back_roundtrip() -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="M")
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0, 133.1],
            "B": [50.0, 49.0, 51.0, 52.0],
        },
        index=dates,
    )

    price_mask = price_availability_mask(prices)
    log_returns = prices_to_log_returns(prices, price_availability=price_mask)
    rebuilt = log_returns_to_prices(
        log_returns,
        prices.iloc[0],
        price_availability=price_mask,
        start_at_first_row=True,
    )

    assert np.allclose(rebuilt.to_numpy(), prices.to_numpy(), equal_nan=True)


def test_log_returns_to_prices_resumes_after_gap() -> None:
    dates = pd.date_range("2021-01-31", periods=3, freq="M")
    log_returns = pd.DataFrame({"A": [0.0, 0.1, 0.2]}, index=dates)
    availability = pd.DataFrame({"A": [True, False, True]}, index=dates)

    rebuilt = log_returns_to_prices(
        log_returns,
        {"A": 100.0},
        price_availability=availability,
        start_at_first_row=False,
    )

    assert rebuilt["A"].tolist()[0] == 100.0
    assert np.isnan(rebuilt["A"].tolist()[1])
    assert np.isclose(rebuilt["A"].tolist()[2], 100.0 * np.exp(0.2))
