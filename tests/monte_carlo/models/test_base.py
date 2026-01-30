import numpy as np
import pandas as pd

from trend_analysis.monte_carlo.models.base import (
    BootstrapPricePathModel,
    apply_missingness_mask,
    build_missingness_mask,
    extract_missingness_mask,
    log_returns_to_prices,
    normalize_price_frequency,
    prices_to_log_returns,
)
from trend_analysis.timefreq import MONTHLY_DATE_FREQ


def _sample_prices(index: pd.DatetimeIndex) -> pd.DataFrame:
    data = {
        "AssetA": np.linspace(100, 130, len(index)),
        "AssetB": np.linspace(50, 80, len(index)),
    }
    return pd.DataFrame(data, index=index)


def test_prices_to_log_returns_round_trip() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = _sample_prices(index)

    log_returns = prices_to_log_returns(prices)
    rebuilt = log_returns_to_prices(log_returns, prices.iloc[0])

    assert np.allclose(rebuilt.to_numpy(), prices.to_numpy())


def test_normalize_price_frequency_monthly() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    prices = _sample_prices(index)

    normalized, summary = normalize_price_frequency(prices, "M")

    assert summary.code == "D"
    assert normalized.index.is_month_end.all()
    assert len(normalized) < len(prices)


def test_normalize_price_frequency_daily() -> None:
    index = pd.date_range("2024-01-31", periods=4, freq=MONTHLY_DATE_FREQ)
    prices = _sample_prices(index)

    normalized, summary = normalize_price_frequency(prices, "D")

    assert summary.code == "M"
    assert pd.infer_freq(normalized.index) == "D"
    assert len(normalized) > len(prices)


def test_normalize_price_frequency_quarterly_maps_monthly() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    prices = _sample_prices(index)

    normalized, summary = normalize_price_frequency(prices, "Q")

    assert summary.code == "D"
    assert normalized.index.is_month_end.all()
    assert len(normalized) < len(prices)


def test_missingness_mask_helpers() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = _sample_prices(index)
    prices.iloc[1, 0] = np.nan

    mask = extract_missingness_mask(prices)
    assert mask.iloc[1, 0]

    future_index = pd.date_range("2024-01-04", periods=5, freq="D")
    projected = build_missingness_mask(mask, future_index)

    assert projected.shape == (5, 2)
    assert projected.iloc[1, 0]
    assert projected.iloc[4, 0]

    masked = apply_missingness_mask(prices.reindex(future_index).ffill(), projected)
    assert masked.isna().iloc[1, 0]


def test_bootstrap_model_output_shape() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = _sample_prices(index)

    model = BootstrapPricePathModel(prices, frequency="D")
    result = model.simulate(n_periods=4, n_paths=3, seed=7)

    assert result.prices.shape == (4, 6)
    assert result.log_returns.shape == (4, 6)
    assert result.prices.columns.nlevels == 2


def test_output_dates_frequency_monthly() -> None:
    index = pd.date_range("2023-12-31", periods=6, freq=MONTHLY_DATE_FREQ)
    prices = _sample_prices(index)

    model = BootstrapPricePathModel(prices, frequency="M")
    result = model.simulate(n_periods=3, n_paths=2, seed=5)

    assert result.prices.index.is_month_end.all()


def test_output_dates_frequency_quarterly_maps_monthly() -> None:
    index = pd.date_range("2023-12-31", periods=6, freq=MONTHLY_DATE_FREQ)
    prices = _sample_prices(index)

    model = BootstrapPricePathModel(prices, frequency="Q")
    result = model.simulate(n_periods=3, n_paths=1, seed=5)

    assert result.frequency == "M"
    assert result.prices.index.is_month_end.all()


def test_prices_are_non_negative() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = _sample_prices(index)

    model = BootstrapPricePathModel(prices, frequency="D")
    result = model.simulate(n_periods=5, n_paths=2, seed=11)

    values = result.prices.to_numpy()
    values = values[np.isfinite(values)]
    assert (values >= 0).all()


def test_missingness_mask_propagation() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    prices = _sample_prices(index)
    prices.iloc[0, 1] = np.nan

    model = BootstrapPricePathModel(prices, frequency="D")
    result = model.simulate(n_periods=4, n_paths=1, seed=3)

    mask = result.missingness_mask
    assert mask.iloc[0, 1]
    assert mask.iloc[2, 1]
    assert result.prices.isna().iloc[0, 1]
    assert result.prices.isna().iloc[2, 1]
