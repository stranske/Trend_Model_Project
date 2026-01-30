import time

import numpy as np
import pandas as pd
import pytest

from trend_analysis.monte_carlo.models.bootstrap import StationaryBootstrapModel

pytestmark = [pytest.mark.performance, pytest.mark.serial]


def _prices_from_log_returns(log_returns: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    prices = np.exp(np.cumsum(log_returns, axis=0)) * 100.0
    columns = [f"Asset{idx}" for idx in range(log_returns.shape[1])]
    return pd.DataFrame(prices, index=index, columns=columns)


def test_stationary_bootstrap_generates_1000_paths_under_10s() -> None:
    rng = np.random.default_rng(42)
    n_obs = 240
    n_assets = 6
    log_returns = rng.normal(0.0, 0.01, size=(n_obs, n_assets))
    index = pd.date_range("2022-01-01", periods=n_obs, freq="D")
    prices = _prices_from_log_returns(log_returns, index)

    model = StationaryBootstrapModel(mean_block_len=6, frequency="D").fit(prices)

    start = time.perf_counter()
    result = model.sample_prices(n_periods=120, n_paths=1000, seed=7)
    elapsed = time.perf_counter() - start

    assert result.prices.shape == (120, 1000 * n_assets)
    assert elapsed < 10.0, f"Performance regression: {elapsed:.2f}s > 10s"
