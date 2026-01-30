import numpy as np
import pandas as pd

from trend_analysis.monte_carlo.models.bootstrap import (
    StationaryBootstrapModel,
    _stationary_bootstrap_indices,
)
from trend_analysis.timefreq import MONTHLY_DATE_FREQ


def _prices_from_log_returns(
    log_returns: np.ndarray, index: pd.DatetimeIndex, columns: list[str]
) -> pd.DataFrame:
    prices = np.exp(np.cumsum(log_returns, axis=0)) * 100.0
    return pd.DataFrame(prices, index=index, columns=columns)


def test_stationary_bootstrap_preserves_correlation() -> None:
    rng = np.random.default_rng(42)
    n_obs = 400
    corr = 0.8
    cov = np.array([[1.0, corr], [corr, 1.0]])
    chol = np.linalg.cholesky(cov)
    raw = rng.standard_normal((n_obs, 2)) @ chol.T * 0.01
    index = pd.date_range("2020-01-01", periods=n_obs, freq="D")
    prices = _prices_from_log_returns(raw, index, ["AssetA", "AssetB"])

    model = StationaryBootstrapModel(mean_block_len=6, frequency="D").fit(prices)
    result = model.sample_prices(n_periods=200, n_paths=50, seed=11)

    hist_returns = np.log(prices / prices.shift(1)).dropna()
    hist_corr = hist_returns.corr().iloc[0, 1]

    n_periods = len(result.log_returns)
    n_assets = 2
    data = result.log_returns.to_numpy().reshape(n_periods, -1, n_assets)
    flattened = data.reshape(-1, n_assets)
    sim = pd.DataFrame(flattened, columns=["AssetA", "AssetB"]).dropna()
    sim_corr = sim.corr().iloc[0, 1]

    assert abs(sim_corr - hist_corr) < 0.15


def test_block_length_distribution_matches_geometric_mean() -> None:
    rng = np.random.default_rng(7)
    mean_block_len = 5.0
    indices = _stationary_bootstrap_indices(
        n_obs=120,
        n_periods=3000,
        n_paths=10,
        mean_block_len=mean_block_len,
        rng=rng,
    )

    run_lengths: list[int] = []
    for path in range(indices.shape[0]):
        current = 1
        for t in range(1, indices.shape[1]):
            prev = indices[path, t - 1]
            expected = (prev + 1) % 120
            if indices[path, t] == expected:
                current += 1
            else:
                run_lengths.append(current)
                current = 1
        run_lengths.append(current)

    mean_run = float(np.mean(run_lengths))
    assert abs(mean_run - mean_block_len) < 0.75


def test_missingness_propagates_into_samples() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    prices = pd.DataFrame(
        {
            "AssetA": np.linspace(100, 120, len(index)),
            "AssetB": np.linspace(50, 65, len(index)),
        },
        index=index,
    )
    prices.iloc[2, 0] = np.nan

    model = StationaryBootstrapModel(mean_block_len=4, frequency="D").fit(prices)
    result = model.sample_prices(n_periods=12, n_paths=5, seed=3)

    assert result.log_returns.isna().any().any()
    assert result.missingness_mask.equals(result.log_returns.isna())
    assert result.prices.isna().equals(result.missingness_mask)


def test_monthly_frequency_flow() -> None:
    index = pd.date_range("2023-01-31", periods=12, freq=MONTHLY_DATE_FREQ)
    prices = pd.DataFrame(
        {
            "AssetA": np.linspace(100, 120, len(index)),
            "AssetB": np.linspace(80, 90, len(index)),
        },
        index=index,
    )

    model = StationaryBootstrapModel(mean_block_len=6, frequency="M").fit(prices)
    result = model.sample_prices(n_periods=4, n_paths=2, seed=9)

    assert result.frequency == "M"
    assert result.prices.index.is_month_end.all()


def test_sample_prices_deterministic_with_seed() -> None:
    index = pd.date_range("2024-01-01", periods=20, freq="D")
    prices = pd.DataFrame(
        {
            "AssetA": np.linspace(100, 130, len(index)),
            "AssetB": np.linspace(60, 70, len(index)),
        },
        index=index,
    )

    model = StationaryBootstrapModel(mean_block_len=3, frequency="D").fit(prices)
    first = model.sample_prices(n_periods=10, n_paths=3, seed=101)
    second = model.sample_prices(n_periods=10, n_paths=3, seed=101)

    pd.testing.assert_frame_equal(first.log_returns, second.log_returns)
    pd.testing.assert_frame_equal(first.prices, second.prices)
