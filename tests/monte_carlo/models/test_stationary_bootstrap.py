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


def test_missingness_preserves_contiguous_nan_segments() -> None:
    index = pd.date_range("2024-02-01", periods=12, freq="D")
    prices = pd.DataFrame(
        {
            "AssetA": np.linspace(100, 130, len(index)),
            "AssetB": np.linspace(80, 95, len(index)),
        },
        index=index,
    )
    prices.loc[index[2:4], "AssetA"] = np.nan
    prices.loc[index[6], "AssetA"] = np.nan
    prices.loc[index[9:11], "AssetA"] = np.nan
    prices.loc[index[1], "AssetB"] = np.nan
    prices.loc[index[4:7], "AssetB"] = np.nan
    prices.loc[index[10], "AssetB"] = np.nan

    model = StationaryBootstrapModel(mean_block_len=5, frequency="D").fit(prices)
    n_periods = 24
    n_paths = 4
    seed = 17
    result = model.sample_prices(n_periods=n_periods, n_paths=n_paths, seed=seed)

    historical = model._log_returns
    assert historical is not None
    n_obs, n_assets = historical.shape
    indices = _stationary_bootstrap_indices(
        n_obs=n_obs,
        n_periods=n_periods,
        n_paths=n_paths,
        mean_block_len=model.mean_block_len,
        rng=np.random.default_rng(seed),
    )
    expected_mask = historical.isna().to_numpy()[indices]
    expected_mask = np.swapaxes(expected_mask, 0, 1)
    expected_frame = pd.DataFrame(
        expected_mask.reshape(n_periods, n_paths * n_assets),
        index=result.log_returns.index,
        columns=result.log_returns.columns,
    )

    pd.testing.assert_frame_equal(result.missingness_mask, expected_frame)

    simulated_mask = result.missingness_mask
    for path in range(n_paths):
        path_indices = indices[path]
        block_starts = [0]
        for t in range(1, n_periods):
            if path_indices[t] != (path_indices[t - 1] + 1) % n_obs:
                block_starts.append(t)
        block_starts.append(n_periods)
        for start, end in zip(block_starts[:-1], block_starts[1:]):
            block_idx = path_indices[start:end]
            for asset_idx, asset in enumerate(historical.columns):
                expected_slice = historical.isna().to_numpy()[block_idx, asset_idx]
                series = simulated_mask[(path, asset)].iloc[start:end].to_numpy()
                assert np.array_equal(series, expected_slice)


def test_missingness_preserves_per_asset_nan_rates() -> None:
    index = pd.date_range("2024-03-01", periods=30, freq="D")
    prices = pd.DataFrame(
        {
            "AssetA": np.linspace(100, 135, len(index)),
            "AssetB": np.linspace(80, 92, len(index)),
            "AssetC": np.linspace(60, 75, len(index)),
        },
        index=index,
    )
    prices.loc[index[[2, 5, 6, 10, 11, 12]], "AssetA"] = np.nan
    prices.loc[index[[1, 3, 4, 7, 18]], "AssetB"] = np.nan
    prices.loc[index[[8, 9, 14, 15, 20, 21, 22, 23]], "AssetC"] = np.nan

    model = StationaryBootstrapModel(mean_block_len=4, frequency="D").fit(prices)
    n_periods = 200
    n_paths = 20
    seed = 23
    result = model.sample_prices(n_periods=n_periods, n_paths=n_paths, seed=seed)

    historical = model._log_returns
    assert historical is not None
    historical_rates = historical.isna().mean()

    n_obs, n_assets = historical.shape
    indices = _stationary_bootstrap_indices(
        n_obs=n_obs,
        n_periods=n_periods,
        n_paths=n_paths,
        mean_block_len=model.mean_block_len,
        rng=np.random.default_rng(seed),
    )
    expected_mask = historical.isna().to_numpy()[indices]
    expected_mask = np.swapaxes(expected_mask, 0, 1)
    expected_rates = {
        asset: expected_mask[:, :, asset_idx].mean()
        for asset_idx, asset in enumerate(historical.columns)
    }

    simulated_mask = result.missingness_mask
    simulated_rates = {
        asset: simulated_mask.xs(asset, level=1, axis=1).to_numpy().mean()
        for asset in historical.columns
    }

    for asset, hist_rate in historical_rates.items():
        sim_rate = simulated_rates[asset]
        expected_rate = expected_rates[asset]
        assert abs(sim_rate - expected_rate) < 1.0e-12
        assert abs(sim_rate - hist_rate) < 0.05


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


def test_quarterly_frequency_normalizes_to_monthly() -> None:
    index = pd.date_range("2022-01-31", periods=12, freq=MONTHLY_DATE_FREQ)
    prices = pd.DataFrame(
        {
            "AssetA": np.linspace(100, 112, len(index)),
            "AssetB": np.linspace(80, 92, len(index)),
        },
        index=index,
    )

    model = StationaryBootstrapModel(mean_block_len=4, frequency="Q").fit(prices)
    result = model.sample_prices(n_periods=3, n_paths=2, frequency="Q", seed=5)

    assert model.frequency == "M"
    assert result.frequency == "M"
