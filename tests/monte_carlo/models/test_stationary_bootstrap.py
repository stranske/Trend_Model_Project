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


def _unique_log_returns(n: int, *, start: float = 0.001, step: float = 0.001) -> np.ndarray:
    return start + step * np.arange(n)


def _infer_indices_from_reference_asset(
    *,
    simulated: pd.DataFrame,
    historical: pd.DataFrame,
    reference_asset: str,
    decimals: int = 12,
) -> np.ndarray:
    reference_history = historical[reference_asset].to_numpy()
    if np.isnan(reference_history).any():
        raise AssertionError("reference asset contains missing values")
    rounded_reference = np.round(reference_history, decimals=decimals)
    if len(np.unique(rounded_reference)) != len(rounded_reference):
        raise AssertionError("reference asset values must be unique")
    lookup = {value: idx for idx, value in enumerate(rounded_reference)}

    reference_simulated = simulated.xs(reference_asset, level=1, axis=1).to_numpy()
    rounded_simulated = np.round(reference_simulated, decimals=decimals)
    indices = np.empty_like(rounded_simulated, dtype=int)
    for i in range(rounded_simulated.shape[0]):
        for j in range(rounded_simulated.shape[1]):
            indices[i, j] = lookup[rounded_simulated[i, j]]
    return indices


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

    run_lengths_arr = np.asarray(run_lengths)
    mean_run = float(run_lengths_arr.mean())
    assert abs(mean_run - mean_block_len) < 0.75

    p = 1.0 / mean_block_len
    max_k = 20
    k = np.arange(1, max_k + 1)
    empirical_cdf = (run_lengths_arr[:, None] <= k[None, :]).mean(axis=0)
    theoretical_cdf = 1.0 - (1.0 - p) ** k
    max_diff = float(np.max(np.abs(empirical_cdf - theoretical_cdf)))
    assert max_diff < 0.06


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
    index = pd.date_range("2024-02-01", periods=14, freq="D")
    n_obs = len(index)
    reference_returns = np.concatenate([[0.0], _unique_log_returns(n_obs - 1)])
    base_returns = np.full(n_obs, 0.002)
    log_returns = np.column_stack([base_returns, reference_returns])
    prices = _prices_from_log_returns(log_returns, index, ["AssetA", "AssetB"])
    prices.loc[index[2:4], "AssetA"] = np.nan
    prices.loc[index[6], "AssetA"] = np.nan
    prices.loc[index[9:11], "AssetA"] = np.nan

    model = StationaryBootstrapModel(mean_block_len=5, frequency="D").fit(prices)
    n_periods = 24
    n_paths = 4
    seed = 17
    result = model.sample_prices(n_periods=n_periods, n_paths=n_paths, seed=seed)

    historical = model._log_returns
    assert historical is not None
    indices = _infer_indices_from_reference_asset(
        simulated=result.log_returns,
        historical=historical,
        reference_asset="AssetB",
    )
    expected_mask = historical["AssetA"].isna().to_numpy()[indices]
    simulated_mask = result.missingness_mask.xs("AssetA", level=1, axis=1).to_numpy()
    assert np.array_equal(simulated_mask, expected_mask)

    n_hist_obs = len(historical)
    for path in range(n_paths):
        path_indices = indices[:, path]
        block_starts = [0]
        for t in range(1, n_periods):
            if path_indices[t] != (path_indices[t - 1] + 1) % n_hist_obs:
                block_starts.append(t)
        block_starts.append(n_periods)
        for start, end in zip(block_starts[:-1], block_starts[1:]):
            block_idx = path_indices[start:end]
            expected_slice = historical["AssetA"].isna().to_numpy()[block_idx]
            series = simulated_mask[start:end, path]
            assert np.array_equal(series, expected_slice)


def test_missingness_preserves_per_asset_nan_rates() -> None:
    index = pd.date_range("2024-03-01", periods=36, freq="D")
    n_obs = len(index)
    reference_returns = np.concatenate([[0.0], _unique_log_returns(n_obs - 1, start=0.002)])
    base_returns = np.full(n_obs, 0.001)
    log_returns = np.column_stack(
        [
            base_returns,
            base_returns * 1.2,
            reference_returns,
        ]
    )
    prices = _prices_from_log_returns(log_returns, index, ["AssetA", "AssetB", "AssetRef"])
    prices.loc[index[[2, 5, 6, 10, 11, 12]], "AssetA"] = np.nan
    prices.loc[index[[1, 3, 4, 7, 18, 19]], "AssetB"] = np.nan

    model = StationaryBootstrapModel(mean_block_len=4, frequency="D").fit(prices)
    n_periods = 200
    n_paths = 20
    seed = 23
    result = model.sample_prices(n_periods=n_periods, n_paths=n_paths, seed=seed)

    historical = model._log_returns
    assert historical is not None
    historical_rates = historical.isna().mean()
    indices = _infer_indices_from_reference_asset(
        simulated=result.log_returns,
        historical=historical,
        reference_asset="AssetRef",
    )
    expected_mask = historical.isna().to_numpy()[indices]
    expected_rates = expected_mask.mean(axis=(0, 1))
    expected_by_asset = dict(zip(historical.columns, expected_rates))

    simulated_mask = result.missingness_mask
    simulated_rates = {
        asset: simulated_mask.xs(asset, level=1, axis=1).to_numpy().mean()
        for asset in historical.columns
    }

    for asset, hist_rate in historical_rates.items():
        sim_rate = simulated_rates[asset]
        expected_rate = expected_by_asset[asset]
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
