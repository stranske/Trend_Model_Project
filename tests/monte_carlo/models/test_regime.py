import numpy as np
import pandas as pd
import pytest

from trend_analysis.monte_carlo.models.bootstrap import StationaryBootstrapModel
from trend_analysis.monte_carlo.models.regime import (
    RegimeConditionedBootstrapModel,
    RegimeLabeler,
    _normalize_transition_matrix,
)


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


def test_regime_labeler_identifies_volatility_shift() -> None:
    n_obs = 40
    calm_n = n_obs // 2
    stress_n = n_obs - calm_n

    calm_proxy = 0.001 * np.array([1, -1] * (calm_n // 2))
    stress_proxy = 0.02 * np.array([1, -1] * (stress_n // 2))
    proxy_returns = np.concatenate([calm_proxy, stress_proxy])

    asset_returns = 0.003 * np.array([1, -1] * (n_obs // 2))
    log_returns = np.column_stack([asset_returns, proxy_returns])

    index = pd.date_range("2022-01-01", periods=n_obs, freq="D")
    prices = _prices_from_log_returns(log_returns, index, ["AssetA", "Proxy"])

    labeler = RegimeLabeler(proxy_column="Proxy", threshold_percentile=70, lookback=4).fit(prices)
    labels = labeler.get_labels()

    mid_point = labels.index[len(labels) // 2]
    calm_share = (labels[labels.index < mid_point] == "stress").mean()
    stress_share = (labels[labels.index >= mid_point] == "stress").mean()

    assert calm_share == 0.0
    assert stress_share > 0.8
    assert set(labels.unique()) == {"calm", "stress"}


def test_regime_labeler_validates_inputs() -> None:
    with pytest.raises(ValueError, match="threshold_percentile"):
        RegimeLabeler(proxy_column="Proxy", threshold_percentile=120)
    with pytest.raises(ValueError, match="lookback"):
        RegimeLabeler(proxy_column="Proxy", lookback=0)


def test_transition_matrix_rows_sum_to_one() -> None:
    rng = np.random.default_rng(4)
    n_obs = 80
    proxy_returns = np.concatenate(
        [
            rng.standard_normal(n_obs // 2) * 0.001,
            rng.standard_normal(n_obs // 2) * 0.02,
        ]
    )
    asset_returns = rng.standard_normal(n_obs) * 0.005
    log_returns = np.column_stack([asset_returns, proxy_returns])

    index = pd.date_range("2021-06-01", periods=n_obs, freq="D")
    prices = _prices_from_log_returns(log_returns, index, ["AssetA", "Proxy"])

    labeler = RegimeLabeler(proxy_column="Proxy", threshold_percentile=70, lookback=5).fit(prices)
    matrix = labeler.get_transition_matrix()

    row_sums = matrix.sum(axis=1).to_numpy()
    assert np.allclose(row_sums, 1.0)


def test_transition_matrix_single_regime_defaults_to_self() -> None:
    n_obs = 40
    index = pd.date_range("2023-01-01", periods=n_obs, freq="D")
    log_returns = np.column_stack([np.zeros(n_obs), np.zeros(n_obs)])
    prices = _prices_from_log_returns(log_returns, index, ["AssetA", "Proxy"])

    labeler = RegimeLabeler(proxy_column="Proxy", threshold_percentile=90, lookback=1).fit(prices)
    matrix = labeler.get_transition_matrix()

    assert matrix.shape == (1, 1)
    assert float(matrix.iloc[0, 0]) == pytest.approx(1.0)


def test_normalize_transition_matrix_handles_empty_rows() -> None:
    transition = np.array([[0.0, 0.0], [0.2, 0.2]])
    normalized = _normalize_transition_matrix(transition)

    assert normalized.shape == transition.shape
    assert np.allclose(normalized.sum(axis=1), 1.0)
    assert np.allclose(normalized[0], np.array([1.0, 0.0]))


def test_regime_conditioned_sampling_preserves_stress_behavior() -> None:
    rng = np.random.default_rng(12)
    n_obs = 200
    calm_n = n_obs // 2
    stress_n = n_obs - calm_n

    def _correlated_returns(n: int, *, vol: float, corr: float) -> np.ndarray:
        cov = np.array([[1.0, corr], [corr, 1.0]]) * (vol**2)
        chol = np.linalg.cholesky(cov)
        return rng.standard_normal((n, 2)) @ chol.T

    calm_returns = _correlated_returns(calm_n, vol=0.004, corr=0.2)
    stress_returns = _correlated_returns(stress_n, vol=0.02, corr=0.85)
    asset_returns = np.vstack([calm_returns, stress_returns])

    proxy_returns = np.concatenate(
        [
            rng.standard_normal(calm_n) * 0.002,
            rng.standard_normal(stress_n) * 0.02,
        ]
    )
    reference_returns = _unique_log_returns(n_obs, start=0.0005, step=0.00001)

    log_returns = np.column_stack(
        [
            asset_returns,
            reference_returns,
            proxy_returns,
        ]
    )

    index = pd.date_range("2020-01-01", periods=n_obs, freq="D")
    prices = _prices_from_log_returns(log_returns, index, ["AssetA", "AssetB", "AssetRef", "Proxy"])

    model = RegimeConditionedBootstrapModel(
        mean_block_len=4,
        frequency="D",
        regime_proxy_column="Proxy",
        threshold_percentile=70,
        lookback=5,
    ).fit(prices)
    result = model.sample_prices(n_periods=120, n_paths=40, seed=21)

    historical = np.log(prices / prices.shift(1)).dropna()
    labels = (
        RegimeLabeler(proxy_column="Proxy", threshold_percentile=70, lookback=5)
        .fit(prices)
        .get_labels()
    )

    indices = _infer_indices_from_reference_asset(
        simulated=result.log_returns,
        historical=historical,
        reference_asset="AssetRef",
    )
    simulated_labels = labels.to_numpy()[indices]

    a_returns = result.log_returns.xs("AssetA", level=1, axis=1).to_numpy()
    b_returns = result.log_returns.xs("AssetB", level=1, axis=1).to_numpy()

    stress_mask = simulated_labels == "stress"
    calm_mask = simulated_labels == "calm"
    assert stress_mask.any()
    assert calm_mask.any()

    a_stress = a_returns[stress_mask]
    b_stress = b_returns[stress_mask]
    a_calm = a_returns[calm_mask]
    b_calm = b_returns[calm_mask]

    assert a_stress.std() > a_calm.std()
    stress_corr = float(np.corrcoef(a_stress, b_stress)[0, 1])
    calm_corr = float(np.corrcoef(a_calm, b_calm)[0, 1])
    assert stress_corr > calm_corr

    rolling_vol = (
        pd.DataFrame(a_returns, index=result.log_returns.index).rolling(5).std(ddof=0).to_numpy()
    )
    vol_mask = ~np.isnan(rolling_vol)
    stress_vol = float(np.nanmean(rolling_vol[stress_mask & vol_mask]))
    calm_vol = float(np.nanmean(rolling_vol[calm_mask & vol_mask]))
    assert stress_vol > calm_vol


def test_regime_sampling_elevates_stress_correlation() -> None:
    rng = np.random.default_rng(99)
    n_obs = 160
    calm_n = n_obs // 2
    stress_n = n_obs - calm_n

    def _correlated_returns(n: int, *, vol: float, corr: float) -> np.ndarray:
        cov = np.array([[1.0, corr], [corr, 1.0]]) * (vol**2)
        chol = np.linalg.cholesky(cov)
        return rng.standard_normal((n, 2)) @ chol.T

    calm_returns = _correlated_returns(calm_n, vol=0.003, corr=0.1)
    stress_returns = _correlated_returns(stress_n, vol=0.018, corr=0.8)
    asset_returns = np.vstack([calm_returns, stress_returns])

    proxy_returns = np.concatenate(
        [
            rng.standard_normal(calm_n) * 0.002,
            rng.standard_normal(stress_n) * 0.02,
        ]
    )
    reference_returns = _unique_log_returns(n_obs, start=0.0004, step=0.00001)

    log_returns = np.column_stack(
        [
            asset_returns,
            reference_returns,
            proxy_returns,
        ]
    )

    index = pd.date_range("2019-01-01", periods=n_obs, freq="D")
    prices = _prices_from_log_returns(log_returns, index, ["AssetA", "AssetB", "AssetRef", "Proxy"])

    model = RegimeConditionedBootstrapModel(
        mean_block_len=4,
        frequency="D",
        regime_proxy_column="Proxy",
        threshold_percentile=70,
        lookback=5,
    ).fit(prices)
    result = model.sample_prices(n_periods=100, n_paths=50, seed=7)

    historical = np.log(prices / prices.shift(1)).dropna()
    labels = (
        RegimeLabeler(proxy_column="Proxy", threshold_percentile=70, lookback=5)
        .fit(prices)
        .get_labels()
    )

    indices = _infer_indices_from_reference_asset(
        simulated=result.log_returns,
        historical=historical,
        reference_asset="AssetRef",
    )
    simulated_labels = labels.to_numpy()[indices]

    a_returns = result.log_returns.xs("AssetA", level=1, axis=1).to_numpy()
    b_returns = result.log_returns.xs("AssetB", level=1, axis=1).to_numpy()

    stress_mask = simulated_labels == "stress"
    calm_mask = simulated_labels == "calm"
    assert stress_mask.any()
    assert calm_mask.any()

    stress_corr = float(np.corrcoef(a_returns[stress_mask], b_returns[stress_mask])[0, 1])
    calm_corr = float(np.corrcoef(a_returns[calm_mask], b_returns[calm_mask])[0, 1])

    assert stress_corr > calm_corr
    assert stress_corr >= 0.7


def test_regime_sampling_deterministic_with_seed() -> None:
    index = pd.date_range("2024-01-01", periods=30, freq="D")
    prices = pd.DataFrame(
        {
            "AssetA": np.linspace(100, 120, len(index)),
            "AssetB": np.linspace(80, 95, len(index)),
            "Proxy": np.linspace(200, 240, len(index)),
        },
        index=index,
    )

    model = RegimeConditionedBootstrapModel(
        mean_block_len=3,
        frequency="D",
        regime_proxy_column="Proxy",
        threshold_percentile=60,
        lookback=3,
    ).fit(prices)

    first = model.sample_prices(n_periods=12, n_paths=3, seed=100)
    second = model.sample_prices(n_periods=12, n_paths=3, seed=100)

    pd.testing.assert_frame_equal(first.log_returns, second.log_returns)
    pd.testing.assert_frame_equal(first.prices, second.prices)


def test_regime_model_falls_back_without_proxy() -> None:
    index = pd.date_range("2024-02-01", periods=25, freq="D")
    prices = pd.DataFrame(
        {
            "AssetA": np.linspace(90, 110, len(index)),
            "AssetB": np.linspace(70, 85, len(index)),
        },
        index=index,
    )

    regime_model = RegimeConditionedBootstrapModel(
        mean_block_len=4,
        frequency="D",
        regime_proxy_column="MissingProxy",
    ).fit(prices)
    fallback = regime_model.sample_prices(n_periods=10, n_paths=2, seed=13)

    stationary_model = StationaryBootstrapModel(mean_block_len=4, frequency="D").fit(prices)
    stationary = stationary_model.sample_prices(n_periods=10, n_paths=2, seed=13)

    pd.testing.assert_frame_equal(fallback.log_returns, stationary.log_returns)
    pd.testing.assert_frame_equal(fallback.prices, stationary.prices)


def test_regime_model_falls_back_when_proxy_returns_missing() -> None:
    index = pd.date_range("2024-03-01", periods=20, freq="D")
    proxy_values = np.full(len(index), np.nan)
    proxy_values[-1] = 200.0
    prices = pd.DataFrame(
        {
            "AssetA": np.linspace(95, 105, len(index)),
            "AssetB": np.linspace(70, 90, len(index)),
            "Proxy": proxy_values,
        },
        index=index,
    )

    regime_model = RegimeConditionedBootstrapModel(
        mean_block_len=3,
        frequency="D",
        regime_proxy_column="Proxy",
    ).fit(prices)
    fallback = regime_model.sample_prices(n_periods=8, n_paths=2, seed=5)

    stationary_model = StationaryBootstrapModel(mean_block_len=3, frequency="D").fit(prices)
    stationary = stationary_model.sample_prices(n_periods=8, n_paths=2, seed=5)

    pd.testing.assert_frame_equal(fallback.log_returns, stationary.log_returns)
    pd.testing.assert_frame_equal(fallback.prices, stationary.prices)
