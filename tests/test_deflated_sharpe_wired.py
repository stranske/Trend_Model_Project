from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import trend_analysis.walk_forward as wf


def _fixed_returns() -> pd.Series:
    return pd.Series(
        np.tile([0.018, 0.012, -0.006, 0.021, 0.009, -0.003, 0.015, 0.006], 10),
        dtype=float,
    )


def _periodic_walk_forward_returns() -> pd.DataFrame:
    index = pd.date_range("2020-01-31", periods=54, freq="ME")
    pattern = np.array([0.020, 0.010, -0.010, 0.015, 0.000, 0.005])
    return pd.DataFrame({"asset": np.resize(pattern, len(index))}, index=index)


def _single_trial_strategy() -> wf.StrategyConfig:
    return wf.StrategyConfig(top_n=1, grid={"lookback": [6]})


def _multi_trial_walk_forward_returns() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    index = pd.date_range("2020-01-31", periods=72, freq="ME")
    return_values = rng.normal(0.004, 0.02, size=(72, 3))
    return_values[:, 0] += np.linspace(-0.004, 0.008, len(index))
    return_values[:, 1] += np.sin(np.arange(len(index)) / 4) * 0.012
    return pd.DataFrame(return_values, index=index, columns=["trend", "cycle", "noise"])


def test_dsr_decreases_with_trial_count() -> None:
    returns = _fixed_returns()

    values = [
        wf._sweep_deflated_sharpe(returns, n_trials=n_trials) for n_trials in (1, 10, 100, 500)
    ]

    assert values == sorted(values, reverse=True)
    undeflated_psr = values[0]
    assert all(value < undeflated_psr for value in values[1:])


def test_overlapping_windows_pool_one_return_per_calendar_date(monkeypatch) -> None:
    returns = _periodic_walk_forward_returns()
    windows = wf.WindowConfig(train=12, test=12, step=6)
    observed: list[tuple[pd.Index, int]] = []
    real_estimate = wf.estimate_sharpe_moments

    def recording_estimate(series):
        moments = real_estimate(series)
        observed.append((series.index.copy(), moments[1]))
        return moments

    monkeypatch.setattr(wf, "estimate_sharpe_moments", recording_estimate)

    wf.evaluate_parameter_grid(returns, windows, _single_trial_strategy())

    splits = wf._window_splits(returns.index, windows)
    expected_index = returns.index[12:]
    summed_fold_lengths = sum(len(test_idx) for _, test_idx in splits)
    assert summed_fold_lengths == 72
    assert len(expected_index) == 42
    assert observed
    assert all(index.equals(expected_index) for index, _ in observed)
    assert all(n_obs == expected_index.nunique() for _, n_obs in observed)
    assert all(n_obs < summed_fold_lengths for _, n_obs in observed)


def test_overlapping_windows_keep_most_recent_fold_prediction() -> None:
    index = pd.date_range("2024-01-31", periods=3, freq="ME")
    older_fold = pd.Series([0.01, 0.02], index=index[:2])
    newer_fold = pd.Series([0.03, 0.04], index=index[1:])

    pooled = wf._pool_fold_returns([older_fold, newer_fold])

    expected = pd.Series([0.01, 0.03, 0.04], index=index)
    pd.testing.assert_series_equal(pooled, expected, check_freq=False)


def test_disjoint_windows_leave_pooled_returns_unchanged(monkeypatch) -> None:
    returns = _periodic_walk_forward_returns()
    windows = wf.WindowConfig(train=12, test=6, step=6)
    observed: list[pd.Series] = []
    real_estimate = wf.estimate_sharpe_moments

    def recording_estimate(series):
        observed.append(series.copy())
        return real_estimate(series)

    monkeypatch.setattr(wf, "estimate_sharpe_moments", recording_estimate)

    wf.evaluate_parameter_grid(returns, windows, _single_trial_strategy())

    splits = wf._window_splits(returns.index, windows)
    expected = pd.concat([returns.loc[test_idx, "asset"] for _, test_idx in splits]).rename(None)
    assert observed
    assert all(series.index.is_unique for series in observed)
    assert all(len(series) == sum(len(test_idx) for _, test_idx in splits) for series in observed)
    for series in observed:
        pd.testing.assert_series_equal(series, expected)


def test_overlapping_windows_reduce_dsr_from_duplicated_pool(monkeypatch) -> None:
    returns = _multi_trial_walk_forward_returns()
    windows = wf.WindowConfig(train=24, test=12, step=6)
    strategy = wf.StrategyConfig(grid={"lookback": [4, 12], "top_n": [1, 2]})

    _, corrected = wf.evaluate_parameter_grid(returns, windows, strategy)

    monkeypatch.setattr(
        wf,
        "_pool_fold_returns",
        lambda fold_returns: pd.concat(fold_returns, ignore_index=True),
    )
    _, duplicated = wf.evaluate_parameter_grid(returns, windows, strategy)

    assert corrected["n_trials"].eq(4).all()
    assert corrected["deflated_sharpe_ratio"].lt(duplicated["deflated_sharpe_ratio"]).all()


def test_parameter_sweep_reports_actual_trial_count_and_dsr(monkeypatch) -> None:
    returns = _multi_trial_walk_forward_returns()
    windows = wf.WindowConfig(train=24, test=12, step=12)
    strategy = wf.StrategyConfig(grid={"lookback": [4, 12], "top_n": [1, 2]})
    captured_calls = []
    real_dsr = wf.deflated_sharpe_ratio

    def recording_dsr(sharpe, n_obs, skew, kurtosis, n_trials, *, sharpe_variance=None):
        captured_calls.append((n_trials, sharpe_variance))
        return real_dsr(
            sharpe,
            n_obs,
            skew,
            kurtosis,
            n_trials,
            sharpe_variance=sharpe_variance,
        )

    monkeypatch.setattr(wf, "deflated_sharpe_ratio", recording_dsr)

    _, summary = wf.evaluate_parameter_grid(returns, windows, strategy)

    assert summary["n_trials"].eq(4).all()
    assert summary["deflated_sharpe_ratio"].notna().all()
    assert summary["deflated_sharpe_ratio"].lt(summary["mean_sharpe"]).all()
    assert (
        summary.columns.get_loc("deflated_sharpe_ratio")
        == summary.columns.get_loc("mean_sharpe") + 1
    )
    trial_variance = summary["trial_sharpe_variance"].iloc[0]
    assert trial_variance > 0.0
    assert captured_calls == [(4, trial_variance)] * 4


def test_single_trial_returns_undeflated_psr_without_raising() -> None:
    returns = _fixed_returns()

    value = wf._sweep_deflated_sharpe(returns, n_trials=1)

    assert np.isfinite(value)


def test_invalid_finite_moments_return_unavailable_dsr() -> None:
    returns = pd.Series([-0.0103, -0.0103, -0.0102, -0.0102], dtype=float)

    assert np.isnan(wf._sweep_deflated_sharpe(returns, n_trials=1))

    with pytest.raises(ValueError, match="n_trials must be at least 1"):
        wf._sweep_deflated_sharpe(_fixed_returns(), n_trials=0)
