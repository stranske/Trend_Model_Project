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


def test_dsr_decreases_with_trial_count() -> None:
    returns = _fixed_returns()

    values = [
        wf._sweep_deflated_sharpe(returns, n_trials=n_trials) for n_trials in (1, 10, 100, 500)
    ]

    assert values == sorted(values, reverse=True)
    undeflated_psr = values[0]
    assert all(value < undeflated_psr for value in values[1:])


def test_parameter_sweep_reports_actual_trial_count_and_dsr(monkeypatch) -> None:
    rng = np.random.default_rng(1)
    index = pd.date_range("2020-01-31", periods=72, freq="ME")
    return_values = rng.normal(0.004, 0.02, size=(72, 3))
    return_values[:, 0] += np.linspace(-0.004, 0.008, len(index))
    return_values[:, 1] += np.sin(np.arange(len(index)) / 4) * 0.012
    returns = pd.DataFrame(return_values, index=index, columns=["trend", "cycle", "noise"])
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
