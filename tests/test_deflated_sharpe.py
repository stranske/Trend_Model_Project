import math
from statistics import NormalDist

import pandas as pd
import pytest

from trend_analysis.metrics import (
    deflated_sharpe_ratio,
    estimate_sharpe_moments,
    probabilistic_sharpe_ratio,
)


def test_psr_matches_known_fixture() -> None:
    expected = NormalDist().cdf(
        (0.8 - 0.25) * math.sqrt(47) / math.sqrt(1 + (((3 - 1) / 4) * 0.8**2))
    )

    got = probabilistic_sharpe_ratio(
        sharpe=0.8,
        n_obs=48,
        skew=0.0,
        kurtosis=3.0,
        sharpe_benchmark=0.25,
    )

    assert got == pytest.approx(expected, abs=1e-6)
    assert got == pytest.approx(0.999484439628, abs=1e-12)


def test_dsr_decreases_with_more_trials() -> None:
    values = [
        deflated_sharpe_ratio(
            sharpe=1.1,
            n_obs=60,
            skew=0.0,
            kurtosis=3.0,
            n_trials=n_trials,
        )
        for n_trials in [1, 5, 25, 100]
    ]

    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values)


def test_dsr_uses_variance_override_for_probability() -> None:
    default = deflated_sharpe_ratio(
        sharpe=0.8,
        n_obs=48,
        skew=0.0,
        kurtosis=3.0,
        n_trials=10,
    )
    wider_variance = deflated_sharpe_ratio(
        sharpe=0.8,
        n_obs=48,
        skew=0.0,
        kurtosis=3.0,
        n_trials=10,
        sharpe_variance=0.25,
    )

    assert wider_variance < default


def test_estimate_sharpe_moments_returns_period_scale_sharpe() -> None:
    returns = pd.Series([0.02, -0.01, 0.03, 0.01, 0.04, -0.02])

    sharpe, n_obs, skew, kurtosis = estimate_sharpe_moments(returns)

    assert sharpe == pytest.approx(float(returns.mean() / returns.std(ddof=1)))
    assert n_obs == 6
    assert skew == pytest.approx(float(returns.skew()))
    assert kurtosis == pytest.approx(float(returns.kurtosis() + 3.0))
