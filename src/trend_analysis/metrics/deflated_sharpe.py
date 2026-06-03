"""Probabilistic and deflated Sharpe ratio helpers."""

from __future__ import annotations

import math
from statistics import NormalDist

import pandas as pd

from trend_analysis.metrics import sharpe_ratio

_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649015329


def _validate_moments(
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> tuple[float, int, float, float]:
    sharpe = float(sharpe)
    n_obs = int(n_obs)
    skew = float(skew)
    kurtosis = float(kurtosis)
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    if not all(math.isfinite(value) for value in (sharpe, skew, kurtosis)):
        raise ValueError("sharpe, skew, and kurtosis must be finite")
    return sharpe, n_obs, skew, kurtosis


def _sharpe_standard_error_variance(
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    numerator = 1.0 - (skew * sharpe) + (((kurtosis - 1.0) / 4.0) * sharpe**2)
    if numerator <= 0.0:
        raise ValueError("invalid Sharpe moment combination")
    return numerator / float(n_obs - 1)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    *,
    sharpe_benchmark: float = 0.0,
) -> float:
    """Return Bailey-Lopez de Prado's probabilistic Sharpe ratio.

    The statistic estimates the probability that the observed Sharpe ratio is
    greater than ``sharpe_benchmark`` after adjusting for sample size, skew,
    and kurtosis.
    """

    sharpe, n_obs, skew, kurtosis = _validate_moments(sharpe, n_obs, skew, kurtosis)
    sharpe_benchmark = float(sharpe_benchmark)
    if not math.isfinite(sharpe_benchmark):
        raise ValueError("sharpe_benchmark must be finite")

    variance = _sharpe_standard_error_variance(sharpe, n_obs, skew, kurtosis)
    z_score = (sharpe - sharpe_benchmark) / math.sqrt(variance)
    return float(_NORMAL.cdf(z_score))


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    *,
    sharpe_variance: float | None = None,
) -> float:
    """Return PSR deflated by the expected maximum Sharpe over ``n_trials``."""

    sharpe, n_obs, skew, kurtosis = _validate_moments(sharpe, n_obs, skew, kurtosis)
    n_trials = int(n_trials)
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")

    if sharpe_variance is None:
        sharpe_variance = _sharpe_standard_error_variance(sharpe, n_obs, skew, kurtosis)
    sharpe_variance = float(sharpe_variance)
    if not math.isfinite(sharpe_variance) or sharpe_variance < 0.0:
        raise ValueError("sharpe_variance must be a finite non-negative value")

    if n_trials == 1 or sharpe_variance == 0.0:
        expected_max_sharpe = 0.0
    else:
        expected_max_sharpe = math.sqrt(sharpe_variance) * (
            (1.0 - _EULER_GAMMA) * _NORMAL.inv_cdf(1.0 - (1.0 / n_trials))
            + _EULER_GAMMA * _NORMAL.inv_cdf(1.0 - (1.0 / (n_trials * math.e)))
        )

    return probabilistic_sharpe_ratio(
        sharpe,
        n_obs,
        skew,
        kurtosis,
        sharpe_benchmark=expected_max_sharpe,
    )


def estimate_sharpe_moments(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> tuple[float, int, float, float]:
    """Estimate annualised Sharpe, observation count, skew, and kurtosis."""

    if not isinstance(returns, pd.Series):
        raise TypeError("estimate_sharpe_moments expects a pandas Series")
    clean = returns.dropna()
    if clean.empty:
        return float("nan"), 0, float("nan"), float("nan")
    sharpe = float(sharpe_ratio(clean, periods_per_year=periods_per_year))
    return (
        sharpe,
        int(clean.shape[0]),
        float(clean.skew()),
        float(clean.kurtosis() + 3.0),
    )
