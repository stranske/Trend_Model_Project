"""Regression gate for benchmark-adjusted alpha metric registration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_analysis.metrics import METRIC_REGISTRY, alpha, annual_return


def test_alpha_ranks_below_raw_return_for_pure_beta_manager() -> None:
    assert "alpha" in METRIC_REGISTRY

    rng = np.random.default_rng(42)
    n = 36
    benchmark = pd.Series(rng.normal(0.008, 0.004, n), name="SPX")

    manager_a = 2.0 * benchmark
    manager_b = benchmark + 0.0012

    returns = pd.DataFrame(
        {
            "pure_beta": manager_a,
            "positive_alpha": manager_b,
        }
    )

    annual_scores = annual_return(returns)
    alpha_scores = alpha(returns, benchmark=benchmark)

    assert annual_scores["pure_beta"] > annual_scores["positive_alpha"]
    assert alpha_scores["pure_beta"] < alpha_scores["positive_alpha"]
    assert alpha_scores["positive_alpha"] > 0.0


def test_alpha_without_benchmark_returns_nan() -> None:
    returns = pd.DataFrame({"manager": [0.01, 0.02, -0.01, 0.015, 0.005]})
    scores = alpha(returns)
    assert np.isnan(scores["manager"]).item()
