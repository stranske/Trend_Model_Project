import numpy as np
import pandas as pd

from trend_analysis.plugins import create_weight_engine, weight_engine_registry


def test_group_upper_bound_is_honored():
    cov = pd.DataFrame(
        np.diag([0.01, 0.01, 0.09, 0.09]),
        index=["low_a", "low_b", "high_a", "high_b"],
        columns=["low_a", "low_b", "high_a", "high_b"],
    )
    engine = create_weight_engine(
        "convex_constrained",
        groups={"low_vol": ["low_a", "low_b"]},
        group_bounds={"low_vol": (0.0, 0.30)},
    )

    weights = engine.weight(cov)

    assert "convex_constrained" in weight_engine_registry.available()
    assert weights.loc[["low_a", "low_b"]].sum() <= 0.30 + 1e-6
    assert abs(weights.sum() - 1.0) <= 1e-6
    assert (weights >= -1e-9).all()


def test_unconstrained_matches_min_variance():
    cov = pd.DataFrame(
        [[0.04, 0.006, 0.0], [0.006, 0.09, 0.0], [0.0, 0.0, 0.16]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    engine = create_weight_engine("convex_constrained")

    weights = engine.weight(cov)

    inverse_cov = np.linalg.inv(cov.to_numpy(dtype=float))
    expected = inverse_cov @ np.ones(len(cov))
    expected = expected / expected.sum()
    assert np.allclose(weights.to_numpy(), expected, atol=1e-6)
