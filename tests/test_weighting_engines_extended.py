"""Additional coverage for weighting engines with challenging inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.weights.equal_risk_contribution import EqualRiskContribution
from trend_analysis.weights.hierarchical_risk_parity import HierarchicalRiskParity
from trend_analysis.weights.risk_parity import RiskParity
from trend_analysis.weights.robust_weighting import RobustMeanVariance, RobustRiskParity


def _make_covariance(num_assets: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    data = rng.normal(scale=0.04, size=(180, num_assets))
    columns = [f"Asset_{idx}" for idx in range(num_assets)]
    frame = pd.DataFrame(data, columns=columns)
    return frame.cov()


def test_equal_risk_contribution_regularises_ill_conditioned_matrix() -> None:
    cov = _make_covariance()
    # Force extreme condition number by shrinking one variance dramatically.
    cov.iloc[0, 0] = cov.iloc[0, 0] * 1e-8
    cov.iloc[0, 1:] = cov.iloc[1:, 0].values  # keep symmetry
    cov.iloc[1:, 0] = cov.iloc[0, 1:].values  # ensure symmetry

    engine = EqualRiskContribution(max_iter=200)
    weights = engine.weight(cov)
    assert pytest.approx(float(weights.sum()), rel=1e-6) == 1.0

    # A covariance matrix with mismatched labels should trigger the validation guard.
    bad_cov = pd.DataFrame([[1.0, 0.1], [0.1, 1.2]], index=["A", "B"], columns=["X", "Y"])
    with pytest.raises(ValueError):
        engine.weight(bad_cov)


def test_hierarchical_risk_parity_handles_zero_variance_assets() -> None:
    cov = _make_covariance()
    cov.iloc[0, 0] = 0.0  # zero variance triggers regularisation path

    engine = HierarchicalRiskParity()
    weights = engine.weight(cov)
    assert pytest.approx(float(weights.sum()), rel=1e-6) == 1.0
    assert (weights >= 0).all()


def test_risk_parity_and_robust_variants_manage_degenerate_cases() -> None:
    cov = _make_covariance()
    cov.iloc[0, 0] = 0.0
    cov.iloc[1, 1] = -1e-6  # non-positive variance triggers guards
    cov.values[:] = (cov.values + cov.values.T) / 2  # ensure symmetry

    base = RiskParity()
    base_weights = base.weight(cov)
    assert pytest.approx(float(base_weights.sum()), rel=1e-6) == 1.0

    robust_rp = RobustRiskParity(condition_threshold=1.0)
    robust_weights = robust_rp.weight(cov)
    assert pytest.approx(float(robust_weights.sum()), rel=1e-6) == 1.0

    mv_safe = RobustMeanVariance(
        shrinkage_method="ledoit_wolf",
        safe_mode="risk_parity",
        condition_threshold=1.0,
    )
    mv_safe_weights = mv_safe.weight(cov)
    assert pytest.approx(float(mv_safe_weights.sum()), rel=1e-6) == 1.0

    mv_regular = RobustMeanVariance(
        shrinkage_method="oas",
        safe_mode="diagonal_mv",
        condition_threshold=1e12,
        min_weight=0.0,
        max_weight=0.8,
    )
    mv_regular_weights = mv_regular.weight(cov)
    assert pytest.approx(float(mv_regular_weights.sum()), rel=1e-6) == 1.0
    assert (mv_regular_weights >= 0).all()


def test_weight_engines_emit_non_negative_weights() -> None:
    cov = _make_covariance()
    engines = [
        ("risk_parity", RiskParity()),
        ("hrp", HierarchicalRiskParity()),
        ("erc", EqualRiskContribution(max_iter=200)),
        ("robust_risk_parity", RobustRiskParity()),
        ("robust_mean_variance", RobustMeanVariance()),
    ]

    for name, engine in engines:
        weights = engine.weight(cov)
        assert pytest.approx(float(weights.sum()), rel=1e-6) == 1.0
        assert (weights >= 0).all(), f"{name} produced negative weights"
