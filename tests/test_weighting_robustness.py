"""Additional coverage for weighting engines with robustness checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.weights.equal_risk_contribution import \
    EqualRiskContribution
from trend_analysis.weights.hierarchical_risk_parity import \
    HierarchicalRiskParity
from trend_analysis.weights.robust_weighting import (RobustMeanVariance,
                                                     RobustRiskParity,
                                                     diagonal_loading,
                                                     ledoit_wolf_shrinkage,
                                                     oas_shrinkage)


def _make_covariance(
    values: np.ndarray, labels: list[str] | None = None
) -> pd.DataFrame:
    """Utility helper returning a symmetric covariance DataFrame."""

    labels = labels or [f"A{i}" for i in range(values.shape[0])]
    return pd.DataFrame(values, index=labels, columns=labels)


class TestEqualRiskContribution:
    def test_weighting_handles_empty_input(self) -> None:
        engine = EqualRiskContribution()
        result = engine.weight(pd.DataFrame())
        assert result.empty

    def test_weighting_requires_square_matrix(self) -> None:
        engine = EqualRiskContribution()
        cov = pd.DataFrame(
            [[0.1, 0.0], [0.0, 0.2]], index=["A", "B"], columns=["A", "C"]
        )
        with pytest.raises(ValueError):
            engine.weight(cov)

    def test_weighting_regularises_ill_conditioned_matrix(self) -> None:
        # Matrix with a negative eigenvalue triggers the regularisation path.
        cov = _make_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]), labels=["A", "B"])
        engine = EqualRiskContribution(max_iter=256, tol=1e-9)
        weights = engine.weight(cov)
        assert weights.sum() == pytest.approx(1.0, rel=1e-9)
        assert (weights >= 0).all()


class TestHierarchicalRiskParity:
    def test_weighting_returns_empty_series_for_empty_covariance(self) -> None:
        engine = HierarchicalRiskParity()
        assert engine.weight(pd.DataFrame()).empty

    def test_weighting_requires_matching_labels(self) -> None:
        engine = HierarchicalRiskParity()
        cov = pd.DataFrame(
            [[0.1, 0.0], [0.0, 0.2]], index=["A", "B"], columns=["A", "C"]
        )
        with pytest.raises(ValueError):
            engine.weight(cov)

    def test_weighting_handles_nan_correlations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "trend_analysis.weights.hierarchical_risk_parity.np.linalg.cond",
            lambda _: 1.0,
        )
        cov = _make_covariance(
            np.array([[0.1, np.nan], [np.nan, 0.2]]), labels=["A", "B"]
        )
        engine = HierarchicalRiskParity()
        weights = engine.weight(cov)
        assert weights.sum() == pytest.approx(1.0, rel=1e-9)
        assert set(weights.index) == {"A", "B"}


class TestShrinkageUtilities:
    def test_ledoit_wolf_shrinkage_bounds_intensity(self) -> None:
        cov = np.array([[0.05, 0.01], [0.01, 0.04]], dtype=float)
        shrunk, intensity = ledoit_wolf_shrinkage(cov, n_samples=100)
        assert shrunk.shape == cov.shape
        assert 0.0 <= intensity <= 1.0

    def test_oas_shrinkage_bounds_intensity(self) -> None:
        cov = np.array([[0.03, 0.002], [0.002, 0.02]], dtype=float)
        shrunk, intensity = oas_shrinkage(cov, n_samples=80)
        assert shrunk.shape == cov.shape
        assert 0.0 <= intensity <= 1.0

    def test_diagonal_loading_inflates_diagonal(self) -> None:
        cov = np.array([[1.0, 0.1], [0.1, 1.5]], dtype=float)
        loaded = diagonal_loading(cov, loading_factor=1e-2)
        assert loaded.shape == cov.shape
        assert np.trace(loaded) > np.trace(cov)


class TestRobustMeanVariance:
    def _covariance(self) -> pd.DataFrame:
        values = np.array(
            [[0.06, 0.01, 0.02], [0.01, 0.05, 0.015], [0.02, 0.015, 0.07]],
            dtype=float,
        )
        return _make_covariance(values, labels=["A", "B", "C"])

    def test_default_shrinkage_computes_weights(self) -> None:
        cov = self._covariance()
        engine = RobustMeanVariance()
        weights = engine.weight(cov)
        assert weights.sum() == pytest.approx(1.0, rel=1e-9)
        assert (weights >= 0).all()

    def test_safe_mode_triggered_by_condition_number(self) -> None:
        cov = _make_covariance(
            np.array([[1.0, 0.0, 0.0], [0.0, 1e-12, 0.0], [0.0, 0.0, 2e-12]]),
            labels=["A", "B", "C"],
        )
        engine = RobustMeanVariance(
            condition_threshold=10.0,
            safe_mode="diagonal_mv",
            shrinkage_method="none",
        )
        weights = engine.weight(cov)
        assert weights.sum() == pytest.approx(1.0, rel=1e-9)
        assert (weights >= 0).all()

    def test_unknown_shrinkage_method_raises(self) -> None:
        cov = self._covariance()
        engine = RobustMeanVariance(shrinkage_method="mystery")
        with pytest.raises(ValueError):
            engine.weight(cov)

    def test_mean_variance_fallback_to_equal_weights(self) -> None:
        cov = _make_covariance(np.array([[1.0, 1.0], [1.0, 1.0]]), labels=["A", "B"])
        engine = RobustMeanVariance(shrinkage_method="none")
        weights = engine.weight(cov)
        assert weights.sum() == pytest.approx(1.0, rel=1e-9)
        assert np.allclose(weights.values, [0.5, 0.5])

    def test_invalid_safe_mode_raises(self) -> None:
        cov = _make_covariance(
            np.array([[1.0, 0.0, 0.0], [0.0, 1e-12, 0.0], [0.0, 0.0, 2e-12]]),
            labels=["A", "B", "C"],
        )
        engine = RobustMeanVariance(
            safe_mode="unsupported",
            condition_threshold=10.0,
            shrinkage_method="none",
        )
        with pytest.raises(ValueError):
            engine.weight(cov)


class TestRobustRiskParity:
    def test_negative_diagonal_triggers_loading(self) -> None:
        cov = _make_covariance(
            np.array([[0.1, 0.02, 0.01], [0.02, 0.0, 0.0], [0.01, 0.0, 0.08]]),
            labels=["A", "B", "C"],
        )
        engine = RobustRiskParity(diagonal_loading_factor=1e-2)
        weights = engine.weight(cov)
        assert weights.sum() == pytest.approx(1.0, rel=1e-9)
        assert (weights >= 0).all()

    def test_high_condition_number_applies_loading(self) -> None:
        cov = _make_covariance(
            np.array([[1.0, 0.0, 0.0], [0.0, 1e-15, 0.0], [0.0, 0.0, 1e-16]]),
            labels=["A", "B", "C"],
        )
        engine = RobustRiskParity(condition_threshold=10.0)
        weights = engine.weight(cov)
        assert weights.sum() == pytest.approx(1.0, rel=1e-9)
        assert (weights >= 0).all()
