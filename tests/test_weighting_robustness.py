"""Additional coverage for weighting engines with robustness checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from trend_analysis.weights.equal_risk_contribution import EqualRiskContribution
from trend_analysis.weights.hierarchical_risk_parity import HierarchicalRiskParity
from trend_analysis.weights.robust_weighting import (
    RobustMeanVariance,
    RobustRiskParity,
    diagonal_loading,
    ledoit_wolf_shrinkage,
    oas_shrinkage,
)


def _make_covariance(values: np.ndarray, labels: list[str] | None = None) -> pd.DataFrame:
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
        cov = pd.DataFrame([[0.1, 0.0], [0.0, 0.2]], index=["A", "B"], columns=["A", "C"])
        with pytest.raises(ValueError):
            engine.weight(cov)

    def test_weighting_regularizes_ill_conditioned_matrix(self) -> None:
        # Matrix with a negative eigenvalue triggers the regularisation path.
        cov = _make_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]), labels=["A", "B"])
        engine = EqualRiskContribution(max_iter=256, tol=1e-9)
        weights = engine.weight(cov)
        assert weights.sum() == pytest.approx(1.0, rel=1e-9)
        assert (weights >= 0).all()

    def test_weighting_handles_iteration_error_and_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Numerical errors mid-iteration should fall back to equal weights."""

        cov = _make_covariance(np.array([[0.05, 0.01], [0.01, 0.04]]), labels=["A", "B"])
        engine = EqualRiskContribution(max_iter=32, tol=1e-9)

        def exploding_max(*args: Any, **kwargs: Any) -> float:
            raise FloatingPointError("boom")

        monkeypatch.setattr("trend_analysis.weights.equal_risk_contribution.np.max", exploding_max)

        with caplog.at_level("WARNING"):
            weights = engine.weight(cov)

        assert np.allclose(weights.values, [0.5, 0.5])
        assert "Numerical error in ERC iteration" in caplog.text
        assert "did not converge" in caplog.text


class TestHierarchicalRiskParity:
    def test_weighting_returns_empty_series_for_empty_covariance(self) -> None:
        engine = HierarchicalRiskParity()
        assert engine.weight(pd.DataFrame()).empty

    def test_weighting_requires_matching_labels(self) -> None:
        engine = HierarchicalRiskParity()
        cov = pd.DataFrame([[0.1, 0.0], [0.0, 0.2]], index=["A", "B"], columns=["A", "C"])
        with pytest.raises(ValueError):
            engine.weight(cov)

    def test_weighting_handles_nan_correlations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "trend_analysis.weights.hierarchical_risk_parity.np.linalg.cond",
            lambda _: 1.0,
        )
        cov = _make_covariance(np.array([[0.1, np.nan], [np.nan, 0.2]]), labels=["A", "B"])
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

    def test_diagonal_loading_empty_matrix_returns_empty(self) -> None:
        empty = np.empty((0, 0), dtype=float)
        loaded = diagonal_loading(empty, loading_factor=1e-2)
        assert loaded.shape == (0, 0)
        assert loaded.size == 0


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

    def test_condition_number_returns_infinity_without_positive_eigenvalues(
        self,
    ) -> None:
        engine = RobustMeanVariance(shrinkage_method="none")
        cov = np.zeros((2, 2), dtype=float)
        assert engine._check_condition_number(cov) == np.inf

    def test_condition_number_handles_linalg_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = RobustMeanVariance(shrinkage_method="none")

        def explode(_: Any) -> np.ndarray:
            raise np.linalg.LinAlgError("failure")

        monkeypatch.setattr("trend_analysis.weights.robust_weighting.np.linalg.cond", explode)

        assert engine._check_condition_number(np.eye(2)) == np.inf

    def test_safe_mode_hrp_triggers_when_condition_exceeds_threshold(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cov = _make_covariance(
            np.array([[1.0, 0.0, 0.0], [0.0, 1e-12, 0.0], [0.0, 0.0, 1e-13]]),
            labels=["A", "B", "C"],
        )
        engine = RobustMeanVariance(
            safe_mode="hrp",
            condition_threshold=10.0,
            shrinkage_method="none",
        )

        with caplog.at_level("WARNING"):
            weights = engine.weight(cov)

        assert weights.sum() == pytest.approx(1.0, rel=1e-9)
        assert set(weights.index) == {"A", "B", "C"}
        assert "safe mode: hrp" in caplog.text

    def test_mean_variance_small_denominator_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        cov = _make_covariance(np.array([[0.1, 0.05], [0.05, 0.05]]), labels=["A", "B"])
        engine = RobustMeanVariance(shrinkage_method="none")

        def fake_inv(_: Any) -> np.ndarray:
            return np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)

        monkeypatch.setattr("trend_analysis.weights.robust_weighting.np.linalg.inv", fake_inv)

        with caplog.at_level("WARNING"):
            weights = engine._mean_variance_weights(cov)

        assert np.allclose(weights.values, [0.5, 0.5])
        assert "Matrix inversion failed in mean-variance" in caplog.text


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

    def test_nan_condition_number_triggers_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cov = _make_covariance(np.array([[0.05, 0.01], [0.01, 0.04]]), labels=["A", "B"])
        engine = RobustRiskParity(condition_threshold=10.0)

        monkeypatch.setattr(
            "trend_analysis.weights.robust_weighting.np.linalg.cond",
            lambda _: np.nan,
        )

        weights = engine.weight(cov)
        assert weights.sum() == pytest.approx(1.0, rel=1e-9)
        assert engine.diagnostics["used_diagonal_loading"] is True
        assert engine.diagnostics["condition_number"] == np.inf
