from typing import Any

import numpy as np
import pandas as pd
import pytest

from trend_analysis.weights.equal_risk_contribution import EqualRiskContribution
from trend_analysis.weights.hierarchical_risk_parity import HierarchicalRiskParity
from trend_analysis.weights.risk_parity import RiskParity
from trend_analysis.weights.robust_weighting import RobustRiskParity


@pytest.fixture
def simple_cov() -> pd.DataFrame:
    data = np.array([[0.04, 0.0], [0.0, 0.09]])
    return pd.DataFrame(data, index=["a", "b"], columns=["a", "b"])


class TestEqualRiskContribution:
    def test_empty_covariance_returns_empty_series(self) -> None:
        engine = EqualRiskContribution()
        result = engine.weight(pd.DataFrame())
        assert result.empty

    def test_requires_square_matrix(self, simple_cov: pd.DataFrame) -> None:
        skewed = simple_cov.rename(columns={"b": "c"})
        engine = EqualRiskContribution()
        with pytest.raises(ValueError):
            engine.weight(skewed)

    def test_regularizes_ill_conditioned_matrix(self) -> None:
        cov = pd.DataFrame(np.diag([1e-13, 1.0]), index=["a", "b"], columns=["a", "b"])
        engine = EqualRiskContribution()
        weights = engine.weight(cov)
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()

    def test_handles_negative_eigenvalues(self) -> None:
        cov = pd.DataFrame(
            [[2.0, 3.0], [3.0, 2.0]], index=["a", "b"], columns=["a", "b"]
        )
        engine = EqualRiskContribution()
        weights = engine.weight(cov)
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()

    def test_eigen_decomposition_failure_returns_equal_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cov = pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.09]], index=["a", "b"], columns=["a", "b"]
        )
        engine = EqualRiskContribution()

        def boom(_: np.ndarray) -> np.ndarray:
            raise np.linalg.LinAlgError("boom")

        monkeypatch.setattr(np.linalg, "eigvals", boom)
        weights = engine.weight(cov)
        assert np.allclose(weights.values, np.array([0.5, 0.5]))

    def test_non_positive_portfolio_variance_returns_equal_weights(self) -> None:
        cov = pd.DataFrame(np.zeros((2, 2)), index=["a", "b"], columns=["a", "b"])
        engine = EqualRiskContribution()
        weights = engine.weight(cov)
        assert np.allclose(weights.values, np.array([0.5, 0.5]))

    def test_non_convergence_emits_equal_weights(self) -> None:
        cov = pd.DataFrame(
            [[0.04, 0.024], [0.024, 0.09]], index=["a", "b"], columns=["a", "b"]
        )
        engine = EqualRiskContribution(max_iter=0)
        weights = engine.weight(cov)
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()


class TestHierarchicalRiskParity:
    def test_empty_covariance_returns_empty_series(self) -> None:
        engine = HierarchicalRiskParity()
        result = engine.weight(pd.DataFrame())
        assert result.empty

    def test_requires_square_matrix(self, simple_cov: pd.DataFrame) -> None:
        skewed = simple_cov.rename(columns={"b": "c"})
        engine = HierarchicalRiskParity()
        with pytest.raises(ValueError):
            engine.weight(skewed)

    def test_zero_standard_deviation_is_regularized(self) -> None:
        cov = pd.DataFrame(
            [[0.0, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.09]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        engine = HierarchicalRiskParity()
        with pytest.warns(RuntimeWarning):
            weights = engine.weight(cov)
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()

    def test_invalid_distance_falls_back_to_equal_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cov = pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.09]], index=["a", "b"], columns=["a", "b"]
        )
        engine = HierarchicalRiskParity()

        def fake_cov_to_corr(_: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                [[1.5, 1.5], [1.5, 1.5]], index=cov.index, columns=cov.columns
            )

        monkeypatch.setattr(
            "trend_analysis.weights.hierarchical_risk_parity._cov_to_corr",
            fake_cov_to_corr,
        )
        with pytest.warns(RuntimeWarning):
            weights = engine.weight(cov)
        assert np.allclose(weights.values, np.array([0.5, 0.5]))

    def test_linkage_failure_falls_back_to_equal_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cov = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]], index=["a", "b"], columns=["a", "b"]
        )
        engine = HierarchicalRiskParity()

        def raise_error(*_: Any, **__: Any) -> None:
            raise ValueError("linkage failed")

        monkeypatch.setattr(
            "trend_analysis.weights.hierarchical_risk_parity.linkage", raise_error
        )
        weights = engine.weight(cov)
        assert np.allclose(weights.values, np.array([0.5, 0.5]))

    def test_cluster_allocation_error_falls_back_to_equal_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cov = pd.DataFrame(
            [[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.16]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        engine = HierarchicalRiskParity()

        call_counter = {"count": 0}
        original_diag = np.diag

        def diag_with_failure(values: Any) -> np.ndarray:
            call_counter["count"] += 1
            if call_counter["count"] >= 2:
                raise ZeroDivisionError("simulated failure")
            return original_diag(values)

        monkeypatch.setattr(np, "diag", diag_with_failure)

        weights = engine.weight(cov)
        assert np.isclose(weights.loc["a"], weights.loc["b"])
        assert np.isclose(weights.loc["a"] + weights.loc["b"], 0.5, atol=1e-8)
        assert np.isclose(weights.sum(), 1.0)


class TestRiskParity:
    def test_empty_covariance_returns_empty_series(self) -> None:
        engine = RiskParity()
        result = engine.weight(pd.DataFrame())
        assert result.empty

    def test_requires_square_matrix(self, simple_cov: pd.DataFrame) -> None:
        skewed = simple_cov.rename(columns={"b": "c"})
        engine = RiskParity()
        with pytest.raises(ValueError):
            engine.weight(skewed)

    def test_non_positive_variances_are_clamped(self) -> None:
        cov = pd.DataFrame(
            [[0.0, 0.0], [0.0, 0.09]], index=["a", "b"], columns=["a", "b"]
        )
        engine = RiskParity()
        weights = engine.weight(cov)
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()

    def test_non_finite_inverse_volatility_returns_equal_weights(self) -> None:
        cov = pd.DataFrame(
            [[0.04, 0.0], [0.0, np.nan]], index=["a", "b"], columns=["a", "b"]
        )
        engine = RiskParity()
        weights = engine.weight(cov)
        assert np.allclose(weights.values, np.array([0.5, 0.5]))


class TestRobustRiskParity:
    def test_empty_covariance_returns_empty_series(self) -> None:
        engine = RobustRiskParity()
        result = engine.weight(pd.DataFrame())
        assert result.empty

    def test_non_positive_diagonal_triggers_diagonal_loading(self) -> None:
        cov = pd.DataFrame(
            [[0.0, 0.0], [0.0, 0.04]], index=["a", "b"], columns=["a", "b"]
        )
        engine = RobustRiskParity()
        weights = engine.weight(cov)
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()

    def test_ill_conditioned_matrix_is_regularized(self) -> None:
        cov = pd.DataFrame(np.diag([1e-13, 1.0]), index=["a", "b"], columns=["a", "b"])
        engine = RobustRiskParity(condition_threshold=1e6)
        weights = engine.weight(cov)
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()
