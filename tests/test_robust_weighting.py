"""Tests for robust weighting methods with shrinkage and condition number
safeguards."""

import logging

import numpy as np
import pandas as pd
import pytest

from trend_analysis.plugins import create_weight_engine
from trend_analysis.weights import robust_weighting as rw


def create_well_conditioned_cov():
    """Create a well-conditioned covariance matrix."""
    return pd.DataFrame(
        [[0.04, 0.002, 0.001], [0.002, 0.09, 0.003], [0.001, 0.003, 0.16]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )


def create_ill_conditioned_cov():
    """Create an ill-conditioned covariance matrix (near-singular)."""
    # Create a matrix with very small eigenvalues
    base = np.array(
        [[1.0, 0.99999, 0.99998], [0.99999, 1.0, 0.99999], [0.99998, 0.99999, 1.0]]
    )
    # Scale to realistic variance levels
    base *= 0.04
    return pd.DataFrame(base, index=["a", "b", "c"], columns=["a", "b", "c"])


def create_singular_cov():
    """Create a singular covariance matrix."""
    # Perfectly correlated assets (rank deficient)
    base = np.ones((3, 3)) * 0.04
    return pd.DataFrame(base, index=["a", "b", "c"], columns=["a", "b", "c"])


def create_negative_eigenvalue_cov():
    """Create a covariance matrix with negative eigenvalues (non-PSD)."""
    # Manually construct a non-positive semi-definite matrix
    base = np.array([[0.04, 0.05, 0.03], [0.05, 0.09, 0.08], [0.03, 0.08, 0.02]])
    return pd.DataFrame(base, index=["a", "b", "c"], columns=["a", "b", "c"])


def create_realistic_cov():
    """Create a realistic covariance matrix with varied volatilities."""
    corr = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]])
    stds = np.array([0.3, 0.2, 0.1])
    cov = np.outer(stds, stds) * corr
    return pd.DataFrame(cov, index=["a", "b", "c"], columns=["a", "b", "c"])


class TestRobustMeanVariance:
    """Tests for RobustMeanVariance weight engine."""

    def test_well_conditioned_matrix_no_shrinkage(self):
        """Test that well-conditioned matrices work without shrinkage."""
        cov = create_well_conditioned_cov()
        engine = create_weight_engine("robust_mv", shrinkage_method="none")
        weights = engine.weight(cov)

        # Basic sanity checks
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()
        assert len(weights) == 3
        assert list(weights.index) == ["a", "b", "c"]

    def test_shrinkage_methods(self):
        """Test different shrinkage methods."""
        cov = create_well_conditioned_cov()

        # Test Ledoit-Wolf shrinkage
        engine_lw = create_weight_engine("robust_mv", shrinkage_method="ledoit_wolf")
        weights_lw = engine_lw.weight(cov)
        assert np.isclose(weights_lw.sum(), 1.0)
        assert (weights_lw >= 0).all()

        # Test OAS shrinkage
        engine_oas = create_weight_engine("robust_mv", shrinkage_method="oas")
        weights_oas = engine_oas.weight(cov)
        assert np.isclose(weights_oas.sum(), 1.0)
        assert (weights_oas >= 0).all()

        # Different shrinkage methods should give different results
        assert not np.allclose(weights_lw.values, weights_oas.values, rtol=1e-3)

    def test_ill_conditioned_safe_mode_hrp(self):
        """Test safe mode fallback to HRP for ill-conditioned matrices."""
        cov = create_ill_conditioned_cov()

        # Use a low condition threshold to trigger safe mode
        engine = create_weight_engine(
            "robust_mv", condition_threshold=1e6, safe_mode="hrp"
        )

        # The engine should handle ill-conditioned matrices gracefully
        weights = engine.weight(cov)

        # Basic sanity checks for HRP fallback
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()
        assert len(weights) == 3

    def test_ill_conditioned_safe_mode_risk_parity(self):
        """Test safe mode fallback to risk parity for ill-conditioned
        matrices."""
        cov = create_ill_conditioned_cov()

        engine = create_weight_engine(
            "robust_mv", condition_threshold=1e6, safe_mode="risk_parity"
        )

        weights = engine.weight(cov)

        # Basic sanity checks for risk parity fallback
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()
        assert len(weights) == 3

    def test_ill_conditioned_safe_mode_diagonal_mv(self):
        """Test safe mode fallback to diagonal-loaded MV for ill-conditioned
        matrices."""
        cov = create_ill_conditioned_cov()

        engine = create_weight_engine(
            "robust_mv",
            condition_threshold=1e6,
            safe_mode="diagonal_mv",
            diagonal_loading_factor=1e-3,
        )

        weights = engine.weight(cov)

        # Basic sanity checks for diagonal-loaded MV fallback
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()
        assert len(weights) == 3

    def test_safe_mode_diagnostics_and_weight_differences(self):
        cov = create_realistic_cov()

        hrp_engine = create_weight_engine(
            "robust_mv",
            condition_threshold=1.0,
            safe_mode="hrp",
            shrinkage_method="none",
        )
        hrp_weights = hrp_engine.weight(cov)
        diag = hrp_engine.diagnostics
        assert diag["used_safe_mode"] is True
        assert diag["condition_number"] > 1.0

        rp_engine = create_weight_engine(
            "robust_mv",
            condition_threshold=1.0,
            safe_mode="risk_parity",
            shrinkage_method="none",
        )
        rp_weights = rp_engine.weight(cov)

        diag_engine = create_weight_engine(
            "robust_mv",
            condition_threshold=1.0,
            safe_mode="diagonal_mv",
            shrinkage_method="none",
            diagonal_loading_factor=1e-4,
        )
        diag_weights = diag_engine.weight(cov)

        assert not np.allclose(
            hrp_weights.values, rp_weights.values, rtol=1e-3, atol=1e-4
        )
        assert not np.allclose(
            rp_weights.values, diag_weights.values, rtol=1e-3, atol=1e-4
        )

    def test_singular_matrix_fallback(self):
        """Test handling of completely singular matrices."""
        cov = create_singular_cov()

        engine = create_weight_engine(
            "robust_mv", condition_threshold=1e10, safe_mode="hrp"
        )

        weights = engine.weight(cov)

        # Should still produce valid weights
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()
        assert len(weights) == 3

    def test_empty_matrix(self):
        """Test handling of empty covariance matrix."""
        cov = pd.DataFrame()
        engine = create_weight_engine("robust_mv")
        weights = engine.weight(cov)

        assert weights.empty
        assert len(weights) == 0

    def test_weight_constraints(self):
        """Test weight constraints are respected."""
        cov = create_well_conditioned_cov()

        engine = create_weight_engine(
            "robust_mv", min_weight=0.1, max_weight=0.6, shrinkage_method="none"
        )

        weights = engine.weight(cov)

        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0.1 - 1e-6).all()  # Allow small numerical tolerance
        assert (weights <= 0.6 + 1e-6).all()

    def test_logging_behavior(self, caplog):
        """Test that appropriate logging occurs."""
        cov = create_ill_conditioned_cov()

        with caplog.at_level(logging.DEBUG):
            engine = create_weight_engine(
                "robust_mv", condition_threshold=1e6, shrinkage_method="ledoit_wolf"
            )
            engine.weight(cov)

        # Should have debug logs about shrinkage and condition numbers
        log_messages = [record.message for record in caplog.records]
        assert any("shrinkage" in msg.lower() for msg in log_messages)


class TestRobustRiskParity:
    """Tests for RobustRiskParity weight engine."""

    def test_well_conditioned_matrix(self):
        """Test robust risk parity with well-conditioned matrix."""
        cov = create_well_conditioned_cov()
        engine = create_weight_engine("robust_risk_parity")
        weights = engine.weight(cov)

        # Should be similar to regular risk parity
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()
        assert len(weights) == 3

    def test_ill_conditioned_matrix_diagonal_loading(self):
        """Test robust risk parity applies diagonal loading when needed."""
        cov = create_ill_conditioned_cov()

        engine = create_weight_engine(
            "robust_risk_parity", condition_threshold=1e6, diagonal_loading_factor=1e-3
        )

        # The engine should handle ill-conditioned matrices gracefully
        weights = engine.weight(cov)

        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()
        assert len(weights) == 3

    def test_zero_diagonal_handling(self):
        """Test handling of zero diagonal elements."""
        # Create matrix with zero variance asset
        cov_data = [[0.04, 0.002, 0.0], [0.002, 0.09, 0.0], [0.0, 0.0, 0.0]]
        cov = pd.DataFrame(cov_data, index=["a", "b", "c"], columns=["a", "b", "c"])

        engine = create_weight_engine("robust_risk_parity")

        weights = engine.weight(cov)

        # Should still produce valid weights
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()
        assert len(weights) == 3

    def test_empty_matrix(self):
        """Test handling of empty covariance matrix."""
        cov = pd.DataFrame()
        engine = create_weight_engine("robust_risk_parity")
        weights = engine.weight(cov)

        assert weights.empty
        assert len(weights) == 0


class TestShrinkageFunctions:
    """Tests for shrinkage utility functions."""

    def test_ledoit_wolf_shrinkage(self):
        """Test Ledoit-Wolf shrinkage function."""
        from trend_analysis.weights.robust_weighting import ledoit_wolf_shrinkage

        cov = create_well_conditioned_cov().values
        shrunk_cov, intensity = ledoit_wolf_shrinkage(cov)

        # Shrunk covariance should be valid
        assert shrunk_cov.shape == cov.shape
        assert 0.0 <= intensity <= 1.0

        # Eigenvalues should be positive
        eigenvals = np.linalg.eigvals(shrunk_cov)
        assert (eigenvals > 0).all()

    def test_oas_shrinkage(self):
        """Test OAS shrinkage function."""
        from trend_analysis.weights.robust_weighting import oas_shrinkage

        cov = create_well_conditioned_cov().values
        shrunk_cov, intensity = oas_shrinkage(cov)

        # Shrunk covariance should be valid
        assert shrunk_cov.shape == cov.shape
        assert 0.0 <= intensity <= 1.0

        # Eigenvalues should be positive
        eigenvals = np.linalg.eigvals(shrunk_cov)
        assert (eigenvals > 0).all()

    def test_diagonal_loading(self):
        """Test diagonal loading function."""
        from trend_analysis.weights.robust_weighting import diagonal_loading

        cov = create_ill_conditioned_cov().values
        original_condition = np.linalg.cond(cov)

        loaded_cov = diagonal_loading(cov, loading_factor=1e-3)
        loaded_condition = np.linalg.cond(loaded_cov)

        # Condition number should improve
        assert loaded_condition < original_condition
        assert loaded_cov.shape == cov.shape

        # Diagonal elements should be larger
        assert (np.diag(loaded_cov) >= np.diag(cov)).all()


class TestRobustWeightingBranchCoverage:
    def test_ledoit_wolf_zero_trace_intensity(self):
        cov = np.zeros((2, 2))
        shrunk, intensity = rw.ledoit_wolf_shrinkage(cov)
        assert intensity == pytest.approx(1.0)
        assert np.allclose(shrunk, np.zeros_like(cov))

    def test_oas_zero_trace_intensity(self):
        cov = np.zeros((3, 3))
        shrunk, intensity = rw.oas_shrinkage(cov)
        assert intensity == pytest.approx(1.0)
        assert np.allclose(shrunk, np.zeros_like(cov))

    def test_robust_mv_unknown_shrinkage(self):
        cov = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"])
        engine = rw.RobustMeanVariance(shrinkage_method="mystery")
        with pytest.raises(ValueError, match="Unknown shrinkage method"):
            engine.weight(cov)

    def test_robust_mv_unknown_safe_mode(self):
        cov = create_ill_conditioned_cov()
        engine = rw.RobustMeanVariance(
            safe_mode="mystery", condition_threshold=1.0, shrinkage_method="none"
        )
        with pytest.raises(ValueError, match="Unknown safe mode"):
            engine.weight(cov)

    def test_robust_mv_non_square_matrix(self):
        cov = pd.DataFrame(
            [[0.1, 0.02], [0.02, 0.1]], index=["a", "b"], columns=["a", "c"]
        )
        engine = rw.RobustMeanVariance()
        with pytest.raises(ValueError, match="Covariance matrix must be square"):
            engine.weight(cov)

    def test_robust_mv_singular_fallback_to_equal_weights(self):
        cov = create_singular_cov()
        engine = rw.RobustMeanVariance(
            shrinkage_method="none",
            condition_threshold=float("inf"),
            min_weight=0.0,
        )
        weights = engine.weight(cov)
        assert pytest.approx(weights.sum()) == 1.0
        assert np.allclose(weights.values, np.repeat(1 / len(cov), len(cov)))

    def test_risk_parity_non_square_matrix(self):
        cov = pd.DataFrame(
            [[0.05, 0.01], [0.01, 0.04]], index=["a", "b"], columns=["a", "c"]
        )
        engine = rw.RobustRiskParity()
        with pytest.raises(ValueError, match="Covariance matrix must be square"):
            engine.weight(cov)

    def test_risk_parity_zero_variance_fallback(self):
        cov = pd.DataFrame(
            np.zeros((3, 3)), index=["a", "b", "c"], columns=["a", "b", "c"]
        )
        engine = rw.RobustRiskParity()
        weights = engine.weight(cov)
        assert pytest.approx(weights.sum()) == 1.0
        assert np.allclose(weights.values, np.repeat(1 / len(cov), len(cov)))

    def test_risk_parity_invalid_inverse_sum(self, monkeypatch):
        cov = create_well_conditioned_cov()
        engine = rw.RobustRiskParity()

        def fake_reciprocal(values):
            return np.full(values.shape, np.nan)

        monkeypatch.setattr(rw.np, "reciprocal", fake_reciprocal)
        weights = engine.weight(cov)
        assert pytest.approx(weights.sum()) == 1.0
        assert np.allclose(weights.values, np.repeat(1 / len(cov), len(cov)))


class TestSyntheticNearSingularCases:
    """Tests with synthetic pathological inputs to ensure stability and
    reproducibility."""

    def test_perfectly_correlated_assets(self):
        """Test handling of perfectly correlated assets."""
        # Create perfectly correlated returns
        returns = pd.DataFrame(
            {
                "asset1": [0.01, 0.02, -0.01, 0.03],
                "asset2": [0.01, 0.02, -0.01, 0.03],  # Identical to asset1
                "asset3": [0.02, 0.04, -0.02, 0.06],  # 2x asset1
            }
        )
        cov = returns.cov()

        # This should be near-singular
        condition_num = np.linalg.cond(cov.values)
        assert condition_num > 1e10

        # Test robust methods handle this gracefully
        for method in ["robust_mv", "robust_risk_parity"]:
            engine = create_weight_engine(method, condition_threshold=1e6)
            weights = engine.weight(cov)

            assert np.isclose(weights.sum(), 1.0)
            assert (weights >= 0).all()
            assert not np.any(np.isnan(weights))
            assert not np.any(np.isinf(weights))

    def test_reproducibility(self):
        """Test that results are reproducible for the same input."""
        cov = create_ill_conditioned_cov()

        # Run the same calculation multiple times
        engine = create_weight_engine(
            "robust_mv", shrinkage_method="ledoit_wolf", condition_threshold=1e8
        )

        weights1 = engine.weight(cov)
        weights2 = engine.weight(cov)
        weights3 = engine.weight(cov)

        # Results should be identical
        assert np.allclose(weights1.values, weights2.values)
        assert np.allclose(weights2.values, weights3.values)

    def test_extreme_variance_differences(self):
        """Test handling of assets with extremely different variances."""
        # Create covariance with very different scales
        cov_data = [
            [1e-8, 0.0, 0.0],  # Very low variance asset
            [0.0, 1.0, 0.0],  # Medium variance asset
            [0.0, 0.0, 100.0],  # Very high variance asset
        ]
        cov = pd.DataFrame(
            cov_data, index=["low", "med", "high"], columns=["low", "med", "high"]
        )

        for method in ["robust_mv", "robust_risk_parity"]:
            engine = create_weight_engine(method)
            weights = engine.weight(cov)

            assert np.isclose(weights.sum(), 1.0)
            assert (weights >= 0).all()
            assert not np.any(np.isnan(weights))

    def test_numerical_stability_under_scaling(self):
        """Test numerical stability when covariance matrix is scaled."""
        base_cov = create_well_conditioned_cov()

        # Test different scaling factors
        for scale in [1e-10, 1e-5, 1.0, 1e5, 1e10]:
            scaled_cov = base_cov * scale

            engine = create_weight_engine("robust_mv", shrinkage_method="ledoit_wolf")
            weights = engine.weight(scaled_cov)

            # Weights should be scale-invariant (minimum variance portfolio property)
            assert np.isclose(weights.sum(), 1.0)
            assert (weights >= 0).all()
            assert not np.any(np.isnan(weights))
            assert not np.any(np.isinf(weights))


class TestIntegrationWithExistingEngines:
    """Tests ensuring new robust engines work with existing plugin system."""

    def test_plugin_registration(self):
        """Test that robust engines are properly registered."""
        from trend_analysis.plugins import weight_engine_registry

        available_engines = weight_engine_registry.available()

        assert "robust_mv" in available_engines
        assert "robust_mean_variance" in available_engines  # alias
        assert "robust_risk_parity" in available_engines

    def test_create_weight_engine_factory(self):
        """Test factory function works with robust engines."""
        # Should be able to create without errors
        engine1 = create_weight_engine("robust_mv")
        engine2 = create_weight_engine("robust_risk_parity")

        assert engine1 is not None
        assert engine2 is not None

        # Should be different instances
        assert engine1 is not engine2

    def test_parameter_passing(self):
        """Test that parameters are properly passed to robust engines."""
        engine = create_weight_engine(
            "robust_mv",
            shrinkage_method="oas",
            condition_threshold=1e8,
            safe_mode="risk_parity",
        )

        assert engine.shrinkage_method == "oas"
        assert engine.condition_threshold == 1e8
        assert engine.safe_mode == "risk_parity"
