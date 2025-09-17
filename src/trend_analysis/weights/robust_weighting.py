"""Robust weighting methods with shrinkage and condition number safeguards."""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

import numpy as np
import pandas as pd

from ..plugins import WeightEngine, weight_engine_registry

logger = logging.getLogger(__name__)


def ledoit_wolf_shrinkage(
    cov: np.ndarray, n_samples: int = None
) -> tuple[np.ndarray, float]:
    """Apply Ledoit-Wolf shrinkage to covariance matrix.

    Args:
        cov: Covariance matrix to shrink
        n_samples: Number of observations used to estimate covariance.
                  If None, uses a heuristic based on matrix properties.

    Returns:
        tuple: (shrunk_cov, shrinkage_intensity)
    """
    p = cov.shape[0]  # Number of assets/variables

    # Sample covariance
    sample_cov = cov.copy()

    # Target: diagonal matrix with average variance
    mu = np.trace(sample_cov) / p
    target = mu * np.eye(p)

    # Estimate sample size if not provided
    if n_samples is None:
        # Heuristic: estimate based on condition number and matrix properties
        # This is an approximation when true sample size is unknown
        n_samples = max(p + 1, int(p * 2))  # Conservative estimate

    # Shrinkage intensity (simplified Ledoit-Wolf approach)
    # When we don't have access to raw data, use matrix-based approximation
    trace_diff = np.trace((sample_cov - target) @ (sample_cov - target))
    trace_sample = np.trace(sample_cov @ sample_cov)

    # Avoid division by zero
    if trace_sample == 0:
        intensity = 1.0
    else:
        # Use estimated sample size rather than matrix dimension
        intensity = min(1.0, trace_diff / (n_samples * trace_sample))

    # Apply shrinkage
    shrunk_cov = (1 - intensity) * sample_cov + intensity * target

    return shrunk_cov, intensity


def oas_shrinkage(cov: np.ndarray, n_samples: int = None) -> tuple[np.ndarray, float]:
    """Apply Oracle Approximating Shrinkage (OAS) to covariance matrix.

    Args:
        cov: Covariance matrix to shrink
        n_samples: Number of observations used to estimate covariance.
                  If None, uses a heuristic based on matrix properties.

    Returns:
        tuple: (shrunk_cov, shrinkage_intensity)
    """
    p = cov.shape[0]  # Number of assets/variables

    # Sample covariance
    sample_cov = cov.copy()

    # Target: diagonal matrix with average variance
    mu = np.trace(sample_cov) / p
    target = mu * np.eye(p)

    # Estimate sample size if not provided
    if n_samples is None:
        # Heuristic: estimate based on matrix properties
        # This is an approximation when true sample size is unknown
        n_samples = max(p + 1, int(p * 2))  # Conservative estimate

    # OAS shrinkage intensity
    trace_sample = np.trace(sample_cov @ sample_cov)
    trace_target = np.trace(target @ target)

    if trace_sample == 0:
        intensity = 1.0
    else:
        # Simplified OAS formula using estimated sample size
        intensity = min(1.0, (trace_target) / (n_samples * trace_sample))

    # Apply shrinkage
    shrunk_cov = (1 - intensity) * sample_cov + intensity * target

    return shrunk_cov, intensity


def diagonal_loading(cov: np.ndarray, loading_factor: float = 1e-6) -> np.ndarray:
    """Apply diagonal loading to improve conditioning.

    Args:
        cov: Covariance matrix
        loading_factor: Factor to add to diagonal elements

    Returns:
        Regularized covariance matrix
    """
    return cov + loading_factor * np.trace(cov) / len(cov) * np.eye(len(cov))


@weight_engine_registry.register("robust_mv")
@weight_engine_registry.register("robust_mean_variance")
class RobustMeanVariance(WeightEngine):
    """Robust mean-variance optimization with shrinkage and safe mode fallback."""

    def __init__(
        self,
        *,
        shrinkage_method: Literal["none", "ledoit_wolf", "oas"] = "ledoit_wolf",
        condition_threshold: float = 1e12,
        safe_mode: Literal["hrp", "risk_parity", "diagonal_mv"] = "hrp",
        diagonal_loading_factor: float = 1e-6,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        log_condition_numbers: bool = False,
        log_shrinkage_intensity: bool = False,
        log_method_switches: bool = False,
    ) -> None:
        """Initialize robust mean-variance optimizer.

        Args:
            shrinkage_method: Type of shrinkage to apply
            condition_threshold: Maximum allowed condition number
            safe_mode: Fallback method when matrix is ill-conditioned
            diagonal_loading_factor: Factor for diagonal loading regularization
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint
            log_condition_numbers: Promote condition number messages to INFO.
            log_shrinkage_intensity: Promote shrinkage logs to INFO.
            log_method_switches: Emit INFO logs describing method selection.
        """
        self.shrinkage_method = shrinkage_method
        self.condition_threshold = float(condition_threshold)
        self.safe_mode = safe_mode
        self.diagonal_loading_factor = float(diagonal_loading_factor)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.log_condition_numbers = bool(log_condition_numbers)
        self.log_shrinkage_intensity = bool(log_shrinkage_intensity)
        self.log_method_switches = bool(log_method_switches)
        self.last_run_info: Optional[Dict[str, Any]] = None

    def _log_condition(self, condition_number: float) -> None:
        level = logging.INFO if self.log_condition_numbers else logging.DEBUG
        logger.log(level, "Covariance matrix condition number: %.2e", condition_number)

    def _log_shrinkage(self, message: str) -> None:
        level = logging.INFO if self.log_shrinkage_intensity else logging.DEBUG
        logger.log(level, message)

    def _check_condition_number(self, cov: np.ndarray) -> float:
        """Compute condition number of covariance matrix."""
        try:
            eigenvals = np.linalg.eigvals(cov)
            eigenvals = eigenvals[eigenvals > 0]  # Only positive eigenvalues
            if len(eigenvals) == 0:
                return np.inf
            return np.max(eigenvals) / np.min(eigenvals)
        except np.linalg.LinAlgError:
            return np.inf

    def _apply_shrinkage(self, cov: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        """Apply shrinkage method to covariance matrix."""
        shrinkage_info = {"method": self.shrinkage_method, "intensity": 0.0}

        if self.shrinkage_method == "none":
            return cov.copy(), shrinkage_info
        elif self.shrinkage_method == "ledoit_wolf":
            shrunk_cov, intensity = ledoit_wolf_shrinkage(cov)
            shrinkage_info["intensity"] = intensity
            self._log_shrinkage(
                f"Applied Ledoit-Wolf shrinkage with intensity {intensity:.4f}"
            )
            return shrunk_cov, shrinkage_info
        elif self.shrinkage_method == "oas":
            shrunk_cov, intensity = oas_shrinkage(cov)
            shrinkage_info["intensity"] = intensity
            self._log_shrinkage(
                f"Applied OAS shrinkage with intensity {intensity:.4f}"
            )
            return shrunk_cov, shrinkage_info
        else:
            raise ValueError(f"Unknown shrinkage method: {self.shrinkage_method}")

    def _safe_mode_weights(self, cov: pd.DataFrame) -> pd.Series:
        """Generate weights using safe mode method."""
        if self.safe_mode == "hrp":
            from .hierarchical_risk_parity import HierarchicalRiskParity

            engine = HierarchicalRiskParity()
            return engine.weight(cov)
        elif self.safe_mode == "risk_parity":
            from .risk_parity import RiskParity

            engine = RiskParity()
            return engine.weight(cov)
        elif self.safe_mode == "diagonal_mv":
            # Use diagonal-loaded covariance for mean-variance
            cov_loaded = diagonal_loading(cov.values, self.diagonal_loading_factor)
            cov_loaded_df = pd.DataFrame(
                cov_loaded, index=cov.index, columns=cov.columns
            )
            return self._mean_variance_weights(cov_loaded_df)
        else:
            raise ValueError(f"Unknown safe mode: {self.safe_mode}")

    def _mean_variance_weights(self, cov: pd.DataFrame) -> pd.Series:
        """Compute mean-variance optimal weights (minimum variance portfolio)."""
        try:
            # Minimum variance portfolio: w = (Σ^-1 * 1) / (1' * Σ^-1 * 1)
            cov_inv = np.linalg.inv(cov.values)
            ones = np.ones(len(cov))
            numerator = cov_inv @ ones
            denominator = ones @ cov_inv @ ones

            if np.abs(denominator) < 1e-12:
                raise np.linalg.LinAlgError("Denominator too small")

            weights = numerator / denominator

            # Apply weight constraints
            weights = np.clip(weights, self.min_weight, self.max_weight)
            weights = weights / np.sum(weights)  # Renormalize

            return pd.Series(weights, index=cov.index)

        except np.linalg.LinAlgError as e:
            logger.warning(f"Matrix inversion failed in mean-variance: {e}")
            # Fall back to equal weights
            n = len(cov)
            return pd.Series(np.ones(n) / n, index=cov.index)

    def weight(self, cov: pd.DataFrame) -> pd.Series:
        """Compute robust portfolio weights with shrinkage and safe mode fallback."""
        if cov.empty:
            return pd.Series(dtype=float)

        if not cov.index.equals(cov.columns):
            raise ValueError("Covariance matrix must be square with matching labels")

        self.last_run_info = None
        # Apply shrinkage
        shrunk_cov_array, shrinkage_info = self._apply_shrinkage(cov.values)
        shrunk_cov = pd.DataFrame(
            shrunk_cov_array, index=cov.index, columns=cov.columns
        )

        # Check condition number
        condition_num = self._check_condition_number(shrunk_cov_array)
        self._log_condition(condition_num)

        run_info: Dict[str, Any] = {
            "engine": "robust_mean_variance",
            "condition_number": float(condition_num),
            "condition_threshold": float(self.condition_threshold),
            "safe_mode": self.safe_mode,
            "shrinkage": shrinkage_info,
        }

        if condition_num > self.condition_threshold:
            message = (
                "Ill-conditioned covariance matrix (condition number: "
                f"{condition_num:.2e} > threshold: {self.condition_threshold:.2e}). "
                f"Switching to safe mode: {self.safe_mode}"
            )
            if self.log_method_switches:
                logger.info(message)
            else:
                logger.warning(message)
            run_info.update(
                {
                    "selected_method": "safe_mode",
                    "method": str(self.safe_mode),
                    "reason": "condition_number_exceeded",
                }
            )
            weights = self._safe_mode_weights(cov)
            run_info["weights_index"] = list(weights.index)
            self.last_run_info = run_info
            return weights

        # Use normal mean-variance optimization
        if self.log_method_switches:
            logger.info(
                "Using mean-variance optimization with %s shrinkage (condition number "
                "%.2e ≤ threshold %.2e)",
                self.shrinkage_method,
                condition_num,
                self.condition_threshold,
            )
        else:
            logger.debug(
                "Using mean-variance optimization with %s shrinkage", self.shrinkage_method
            )
        run_info.update(
            {
                "selected_method": "mean_variance",
                "method": "mean_variance",
                "reason": "condition_number_within_threshold",
            }
        )
        weights = self._mean_variance_weights(shrunk_cov)
        run_info["weights_index"] = list(weights.index)
        self.last_run_info = run_info
        return weights


@weight_engine_registry.register("robust_risk_parity")
class RobustRiskParity(WeightEngine):
    """Risk parity weighting with robustness checks."""

    def __init__(
        self,
        *,
        condition_threshold: float = 1e12,
        diagonal_loading_factor: float = 1e-6,
        log_condition_numbers: bool = False,
        log_method_switches: bool = False,
    ) -> None:
        """Initialize robust risk parity.

        Args:
            condition_threshold: Maximum allowed condition number
            diagonal_loading_factor: Factor for diagonal loading when needed
            log_condition_numbers: Promote condition logs to INFO level.
            log_method_switches: Emit INFO logs when applying diagonal loading.
        """
        self.condition_threshold = float(condition_threshold)
        self.diagonal_loading_factor = float(diagonal_loading_factor)
        self.log_condition_numbers = bool(log_condition_numbers)
        self.log_method_switches = bool(log_method_switches)
        self.last_run_info: Optional[Dict[str, Any]] = None

    def weight(self, cov: pd.DataFrame) -> pd.Series:
        """Compute risk parity weights with robustness checks."""
        if cov.empty:
            return pd.Series(dtype=float)

        if not cov.index.equals(cov.columns):
            raise ValueError("Covariance matrix must be square with matching labels")

        # Check for problematic values
        cov_array = cov.values
        self.last_run_info = {
            "engine": "robust_risk_parity",
            "condition_threshold": float(self.condition_threshold),
            "diagonal_loading_factor": float(self.diagonal_loading_factor),
        }

        info = self.last_run_info

        # Check for non-positive diagonal elements
        diag_vals = np.diag(cov_array)
        if np.any(diag_vals <= 0):
            logger.warning(
                "Non-positive diagonal elements detected. Applying diagonal loading."
            )
            cov_array = diagonal_loading(cov_array, self.diagonal_loading_factor)
            info.update(
                {
                    "selected_method": "diagonal_loading",
                    "reason": "non_positive_diagonal",
                }
            )

        # Check condition number
        condition_num = np.linalg.cond(cov_array)
        level = logging.INFO if self.log_condition_numbers else logging.DEBUG
        logger.log(level, f"Risk parity covariance condition number: {condition_num:.2e}")

        if condition_num > self.condition_threshold:
            message = (
                "Ill-conditioned covariance matrix (condition number: "
                f"{condition_num:.2e} > threshold: {self.condition_threshold:.2e}). "
                "Applying diagonal loading."
            )
            if self.log_method_switches:
                logger.info(message)
            else:
                logger.warning(message)
            cov_array = diagonal_loading(cov_array, self.diagonal_loading_factor)
            info.update(
                {
                    "selected_method": "diagonal_loading",
                    "reason": "condition_number_exceeded",
                }
            )
        else:
            info.setdefault("selected_method", "risk_parity")
            info.setdefault("reason", "condition_number_within_threshold")

        info["condition_number"] = float(condition_num)

        # Compute inverse volatility weights
        std_devs = np.sqrt(np.diag(cov_array))

        # Handle zero or very small standard deviations
        min_std = np.max(std_devs) * 1e-8
        std_devs = np.maximum(std_devs, min_std)

        inv_vol = 1.0 / std_devs
        weights = inv_vol / np.sum(inv_vol)

        logger.debug("Successfully computed robust risk parity weights")
        info.setdefault("selected_method", "risk_parity")
        info.setdefault("reason", "condition_number_within_threshold")
        info["weights_index"] = list(cov.index)
        self.last_run_info = info
        return pd.Series(weights, index=cov.index)
