"""Robust weighting methods with shrinkage and condition number safeguards."""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..plugins import WeightEngine, weight_engine_registry

logger = logging.getLogger(__name__)

NDArrayFloat = npt.NDArray[np.float64]


def ledoit_wolf_shrinkage(
    cov: NDArrayFloat, n_samples: int | None = None
) -> tuple[NDArrayFloat, float]:
    """Apply Ledoit-Wolf shrinkage to covariance matrix.

    Args:
        cov: Covariance matrix to shrink
        n_samples: Number of observations used to estimate covariance.
                  If None, uses a heuristic based on matrix properties.

    Returns:
        tuple: (shrunk_cov, shrinkage_intensity)
    """
    sample_cov = np.asarray(cov, dtype=float)
    p = sample_cov.shape[0]  # Number of assets/variables

    # Target: diagonal matrix with average variance
    mu = float(np.trace(sample_cov)) / float(p)
    target = mu * np.eye(p, dtype=float)

    # Estimate sample size if not provided
    if n_samples is None:
        # Heuristic: estimate based on condition number and matrix properties
        # This is an approximation when true sample size is unknown
        n_samples = max(p + 1, int(p * 2))  # Conservative estimate

    # Shrinkage intensity (simplified Ledoit-Wolf approach)
    # When we don't have access to raw data, use matrix-based approximation
    trace_diff = float(np.trace((sample_cov - target) @ (sample_cov - target)))
    trace_sample = float(np.trace(sample_cov @ sample_cov))

    # Avoid division by zero
    if trace_sample == 0:
        intensity = 1.0
    else:
        # Use estimated sample size rather than matrix dimension
        intensity = min(1.0, trace_diff / (n_samples * trace_sample))

    # Apply shrinkage
    shrunk_cov = (1.0 - intensity) * sample_cov + intensity * target

    return shrunk_cov.astype(float, copy=False), float(intensity)


def oas_shrinkage(
    cov: NDArrayFloat, n_samples: int | None = None
) -> tuple[NDArrayFloat, float]:
    """Apply Oracle Approximating Shrinkage (OAS) to covariance matrix.

    Args:
        cov: Covariance matrix to shrink
        n_samples: Number of observations used to estimate covariance.
                  If None, uses a heuristic based on matrix properties.

    Returns:
        tuple: (shrunk_cov, shrinkage_intensity)
    """
    sample_cov = np.asarray(cov, dtype=float)
    p = sample_cov.shape[0]  # Number of assets/variables

    # Target: diagonal matrix with average variance
    mu = float(np.trace(sample_cov)) / float(p)
    target = mu * np.eye(p, dtype=float)

    # Estimate sample size if not provided
    if n_samples is None:
        # Heuristic: estimate based on matrix properties
        # This is an approximation when true sample size is unknown
        n_samples = max(p + 1, int(p * 2))  # Conservative estimate

    # OAS shrinkage intensity
    trace_sample = float(np.trace(sample_cov @ sample_cov))
    trace_target = float(np.trace(target @ target))

    if trace_sample == 0:
        intensity = 1.0
    else:
        # Simplified OAS formula using estimated sample size
        intensity = min(1.0, trace_target / (n_samples * trace_sample))

    # Apply shrinkage
    shrunk_cov = (1.0 - intensity) * sample_cov + intensity * target

    return shrunk_cov.astype(float, copy=False), float(intensity)


def diagonal_loading(cov: NDArrayFloat, loading_factor: float = 1e-6) -> NDArrayFloat:
    """Apply diagonal loading to improve conditioning.

    Args:
        cov: Covariance matrix
        loading_factor: Factor to add to diagonal elements

    Returns:
        Regularized covariance matrix
    """
    base = np.asarray(cov, dtype=float)
    if base.size == 0:
        return base
    scale = float(loading_factor) * float(np.trace(base)) / float(base.shape[0])
    return base + scale * np.eye(base.shape[0], dtype=float)


@weight_engine_registry.register("robust_mv")
@weight_engine_registry.register("robust_mean_variance")
class RobustMeanVariance(WeightEngine):
    """Robust mean-variance optimization with shrinkage and safe mode
    fallback."""

    def __init__(
        self,
        *,
        shrinkage_method: Literal["none", "ledoit_wolf", "oas"] = "ledoit_wolf",
        condition_threshold: float = 1e12,
        safe_mode: Literal["hrp", "risk_parity", "diagonal_mv"] = "hrp",
        diagonal_loading_factor: float = 1e-6,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        log_condition_numbers: bool = True,
        log_method_switches: bool = True,
        log_shrinkage_intensity: bool = True,
    ) -> None:
        """Initialize robust mean-variance optimizer.

        Args:
            shrinkage_method: Type of shrinkage to apply
            condition_threshold: Maximum allowed condition number
            safe_mode: Fallback method when matrix is ill-conditioned
            diagonal_loading_factor: Factor for diagonal loading regularization
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint
        """
        self.shrinkage_method = shrinkage_method
        self.condition_threshold = float(condition_threshold)
        self.safe_mode = safe_mode
        self.diagonal_loading_factor = float(diagonal_loading_factor)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.log_condition_numbers = bool(log_condition_numbers)
        self.log_method_switches = bool(log_method_switches)
        self.log_shrinkage_intensity = bool(log_shrinkage_intensity)
        self.diagnostics: Dict[str, Any] = {}

    def _check_condition_number(self, cov: NDArrayFloat) -> float:
        """Compute condition number of covariance matrix."""
        try:
            condition_num = float(np.linalg.cond(cov))
            if not np.isfinite(condition_num):
                return np.inf
            return condition_num
        except np.linalg.LinAlgError:
            return np.inf

    def _apply_shrinkage(
        self, cov: NDArrayFloat
    ) -> tuple[NDArrayFloat, Dict[str, Any]]:
        """Apply shrinkage method to covariance matrix."""
        shrinkage_info = {"method": self.shrinkage_method, "intensity": 0.0}

        if self.shrinkage_method == "none":
            return cov.copy(), shrinkage_info
        elif self.shrinkage_method == "ledoit_wolf":
            shrunk_cov, intensity = ledoit_wolf_shrinkage(cov)
            shrinkage_info["intensity"] = intensity
            if self.log_shrinkage_intensity:
                logger.debug(
                    f"Applied Ledoit-Wolf shrinkage with intensity {intensity:.4f}"
                )
            return shrunk_cov, shrinkage_info
        elif self.shrinkage_method == "oas":
            shrunk_cov, intensity = oas_shrinkage(cov)
            shrinkage_info["intensity"] = intensity
            if self.log_shrinkage_intensity:
                logger.debug(f"Applied OAS shrinkage with intensity {intensity:.4f}")
            return shrunk_cov, shrinkage_info
        else:
            raise ValueError(f"Unknown shrinkage method: {self.shrinkage_method}")

    def _safe_mode_weights(self, cov: pd.DataFrame) -> pd.Series:
        """Generate weights using safe mode method."""
        if self.safe_mode == "hrp":
            from .hierarchical_risk_parity import HierarchicalRiskParity

            hrp_engine = HierarchicalRiskParity()
            return hrp_engine.weight(cov)
        elif self.safe_mode == "risk_parity":
            from .risk_parity import RiskParity

            rp_engine = RiskParity()
            return rp_engine.weight(cov)
        elif self.safe_mode == "diagonal_mv":
            # Use diagonal-loaded covariance for mean-variance
            cov_loaded = diagonal_loading(
                np.asarray(cov.values, dtype=float), self.diagonal_loading_factor
            )
            cov_loaded_df = pd.DataFrame(
                cov_loaded, index=cov.index, columns=cov.columns
            )
            return self._mean_variance_weights(cov_loaded_df)
        else:
            raise ValueError(f"Unknown safe mode: {self.safe_mode}")

    def _mean_variance_weights(self, cov: pd.DataFrame) -> pd.Series:
        """Compute mean-variance optimal weights (minimum variance
        portfolio)."""
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
        """Compute robust portfolio weights with shrinkage and safe mode
        fallback."""
        if cov.empty:
            return pd.Series(dtype=float)

        if not cov.index.equals(cov.columns):
            raise ValueError("Covariance matrix must be square with matching labels")

        # Apply shrinkage
        cov_array = np.asarray(cov.values, dtype=float)
        shrunk_cov_array, shrinkage_info = self._apply_shrinkage(cov_array)
        shrunk_cov = pd.DataFrame(
            shrunk_cov_array, index=cov.index, columns=cov.columns
        )

        # Check condition numbers on both the raw and post-shrinkage matrices.
        raw_condition_num = self._check_condition_number(cov_array)
        shrunk_condition_num = self._check_condition_number(shrunk_cov_array)
        if self.shrinkage_method != "none":
            if raw_condition_num >= shrunk_condition_num:
                condition_num = raw_condition_num
                condition_source = "raw_cov"
            else:
                condition_num = shrunk_condition_num
                condition_source = "shrunk_cov"
        else:
            condition_num = raw_condition_num
            condition_source = "raw_cov"

        if self.log_condition_numbers:
            logger.debug(
                f"Raw covariance matrix condition number: {raw_condition_num:.2e}"
            )
            if self.shrinkage_method != "none":
                logger.debug(
                    "Shrinkage-adjusted covariance matrix condition number: "
                    f"{shrunk_condition_num:.2e}"
                )

        used_safe_mode = condition_num > self.condition_threshold
        self.diagnostics = {
            "condition_number": condition_num,
            "raw_condition_number": raw_condition_num,
            "shrunk_condition_number": shrunk_condition_num,
            "condition_source": condition_source,
            "condition_threshold": self.condition_threshold,
            "safe_mode": self.safe_mode,
            "used_safe_mode": used_safe_mode,
            "shrinkage": shrinkage_info,
        }

        if used_safe_mode:
            self.diagnostics["fallback_reason"] = "condition_threshold_exceeded"
            if self.log_method_switches:
                logger.warning(
                    "Ill-conditioned covariance matrix (%s condition number: %.2e; "
                    "raw: %.2e; shrunk: %.2e > threshold: %.2e). Switching to safe "
                    "mode: %s",
                    condition_source,
                    condition_num,
                    raw_condition_num,
                    shrunk_condition_num,
                    self.condition_threshold,
                    self.safe_mode,
                )
            return self._safe_mode_weights(shrunk_cov)

        # Use normal mean-variance optimization
        logger.debug(
            f"Using mean-variance optimization with {self.shrinkage_method} shrinkage"
        )
        return self._mean_variance_weights(shrunk_cov)


@weight_engine_registry.register("robust_risk_parity")
class RobustRiskParity(WeightEngine):
    """Risk parity weighting with robustness checks."""

    def __init__(
        self,
        *,
        condition_threshold: float = 1e12,
        diagonal_loading_factor: float = 1e-6,
    ) -> None:
        """Initialize robust risk parity.

        Args:
            condition_threshold: Maximum allowed condition number
            diagonal_loading_factor: Factor for diagonal loading when needed
        """
        self.condition_threshold = float(condition_threshold)
        self.diagonal_loading_factor = float(diagonal_loading_factor)
        self.diagnostics: Dict[str, Any] = {}

    def weight(self, cov: pd.DataFrame) -> pd.Series:
        """Compute risk parity weights with robustness checks."""
        if cov.empty:
            return pd.Series(dtype=float)

        if not cov.index.equals(cov.columns):
            raise ValueError("Covariance matrix must be square with matching labels")

        # Check for problematic values
        cov_array = cov.values.astype(float, copy=True)

        # Check for non-positive diagonal elements
        diag_vals = np.diag(cov_array)
        if np.any(diag_vals <= 0):
            logger.warning(
                "Non-positive diagonal elements detected. Applying diagonal loading."
            )
            cov_array = diagonal_loading(cov_array, self.diagonal_loading_factor)
            diag_vals = np.diag(cov_array)

        # Check condition number
        condition_num = np.linalg.cond(cov_array)

        if condition_num > self.condition_threshold:
            logger.warning(
                f"Ill-conditioned covariance matrix (condition number: {condition_num:.2e}). "
                f"Applying diagonal loading."
            )
            cov_array = diagonal_loading(cov_array, self.diagonal_loading_factor)
            diag_vals = np.diag(cov_array)

        # Compute inverse volatility weights
        diag_vals = np.where(diag_vals > 0, diag_vals, 0.0)
        std_devs = np.sqrt(diag_vals)
        std_devs = np.nan_to_num(std_devs, nan=0.0, posinf=0.0, neginf=0.0)

        max_std = float(np.max(std_devs)) if std_devs.size else 0.0
        if max_std <= 0.0:
            # Fallback to equal weights when the covariance matrix collapses.
            logger.warning("Falling back to equal weights due to zero variance inputs")
            return pd.Series(
                np.full(len(cov.index), 1.0 / len(cov.index)), index=cov.index
            )

        # Handle zero or very small standard deviations
        min_std = max_std * 1e-8 if max_std > 0.0 else np.finfo(float).eps
        std_devs = np.where(std_devs < min_std, min_std, std_devs)

        inv_vol = np.reciprocal(std_devs)
        total = float(np.sum(inv_vol))
        if not np.isfinite(total) or total <= 0.0:
            logger.warning(
                "Falling back to equal weights due to invalid inverse volatility sum"
            )
            return pd.Series(
                np.full(len(cov.index), 1.0 / len(cov.index)), index=cov.index
            )

        weights = inv_vol / total

        self.diagnostics = {
            "condition_number": condition_num,
            "condition_threshold": self.condition_threshold,
            "used_diagonal_loading": condition_num > self.condition_threshold,
            "diagonal_loading_factor": self.diagonal_loading_factor,
        }

        logger.debug("Successfully computed robust risk parity weights")
        return pd.Series(weights, index=cov.index)
