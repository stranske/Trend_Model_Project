"""Robust weighting methods with shrinkage and condition number safeguards."""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

import numpy as np
import pandas as pd

from ..plugins import WeightEngine, weight_engine_registry

logger = logging.getLogger(__name__)


def ledoit_wolf_shrinkage(cov: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply Ledoit-Wolf shrinkage to covariance matrix.
    
    Returns:
        tuple: (shrunk_cov, shrinkage_intensity)
    """
    n, p = cov.shape[0], cov.shape[1]
    
    # Sample covariance
    sample_cov = cov.copy()
    
    # Target: diagonal matrix with average variance
    mu = np.trace(sample_cov) / p
    target = mu * np.eye(p)
    
    # Shrinkage intensity (simplified Ledoit-Wolf formula)
    # In practice, this would use more sophisticated estimation
    trace_diff = np.trace((sample_cov - target) @ (sample_cov - target))
    trace_sample = np.trace(sample_cov @ sample_cov)
    
    # Avoid division by zero
    if trace_sample == 0:
        intensity = 1.0
    else:
        intensity = min(1.0, trace_diff / (n * trace_sample))
    
    # Apply shrinkage
    shrunk_cov = (1 - intensity) * sample_cov + intensity * target
    
    return shrunk_cov, intensity


def oas_shrinkage(cov: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply Oracle Approximating Shrinkage (OAS) to covariance matrix.
    
    Returns:
        tuple: (shrunk_cov, shrinkage_intensity)
    """
    n, p = cov.shape[0], cov.shape[1]
    
    # Sample covariance
    sample_cov = cov.copy()
    
    # Target: diagonal matrix with average variance
    mu = np.trace(sample_cov) / p
    target = mu * np.eye(p)
    
    # OAS shrinkage intensity
    trace_sample = np.trace(sample_cov @ sample_cov)
    trace_target = np.trace(target @ target)
    
    if trace_sample == 0:
        intensity = 1.0
    else:
        # Simplified OAS formula
        intensity = min(1.0, (trace_target) / (n * trace_sample))
    
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
            logger.debug(f"Applied Ledoit-Wolf shrinkage with intensity {intensity:.4f}")
            return shrunk_cov, shrinkage_info
        elif self.shrinkage_method == "oas":
            shrunk_cov, intensity = oas_shrinkage(cov)
            shrinkage_info["intensity"] = intensity
            logger.debug(f"Applied OAS shrinkage with intensity {intensity:.4f}")
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
            cov_loaded_df = pd.DataFrame(cov_loaded, index=cov.index, columns=cov.columns)
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
        
        # Apply shrinkage
        shrunk_cov_array, shrinkage_info = self._apply_shrinkage(cov.values)
        shrunk_cov = pd.DataFrame(shrunk_cov_array, index=cov.index, columns=cov.columns)
        
        # Check condition number
        condition_num = self._check_condition_number(shrunk_cov_array)
        
        logger.debug(f"Covariance matrix condition number: {condition_num:.2e}")
        
        if condition_num > self.condition_threshold:
            logger.warning(
                f"Ill-conditioned covariance matrix (condition number: {condition_num:.2e} > "
                f"threshold: {self.condition_threshold:.2e}). Switching to safe mode: {self.safe_mode}"
            )
            return self._safe_mode_weights(cov)
        
        # Use normal mean-variance optimization
        logger.info(f"Using mean-variance optimization with {self.shrinkage_method} shrinkage")
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
    
    def weight(self, cov: pd.DataFrame) -> pd.Series:
        """Compute risk parity weights with robustness checks."""
        if cov.empty:
            return pd.Series(dtype=float)
        
        if not cov.index.equals(cov.columns):
            raise ValueError("Covariance matrix must be square with matching labels")
        
        # Check for problematic values
        cov_array = cov.values
        
        # Check for non-positive diagonal elements
        diag_vals = np.diag(cov_array)
        if np.any(diag_vals <= 0):
            logger.warning("Non-positive diagonal elements detected. Applying diagonal loading.")
            cov_array = diagonal_loading(cov_array, self.diagonal_loading_factor)
        
        # Check condition number
        condition_num = np.linalg.cond(cov_array)
        
        if condition_num > self.condition_threshold:
            logger.warning(
                f"Ill-conditioned covariance matrix (condition number: {condition_num:.2e}). "
                f"Applying diagonal loading."
            )
            cov_array = diagonal_loading(cov_array, self.diagonal_loading_factor)
        
        # Compute inverse volatility weights
        std_devs = np.sqrt(np.diag(cov_array))
        
        # Handle zero or very small standard deviations
        min_std = np.max(std_devs) * 1e-8
        std_devs = np.maximum(std_devs, min_std)
        
        inv_vol = 1.0 / std_devs
        weights = inv_vol / np.sum(inv_vol)
        
        logger.debug("Successfully computed robust risk parity weights")
        return pd.Series(weights, index=cov.index)