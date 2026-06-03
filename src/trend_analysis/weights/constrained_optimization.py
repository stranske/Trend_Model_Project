from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..plugins import WeightEngine, weight_engine_registry


@weight_engine_registry.register("convex_constrained")
class ConstrainedConvexWeighting(WeightEngine):
    """Minimum-variance weighting with simple convex constraints."""

    def __init__(
        self,
        *,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        groups: Mapping[str, Sequence[str]] | None = None,
        group_bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.groups = dict(groups or {})
        self.group_bounds = dict(group_bounds or {})

        if self.min_weight > self.max_weight:
            raise ValueError("min_weight must be <= max_weight")

    def weight(self, cov: pd.DataFrame) -> pd.Series:
        if cov.empty:
            return pd.Series(dtype=float)
        if not cov.index.equals(cov.columns):
            raise ValueError("Covariance matrix must be square with matching labels")

        labels = list(cov.index)
        label_positions = {label: idx for idx, label in enumerate(labels)}
        matrix = cov.to_numpy(dtype=float)
        matrix = (matrix + matrix.T) / 2.0
        n_assets = len(labels)

        constraints = [{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}]
        for group_name, (lower, upper) in self.group_bounds.items():
            try:
                group_members = self.groups[group_name]
            except KeyError as exc:
                raise ValueError(f"group_bounds references unknown group {group_name!r}") from exc

            missing = [asset for asset in group_members if asset not in label_positions]
            if missing:
                raise ValueError(
                    f"group {group_name!r} references assets missing from covariance: {missing}"
                )

            group_idx = np.array([label_positions[asset] for asset in group_members], dtype=int)
            lower_bound = float(lower)
            upper_bound = float(upper)
            constraints.extend(
                [
                    {
                        "type": "ineq",
                        "fun": lambda weights, idx=group_idx, lb=lower_bound: float(
                            np.sum(weights[idx]) - lb
                        ),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda weights, idx=group_idx, ub=upper_bound: float(
                            ub - np.sum(weights[idx])
                        ),
                    },
                ]
            )

        bounds = [(self.min_weight, self.max_weight)] * n_assets
        initial = np.repeat(1.0 / n_assets, n_assets)

        result = minimize(
            lambda weights: float(weights @ matrix @ weights),
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            raise ValueError(f"Constrained optimization failed: {result.message}")

        weights = pd.Series(result.x, index=cov.index, dtype=float)
        weights[weights.abs() < 1e-12] = 0.0
        return weights / weights.sum()
