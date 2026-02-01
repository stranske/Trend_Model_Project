"""Deterministic seeding utilities for Monte Carlo simulations."""

from __future__ import annotations

import numpy as np


class SeedManager:
    """Create deterministic RNG streams from a scenario seed.

    Seeding scheme:
    - Path seed: hash((master_seed, path_id)) & 0xFFFFFFFF
    - Strategy seed: hash((master_seed, path_id, strategy_name)) & 0xFFFFFFFF

    The deterministic seed is then used to create an independent NumPy RNG via
    ``np.random.default_rng``.
    """

    def __init__(self, master_seed: int) -> None:
        if master_seed is None:
            raise ValueError("master_seed is required for deterministic seeding")
        self.master_seed = self._coerce_seed(master_seed, label="master_seed")

    def get_path_seed(self, path_id: int) -> int:
        """Return a deterministic seed for a given path."""
        path_id = self._coerce_seed(path_id, label="path_id")
        return hash((self.master_seed, path_id)) & 0xFFFFFFFF

    def get_strategy_seed(self, path_id: int, strategy_name: str) -> int:
        """Return a deterministic seed for a strategy-specific RNG stream."""
        path_id = self._coerce_seed(path_id, label="path_id")
        if not isinstance(strategy_name, str) or not strategy_name.strip():
            raise ValueError("strategy_name must be a non-empty string")
        normalized = strategy_name.strip()
        return hash((self.master_seed, path_id, normalized)) & 0xFFFFFFFF

    def get_path_rng(self, path_id: int) -> np.random.Generator:
        """Get deterministic RNG for a path."""
        return np.random.default_rng(self.get_path_seed(path_id))

    def get_strategy_rng(self, path_id: int, strategy_name: str) -> np.random.Generator:
        """Get deterministic RNG for strategy-specific randomness (e.g., costs)."""
        return np.random.default_rng(self.get_strategy_seed(path_id, strategy_name))

    @staticmethod
    def _coerce_seed(value: int, *, label: str) -> int:
        try:
            seed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer") from exc
        if seed < 0:
            raise ValueError(f"{label} must be >= 0")
        return seed
