"""Interfaces for Monte Carlo price path generation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .utils import log_returns_to_prices


@dataclass(slots=True)
class ReturnPath:
    """Container for simulated log returns with availability metadata."""

    log_returns: pd.DataFrame
    availability: pd.DataFrame

    def to_prices(self, start_prices: pd.Series) -> pd.DataFrame:
        """Reconstruct prices from log returns using the availability mask."""

        return log_returns_to_prices(
            self.log_returns,
            start_prices,
            price_availability=self.availability,
            start_at_first_row=False,
        )


@runtime_checkable
class PricePathModel(Protocol):
    """Interface for models that simulate log-return paths."""

    def fit(
        self,
        prices: pd.DataFrame,
        *,
        availability: pd.DataFrame | None = None,
    ) -> "PricePathModel":
        """Fit the model to historical prices."""

    def simulate(
        self,
        n_steps: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> ReturnPath:
        """Simulate a log-return path with an availability mask."""


__all__ = ["PricePathModel", "ReturnPath"]
