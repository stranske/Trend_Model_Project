"""Schema definitions for Monte Carlo scenario configurations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MonteCarloSettings:
    """Configuration settings for Monte Carlo path generation."""

    mode: str
    n_paths: int
    horizon_years: float
    frequency: str
    seed: int | None
    jobs: int | None
