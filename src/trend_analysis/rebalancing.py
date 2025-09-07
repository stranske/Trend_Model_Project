"""Backward compatibility shim for rebalancing strategies.

The canonical implementations live in :mod:`trend_analysis.rebalancing.strategies`.
This module re-exports those classes so legacy imports of
``trend_analysis.rebalancing`` continue to work without registering
duplicate plugins.
"""

from __future__ import annotations

from .rebalancing.strategies import (
    DrawdownGuardStrategy,
    DriftBandStrategy,
    PeriodicRebalanceStrategy,
    RebalancingStrategy,
    TurnoverCapStrategy,
    VolTargetRebalanceStrategy,
    apply_rebalancing_strategies,
    create_rebalancing_strategy,
    rebalancer_registry,
)

__all__ = [
    "RebalancingStrategy",
    "TurnoverCapStrategy",
    "PeriodicRebalanceStrategy",
    "DriftBandStrategy",
    "VolTargetRebalanceStrategy",
    "DrawdownGuardStrategy",
    "create_rebalancing_strategy",
    "apply_rebalancing_strategies",
    "rebalancer_registry",
]
