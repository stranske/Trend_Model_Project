"""Backward compatibility shim for rebalancing strategies.

The canonical implementations live in :mod:`trend_analysis.rebalancing.strategies`.
This module re-exports those classes so legacy imports of
``trend_analysis.rebalancing`` continue to work without registering
duplicate plugins.
"""

from __future__ import annotations

from .plugins import rebalancer_registry

# Import canonical implementations from the package so this shim
# can re-export them without triggering circular imports or relying
# on a non-existent top-level ``strategies`` module.
from .rebalancing.strategies import (
    TURNOVER_EPSILON,
    DrawdownGuardStrategy,
    DriftBandStrategy,
    PeriodicRebalanceStrategy,
    RebalancingStrategy,
    TurnoverCapStrategy,
    VolTargetRebalanceStrategy,
    apply_rebalancing_strategies,
    create_rebalancing_strategy,
)


def get_rebalancing_strategies() -> dict[str, type]:
    """Return mapping of registered strategy names to classes."""

    # ``PluginRegistry`` exposes its internal mapping via the private
    # ``_plugins`` attribute. Accessing it directly lets this shim provide a
    # snapshot of the current registrations without requiring additional API
    # surface on the registry itself.
    return {name: cls for name, cls in rebalancer_registry._plugins.items()}


# Snapshot of available strategies for external introspection
REBALANCING_STRATEGIES = get_rebalancing_strategies()


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
    "TURNOVER_EPSILON",
]
