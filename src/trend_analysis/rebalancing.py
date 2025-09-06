"""Backward-compatible rebalancing strategy API.

This module used to host the concrete rebalancing strategy implementations.
The implementations now live in :mod:`trend_analysis.rebalancing.strategies`.
To avoid confusing multiple class definitions (which previously caused
``isinstance`` checks to fail), this shim simply re-exports the strategy
classes and helpers from the canonical ``strategies`` module.
"""

from __future__ import annotations

from typing import Dict

from . import strategies as _strategies
from .plugins import rebalancer_registry

# Re-export public classes and helpers from the strategies module
RebalancingStrategy = _strategies.RebalancingStrategy
TurnoverCapStrategy = _strategies.TurnoverCapStrategy
PeriodicRebalanceStrategy = _strategies.PeriodicRebalanceStrategy
DriftBandStrategy = _strategies.DriftBandStrategy
VolTargetRebalanceStrategy = _strategies.VolTargetRebalanceStrategy
DrawdownGuardStrategy = _strategies.DrawdownGuardStrategy
create_rebalancing_strategy = _strategies.create_rebalancing_strategy
apply_rebalancing_strategies = _strategies.apply_rebalancing_strategies
TURNOVER_EPSILON = _strategies.TURNOVER_EPSILON


def get_rebalancing_strategies() -> Dict[str, type]:
    """Return mapping of registered strategy names to classes."""

    return {name: rebalancer_registry.get(name) for name in rebalancer_registry.available()}


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
    "TURNOVER_EPSILON",
    "rebalancer_registry",
    "get_rebalancing_strategies",
    "REBALANCING_STRATEGIES",
]

