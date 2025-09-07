"""Backward compatibility shim for rebalancing strategies.

The canonical implementations live in :mod:`trend_analysis.rebalancing.strategies`.
This module re-exports those classes so legacy imports of
``trend_analysis.rebalancing`` continue to work without registering
duplicate plugins.
"""

from __future__ import annotations

from typing import Dict

from .strategies import strategies as _strategies
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

    # PluginRegistry exposes registered names via ``available`` but does not
    # provide a public accessor for the classes themselves. The registry keeps
    # them in the private ``_plugins`` dict, which we copy here for introspection
    # without mutating the original mapping.
    # NOTE: Accessing the private attribute ``_plugins`` violates encapsulation principles
    # (see CodeQL rule: "Accessing private attributes (`_plugins`) violates encapsulation principles").
    # This is a necessary workaround because the registry design cannot be modified here.
    return rebalancer_registry._plugins.copy()


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
]
