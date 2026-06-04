from typing import Dict

from .strategies import (
    CashPolicy,
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


def get_rebalancing_strategies() -> Dict[str, type]:
    """Return a snapshot mapping of registered strategy names to classes."""

    # ``PluginRegistry`` exposes its internal mapping via the private
    # ``_plugins`` attribute. Accessing it directly provides a snapshot of the
    # current registrations without requiring additional API surface on the
    # registry itself.
    return {name: cls for name, cls in rebalancer_registry._plugins.items()}


__all__ = (
    "CashPolicy",
    "RebalancingStrategy",
    "TurnoverCapStrategy",
    "PeriodicRebalanceStrategy",
    "DriftBandStrategy",
    "VolTargetRebalanceStrategy",
    "DrawdownGuardStrategy",
    "create_rebalancing_strategy",
    "apply_rebalancing_strategies",
    "get_rebalancing_strategies",
    "rebalancer_registry",
)
