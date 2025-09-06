"""Light‑weight plugin system used across the project.

Historically the package exposed registries for selector and rebalancing
strategies only.  Tests such as ``test_weight_engines.py`` expect a similar
interface for portfolio weight engines (risk parity, ERC, etc.) which was
missing and resulted in ``ImportError`` during collection.  This module now
provides a generic :class:`PluginRegistry` plus concrete base classes and
factory helpers for selectors, rebalancers and weight engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Type, TypeVar

import pandas as pd

T = TypeVar("T", bound="Plugin")


class Plugin(ABC):
    """Base class for all plugins."""


class PluginRegistry(Generic[T]):
    """Simple in-memory registry mapping names to plugin classes."""

    def __init__(self) -> None:  # pragma: no cover - trivial container
        self._plugins: Dict[str, Type[T]] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Register ``cls`` under ``name`` using a decorator."""

        def decorator(cls: Type[T]) -> Type[T]:
            self._plugins[name] = cls
            return cls

        return decorator

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Instantiate the plugin registered under ``name``."""
        try:
            cls = self._plugins[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Unknown plugin: {name}. Available: {list(self._plugins.keys())}"
            ) from exc
        return cls(*args, **kwargs)

    def available(self) -> List[str]:
        """Return a list of registered plugin names."""
        return list(self._plugins.keys())


# --- Selector plugins ---------------------------------------------------------------
class Selector(Plugin):
    """Base class for selector plugins."""

    @abstractmethod
    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Select assets from ``score_frame`` returning (selected, log)."""


class Rebalancer(Plugin):
    """Base class for rebalancing strategy plugins."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def apply(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        **kwargs: Any,
    ) -> tuple[pd.Series, float]:
        """Return new weights and total cost for the rebalance."""


class WeightEngine(Plugin):
    """Base class for risk-based weight engine plugins."""

    @abstractmethod
    def weight(self, cov: pd.DataFrame) -> pd.Series:
        """Return portfolio weights given a covariance matrix."""


selector_registry: PluginRegistry[Selector] = PluginRegistry()
rebalancer_registry: PluginRegistry[Rebalancer] = PluginRegistry()
weight_engine_registry: PluginRegistry[WeightEngine] = PluginRegistry()


def create_selector(name: str, **params: Any) -> Selector:
    """Instantiate a selector plugin by ``name``."""
    return selector_registry.create(name, **params)


def create_rebalancer(name: str, params: Dict[str, Any] | None = None) -> Rebalancer:
    """Instantiate a rebalancer plugin by ``name``."""
    return rebalancer_registry.create(name, params or {})


def create_weight_engine(name: str, **params: Any) -> WeightEngine:
    """Instantiate a weight engine plugin by ``name``."""
    return weight_engine_registry.create(name, **params)


# Import built-in weight engine modules so that they register with the plugin registry
from ..weights import \
    equal_risk_contribution as _equal_risk_contribution  # noqa: F401, E402
from ..weights import hierarchical_risk_parity as _hierarchical_risk_parity  # noqa: F401, E402
from ..weights import risk_parity as _risk_parity  # noqa: F401, E402

__all__ = [
    "Plugin",
    "PluginRegistry",
    "Selector",
    "Rebalancer",
    "WeightEngine",
    "selector_registry",
    "rebalancer_registry",
    "weight_engine_registry",
    "create_selector",
    "create_rebalancer",
    "create_weight_engine",
    "_equal_risk_contribution",
    "_hierarchical_risk_parity",  # ensures HRP is exported
    "_risk_parity",
]
