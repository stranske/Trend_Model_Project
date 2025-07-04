from __future__ import annotations

from collections.abc import Iterable
from typing import Type

_PLUGIN_REGISTRY: list[Type[object]] = []


def register_plugin(cls: Type[object]) -> None:
    if cls not in _PLUGIN_REGISTRY:
        _PLUGIN_REGISTRY.append(cls)


def iter_plugins() -> Iterable[Type[object]]:
    yield from _PLUGIN_REGISTRY


def discover_plugins() -> None:
    import importlib.metadata as metadata

    for ep in metadata.entry_points(group="trend_analysis.gui_plugins"):
        plugin_cls = ep.load()
        register_plugin(plugin_cls)


__all__ = ["register_plugin", "iter_plugins", "discover_plugins"]
