"""Signal-only view of the canonical YAML-backed preset registry.

``trend_analysis.presets`` owns full preset payloads.  This module preserves
the small signal-only type used by older CLI call sites without maintaining a
second authoritative name map.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

from .presets import _preset_registry, get_trend_preset, list_trend_presets
from .signals import TrendSpec


@dataclass(frozen=True, slots=True)
class TrendSpecPreset:
    """Container describing a preset TrendSpec configuration."""

    name: str
    description: str
    spec: TrendSpec

    def as_signal_config(self) -> Dict[str, object]:
        """Return a mapping suitable for config ``signals`` sections."""

        payload: Dict[str, object] = {
            "kind": self.spec.kind,
            "window": self.spec.window,
            "lag": self.spec.lag,
            "vol_adjust": self.spec.vol_adjust,
            "zscore": self.spec.zscore,
        }
        if self.spec.min_periods is not None:
            payload["min_periods"] = self.spec.min_periods
        if self.spec.vol_target is not None:
            payload["vol_target"] = self.spec.vol_target
        return payload

    def form_defaults(self) -> Dict[str, object]:
        """Return defaults for interactive forms (min periods/vol target
        optional)."""

        defaults: Dict[str, object] = {
            "window": self.spec.window,
            "min_periods": self.spec.min_periods or 0,
            "lag": self.spec.lag,
            "vol_adjust": self.spec.vol_adjust,
            "vol_target": self.spec.vol_target or 0.0,
            "zscore": self.spec.zscore,
        }
        return defaults


_DEFAULT_PRESET_NAME = "Balanced"
_SIGNAL_PRESET_SLUGS = ("aggressive", "balanced", "conservative")


def _ordered_presets_items() -> tuple[tuple[str, TrendSpecPreset], ...]:
    """Return the canonical signal-compatible presets by stable slug.

    The full registry also includes full-config-only choices such as the cash
    constrained preset.  This tuple is a compatibility surface selection, not
    a second registry: each returned payload is still derived from the single
    YAML-backed owner.
    """

    available = {preset.slug for preset in list_trend_presets()}
    registry_identity = id(_preset_registry())
    return tuple(
        (slug, _signal_view(slug, registry_identity))
        for slug in _SIGNAL_PRESET_SLUGS
        if slug in available
    )


def _ordered_presets() -> tuple[TrendSpecPreset, ...]:
    """Return canonical signal presets in the same order as their keys."""

    return tuple(preset for _, preset in _ordered_presets_items())


@lru_cache(maxsize=None)
def _signal_view(slug: str, registry_identity: int) -> TrendSpecPreset:
    del registry_identity  # It keys this cache to the current canonical registry generation.
    preset = get_trend_preset(slug)
    return TrendSpecPreset(
        name=preset.label,
        description=preset.description,
        spec=preset.trend_spec,
    )


def default_preset_name() -> str:
    """Return the default preset name used by the application."""

    return _DEFAULT_PRESET_NAME


def list_trend_spec_presets() -> List[str]:
    """Return the available TrendSpec preset names (title case)."""

    return [preset.name for preset in _ordered_presets()]


def list_trend_spec_keys() -> List[str]:
    """Return canonical keys for TrendSpec presets (lower case)."""

    return [key for key, _ in _ordered_presets_items()]


def get_trend_spec_preset(name: str) -> TrendSpecPreset:
    """Look up a preset by name (case-insensitive)."""

    preset = get_trend_preset(name)
    if preset.slug not in _SIGNAL_PRESET_SLUGS:
        raise KeyError(f"Unknown trend preset: {name}")
    return _signal_view(preset.slug, id(_preset_registry()))


def resolve_trend_spec(name: str | None) -> TrendSpecPreset:
    """Return preset by name falling back to the default when ``name`` is
    falsy."""

    return get_trend_spec_preset(name or _DEFAULT_PRESET_NAME)


__all__ = [
    "TrendSpecPreset",
    "default_preset_name",
    "get_trend_spec_preset",
    "list_trend_spec_presets",
    "list_trend_spec_keys",
    "resolve_trend_spec",
]
