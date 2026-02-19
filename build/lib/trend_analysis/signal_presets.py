"""Named TrendSpec presets shared between CLI and UI layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

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

_PRESETS: Dict[str, TrendSpecPreset] = {
    "conservative": TrendSpecPreset(
        name="Conservative",
        description="Longer window with heavier smoothing and lower volatility target.",
        spec=TrendSpec(
            window=126,
            min_periods=90,
            lag=1,
            vol_adjust=True,
            vol_target=0.08,
            zscore=True,
        ),
    ),
    "balanced": TrendSpecPreset(
        name="Balanced",
        description="Default configuration offering a balance between responsiveness and stability.",
        spec=TrendSpec(
            window=84,
            min_periods=63,
            lag=1,
            vol_adjust=True,
            vol_target=0.10,
            zscore=True,
        ),
    ),
    "aggressive": TrendSpecPreset(
        name="Aggressive",
        description="Shorter window prioritising responsiveness with higher volatility allowance.",
        spec=TrendSpec(
            window=42,
            min_periods=30,
            lag=1,
            vol_adjust=True,
            vol_target=0.15,
            zscore=False,
        ),
    ),
}


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

    key = name.strip().lower()
    if key not in _PRESETS:
        raise KeyError(f"Unknown TrendSpec preset: {name}")
    return _PRESETS[key]


def resolve_trend_spec(name: str | None) -> TrendSpecPreset:
    """Return preset by name falling back to the default when ``name`` is
    falsy."""

    if not name:
        return _PRESETS[_DEFAULT_PRESET_NAME.lower()]
    return get_trend_spec_preset(name)


def _ordered_presets() -> Iterable[TrendSpecPreset]:
    for _, preset in _ordered_presets_items():
        yield preset


def _ordered_presets_items() -> List[tuple[str, TrendSpecPreset]]:
    return sorted(_PRESETS.items(), key=lambda item: item[0])


__all__ = [
    "TrendSpecPreset",
    "default_preset_name",
    "get_trend_spec_preset",
    "list_trend_spec_presets",
    "list_trend_spec_keys",
    "resolve_trend_spec",
]
