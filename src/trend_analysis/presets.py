"""Registry of named TrendSpec presets shared across CLI and Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping

import yaml

from .signals import TrendSpec

PRESETS_DIR = Path(__file__).resolve().parents[2] / "config" / "presets"

# Metric aliases exposed for UI components and pipeline wiring.
UI_METRIC_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "sharpe_ratio": "sharpe",
        "sharpe": "sharpe",
        "return_ann": "return_ann",
        "annual_return": "return_ann",
        "max_drawdown": "drawdown",
        "drawdown": "drawdown",
        "volatility": "vol",
        "vol": "vol",
    }
)

PIPELINE_METRIC_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "sharpe": "Sharpe",
        "return_ann": "AnnualReturn",
        "drawdown": "MaxDrawdown",
        "max_drawdown": "MaxDrawdown",
        "vol": "Volatility",
        "volatility": "Volatility",
    }
)


def _freeze_mapping(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an immutable view over ``data`` suitable for storage on
    presets."""

    return MappingProxyType({k: data[k] for k in data})


def _normalise_metric_weights(raw: Mapping[str, Any]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for key, value in raw.items():
        alias = normalise_metric_key(str(key))
        if alias is None:
            continue
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        weights[alias] = weight
    return weights


def _coerce_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return max(coerced, minimum)


def _coerce_optional_int(value: Any | None, minimum: int = 1) -> int | None:
    if value is None:
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    if coerced < minimum:
        return None
    return coerced


def _coerce_optional_float(value: Any | None, minimum: float = 0.0) -> float | None:
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if coerced < minimum:
        return None
    return coerced


def _build_trend_spec(config: Mapping[str, Any]) -> TrendSpec:
    signals = config.get("signals")
    if not isinstance(signals, Mapping):
        signals = {}

    window = _coerce_int(signals.get("window"), default=63, minimum=1)
    min_periods = _coerce_optional_int(signals.get("min_periods"))
    if min_periods is not None and min_periods > window:
        min_periods = window
    lag = _coerce_int(signals.get("lag"), default=1, minimum=1)
    vol_adjust = bool(signals.get("vol_adjust", False))
    vol_target = _coerce_optional_float(signals.get("vol_target"))
    zscore = bool(signals.get("zscore", False))

    return TrendSpec(
        window=window,
        min_periods=min_periods,
        lag=lag,
        vol_adjust=vol_adjust,
        vol_target=vol_target,
        zscore=zscore,
    )


@dataclass(frozen=True)
class TrendPreset:
    """Container holding preset metadata and normalised configuration."""

    slug: str
    label: str
    description: str
    trend_spec: TrendSpec
    _config: Mapping[str, Any]

    def form_defaults(self) -> dict[str, Any]:
        """Return UI-ready defaults derived from the preset."""

        preset = self._config
        portfolio = preset.get("portfolio")
        if not isinstance(portfolio, Mapping):
            portfolio = {}

        defaults = {
            "lookback_months": _coerce_int(
                preset.get("lookback_months"), default=36, minimum=1
            ),
            "rebalance_frequency": str(preset.get("rebalance_frequency", "monthly")),
            "min_track_months": _coerce_int(
                preset.get("min_track_months"), default=24, minimum=1
            ),
            "selection_count": _coerce_int(
                preset.get("selection_count"), default=10, minimum=1
            ),
            "risk_target": _coerce_optional_float(
                preset.get("risk_target"), minimum=0.0
            )
            or 0.1,
            "weighting_scheme": str(portfolio.get("weighting_scheme", "equal")),
            "cooldown_months": _coerce_int(
                portfolio.get("cooldown_months"), default=3, minimum=0
            ),
            "metrics": _normalise_metric_weights(
                preset.get("metrics", {})
                if isinstance(preset.get("metrics"), Mapping)
                else {}
            ),
        }
        return defaults

    def signals_mapping(self) -> dict[str, Any]:
        """Return a mapping suitable for embedding into configuration."""

        spec = self.trend_spec
        mapping: dict[str, Any] = {
            "kind": spec.kind,
            "window": spec.window,
            "lag": spec.lag,
            "vol_adjust": spec.vol_adjust,
            "zscore": spec.zscore,
        }
        if spec.min_periods is not None:
            mapping["min_periods"] = spec.min_periods
        if spec.vol_target is not None:
            mapping["vol_target"] = spec.vol_target
        return mapping

    def vol_adjust_defaults(self) -> dict[str, Any]:
        """Return a shallow copy of vol adjustment overrides."""

        preset = self._config.get("vol_adjust")
        base: dict[str, Any] = {}
        if isinstance(preset, Mapping):
            base.update(preset)
        if "enabled" not in base:
            base["enabled"] = self.trend_spec.vol_adjust
        base.setdefault("target_vol", self.trend_spec.vol_target)
        window = base.get("window")
        if isinstance(window, MutableMapping):
            window = dict(window)
        elif isinstance(window, Mapping):
            window = dict(window.items())
        else:
            window = {}
        window.setdefault("length", self.trend_spec.window)
        base["window"] = window
        return base

    def metrics_pipeline(self) -> dict[str, float]:
        """Return metrics mapped to pipeline registry names."""

        weights = self.form_defaults()["metrics"]
        return {
            pipeline_metric_key(metric) or metric: float(weight)
            for metric, weight in weights.items()
        }


def _load_yaml(path: Path) -> Mapping[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        return {}
    return data


@lru_cache(maxsize=None)
def _preset_registry() -> Mapping[str, TrendPreset]:
    registry: dict[str, TrendPreset] = {}
    if not PRESETS_DIR.exists():
        return MappingProxyType(registry)
    for path in sorted(PRESETS_DIR.glob("*.yml")):
        slug = path.stem.lower()
        raw = _load_yaml(path)
        if not raw:
            continue
        label = str(raw.get("name") or slug.title())
        description = str(raw.get("description") or "")
        spec = _build_trend_spec(raw)
        preset = TrendPreset(
            slug=slug,
            label=label,
            description=description,
            trend_spec=spec,
            _config=_freeze_mapping(dict(raw)),
        )
        registry[slug] = preset
    return MappingProxyType(registry)


def list_trend_presets() -> tuple[TrendPreset, ...]:
    """Return all presets sorted by display label."""

    registry = _preset_registry().values()
    return tuple(sorted(registry, key=lambda preset: preset.label.lower()))


def list_preset_slugs() -> tuple[str, ...]:
    """Return available preset slugs (lowercase identifiers)."""

    return tuple(sorted(_preset_registry().keys()))


def get_trend_preset(name: str) -> TrendPreset:
    """Look up a preset by slug or display label."""

    if not name:
        raise KeyError("Preset name must be provided")
    lowered = name.lower()
    registry = _preset_registry()
    if lowered in registry:
        return registry[lowered]
    for preset in registry.values():
        if preset.label.lower() == lowered:
            return preset
    raise KeyError(f"Unknown trend preset: {name}")


def normalise_metric_key(name: str) -> str | None:
    """Return the UI metric key corresponding to ``name`` or ``None``."""

    if not name:
        return None
    key = UI_METRIC_ALIASES.get(name.lower())
    return key if key is not None else None


def pipeline_metric_key(name: str) -> str | None:
    """Return the pipeline metric registry name for ``name`` when available."""

    if not name:
        return None
    key = PIPELINE_METRIC_ALIASES.get(name.lower())
    return key if key is not None else None


def apply_trend_preset(config: Any, preset: TrendPreset) -> None:
    """Mutate ``config`` so future pipeline runs use ``preset`` parameters."""

    signals_mapping = preset.signals_mapping()
    current_signals = getattr(config, "signals", None)
    if isinstance(current_signals, Mapping):
        merged = {**current_signals, **signals_mapping}
    else:
        merged = dict(signals_mapping)
    setattr(config, "signals", merged)

    vol_adjust_cfg = getattr(config, "vol_adjust", {})
    if isinstance(vol_adjust_cfg, MutableMapping):
        vol_adjust = vol_adjust_cfg
    elif isinstance(vol_adjust_cfg, Mapping):
        vol_adjust = dict(vol_adjust_cfg)
    else:
        vol_adjust = {}

    defaults = preset.vol_adjust_defaults()
    vol_adjust.update({k: v for k, v in defaults.items() if v is not None})
    setattr(config, "vol_adjust", vol_adjust)

    run_section = getattr(config, "run", {})
    if isinstance(run_section, MutableMapping):
        run_cfg = run_section
    elif isinstance(run_section, Mapping):
        run_cfg = dict(run_section)
    else:
        run_cfg = {}
    run_cfg["trend_preset"] = preset.slug
    setattr(config, "run", run_cfg)


__all__ = [
    "TrendPreset",
    "apply_trend_preset",
    "get_trend_preset",
    "list_trend_presets",
    "list_preset_slugs",
    "normalise_metric_key",
    "pipeline_metric_key",
    "UI_METRIC_ALIASES",
    "PIPELINE_METRIC_ALIASES",
]
