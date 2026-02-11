"""Cost process utilities for Monte Carlo simulations.

Canonical schema:
    costs:
      kind: regime_stochastic
      calm:
        trade_cost_bps:
          dist: lognormal
          mean: 6
          sigma: 0.25
      stress:
        trade_cost_bps:
          dist: lognormal
          mean: 18
          sigma: 0.35
        slippage_multiplier: 1.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CostDistribution",
    "FixedCostDistribution",
    "NormalCostDistribution",
    "LognormalCostDistribution",
    "RegimeCostSpec",
    "CostProcessOutput",
    "CostProcess",
]


def _coerce_float(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return number


def _coerce_optional_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _coerce_float(value, field)


@dataclass(frozen=True)
class CostDistribution:
    """Base cost distribution for sampling basis point costs."""

    kind: str
    clip_min: float | None = None
    clip_max: float | None = None

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        raise NotImplementedError

    def _apply_clip(self, values: np.ndarray) -> np.ndarray:
        if self.clip_min is not None:
            values = np.maximum(values, float(self.clip_min))
        if self.clip_max is not None:
            values = np.minimum(values, float(self.clip_max))
        return values


@dataclass(frozen=True)
class FixedCostDistribution(CostDistribution):
    value: float = 0.0

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if size <= 0:
            return np.array([], dtype=float)
        values = np.full(size, float(self.value), dtype=float)
        return self._apply_clip(values)


@dataclass(frozen=True)
class NormalCostDistribution(CostDistribution):
    mean: float = 0.0
    std: float = 0.0

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if size <= 0:
            return np.array([], dtype=float)
        values = rng.normal(loc=float(self.mean), scale=float(self.std), size=size)
        return self._apply_clip(values)


@dataclass(frozen=True)
class LognormalCostDistribution(CostDistribution):
    mean: float = 0.0
    sigma: float = 1.0

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if size <= 0:
            return np.array([], dtype=float)
        values = rng.lognormal(mean=float(self.mean), sigma=float(self.sigma), size=size)
        return self._apply_clip(values)


@dataclass(frozen=True)
class RegimeCostSpec:
    """Regime-specific cost specification."""

    distribution: CostDistribution
    slippage_multiplier: float = 1.0


@dataclass(frozen=True)
class CostProcessOutput:
    """Sampled cost results for a single Monte Carlo path."""

    regimes: pd.Series
    cost_bps: pd.Series
    slippage_multiplier: pd.Series
    turnover: pd.Series
    transaction_costs: pd.Series
    cost_drag: pd.Series
    total_cost_drag: float


class CostProcess:
    """Sample per-period transaction costs conditional on regime.

    `from_config` accepts the canonical `kind: regime_stochastic` shape with
    top-level regime blocks (`calm`, `stress`, etc.) containing `trade_cost_bps`.
    Legacy aliases remain supported (`regimes`, `distribution`, `dist`, numeric
    shorthand).
    """

    def __init__(
        self,
        regimes: Mapping[str, RegimeCostSpec],
        *,
        default_regime: str = "calm",
        allow_unknown: bool = True,
    ) -> None:
        if not regimes:
            raise ValueError("CostProcess requires at least one regime specification")
        cleaned: dict[str, RegimeCostSpec] = {}
        for key, spec in regimes.items():
            label = str(key).strip()
            if not label:
                raise ValueError("Regime labels must be non-empty strings")
            if not isinstance(spec, RegimeCostSpec):
                raise TypeError("RegimeCostSpec required for regime definitions")
            cleaned[label] = spec
        default_label = str(default_regime).strip() or "calm"
        if default_label not in cleaned:
            default_label = next(iter(cleaned.keys()))
        self._regimes = cleaned
        self._default = default_label
        self._allow_unknown = bool(allow_unknown)

    @property
    def default_regime(self) -> str:
        return self._default

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "CostProcess | None":
        if not config:
            return None
        if not isinstance(config, Mapping):
            raise ValueError("cost_process config must be a mapping")
        if "enabled" in config and not bool(config.get("enabled")):
            return None

        default_regime = str(config.get("default_regime") or "calm").strip() or "calm"
        regimes: dict[str, RegimeCostSpec] = {}

        regimes_cfg = config.get("regimes")
        if isinstance(regimes_cfg, Mapping):
            for label, spec in regimes_cfg.items():
                regimes[str(label)] = _parse_regime_spec(spec, name=str(label))

        top_level_regimes = _extract_top_level_regimes(config)
        for label, spec in top_level_regimes.items():
            regimes[str(label)] = _parse_regime_spec(spec, name=str(label))

        if not regimes:
            fallback_spec = config.get("default") or config.get("distribution") or config
            regimes[default_regime] = _parse_regime_spec(fallback_spec, name=default_regime)

        return cls(regimes, default_regime=default_regime)

    def sample(
        self,
        *,
        regimes: pd.Series | Sequence[str] | None,
        turnover: pd.Series | float | None,
        index: pd.Index | None,
        rng: np.random.Generator,
    ) -> CostProcessOutput:
        series_index = _resolve_index(regimes, turnover, index)
        regime_series = _coerce_regime_series(regimes, series_index, self._default)
        turnover_series = _coerce_turnover_series(turnover, series_index)

        cost_bps = self._sample_cost_bps(regime_series, rng)
        slippage = self._sample_slippage_multiplier(regime_series)

        transaction_costs = turnover_series * (cost_bps / 10000.0) * slippage
        cost_drag = transaction_costs.copy()
        total_cost_drag = float(cost_drag.sum()) if not cost_drag.empty else 0.0

        return CostProcessOutput(
            regimes=regime_series,
            cost_bps=cost_bps,
            slippage_multiplier=slippage,
            turnover=turnover_series,
            transaction_costs=transaction_costs,
            cost_drag=cost_drag,
            total_cost_drag=total_cost_drag,
        )

    def _sample_cost_bps(self, regime_series: pd.Series, rng: np.random.Generator) -> pd.Series:
        values = np.empty(len(regime_series), dtype=float)
        for label in pd.unique(regime_series):
            spec = self._select_spec(label)
            mask = regime_series == label
            count = int(mask.sum())
            if count <= 0:
                continue
            values[mask.to_numpy()] = spec.distribution.sample(rng, count)
        return pd.Series(values, index=regime_series.index, name="cost_bps")

    def _sample_slippage_multiplier(self, regime_series: pd.Series) -> pd.Series:
        values = np.empty(len(regime_series), dtype=float)
        for label in pd.unique(regime_series):
            spec = self._select_spec(label)
            mask = regime_series == label
            values[mask.to_numpy()] = float(spec.slippage_multiplier)
        return pd.Series(values, index=regime_series.index, name="slippage_multiplier")

    def _select_spec(self, label: str) -> RegimeCostSpec:
        key = str(label).strip() if label is not None else ""
        if key in self._regimes:
            return self._regimes[key]
        if self._allow_unknown:
            return self._regimes[self._default]
        raise KeyError(f"Unknown regime '{label}'")


def _resolve_index(
    regimes: pd.Series | Sequence[str] | None,
    turnover: pd.Series | float | None,
    index: pd.Index | None,
) -> pd.Index:
    if index is not None:
        return index
    if isinstance(regimes, pd.Series):
        return regimes.index
    if isinstance(turnover, pd.Series):
        return turnover.index
    raise ValueError("index is required when regimes and turnover are not Series")


def _coerce_regime_series(
    regimes: pd.Series | Sequence[str] | None,
    index: pd.Index,
    default_regime: str,
) -> pd.Series:
    if regimes is None:
        return pd.Series([default_regime] * len(index), index=index, dtype="string")
    if isinstance(regimes, pd.Series):
        series = regimes.reindex(index)
        return series.fillna(default_regime).astype("string")
    values = [str(label) if label is not None else default_regime for label in regimes]
    if len(values) != len(index):
        raise ValueError("regime label length must match index length")
    return pd.Series(values, index=index, dtype="string")


def _coerce_turnover_series(turnover: pd.Series | float | None, index: pd.Index) -> pd.Series:
    if turnover is None:
        return pd.Series(0.0, index=index, name="turnover")
    if isinstance(turnover, pd.Series):
        return turnover.reindex(index).fillna(0.0).astype(float)
    return pd.Series(float(turnover), index=index, name="turnover")


def _parse_regime_spec(spec: Any, *, name: str) -> RegimeCostSpec:
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return RegimeCostSpec(
            distribution=FixedCostDistribution(kind="fixed", value=float(spec)),
            slippage_multiplier=1.0,
        )
    if not isinstance(spec, Mapping):
        raise ValueError(f"regime '{name}' spec must be a mapping or number")
    dist_cfg = spec.get("trade_cost_bps", spec.get("distribution", spec))
    distribution = _parse_distribution(dist_cfg, regime=name)
    slippage = _coerce_float(
        spec.get("slippage_multiplier", 1.0), "slippage_multiplier", minimum=0.0
    )
    return RegimeCostSpec(distribution=distribution, slippage_multiplier=slippage)


def _parse_distribution(spec: Any, *, regime: str) -> CostDistribution:
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return FixedCostDistribution(kind="fixed", value=float(spec))
    if not isinstance(spec, Mapping):
        raise ValueError(f"distribution for regime '{regime}' must be a mapping or number")
    kind = str(spec.get("kind") or spec.get("dist") or "fixed").strip().lower()
    clip_min = _coerce_optional_float(spec.get("clip_min"), "clip_min")
    clip_max = _coerce_optional_float(spec.get("clip_max"), "clip_max")

    if kind == "fixed":
        value = _coerce_float(spec.get("value", spec.get("bps", 0.0)), "value")
        return FixedCostDistribution(kind=kind, value=value, clip_min=clip_min, clip_max=clip_max)

    if kind == "normal":
        mean = _coerce_float(spec.get("mean", 0.0), "mean")
        std = _coerce_float(spec.get("std", spec.get("sigma", 0.0)), "std", minimum=0.0)
        return NormalCostDistribution(
            kind=kind,
            mean=mean,
            std=std,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    if kind == "lognormal":
        mean = _coerce_float(spec.get("mean", 0.0), "mean")
        sigma = _coerce_float(spec.get("sigma", 1.0), "sigma", minimum=0.0)
        return LognormalCostDistribution(
            kind=kind,
            mean=mean,
            sigma=sigma,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    raise ValueError(f"Unsupported distribution kind '{kind}' for regime '{regime}'")


def _extract_top_level_regimes(config: Mapping[str, Any]) -> dict[str, Any]:
    reserved = {
        "kind",
        "enabled",
        "default_regime",
        "allow_unknown",
        "regimes",
        "default",
        "distribution",
    }
    top_level: dict[str, Any] = {}
    for key, value in config.items():
        label = str(key).strip()
        if not label or label in reserved:
            continue
        if isinstance(value, Mapping) and (
            "trade_cost_bps" in value or "distribution" in value or "slippage_multiplier" in value
        ):
            top_level[label] = value
    return top_level
