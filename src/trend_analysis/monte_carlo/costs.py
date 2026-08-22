"""Cost process utilities for Monte Carlo simulations.

Canonical schema:
    costs:
      kind: regime_stochastic
      calm:
        trade_cost_bps:
          kind: lognormal
          mean: 6
          sigma: 0.25
      stress:
        trade_cost_bps:
          kind: lognormal
          mean: 18
          sigma: 0.35
        slippage_multiplier: 1.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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

    def sample(self, rng: np.random.Generator, size: int) -> NDArray[np.float64]:
        raise NotImplementedError

    def _apply_clip(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.clip_min is not None:
            values = np.maximum(values, float(self.clip_min))
        if self.clip_max is not None:
            values = np.minimum(values, float(self.clip_max))
        return values


@dataclass(frozen=True)
class FixedCostDistribution(CostDistribution):
    value: float = 0.0

    def sample(self, rng: np.random.Generator, size: int) -> NDArray[np.float64]:
        if size <= 0:
            return np.array([], dtype=float)
        values = np.full(size, float(self.value), dtype=float)
        return self._apply_clip(values)


@dataclass(frozen=True)
class NormalCostDistribution(CostDistribution):
    mean: float = 0.0
    std: float = 0.0

    def sample(self, rng: np.random.Generator, size: int) -> NDArray[np.float64]:
        if size <= 0:
            return np.array([], dtype=float)
        values = rng.normal(loc=float(self.mean), scale=float(self.std), size=size)
        return self._apply_clip(values)


@dataclass(frozen=True)
class LognormalCostDistribution(CostDistribution):
    mean: float = 0.0
    sigma: float = 1.0
    log_mean: float | None = None

    def sample(self, rng: np.random.Generator, size: int) -> NDArray[np.float64]:
        if size <= 0:
            return np.array([], dtype=float)
        log_mean = (
            float(self.log_mean)
            if self.log_mean is not None
            else _lognormal_mu_from_arithmetic_mean(float(self.mean), float(self.sigma))
        )
        values = rng.lognormal(mean=log_mean, sigma=float(self.sigma), size=size)
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
    top-level regime blocks (`calm`, `stress`, etc.) containing a
    `trade_cost_bps` mapping whose distribution discriminator is `kind`.
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

        kind = config.get("kind")
        if kind != "regime_stochastic":
            raise ValueError("costs.kind must be 'regime_stochastic'")

        legacy_keys = {"regimes", "default", "distribution", "allow_unknown"} & set(config)
        if legacy_keys:
            keys = ", ".join(sorted(legacy_keys))
            raise ValueError(
                f"unsupported legacy costs key(s): {keys}; place regime mappings directly under costs"
            )

        default_regime = str(config.get("default_regime") or "calm").strip() or "calm"
        regimes: dict[str, RegimeCostSpec] = {}

        for label, spec in _extract_top_level_regimes(config).items():
            regimes[label] = _parse_regime_spec(spec, name=label)

        if not regimes:
            raise ValueError("costs must define at least one top-level regime mapping")

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
    if not isinstance(spec, Mapping):
        raise ValueError(f"regime '{name}' spec must be a mapping with trade_cost_bps")
    unexpected = set(spec) - {"trade_cost_bps", "slippage_multiplier"}
    if unexpected:
        keys = ", ".join(sorted(str(key) for key in unexpected))
        raise ValueError(f"regime '{name}' has unsupported key(s): {keys}")
    if "trade_cost_bps" not in spec:
        raise ValueError(f"regime '{name}' must define trade_cost_bps")
    dist_cfg = spec["trade_cost_bps"]
    distribution = _parse_distribution(dist_cfg, regime=name)
    slippage = _coerce_float(
        spec.get("slippage_multiplier", 1.0), "slippage_multiplier", minimum=0.0
    )
    return RegimeCostSpec(distribution=distribution, slippage_multiplier=slippage)


def _parse_distribution(spec: Any, *, regime: str) -> CostDistribution:
    if not isinstance(spec, Mapping):
        raise ValueError(
            f"trade_cost_bps for regime '{regime}' must be a mapping with a kind field"
        )
    if "dist" in spec or "distribution" in spec:
        raise ValueError(
            f"trade_cost_bps for regime '{regime}' uses a legacy discriminator; use kind"
        )
    if "kind" not in spec:
        raise ValueError(f"trade_cost_bps for regime '{regime}' must define kind")
    kind = str(spec["kind"]).strip().lower()
    clip_min = _coerce_optional_float(spec.get("clip_min"), "clip_min")
    clip_max = _coerce_optional_float(spec.get("clip_max"), "clip_max")

    if kind == "fixed":
        unexpected = set(spec) - {"kind", "value", "clip_min", "clip_max"}
        if unexpected:
            raise ValueError(f"fixed distribution has unsupported key(s): {sorted(unexpected)}")
        if "value" not in spec:
            raise ValueError("fixed distribution must define value")
        value = _coerce_float(spec["value"], "value")
        return FixedCostDistribution(kind=kind, value=value, clip_min=clip_min, clip_max=clip_max)

    if kind == "normal":
        unexpected = set(spec) - {"kind", "mean", "std", "clip_min", "clip_max"}
        if unexpected:
            raise ValueError(f"normal distribution has unsupported key(s): {sorted(unexpected)}")
        mean = _coerce_float(spec.get("mean", 0.0), "mean")
        std = _coerce_float(spec.get("std", 0.0), "std", minimum=0.0)
        return NormalCostDistribution(
            kind=kind,
            mean=mean,
            std=std,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    if kind == "lognormal":
        unexpected = set(spec) - {
            "kind",
            "mean",
            "sigma",
            "log_mean",
            "clip_min",
            "clip_max",
        }
        if unexpected:
            raise ValueError(f"lognormal distribution has unsupported key(s): {sorted(unexpected)}")
        mean = _coerce_float(spec.get("mean", 0.0), "mean", minimum=0.0)
        sigma = _coerce_float(spec.get("sigma", 1.0), "sigma", minimum=0.0)
        if "log_mean" in spec:
            log_mean = _coerce_float(spec["log_mean"], "log_mean")
            mean = float(np.exp(log_mean + 0.5 * sigma * sigma))
        else:
            log_mean = _lognormal_mu_from_arithmetic_mean(mean, sigma)
        return LognormalCostDistribution(
            kind=kind,
            mean=mean,
            sigma=sigma,
            log_mean=log_mean,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    raise ValueError(f"Unsupported distribution kind '{kind}' for regime '{regime}'")


def _lognormal_mu_from_arithmetic_mean(mean_bps: float, sigma: float) -> float:
    if mean_bps <= 0.0:
        raise ValueError("lognormal mean must be > 0 basis points")
    return float(np.log(mean_bps) - 0.5 * sigma * sigma)


def _extract_top_level_regimes(config: Mapping[str, Any]) -> dict[str, Any]:
    reserved = {"kind", "enabled", "default_regime"}
    top_level: dict[str, Any] = {}
    for key, value in config.items():
        label = str(key).strip()
        if not label or label in reserved:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"costs.{label} must be a regime mapping with trade_cost_bps")
        top_level[label] = value
    return top_level
