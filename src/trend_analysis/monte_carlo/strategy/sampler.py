"""Sampling utilities for Monte Carlo strategy variants."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from trend_analysis.monte_carlo.strategy.variant import StrategyVariant

logger = logging.getLogger(__name__)

ConstraintResult = bool | tuple[bool, str]
ConstraintFn = Callable[[Mapping[str, Any]], ConstraintResult]


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _require_non_empty_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _coerce_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    return value


def _set_override(overrides: dict[str, Any], path: Sequence[str], value: Any) -> None:
    if not path:
        raise ValueError("sampling key path cannot be empty")
    cursor: dict[str, Any] = overrides
    for segment in path[:-1]:
        if not segment:
            raise ValueError("sampling key path cannot contain empty segments")
        if segment not in cursor or not isinstance(cursor[segment], Mapping):
            cursor[segment] = {}
        cursor = cursor[segment]
    if not path[-1]:
        raise ValueError("sampling key path cannot contain empty segments")
    cursor[path[-1]] = value


@dataclass(frozen=True)
class CategoricalDistribution:
    values: tuple[Any, ...]

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.values)

    def finite_values(self) -> tuple[Any, ...] | None:
        return self.values


@dataclass(frozen=True)
class DiscreteDistribution:
    values: tuple[Any, ...]

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.values)

    def finite_values(self) -> tuple[Any, ...] | None:
        return self.values


@dataclass(frozen=True)
class UniformDistribution:
    low: float
    high: float

    def sample(self, rng: random.Random) -> float:
        if self.low == self.high:
            return self.low
        return rng.uniform(self.low, self.high)

    def finite_values(self) -> tuple[Any, ...] | None:
        return None


Distribution = CategoricalDistribution | DiscreteDistribution | UniformDistribution


def parse_distribution(spec: Mapping[str, Any], *, path: str) -> Distribution:
    dist = _require_non_empty_str(spec.get("dist"), f"{path}.dist").lower()

    if dist == "categorical":
        values = spec.get("values")
        if not isinstance(values, list):
            raise ValueError(f"{path}.values must be a list")
        if not values:
            raise ValueError(f"{path}.values must be non-empty")
        return CategoricalDistribution(tuple(values))

    if dist == "discrete":
        values = spec.get("values")
        if values is not None:
            if not isinstance(values, list):
                raise ValueError(f"{path}.values must be a list")
            if not values:
                raise ValueError(f"{path}.values must be non-empty")
            return DiscreteDistribution(tuple(values))

        low = _coerce_float(spec.get("low"), f"{path}.low")
        high = _coerce_float(spec.get("high"), f"{path}.high")
        if high < low:
            raise ValueError(f"{path}.high must be >= {path}.low")
        step_value = spec.get("step", 1)
        step = _coerce_float(step_value, f"{path}.step")
        if step <= 0:
            raise ValueError(f"{path}.step must be > 0")
        values_list: list[Any] = []
        current = low
        epsilon = step / 1e6
        all_integer = all(
            _is_number(value) and float(value).is_integer() for value in (low, high, step)
        )
        while current <= high + epsilon:
            values_list.append(int(round(current)) if all_integer else current)
            current += step
        if not values_list:
            raise ValueError(f"{path} generated no discrete values")
        return DiscreteDistribution(tuple(values_list))

    if dist == "uniform":
        low = _coerce_float(spec.get("low"), f"{path}.low")
        high = _coerce_float(spec.get("high"), f"{path}.high")
        if high < low:
            raise ValueError(f"{path}.high must be >= {path}.low")
        return UniformDistribution(low=low, high=high)

    raise ValueError(f"{path}.dist must be categorical, discrete, or uniform (got {dist!r})")


def parse_sampling_config(sampling: Mapping[str, Any]) -> dict[str, Distribution]:
    sampling = _require_mapping(sampling, "sampling")
    parsed: dict[str, Distribution] = {}
    for key, spec in sampling.items():
        if not isinstance(key, str):
            key = str(key)
        path = _require_non_empty_str(key, "sampling key")
        if any(not segment for segment in path.split(".")):
            raise ValueError("sampling key cannot contain empty path segments")
        spec_map = _require_mapping(spec, f"sampling.{path}")
        parsed[path] = parse_distribution(spec_map, path=f"sampling.{path}")
    if not parsed:
        raise ValueError("sampling must define at least one distribution")
    return parsed


def _count_finite_combinations(distributions: Iterable[Distribution]) -> int | None:
    sizes: list[int] = []
    for dist in distributions:
        values = dist.finite_values()
        if values is None:
            return None
        unique_values = {_freeze(value) for value in values}
        sizes.append(len(unique_values))
    total = 1
    for size in sizes:
        total *= size
    return total


def _next_name(prefix: str, index: int, existing: set[str]) -> tuple[str, int]:
    while True:
        name = f"{prefix}_{index:03d}"
        if name not in existing:
            existing.add(name)
            return name, index + 1
        index += 1


def _coerce_constraints(constraints: Sequence[ConstraintFn] | None) -> tuple[ConstraintFn, ...]:
    if constraints is None:
        return ()
    return tuple(constraints)


def sample_strategy_variants(
    sampling: Mapping[str, Any],
    n_strategies: int,
    *,
    seed: int | None = None,
    constraints: Sequence[ConstraintFn] | None = None,
    max_rejection_attempts: int = 1000,
    name_prefix: str = "sampled",
    existing_names: Sequence[str] | None = None,
) -> list[StrategyVariant]:
    if n_strategies < 1:
        raise ValueError("n_strategies must be >= 1")
    if max_rejection_attempts < 0:
        raise ValueError("max_rejection_attempts must be >= 0")

    distributions = parse_sampling_config(sampling)
    total_combinations = _count_finite_combinations(distributions.values())
    if total_combinations is not None and n_strategies > total_combinations:
        raise ValueError(
            "n_strategies exceeds available unique combinations "
            f"({n_strategies} > {total_combinations})"
        )

    rng = random.Random(seed)
    constraint_fns = _coerce_constraints(constraints)
    names = set(existing_names or [])
    name_index = 1
    accepted: list[StrategyVariant] = []
    seen_overrides: set[Any] = set()
    rejection_count = 0

    while len(accepted) < n_strategies:
        overrides: dict[str, Any] = {}
        for key, dist in distributions.items():
            path = key.split(".")
            _set_override(overrides, path, dist.sample(rng))

        frozen = _freeze(overrides)
        if frozen in seen_overrides:
            rejection_count += 1
            logger.info("Rejected duplicate sampled config (attempt %s)", rejection_count)
        else:
            rejected = False
            for idx, constraint in enumerate(constraint_fns):
                result = constraint(overrides)
                ok = result
                reason = "constraint failed"
                if isinstance(result, tuple):
                    ok, reason = result
                if not ok:
                    rejection_count += 1
                    logger.info("Rejected sampled config (attempt %s): %s", rejection_count, reason)
                    rejected = True
                    break
            if not rejected:
                seen_overrides.add(frozen)
                name, name_index = _next_name(name_prefix, name_index, names)
                accepted.append(StrategyVariant(name=name, overrides=overrides))

        if rejection_count >= max_rejection_attempts and len(accepted) < n_strategies:
            raise RuntimeError(
                "Exceeded max_rejection_attempts while sampling strategies "
                f"({max_rejection_attempts})"
            )

    return accepted


__all__ = [
    "CategoricalDistribution",
    "DiscreteDistribution",
    "UniformDistribution",
    "parse_distribution",
    "parse_sampling_config",
    "sample_strategy_variants",
]
