"""Schema definitions for Monte Carlo scenario configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_ALLOWED_MODES = {"two_layer", "mixture"}
_ALLOWED_FREQUENCIES = {"D", "W", "M", "Q", "Y"}


def _require_value(value: Any, field: str) -> Any:
    if value is None:
        raise ValueError(f"{field} is required")
    return value


def _require_non_empty_str(value: Any, field: str) -> str:
    _require_value(value, field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _coerce_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    _require_value(value, field)
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return number


def _coerce_float(value: Any, field: str, *, minimum: float | None = None) -> float:
    _require_value(value, field)
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return number


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    _require_value(value, field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


@dataclass
class MonteCarloSettings:
    """Configuration settings for Monte Carlo path generation.

    Use this dataclass directly or pass a mapping when building
    :class:`MonteCarloScenario`, which will coerce and validate fields.

    This schema enforces required fields, permissible values, and numeric bounds
    for the Monte Carlo simulation engine.

    Attributes:
        mode: Simulation mode, either ``two_layer`` or ``mixture`` (required).
        n_paths: Number of simulation paths to generate (integer, >= 1).
        horizon_years: Forecast horizon in years (float, > 0).
        frequency: Sampling frequency for generated paths. Allowed values are
            ``D``, ``W``, ``M``, ``Q``, or ``Y``.
        seed: Optional random seed for reproducibility (integer, >= 0 when set).
        jobs: Optional parallel job count (integer, >= 1 when set).
    """

    mode: str | None = None
    n_paths: int | None = None
    horizon_years: float | None = None
    frequency: str | None = None
    seed: int | None = None
    jobs: int | None = None

    def __post_init__(self) -> None:
        self.mode = _require_non_empty_str(self.mode, "mode").lower()
        if self.mode not in _ALLOWED_MODES:
            allowed = ", ".join(sorted(_ALLOWED_MODES))
            raise ValueError(f"mode must be one of: {allowed}")

        self.n_paths = _coerce_int(self.n_paths, "n_paths", minimum=1)
        self.horizon_years = _coerce_float(self.horizon_years, "horizon_years", minimum=0.0)
        if self.horizon_years <= 0.0:
            raise ValueError("horizon_years must be > 0")

        self.frequency = _require_non_empty_str(self.frequency, "frequency").upper()
        if self.frequency not in _ALLOWED_FREQUENCIES:
            allowed = ", ".join(sorted(_ALLOWED_FREQUENCIES))
            raise ValueError(f"frequency must be one of: {allowed}")

        if self.seed is not None:
            self.seed = _coerce_int(self.seed, "seed", minimum=0)

        if self.jobs is not None:
            self.jobs = _coerce_int(self.jobs, "jobs", minimum=1)


@dataclass(frozen=True)
class MonteCarloScenario:
    """Resolved Monte Carlo scenario configuration."""

    name: str
    description: str | None
    version: str
    base_config: Path
    monte_carlo: Mapping[str, Any]
    strategy_set: Mapping[str, Any] | None
    outputs: Mapping[str, Any] | None
    path: Path
    raw: Mapping[str, Any]
