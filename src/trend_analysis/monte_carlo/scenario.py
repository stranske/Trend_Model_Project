"""Schema definitions for Monte Carlo scenario configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

_ALLOWED_MODES = {"two_layer", "mixture"}
_ALLOWED_FREQUENCIES = {"D", "W", "M", "Q", "Y"}


def _require_non_empty_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _coerce_int(value: Any, field: str, *, minimum: int | None = None) -> int:
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
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


@dataclass
class MonteCarloSettings:
    """Configuration settings for Monte Carlo path generation.

    Use this dataclass directly or pass a mapping when building
    :class:`MonteCarloScenario`, which will coerce and validate fields.

    Attributes:
        mode: Simulation mode, either ``two_layer`` or ``mixture``.
        n_paths: Number of simulation paths to generate (must be >= 1).
        horizon_years: Forecast horizon in years (must be > 0).
        frequency: Sampling frequency for generated paths (D/W/M/Q/Y).
        seed: Optional random seed for reproducibility (must be >= 0 when set).
        jobs: Optional parallel job count (must be >= 1 when set).
    """

    mode: str
    n_paths: int
    horizon_years: float
    frequency: str
    seed: int | None
    jobs: int | None

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


@dataclass
class MonteCarloScenario:
    """Scenario configuration for Monte Carlo simulations.

    The schema accepts nested dictionaries for the ``monte_carlo`` field and
    validates required fields for all top-level mappings.

    Attributes:
        name: Scenario identifier.
        description: Human-readable description of the scenario.
        base_config: Path to the base configuration file to extend.
        monte_carlo: Monte Carlo settings (or a mapping to build them from).
        return_model: Return model configuration mapping.
        strategy_set: Strategy selection configuration mapping.
        folds: Cross-validation fold configuration mapping.
        outputs: Output configuration mapping.
    """

    name: str
    description: str
    base_config: str
    monte_carlo: MonteCarloSettings
    return_model: Mapping[str, Any]
    strategy_set: Mapping[str, Any]
    folds: Mapping[str, Any]
    outputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.name = _require_non_empty_str(self.name, "name")
        self.description = _require_non_empty_str(self.description, "description")
        self.base_config = _require_non_empty_str(self.base_config, "base_config")

        if isinstance(self.monte_carlo, Mapping):
            self.monte_carlo = MonteCarloSettings(**self.monte_carlo)
        if not isinstance(self.monte_carlo, MonteCarloSettings):
            raise ValueError("monte_carlo must be a MonteCarloSettings instance or mapping")

        self.return_model = dict(_require_mapping(self.return_model, "return_model"))
        self.strategy_set = dict(_require_mapping(self.strategy_set, "strategy_set"))
        self.folds = dict(_require_mapping(self.folds, "folds"))
        self.outputs = dict(_require_mapping(self.outputs, "outputs"))
