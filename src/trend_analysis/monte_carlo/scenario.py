"""Schema definitions for Monte Carlo scenario configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_ALLOWED_MODES = {"two_layer", "mixture"}
_ALLOWED_FREQUENCIES = {"D", "W", "M", "Q", "Y"}
_MISSING = object()


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

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, str):
            return False
        return item in self.__dict__


@dataclass
class MonteCarloScenario:
    """Resolved Monte Carlo scenario configuration.

    This dataclass represents the validated schema for a Monte Carlo scenario.
    Required fields such as ``name``, ``base_config``, and ``monte_carlo`` are
    enforced, while nested mappings are coerced and validated when present.

    Attributes:
        name: Scenario identifier used for registry lookups.
        description: Optional free-form description of the scenario.
        version: Optional version string for scenario definitions.
        base_config: Path to the base configuration file for defaults.
        monte_carlo: Monte Carlo settings or a mapping of settings fields.
        return_model: Optional mapping describing the return model configuration.
        strategy_set: Optional mapping for curated/sampled strategy lists.
        folds: Optional mapping describing fold definitions and calibration.
        outputs: Optional mapping describing output locations and formats.
        path: Optional source path for the scenario definition file.
        raw: Optional raw scenario payload for traceability.
    """

    name: str | None = None
    description: str | None = None
    version: str | None = None
    base_config: Path | str | None = None
    monte_carlo: MonteCarloSettings | Mapping[str, Any] | None = None
    return_model: Mapping[str, Any] | None | object = _MISSING
    strategy_set: Mapping[str, Any] | None | object = _MISSING
    folds: Mapping[str, Any] | None | object = _MISSING
    outputs: Mapping[str, Any] | None | object = _MISSING
    path: Path | str | None = None
    raw: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.name = _require_non_empty_str(self.name, "name")
        if self.description is not None and not isinstance(self.description, str):
            self.description = str(self.description)
        if self.version is not None:
            self.version = _require_non_empty_str(self.version, "version")

        base_value = _require_value(self.base_config, "base_config")
        if isinstance(base_value, Path):
            self.base_config = base_value
        else:
            self.base_config = Path(_require_non_empty_str(base_value, "base_config"))

        monte_carlo_value = _require_value(self.monte_carlo, "monte_carlo")
        if isinstance(monte_carlo_value, MonteCarloSettings):
            self.monte_carlo = monte_carlo_value
        else:
            monte_carlo_map = _require_mapping(monte_carlo_value, "monte_carlo")
            self.monte_carlo = MonteCarloSettings(**monte_carlo_map)

        if self.return_model is _MISSING:
            self.return_model = None
        elif self.return_model is None:
            raise ValueError("return_model is required")
        else:
            self.return_model = _require_mapping(self.return_model, "return_model")

        self.strategy_set = _coerce_optional_mapping(self.strategy_set, "strategy_set")
        self.folds = _coerce_optional_mapping(self.folds, "folds")
        self.outputs = _coerce_optional_mapping(self.outputs, "outputs")

        if self.path is not None and not isinstance(self.path, Path):
            self.path = Path(str(self.path))
        if self.raw is not None:
            self.raw = _require_mapping(self.raw, "raw")


def _coerce_optional_mapping(value: object, field: str) -> Mapping[str, Any] | None:
    if value is _MISSING or value is None:
        return None
    return _require_mapping(value, field)
