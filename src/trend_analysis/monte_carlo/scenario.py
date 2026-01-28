"""Schema definitions for Monte Carlo scenario configurations."""

from __future__ import annotations

import math
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
        raise ValueError(f"{field} must be >= {minimum} (got {number})")
    return number


def _coerce_float(value: Any, field: str, *, minimum: float | None = None) -> float:
    _require_value(value, field)
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be >= {minimum} (got {number})")
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

    Required fields
    ---------------
    - ``mode``: Simulation mode, either ``two_layer`` or ``mixture`` (case-insensitive).
    - ``n_paths``: Integer count of Monte Carlo paths (>= 1).
    - ``horizon_years``: Forecast horizon in years (float, > 0).
    - ``frequency``: Sampling frequency for generated paths. Allowed values are
      ``D``, ``W``, ``M``, ``Q``, or ``Y`` (case-insensitive).

    Optional fields
    ---------------
    - ``seed``: Random seed for reproducibility (integer, >= 0 when set).
    - ``jobs``: Parallel job count for simulation execution (integer, >= 1 when set).

    Validation and normalization
    ----------------------------
    - Required fields must be present and non-empty.
    - ``mode`` is normalized to lowercase and must be in the allowed set.
    - ``frequency`` is normalized to uppercase and must be in the allowed set.
    - ``n_paths`` and ``jobs`` are coerced to integers.
    - ``horizon_years`` is coerced to a float and must be > 0.
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
            raise ValueError(f"mode must be one of: {allowed} (got {self.mode!r})")

        self.n_paths = _coerce_int(self.n_paths, "n_paths", minimum=1)
        self.horizon_years = _coerce_float(self.horizon_years, "horizon_years")
        if self.horizon_years <= 0.0:
            raise ValueError(f"horizon_years must be > 0 (got {self.horizon_years})")

        self.frequency = _require_non_empty_str(self.frequency, "frequency").upper()
        if self.frequency not in _ALLOWED_FREQUENCIES:
            allowed = ", ".join(sorted(_ALLOWED_FREQUENCIES))
            raise ValueError(f"frequency must be one of: {allowed} (got {self.frequency!r})")

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

    Required fields
    ---------------
    - ``name``: Scenario identifier used for registry lookups (non-empty string).
    - ``base_config``: Path to the base configuration file for defaults
      (string or ``Path``).
    - ``monte_carlo``: Monte Carlo settings or a mapping of settings fields.

    Optional fields
    ---------------
    - ``description``: Free-form description of the scenario (string when set).
    - ``version``: Version string for scenario definitions (non-empty when set).
    - ``return_model``: Mapping describing the return model configuration. When
      supplied, it must be a mapping (explicit ``null`` is invalid).
    - ``strategy_set``: Mapping for curated/sampled strategy lists (mapping or
      omitted/``None``).
    - ``folds``: Mapping describing fold definitions and calibration (mapping or
      omitted/``None``).
    - ``outputs``: Mapping describing output locations and formats (mapping or
      omitted/``None``).
    - ``path``: Source path for the scenario definition file (``Path`` or string).
    - ``raw``: Raw scenario payload for traceability (mapping when set).

    Validation and normalization
    ----------------------------
    - ``base_config`` is stored as a ``Path`` after validation.
    - ``monte_carlo`` mappings are coerced into ``MonteCarloSettings`` with
      nested validation errors prefixed (e.g., ``monte_carlo.n_paths``).
    - Optional mappings must be provided as mappings when present.
    - ``raw`` must be a mapping when provided.
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
            try:
                self.monte_carlo = MonteCarloSettings(**monte_carlo_map)
            except ValueError as exc:
                raise ValueError(_prefix_error("monte_carlo", str(exc))) from exc

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


def _prefix_error(prefix: str, message: str) -> str:
    if message.startswith(f"{prefix}.") or message.startswith(f"{prefix} "):
        return message
    return f"{prefix}.{message}"
