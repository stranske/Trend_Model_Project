"""Lightweight configuration schema shared by the CLI and Streamlit app.

The main ``trend_analysis`` configuration model is powered by Pydantic, which
pulls in a long dependency chain and validates hundreds of fields.  The CLI and
Streamlit app only need a tiny subset of that surface area to fail fast when the
inputs are clearly wrong.  Re-implementing the small contract with stdlib
building blocks keeps startup lean while providing actionable error messages for
common mistakes (missing CSV path, typos in the universe membership file, wrong
frequency, etc.).
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from utils.paths import proj_path

try:  # pragma: no cover - optional instrumentation
    from trend_analysis.config.coverage import (
        ConfigCoverageTracker,
        get_config_coverage_tracker,
    )
except Exception:  # pragma: no cover - defensive fallback

    def get_config_coverage_tracker() -> ConfigCoverageTracker | None:
        return None

    ConfigCoverageTracker = None  # type: ignore[assignment,misc]


__all__ = [
    "CoreConfig",
    "CoreConfigError",
    "CostSettings",
    "DataSettings",
    "load_core_config",
    "validate_core_config",
]

_ALLOWED_FREQUENCIES = {"D", "W", "M", "ME"}
_GLOB_CHARS = {"*", "?", "[", "]"}
_DEFAULT_BASE = proj_path()
_DEFAULT_DATE_COLUMN = "Date"
_DEFAULT_FREQUENCY = "M"
_DEFAULT_PER_TRADE_COST = 0.0
_DEFAULT_HALF_SPREAD = 0.0


class CoreConfigError(ValueError):
    """Raised when the lightweight configuration contract is violated."""


@dataclass(frozen=True, slots=True)
class DataSettings:
    """Resolved data paths and time-series settings."""

    csv_path: Path | None
    managers_glob: str | None
    date_column: str
    frequency: str
    universe_membership_path: Path | None


@dataclass(frozen=True, slots=True)
class CostSettings:
    """Portfolio cost parameters validated at startup."""

    per_trade_bps: float
    half_spread_bps: float


@dataclass(frozen=True, slots=True)
class CoreConfig:
    """Bundle of the minimal configuration knobs used by the CLI/UI."""

    data: DataSettings
    costs: CostSettings

    def to_payload(self) -> dict[str, Any]:
        """Serialise the validated configuration back to simple dictionaries."""

        csv_path = str(self.data.csv_path) if self.data.csv_path is not None else None
        membership = (
            str(self.data.universe_membership_path)
            if self.data.universe_membership_path is not None
            else None
        )
        data_section: dict[str, Any] = {
            "universe_membership_path": membership,
            "date_column": self.data.date_column,
            "frequency": self.data.frequency,
        }
        if csv_path is not None:
            data_section["csv_path"] = csv_path
        if self.data.managers_glob is not None:
            data_section["managers_glob"] = self.data.managers_glob

        return {
            "data": data_section,
            "portfolio": {
                "cost_model": {
                    "per_trade_bps": self.costs.per_trade_bps,
                    "half_spread_bps": self.costs.half_spread_bps,
                },
            },
        }


def _as_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise CoreConfigError(f"{field} section must be a mapping")


def _normalise_path(
    value: Any,
    *,
    field: str,
    base_path: Path | None,
    required: bool,
) -> Path | None:
    if value in (None, ""):
        if required:
            raise CoreConfigError(f"{field} is required")
        return None
    if isinstance(value, Path):
        candidate = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise CoreConfigError(f"{field} must be a non-empty string")
        candidate = Path(stripped)
    else:
        raise CoreConfigError(f"{field} must be a path-like string")
    candidate = candidate.expanduser()
    base = base_path or _DEFAULT_BASE
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise CoreConfigError(f"{field} '{candidate}' does not exist")
    if not candidate.is_file():
        raise CoreConfigError(f"{field} '{candidate}' must point to a file")
    return candidate


def _normalise_glob(
    value: Any,
    *,
    field: str,
    base_path: Path | None,
) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (str, Path)):
        raw = str(value).strip()
    else:
        raise CoreConfigError(f"{field} must be a string")
    if not raw:
        raise CoreConfigError(f"{field} must be a non-empty string")
    base = base_path or _DEFAULT_BASE
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    else:
        candidate = candidate.resolve()
    pattern = str(candidate)
    contains_wildcard = any(ch in pattern for ch in _GLOB_CHARS)
    if not contains_wildcard:
        path = _normalise_path(candidate, field=field, base_path=None, required=True)
        return str(path) if path is not None else None
    matches = [Path(match) for match in glob.glob(pattern)]
    files = [match for match in matches if match.is_file()]
    if not files:
        raise CoreConfigError(f"{field} '{value}' did not match any files")
    return pattern


def _normalise_string(value: Any, *, field: str, default: str) -> str:
    if value in (None, ""):
        return default
    if not isinstance(value, str):
        raise CoreConfigError(f"{field} must be a string")
    stripped = value.strip()
    if not stripped:
        raise CoreConfigError(f"{field} must be a non-empty string")
    return stripped


def _normalise_frequency(value: Any) -> str:
    freq = _normalise_string(
        value,
        field="data.frequency",
        default=_DEFAULT_FREQUENCY,
    ).upper()
    if freq not in _ALLOWED_FREQUENCIES:
        allowed = ", ".join(sorted(_ALLOWED_FREQUENCIES))
        raise CoreConfigError(
            f"data.frequency '{value}' is not supported (choose one of {allowed})"
        )
    return freq


def _coerce_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CoreConfigError(f"{field} must be numeric") from exc
    if parsed < 0:
        raise CoreConfigError(f"{field} cannot be negative")
    return parsed


def validate_core_config(
    payload: Mapping[str, Any], *, base_path: Path | None = None
) -> CoreConfig:
    """Validate the minimal configuration contract and return dataclasses."""

    if not isinstance(payload, Mapping):
        raise CoreConfigError("Configuration payload must be a mapping")

    data_section = _as_mapping(payload.get("data"), field="data")
    tracker = get_config_coverage_tracker()
    csv_path = _normalise_path(
        data_section.get("csv_path"),
        field="data.csv_path",
        base_path=base_path,
        required=False,
    )
    if tracker is not None:
        tracker.track_validated("data.csv_path")
    managers_glob = _normalise_glob(
        data_section.get("managers_glob"),
        field="data.managers_glob",
        base_path=base_path,
    )
    if tracker is not None:
        tracker.track_validated("data.managers_glob")
    if csv_path is None and managers_glob is None:
        raise CoreConfigError("Provide data.csv_path or data.managers_glob to locate return series")
    universe_path = _normalise_path(
        data_section.get("universe_membership_path"),
        field="data.universe_membership_path",
        base_path=base_path,
        required=False,
    )
    if tracker is not None:
        tracker.track_validated("data.universe_membership_path")
    date_column = _normalise_string(
        data_section.get("date_column", _DEFAULT_DATE_COLUMN),
        field="data.date_column",
        default=_DEFAULT_DATE_COLUMN,
    )
    if tracker is not None:
        tracker.track_validated("data.date_column")
    frequency = _normalise_frequency(data_section.get("frequency", _DEFAULT_FREQUENCY))
    if tracker is not None:
        tracker.track_validated("data.frequency")

    regime_section = _as_mapping(payload.get("regime", {}), field="regime")
    _normalise_string(
        regime_section.get("model", "binary_threshold"),
        field="regime.model",
        default="binary_threshold",
    )
    if tracker is not None:
        tracker.track_validated("regime.model")

    portfolio_section = _as_mapping(payload.get("portfolio"), field="portfolio")
    for removed in ("transaction_cost_bps", "slippage_bps"):
        if removed in portfolio_section:
            raise CoreConfigError(f"portfolio.{removed} was removed; use portfolio.cost_model")
    cost_model_section = portfolio_section.get("cost_model") or {}
    cost_model = _as_mapping(cost_model_section, field="portfolio.cost_model")
    for removed in ("bps_per_trade", "slippage_bps"):
        if removed in cost_model:
            raise CoreConfigError(
                f"portfolio.cost_model.{removed} was removed; use per_trade_bps and half_spread_bps"
            )
    per_trade_bps = _coerce_float(
        cost_model.get("per_trade_bps", _DEFAULT_PER_TRADE_COST),
        field="portfolio.cost_model.per_trade_bps",
    )
    if tracker is not None:
        tracker.track_validated("portfolio.cost_model.per_trade_bps")
    half_spread_bps = _coerce_float(
        cost_model.get("half_spread_bps", _DEFAULT_HALF_SPREAD),
        field="portfolio.cost_model.half_spread_bps",
    )
    if tracker is not None:
        tracker.track_validated("portfolio.cost_model.half_spread_bps")

    data_settings = DataSettings(
        csv_path=csv_path,
        managers_glob=managers_glob,
        date_column=date_column,
        frequency=frequency,
        universe_membership_path=universe_path,
    )
    cost_settings = CostSettings(
        per_trade_bps=per_trade_bps,
        half_spread_bps=half_spread_bps,
    )
    return CoreConfig(data=data_settings, costs=cost_settings)


def load_core_config(path: str | Path) -> CoreConfig:
    """Load a YAML configuration file and validate the lightweight schema."""

    cfg_path = Path(path).expanduser().resolve()
    text = cfg_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise CoreConfigError("Configuration files must contain a mapping at the top level")
    return validate_core_config(data, base_path=cfg_path.parent)
