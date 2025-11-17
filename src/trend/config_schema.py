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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "CoreConfig",
    "CoreConfigError",
    "CostSettings",
    "DataSettings",
    "load_core_config",
    "validate_core_config",
]

_ALLOWED_FREQUENCIES = {"D", "W", "M", "ME"}
_DEFAULT_BASE = Path.cwd()
_DEFAULT_DATE_COLUMN = "Date"
_DEFAULT_FREQUENCY = "M"
_DEFAULT_TRANSACTION_COST = 0.0
_DEFAULT_SLIPPAGE = 0.0


class CoreConfigError(ValueError):
    """Raised when the lightweight configuration contract is violated."""


@dataclass(frozen=True, slots=True)
class DataSettings:
    """Resolved data paths and time-series settings."""

    csv_path: Path
    date_column: str
    frequency: str
    universe_membership_path: Path | None


@dataclass(frozen=True, slots=True)
class CostSettings:
    """Portfolio cost parameters validated at startup."""

    transaction_cost_bps: float
    bps_per_trade: float
    slippage_bps: float


@dataclass(frozen=True, slots=True)
class CoreConfig:
    """Bundle of the minimal configuration knobs used by the CLI/UI."""

    data: DataSettings
    costs: CostSettings

    def to_payload(self) -> dict[str, Any]:
        """Serialise the validated configuration back to simple dictionaries."""

        membership = (
            str(self.data.universe_membership_path)
            if self.data.universe_membership_path is not None
            else None
        )
        return {
            "data": {
                "csv_path": str(self.data.csv_path),
                "universe_membership_path": membership,
                "date_column": self.data.date_column,
                "frequency": self.data.frequency,
            },
            "portfolio": {
                "transaction_cost_bps": self.costs.transaction_cost_bps,
                "cost_model": {
                    "bps_per_trade": self.costs.bps_per_trade,
                    "slippage_bps": self.costs.slippage_bps,
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
    csv_path = _normalise_path(
        data_section.get("csv_path"),
        field="data.csv_path",
        base_path=base_path,
        required=True,
    )
    universe_path = _normalise_path(
        data_section.get("universe_membership_path"),
        field="data.universe_membership_path",
        base_path=base_path,
        required=False,
    )
    date_column = _normalise_string(
        data_section.get("date_column", _DEFAULT_DATE_COLUMN),
        field="data.date_column",
        default=_DEFAULT_DATE_COLUMN,
    )
    frequency = _normalise_frequency(data_section.get("frequency", _DEFAULT_FREQUENCY))

    portfolio_section = _as_mapping(payload.get("portfolio"), field="portfolio")
    transaction_cost = _coerce_float(
        portfolio_section.get("transaction_cost_bps", _DEFAULT_TRANSACTION_COST),
        field="portfolio.transaction_cost_bps",
    )
    cost_model_section = portfolio_section.get("cost_model") or {}
    cost_model = _as_mapping(cost_model_section, field="portfolio.cost_model")
    bps_per_trade = _coerce_float(
        cost_model.get("bps_per_trade", transaction_cost),
        field="portfolio.cost_model.bps_per_trade",
    )
    slippage_bps = _coerce_float(
        cost_model.get("slippage_bps", _DEFAULT_SLIPPAGE),
        field="portfolio.cost_model.slippage_bps",
    )

    data_settings = DataSettings(
        csv_path=csv_path,
        date_column=date_column,
        frequency=frequency,
        universe_membership_path=universe_path,
    )
    cost_settings = CostSettings(
        transaction_cost_bps=transaction_cost,
        bps_per_trade=bps_per_trade,
        slippage_bps=slippage_bps,
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
