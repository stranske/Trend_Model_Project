"""Minimal configuration model used for startup validation.

The production configuration model in :mod:`trend_analysis.config.models`
remains the source of truth for the full schema.  This module defines a small
subset that captures the fields required to safely start the application.  We
leverage Pydantic so the same validation logic can run in both the command line
entry points and the Streamlit UI before the heavy pipeline code is invoked.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
_GLOB_CHARS = {"*", "?", "[", "]"}


def _resolve_path(value: str | os.PathLike[str], *, base_dir: Path | None) -> Path:
    """Resolve ``value`` relative to ``base_dir`` and ensure it exists.

    Parameters
    ----------
    value:
        Path-like value supplied by the configuration.  Strings are expanded to
        user directories before resolution.
    base_dir:
        Directory that should be treated as the root for relative paths.  When
        ``None`` the current working directory is used instead.
    """

    raw = Path(value).expanduser()
    if raw.is_absolute():
        path = raw.resolve()
    else:
        roots: list[Path] = []
        if base_dir is not None:
            roots.append(base_dir)
            roots.append(base_dir.parent)
        roots.append(Path.cwd())
        for root in roots:
            candidate = (root / raw).resolve()
            if candidate.exists():
                path = candidate
                break
        else:
            path = (base_dir or Path.cwd()) / raw
            path = path.resolve()
    if any(ch in str(raw) for ch in _GLOB_CHARS):
        # Globs are not supported because downstream readers expect a concrete
        # CSV file.  Raising here keeps the failure actionable.
        raise ValueError(
            f"Input path '{value}' contains wildcard characters. Provide a "
            "single CSV file instead of a glob pattern."
        )
    if not path.exists():
        raise ValueError(
            f"Input path '{value}' does not exist. Update the configuration or "
            "generate the dataset before launching the analysis."
        )
    if path.is_dir():
        raise ValueError(
            f"Input path '{value}' points to a directory. Provide the full path "
            "to the CSV file containing returns data."
        )
    return path


# ---------------------------------------------------------------------------
# Pydantic models covering the minimal runtime contract
# ---------------------------------------------------------------------------


class DataSettings(BaseModel):
    """Data input configuration validated at startup."""

    csv_path: Path = Field()
    date_column: str = Field()
    frequency: Literal["D", "W", "M", "ME"] = Field()

    model_config = ConfigDict(extra="ignore")

    @field_validator("csv_path", mode="before")
    @classmethod
    def _validate_csv_path(cls, value: Any, info: Any) -> Path:
        if value in (None, ""):
            raise ValueError("data.csv_path must point to the returns CSV file.")
        base_dir = None
        if info.context:
            base_dir = info.context.get("base_path")
        return _resolve_path(value, base_dir=base_dir)

    @field_validator("date_column")
    @classmethod
    def _validate_date_column(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "data.date_column must be a non-empty string identifying the date column."
            )
        return value

    @field_validator("frequency", mode="before")
    @classmethod
    def _normalize_frequency(cls, value: Any) -> str:
        if value is None:
            raise ValueError("data.frequency must be provided (D, W or M).")
        freq = str(value).strip().upper()
        if freq == "ME":
            # Allow existing configs that still rely on the month-end alias.
            return "ME"
        allowed = {"D", "W", "M"}
        if freq not in allowed:
            allowed_list = ", ".join(sorted(allowed))
            raise ValueError(
                f"data.frequency '{value}' is not supported. Choose one of {allowed_list}."
            )
        return freq


class PortfolioSettings(BaseModel):
    """Portfolio controls validated before running analyses."""

    rebalance_calendar: str
    max_turnover: float
    transaction_cost_bps: float

    model_config = ConfigDict(extra="ignore")

    @field_validator("rebalance_calendar")
    @classmethod
    def _validate_calendar(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "portfolio.rebalance_calendar must name a valid trading calendar (e.g. 'NYSE')."
            )
        return value

    @field_validator("max_turnover", mode="before")
    @classmethod
    def _validate_turnover(cls, value: Any) -> float:
        try:
            turnover = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("portfolio.max_turnover must be numeric.") from exc
        if turnover < 0:
            raise ValueError("portfolio.max_turnover cannot be negative.")
        if turnover > 1:
            raise ValueError(
                "portfolio.max_turnover must be between 0 and 1.0 inclusive to cap per-period turnover."
            )
        return turnover

    @field_validator("transaction_cost_bps", mode="before")
    @classmethod
    def _validate_cost(cls, value: Any) -> float:
        try:
            cost = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("portfolio.transaction_cost_bps must be numeric.") from exc
        if cost < 0:
            raise ValueError("portfolio.transaction_cost_bps cannot be negative.")
        return cost


class RiskSettings(BaseModel):
    """Risk target configuration for volatility control."""

    target_vol: float = Field()

    model_config = ConfigDict(extra="ignore")

    @field_validator("target_vol", mode="before")
    @classmethod
    def _validate_target(cls, value: Any) -> float:
        try:
            target = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("vol_adjust.target_vol must be numeric.") from exc
        if target <= 0:
            raise ValueError("vol_adjust.target_vol must be greater than zero.")
        return target


class TrendConfig(BaseModel):
    """Subset of configuration validated at application startup."""

    data: DataSettings
    portfolio: PortfolioSettings
    vol_adjust: RiskSettings

    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _resolve_config_path(candidate: str | os.PathLike[str] | None) -> Path:
    if candidate in (None, ""):
        env_override = os.environ.get("TREND_CONFIG") or os.environ.get("TREND_CFG")
        if env_override:
            candidate = env_override
        else:
            candidate = "demo.yml"
    path = Path(candidate)
    if not path.suffix:
        path = path.with_suffix(".yml")
    if not path.is_absolute():
        within_repo = _CONFIG_DIR / path
        if within_repo.exists():
            return within_repo.resolve()
    if path.exists():
        return path.resolve()
    raise FileNotFoundError(
        f"Configuration file '{candidate}' was not found. Provide an absolute path "
        f"or place the file inside '{_CONFIG_DIR}'."
    )


def validate_trend_config(data: dict[str, Any], *, base_path: Path) -> TrendConfig:
    """Validate ``data`` against :class:`TrendConfig` with helpful errors."""

    try:
        return TrendConfig.model_validate(data, context={"base_path": base_path})
    except ValidationError as exc:
        first_error = exc.errors()[0] if exc.errors() else {}
        message = first_error.get("msg") or str(exc)
        loc = first_error.get("loc") or ()
        if loc:
            joined = ".".join(str(part) for part in loc)
            if joined:
                message = f"{joined}: {message}"
        raise ValueError(message) from exc


def load_trend_config(
    candidate: str | os.PathLike[str] | None = None,
) -> tuple[TrendConfig, Path]:
    """Load and validate the minimal configuration model."""

    cfg_path = _resolve_config_path(candidate)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Configuration files must contain a mapping at the root level.")
    cfg = validate_trend_config(raw, base_path=cfg_path.parent)
    return cfg, cfg_path


__all__ = [
    "TrendConfig",
    "load_trend_config",
    "validate_trend_config",
    "DataSettings",
    "PortfolioSettings",
    "RiskSettings",
]
