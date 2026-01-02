"""Minimal configuration model used for startup validation.

The production configuration model in :mod:`trend_analysis.config.models`
remains the source of truth for the full schema.  This module defines a small
subset that captures the fields required to safely start the application.  We
leverage Pydantic so the same validation logic can run in both the command line
entry points and the Streamlit UI before the heavy pipeline code is invoked.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from utils.paths import proj_path

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
        ``None`` the repository root and current working directory are checked.
    """

    # Strip whitespace from string paths to handle copy-paste artifacts
    if isinstance(value, str):
        value = value.strip()

    raw = Path(value).expanduser()
    if raw.is_absolute():
        path = raw.resolve()
    else:
        roots: list[Path] = []
        if base_dir is not None:
            roots.append(base_dir)
            roots.append(base_dir.parent)
        repo_root = proj_path()
        if repo_root not in roots:
            roots.append(repo_root)
        cwd = Path.cwd()
        if cwd not in roots:
            roots.append(cwd)
        for root in roots:
            candidate = (root / raw).resolve()
            if candidate.exists():
                path = candidate
                break
        else:
            path = (base_dir or proj_path()) / raw
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
# Glob helpers
# ---------------------------------------------------------------------------


def _candidate_roots(base_dir: Path | None) -> Iterable[Path]:
    """Yield roots that should be considered when resolving relative paths."""

    seen: set[Path] = set()
    if base_dir is not None:
        for candidate in (base_dir, base_dir.parent):
            if candidate not in seen:
                seen.add(candidate)
                yield candidate
    repo_root = proj_path()
    if repo_root not in seen:
        seen.add(repo_root)
        yield repo_root
    cwd = Path.cwd()
    if cwd not in seen:
        seen.add(cwd)
        yield cwd


def _expand_pattern(pattern: str, *, base_dir: Path | None) -> list[Path]:
    """Expand ``pattern`` relative to plausible search roots."""

    raw_pattern = Path(os.path.expandvars(pattern)).expanduser()
    if raw_pattern.is_absolute():
        return [raw_pattern]

    expanded: list[Path] = []
    seen: set[Path] = set()
    for root in _candidate_roots(base_dir):
        candidate = root / raw_pattern
        # Avoid duplicates when base_dir and cwd are identical.
        if candidate in seen:
            continue
        seen.add(candidate)
        expanded.append(candidate)
    return expanded


def _ensure_glob_matches(pattern: str, *, base_dir: Path | None) -> None:
    """Ensure ``pattern`` matches at least one CSV file."""

    expanded = _expand_pattern(pattern, base_dir=base_dir)
    matched: list[Path] = []
    recursive = "**" in pattern
    for candidate in expanded:
        matches = glob.glob(str(candidate), recursive=recursive)
        matched.extend(Path(match) for match in matches)

    files = [path for path in matched if path.is_file()]
    if not files:
        base_hint = base_dir or proj_path()
        raise ValueError(
            "data.managers_glob did not match any CSV files. "
            f"Update the glob '{pattern}' relative to '{base_hint}' or "
            "generate the manager inputs before running the analysis."
        )
    csv_files = [path for path in files if path.suffix.lower() == ".csv"]
    if not csv_files:
        found = ", ".join(str(path.name) for path in files)
        raise ValueError(
            "data.managers_glob must resolve to CSV files. "
            f"The pattern '{pattern}' matched non-CSV inputs: {found}."
        )


# ---------------------------------------------------------------------------
# Pydantic models covering the minimal runtime contract
# ---------------------------------------------------------------------------


class DataSettings(BaseModel):
    """Data input configuration validated at startup."""

    csv_path: Path | None = Field(default=None)
    universe_membership_path: Path | None = Field(default=None)
    managers_glob: str | None = Field(default=None)
    date_column: str = Field()
    frequency: Literal["D", "W", "M", "ME"] = Field()
    missing_policy: str | Mapping[str, str] | None = Field(default=None)
    missing_limit: int | Mapping[str, int | None] | None = Field(default=None)
    risk_free_column: str | None = Field(default=None)
    allow_risk_free_fallback: bool | None = Field(default=None)

    model_config = ConfigDict(extra="ignore")

    @field_validator("csv_path", mode="before")
    @classmethod
    def _validate_csv_path(cls, value: Any, info: Any) -> Path | None:
        if value in (None, ""):
            return None
        base_dir = None
        if info.context:
            base_dir = info.context.get("base_path")
        return _resolve_path(value, base_dir=base_dir)

    @field_validator("managers_glob", mode="before")
    @classmethod
    def _validate_managers_glob(cls, value: Any, info: Any) -> str | None:
        if value in (None, ""):
            return None
        if isinstance(value, os.PathLike):
            pattern = str(Path(value))
        elif isinstance(value, str):
            pattern = value
        else:
            raise ValueError("data.managers_glob must be a string if provided.")
        base_dir = None
        if info.context:
            base_dir = info.context.get("base_path")
        if any(ch in pattern for ch in _GLOB_CHARS):
            _ensure_glob_matches(pattern, base_dir=base_dir)
            return pattern
        resolved = _resolve_path(pattern, base_dir=base_dir)
        return str(resolved)

    @field_validator("universe_membership_path", mode="before")
    @classmethod
    def _validate_membership_path(cls, value: Any, info: Any) -> Path | None:
        if value in (None, ""):
            return None
        base_dir = None
        if info.context:
            base_dir = info.context.get("base_path")
        return _resolve_path(value, base_dir=base_dir)

    @field_validator("risk_free_column", mode="before")
    @classmethod
    def _validate_risk_free_column(cls, value: Any) -> str | None:
        if value in (None, ""):
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        raise ValueError("data.risk_free_column must be a string when provided")

    @field_validator("allow_risk_free_fallback", mode="before")
    @classmethod
    def _validate_allow_risk_free_fallback(cls, value: Any) -> bool | None:
        if value in (None, ""):
            return False
        if isinstance(value, bool):
            return value
        raise ValueError(
            "data.allow_risk_free_fallback must be a boolean when provided"
        )

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

    @field_validator("missing_policy", mode="before")
    @classmethod
    def _validate_missing_policy(cls, value: Any) -> str | Mapping[str, str] | None:
        if value in (None, ""):
            return None
        if isinstance(value, (str, Mapping)):
            return value
        raise ValueError("data.missing_policy must be a string or mapping.")

    @field_validator("missing_limit", mode="before")
    @classmethod
    def _validate_missing_limit(
        cls, value: Any
    ) -> int | Mapping[str, int | None] | None:
        if value in (None, "", "null"):
            return None
        if isinstance(value, Mapping):
            return value
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "data.missing_limit must be an integer, mapping, or null."
            ) from exc

    @model_validator(mode="after")
    def _ensure_source(self) -> "DataSettings":
        if self.csv_path is None:
            managers = (self.managers_glob or "").strip()
            if not managers:
                raise ValueError(
                    "data.csv_path must point to the returns CSV file or provide data.managers_glob."
                )
        return self


class CostModelSettings(BaseModel):
    """Linear cost and slippage parameters."""

    bps_per_trade: float = Field(default=0.0)
    slippage_bps: float = Field(default=0.0)
    per_trade_bps: float | None = Field(default=None)
    half_spread_bps: float | None = Field(default=None)

    model_config = ConfigDict(extra="ignore")

    @field_validator("bps_per_trade", "slippage_bps", mode="before")
    @classmethod
    def _validate_cost(cls, value: Any, info: ValidationInfo[Any]) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"portfolio.cost_model.{info.field_name} must be numeric."
            ) from exc
        if parsed < 0:
            raise ValueError(
                f"portfolio.cost_model.{info.field_name} cannot be negative."
            )
        return parsed

    @field_validator("per_trade_bps", "half_spread_bps", mode="before")
    @classmethod
    def _validate_optional_cost(
        cls, value: Any, info: ValidationInfo[Any]
    ) -> float | None:
        if value in (None, "", "null"):
            return None
        return cls._validate_cost(value, info)


class PortfolioSettings(BaseModel):
    """Portfolio controls validated before running analyses."""

    rebalance_calendar: str
    rebalance_freq: str | None = Field(default=None)
    max_turnover: float
    transaction_cost_bps: float
    lambda_tc: float = Field(default=0.0)
    ci_level: float = Field(
        default=0.0,
        description="Reporting-only confidence interval level (0 disables CI annotations).",
    )
    cost_model: CostModelSettings | None = None
    turnover_cap: float | None = None
    weight_policy: dict[str, Any] | None = None
    cooldown_periods: int | None = None
    cooldown_months: int | None = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("rebalance_calendar")
    @classmethod
    def _validate_calendar(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "portfolio.rebalance_calendar must name a valid trading calendar (e.g. 'NYSE')."
            )
        return value

    @field_validator("rebalance_freq", mode="before")
    @classmethod
    def _validate_rebalance_freq(cls, value: Any) -> str | None:
        if value in (None, "", "null", "none"):
            return None
        if not isinstance(value, str):
            raise ValueError("portfolio.rebalance_freq must be a string")
        cleaned = value.strip()
        return cleaned if cleaned else None

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

    @field_validator("turnover_cap", mode="before")
    @classmethod
    def _validate_turnover_cap(cls, value: Any) -> float | None:
        if value in (None, "", "null"):
            return None
        return cls._validate_turnover(value)

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

    @field_validator("lambda_tc", mode="before")
    @classmethod
    def _validate_lambda_tc(cls, value: Any) -> float:
        if value in (None, "", "null"):
            return 0.0
        try:
            lam = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("portfolio.lambda_tc must be numeric.") from exc
        if lam < 0 or lam > 1:
            raise ValueError("portfolio.lambda_tc must be between 0 and 1 inclusive.")
        return lam

    @field_validator("cooldown_periods", "cooldown_months", mode="before")
    @classmethod
    def _validate_cooldown(cls, value: Any, info: ValidationInfo[Any]) -> int | None:
        if value in (None, "", "null"):
            return None
        try:
            cooldown = int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"portfolio.{info.field_name} must be an integer."
            ) from exc
        if cooldown < 0:
            raise ValueError(f"portfolio.{info.field_name} cannot be negative.")
        return cooldown

    @field_validator("ci_level", mode="before")
    @classmethod
    def _validate_ci_level(cls, value: Any) -> float:
        if value in (None, "", "null"):
            return 0.0
        try:
            level = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("portfolio.ci_level must be numeric.") from exc
        if level < 0:
            raise ValueError("portfolio.ci_level cannot be negative.")
        if level > 1:
            raise ValueError("portfolio.ci_level must be between 0 and 1 inclusive.")
        return level


class RiskSettings(BaseModel):
    """Risk target configuration for volatility control."""

    target_vol: float = Field()
    floor_vol: float = Field(default=0.015)
    warmup_periods: int = Field(default=0)

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

    @field_validator("floor_vol", mode="before")
    @classmethod
    def _validate_floor(cls, value: Any) -> float:
        try:
            floor = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("vol_adjust.floor_vol must be numeric.") from exc
        if floor < 0:
            raise ValueError("vol_adjust.floor_vol cannot be negative.")
        return floor

    @field_validator("warmup_periods", mode="before")
    @classmethod
    def _validate_warmup(cls, value: Any) -> int:
        try:
            warmup = int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("vol_adjust.warmup_periods must be an integer.") from exc
        if warmup < 0:
            raise ValueError("vol_adjust.warmup_periods cannot be negative.")
        return warmup


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
    candidate_value: str | os.PathLike[str]
    if candidate is None or candidate == "":
        env_override = os.environ.get("TREND_CONFIG") or os.environ.get("TREND_CFG")
        if env_override:
            candidate_value = env_override
        else:
            candidate_value = "demo.yml"
    else:
        candidate_value = candidate
    path = Path(candidate_value)
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
        errors = exc.errors()
        message = str(exc)
        if errors:
            first_error = errors[0]
            message = str(first_error.get("msg") or message)
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
