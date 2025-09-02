"""Configuration models for Streamlit Configure page validation.

This module supports environments with or without Pydantic installed.
Tests import the module twice (with and without Pydantic) and expect the
symbol ``_HAS_PYDANTIC`` to reflect availability.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, cast
from collections.abc import Mapping

import yaml

# Pydantic import (optional in tests)
# Use temporary underscored names within the branch, then export public names
try:  # pragma: no cover - exercised via tests toggling availability
    import pydantic as _pyd

    _BaseModel = _pyd.BaseModel
    _Field = _pyd.Field
    _ValidationInfo = _pyd.ValidationInfo
    _field_validator = _pyd.field_validator

    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover
    _BaseModel = object

    def _Field(*_args: Any, **_kwargs: Any) -> None:  # noqa: D401 - simple fallback
        """Fallback Field when Pydantic is unavailable."""
        return None

    _ValidationInfo = object

    def _field_validator(*_args: Any, **_kwargs: Any) -> Any:
        def _decorator(func: Any) -> Any:
            return func

        return _decorator

    _HAS_PYDANTIC = False

# Export names with broad Any types for static checkers
BaseModel: Any = cast(Any, _BaseModel)
Field: Any = cast(Any, _Field)
ValidationInfo: Any = cast(Any, _ValidationInfo)
field_validator: Any = cast(Any, _field_validator)


# Simple BaseModel that works without pydantic (used by fallback Config)
class SimpleBaseModel:
    """Simple base model for configuration validation."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with validation."""
        defaults = self._get_defaults()
        for key, value in defaults.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._validate()

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default values for this model."""
        return {}

    def _validate(self) -> None:
        """Validate the configuration."""
        pass


def _find_config_directory() -> Path:
    """Locate the project's configuration directory.

    Starting from this file's location, walk up the directory tree until a
    ``config`` directory containing ``defaults.yml`` is found. If no suitable
    directory is discovered, a :class:`FileNotFoundError` is raised.
    """

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "config"
        if candidate.is_dir() and (candidate / "defaults.yml").exists():
            return candidate

    raise FileNotFoundError("Could not find 'config' directory")


def _validate_version_value(v: Any) -> str:
    """Validate the ``version`` field for both pydantic and fallback modes."""
    if not isinstance(v, str):
        raise TypeError("version must be a string")
    if len(v) == 0:
        # Match pydantic's wording for empty strings
        raise ValueError("String should have at least 1 character")
    if not v.strip():
        raise ValueError("Version field cannot be empty")
    return v


if _HAS_PYDANTIC:
    # Cache class identity across re-imports to keep isinstance checks stable
    import builtins as _bi  # type: ignore

    _cached = getattr(_bi, "_TREND_CONFIG_CLASS", None)

    if _cached is not None:
        Config = _cached  # type: ignore[assignment]
    else:

        class Config(BaseModel):  # type: ignore[valid-type,misc]
            """Typed access to the YAML configuration (Pydantic mode)."""

            # Use a plain dict for model_config to avoid type-checker issues when
            # Pydantic is not installed (tests toggle availability).
            model_config = {"extra": "ignore"}
            # ``version`` must be a non-empty string. ``min_length`` handles the empty
            # string case and produces the standard pydantic error message
            # "String should have at least 1 character". A separate validator below
            # ensures the field isn't composed solely of whitespace.
            version: str = Field(min_length=1)
            data: dict[str, Any] = Field(default_factory=dict)
            preprocessing: dict[str, Any] = Field(default_factory=dict)
            vol_adjust: dict[str, Any] = Field(default_factory=dict)
            sample_split: dict[str, Any] = Field(default_factory=dict)
            portfolio: dict[str, Any] = Field(default_factory=dict)
            benchmarks: dict[str, str] = Field(default_factory=dict)
            metrics: dict[str, Any] = Field(default_factory=dict)
            export: dict[str, Any] = Field(default_factory=dict)
            output: dict[str, Any] | None = None
            run: dict[str, Any] = Field(default_factory=dict)
            multi_period: dict[str, Any] | None = None
            jobs: int | None = None
            checkpoint_dir: str | None = None
            seed: int = 42

            @field_validator("version", mode="before")
            def _ensure_version_str(cls, v: Any) -> str:
                """Ensure ``version`` is always a string."""
                if not isinstance(v, str):
                    raise TypeError("version must be a string")
                return v

            @field_validator("version")
            def _ensure_version_not_whitespace(cls, v: str) -> str:
                """Reject strings that consist only of whitespace."""
                if not v.strip():
                    raise ValueError("Version field cannot be empty")
                return v

            @field_validator(
                "data",
                "preprocessing",
                "vol_adjust",
                "sample_split",
                "portfolio",
                "metrics",
                "export",
                "run",
                mode="before",
            )
            def _ensure_dict(cls, v: Any, info: Any) -> dict[str, Any]:
                if not isinstance(v, dict):
                    raise TypeError(f"{info.field_name} must be a dictionary")
                return v

        setattr(_bi, "_TREND_CONFIG_CLASS", Config)
        # Keep package-level alias in sync for consistent isinstance checks
        import sys as _sys

        _pkg = _sys.modules.get("trend_analysis.config")
        if _pkg is not None:  # pragma: no cover
            setattr(_pkg, "Config", Config)
else:  # Fallback mode for tests without pydantic

    class Config(SimpleBaseModel):  # type: ignore[no-redef]
        """Simplified Config for environments without Pydantic."""

        # Attribute declarations for linters/type-checkers
        version: str
        data: Dict[str, Any]
        preprocessing: Dict[str, Any]
        vol_adjust: Dict[str, Any]
        sample_split: Dict[str, Any]
        portfolio: Dict[str, Any]
        benchmarks: Dict[str, str]
        metrics: Dict[str, Any]
        export: Dict[str, Any]
        output: Dict[str, Any] | None
        run: Dict[str, Any]
        multi_period: Dict[str, Any] | None
        jobs: int | None
        checkpoint_dir: str | None
        seed: int

        def _get_defaults(self) -> Dict[str, Any]:
            return {
                "data": {},
                "preprocessing": {},
                "vol_adjust": {},
                "sample_split": {},
                "portfolio": {},
                "benchmarks": {},
                "metrics": {},
                "export": {},
                "output": None,
                "run": {},
                "multi_period": None,
                "jobs": None,
                "checkpoint_dir": None,
                "seed": 42,
            }

        def _validate(self) -> None:  # Simple runtime validation
            if getattr(self, "version", None) is None:
                raise ValueError("version field is required")
            if not isinstance(self.version, str):
                raise TypeError("version must be a string")
            if len(self.version) == 0:
                raise ValueError("String should have at least 1 character")
            if not self.version.strip():
                raise ValueError("Version field cannot be empty")

        # Provide a similar API surface to pydantic for callers
        def model_dump(self) -> Dict[str, Any]:
            return {
                k: getattr(self, k)
                for k in [
                    "version",
                    "data",
                    "preprocessing",
                    "vol_adjust",
                    "sample_split",
                    "portfolio",
                    "benchmarks",
                    "metrics",
                    "export",
                    "output",
                    "run",
                    "multi_period",
                    "jobs",
                    "checkpoint_dir",
                    "seed",
                ]
            }

    # Keep package-level alias in sync when using fallback
    import sys as _sys

    _pkg = _sys.modules.get("trend_analysis.config")
    if _pkg is not None:  # pragma: no cover
        setattr(_pkg, "Config", Config)


class PresetConfig(SimpleBaseModel):
    """Configuration preset with validation."""

    name: str
    description: str
    data: Dict[str, Any]
    preprocessing: Dict[str, Any]
    vol_adjust: Dict[str, Any]
    sample_split: Dict[str, Any]
    portfolio: Dict[str, Any]
    metrics: Dict[str, Any]
    export: Dict[str, Any]
    run: Dict[str, Any]

    def _get_defaults(self) -> Dict[str, Any]:
        return {
            "data": {},
            "preprocessing": {},
            "vol_adjust": {},
            "sample_split": {},
            "portfolio": {},
            "metrics": {},
            "export": {},
            "run": {},
        }

    def _validate(self) -> None:
        """Validate preset configuration."""
        if not self.name or not self.name.strip():
            raise ValueError("Preset name must be specified")


class ColumnMapping(SimpleBaseModel):
    """Column mapping configuration for uploaded data."""

    # Attribute declarations for type checkers
    date_column: str
    return_columns: List[str]
    benchmark_column: str | None
    risk_free_column: str | None
    column_display_names: Dict[str, str]
    column_tickers: Dict[str, str]

    def __init__(
        self,
        date_column: str = "",
        return_columns: List[str] | None = None,
        benchmark_column: str | None = None,
        risk_free_column: str | None = None,
        column_display_names: Dict[str, str] | None = None,
        column_tickers: Dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        if return_columns is None:
            return_columns = []
        if column_display_names is None:
            column_display_names = {}
        if column_tickers is None:
            column_tickers = {}
        super().__init__(
            date_column=date_column,
            return_columns=return_columns,
            benchmark_column=benchmark_column,
            risk_free_column=risk_free_column,
            column_display_names=column_display_names,
            column_tickers=column_tickers,
            **kwargs,
        )

    def _get_defaults(self) -> Dict[str, Any]:
        return {
            "date_column": "",
            "return_columns": [],
            "benchmark_column": None,
            "risk_free_column": None,
            "column_display_names": {},
            "column_tickers": {},
        }

    def _validate(self) -> None:
        """Validate column mapping."""
        if not self.date_column or not self.date_column.strip():
            raise ValueError("Date column must be specified")

        if not self.return_columns:
            raise ValueError("At least one return column must be specified")


class ConfigurationState(SimpleBaseModel):
    """Complete configuration state for the Streamlit app."""

    preset_name: str
    column_mapping: ColumnMapping | None
    config_dict: Dict[str, Any]
    uploaded_data: Any
    analysis_results: Any

    def _get_defaults(self) -> Dict[str, Any]:
        return {
            "preset_name": "",
            "column_mapping": None,
            "config_dict": {},
            "uploaded_data": None,
            "analysis_results": None,
        }

    def _validate(self) -> None:
        """Validate configuration state."""
        pass


def load_preset(preset_name: str) -> PresetConfig:
    """Load a preset configuration from file."""
    # Find the config directory relative to this file
    config_dir = _find_config_directory()
    preset_file = config_dir / f"{preset_name}.yml"

    if not preset_file.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_file}")

    with preset_file.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise TypeError("Preset file must contain a mapping")

    data["name"] = preset_name
    return PresetConfig(**data)


def list_available_presets() -> List[str]:
    """List all available preset names."""
    config_dir = _find_config_directory()

    if not config_dir.exists():
        return []

    presets = []
    for yml_file in config_dir.glob("*.yml"):
        if yml_file.name not in ["defaults.yml"]:  # Exclude defaults
            presets.append(yml_file.stem)

    return sorted(presets)


DEFAULTS = Path(__file__).resolve().parents[3] / "config" / "defaults.yml"


def load_config(cfg: Mapping[str, Any] | str | Path) -> Config:
    """Load configuration from a mapping or file path."""
    if isinstance(cfg, (str, Path)):
        return load(cfg)
    if isinstance(cfg, Mapping):
        return Config(**cfg)
    raise TypeError("cfg must be a mapping or path")


def load(path: str | Path | None = None) -> Config:
    """Load configuration from ``path`` or ``DEFAULTS``.
    If ``path`` is ``None``, the ``TREND_CFG`` environment variable is
    consulted before falling back to ``DEFAULTS``.
    If ``path`` is a dict, it is used directly as configuration data.
    """
    if isinstance(path, dict):
        data = path.copy()
    elif path is None:
        env = os.environ.get("TREND_CFG")
        cfg_path = Path(env) if env else DEFAULTS
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            if not isinstance(data, dict):
                raise TypeError("Config file must contain a mapping")
    else:
        cfg_path = Path(path)
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            if not isinstance(data, dict):
                raise TypeError("Config file must contain a mapping")

    out_cfg = data.pop("output", None)
    if isinstance(out_cfg, dict):
        export_cfg = data.setdefault("export", {})
        fmt = out_cfg.get("format")
        if fmt:
            export_cfg["formats"] = [fmt] if isinstance(fmt, str) else list(fmt)
        path_val = out_cfg.get("path")
        if path_val:
            p = Path(path_val)
            export_cfg.setdefault("directory", str(p.parent) if p.parent else ".")
            export_cfg.setdefault("filename", p.name)

    return Config(**data)


__all__ = [
    "Config",
    "load",
    "PresetConfig",
    "ColumnMapping",
    "ConfigurationState",
    "load_preset",
    "list_available_presets",
    "load_config",
    "DEFAULTS",
    "_find_config_directory",
    "_HAS_PYDANTIC",
]
