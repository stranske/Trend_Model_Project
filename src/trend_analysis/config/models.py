"""Configuration models for Streamlit Configure page validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List
from collections.abc import Mapping

import yaml  # type: ignore[import-untyped]

# Conditional pydantic imports with fallback stubs
try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

    _HAS_PYDANTIC = True
except ImportError:
    # Fallback stubs when pydantic is not available
    _HAS_PYDANTIC = False

    class BaseModel:
        """Fallback BaseModel stub when pydantic is unavailable."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class ConfigDict:
        """Fallback ConfigDict stub."""

        def __init__(self, **kwargs):
            pass

    def Field(default=None, default_factory=None, **kwargs):
        """Fallback Field function."""
        if default_factory is not None:
            return default_factory()
        return default

    class ValidationInfo:
        """Fallback ValidationInfo stub."""

        def __init__(self, field_name=None):
            self.field_name = field_name

    def field_validator(*args, **kwargs):
        """Fallback field_validator decorator."""

        def decorator(func):
            return func

        return decorator


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
        raise ValueError("version must be a string")
    if len(v) == 0:
        # Match pydantic's wording for empty strings
        raise ValueError("String should have at least 1 character")
    if not v.strip():
        raise ValueError("Version field cannot be empty")
    return v


class Config(BaseModel):
    """Typed access to the YAML configuration."""

    model_config = ConfigDict(extra="ignore")
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
    @classmethod
    def _validate_version_type_and_value(cls, v: Any) -> str:
        """Validate version field type and value."""
        if v is None:
            raise ValueError("version field is required")
        if not isinstance(v, str):
            raise ValueError("version must be a string")
        if len(v) == 0:
            raise ValueError("String should have at least 1 character")
        if not v.strip():
            raise ValueError("Version field cannot be empty")
        return v

    if not _HAS_PYDANTIC:
        # When pydantic is not installed the validators above will not run.
        # Perform minimal validation in the ``__init__`` method to keep
        # behaviour consistent with the pydantic-backed model.
        def __init__(self, **kwargs: Any) -> None:  # type: ignore[override]
            version_value = kwargs.get("version")
            if version_value is None:
                raise ValueError("version field is required")
            _validate_version_value(version_value)
            super().__init__(**kwargs)

            v = getattr(self, "version", None)
            if not isinstance(v, str):
                raise ValueError("version must be a string")
            if len(v) == 0:
                raise ValueError("String should have at least 1 character")
            if not v.strip():
                raise ValueError("Version field cannot be empty")

            # Dynamically find all attributes whose default value is a dict
            dict_field_names = [
                name
                for name, value in type(self).__dict__.items()
                if isinstance(value, dict)
            ]
            for name in dict_field_names:
                val = getattr(self, name, {})
                if not isinstance(val, dict):
                    raise ValueError(f"{name} must be a dictionary")

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
    @classmethod
    def _ensure_dict(cls, v: Any, info: ValidationInfo) -> dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError(f"{info.field_name} must be a dictionary")
        return v


# Simple BaseModel that works without pydantic
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

    def __init__(
        self,
        date_column: str = "",
        return_columns: List[str] = None,
        benchmark_column: str | None = None,
        risk_free_column: str | None = None,
        column_display_names: Dict[str, str] = None,
        column_tickers: Dict[str, str] = None,
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
]
