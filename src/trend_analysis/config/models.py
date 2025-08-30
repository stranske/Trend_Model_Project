"""Configuration models for Streamlit Configure page validation."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any
import os
import yaml

from pydantic import BaseModel, Field, ConfigDict, StrictStr


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
        **kwargs: Any
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
            **kwargs
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
    config_dir = Path(__file__).parent.parent.parent.parent.parent / "config"
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
    config_dir = Path(__file__).parent.parent.parent.parent.parent / "config"

    if not config_dir.exists():
        return []

    presets = []
    for yml_file in config_dir.glob("*.yml"):
        if yml_file.name not in ["defaults.yml"]:  # Exclude defaults
            presets.append(yml_file.stem)

    return sorted(presets)


class Config(BaseModel):
    """Typed access to the YAML configuration."""

    model_config = ConfigDict(extra="forbid")

    version: StrictStr
    data: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    vol_adjust: dict[str, Any] = Field(default_factory=dict)
    sample_split: dict[str, Any] = Field(default_factory=dict)
    portfolio: dict[str, Any] = Field(default_factory=dict)
    benchmarks: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    export: dict[str, Any] = Field(default_factory=dict)
    run: dict[str, Any] = Field(default_factory=dict)
    multi_period: dict[str, Any] | None = None
    jobs: int | None = None
    checkpoint_dir: str | None = None
    random_seed: int | None = None


def _find_config_directory() -> Path:
    """Find config directory by searching up from current file.
    
    This provides a more robust alternative to hardcoded parent navigation.
    Searches for a 'config' directory starting from the current file location
    and working up the directory tree, but skips the config package directory itself.
    
    Returns:
        Path to the config directory
        
    Raises:
        FileNotFoundError: If config directory cannot be found
    """
    current = Path(__file__).resolve()
    current_config_package = current.parent  # Skip the config package directory itself
    
    # Search up the directory tree for a config directory
    for parent in current.parents:
        # Skip the config package directory itself
        if parent == current_config_package:
            continue
            
        config_dir = parent / "config"
        if config_dir.is_dir() and (config_dir / "defaults.yml").exists():
            return config_dir
    
    # Fallback: search all parent directories for a "config" directory with "defaults.yml"
    for parent in current.parents:
        config_dir = parent / "config"
        if config_dir.is_dir() and (config_dir / "defaults.yml").exists():
            return config_dir
    raise FileNotFoundError(
        f"Could not find 'config' directory with defaults.yml in any parent of {current}. "
        "Please ensure the config directory exists in the project structure."
    )


DEFAULTS = _find_config_directory() / "defaults.yml"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TypeError("Config file must contain a mapping")
    return data


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_config(src: str | Path | dict[str, Any] | None = None) -> Config:
    """Load configuration from ``src`` applying defaults and validation.

    ``src`` may be a path to a YAML file, a pre-parsed dictionary or ``None`` to
    load the default configuration.  Environment variable ``TREND_CFG`` is
    honoured when ``src`` is ``None``.
    """

    if src is None:
        env = os.environ.get("TREND_CFG")
        data = _read_yaml(Path(env) if env else DEFAULTS)
    else:
        if isinstance(src, (str, Path)):
            user_data = _read_yaml(Path(src))
        elif isinstance(src, dict):
            user_data = src
        else:  # pragma: no cover - defensive
            raise TypeError("src must be path or mapping")
        
        # Process output section BEFORE merging defaults to avoid conflicts
        user_export = user_data.get("export", {})
        out_cfg = user_data.pop("output", None)
        if isinstance(out_cfg, dict):
            fmt = out_cfg.get("format")
            if fmt:
                # formats from output always override (consistent with legacy behavior)
                user_export["formats"] = [fmt] if isinstance(fmt, str) else list(fmt)
            path_val = out_cfg.get("path")
            if path_val:
                p = Path(path_val)
                # Only set directory/filename from output if not explicitly provided in export
                if "directory" not in user_export:
                    user_export["directory"] = str(p.parent) if p.parent else "."
                if "filename" not in user_export:
                    user_export["filename"] = p.name
        
        # Update user_data with processed export
        if user_export:
            user_data["export"] = user_export
            
        data = _deep_update(_read_yaml(DEFAULTS), user_data)

    return Config.model_validate(data)


# Backwards compatible name
load = load_config


__all__ = [
    "PresetConfig",
    "ColumnMapping",
    "ConfigurationState",
    "load_preset",
    "list_available_presets",
    "Config",
    "load_config",
    "load",
    "DEFAULTS",
]
