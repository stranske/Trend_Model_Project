"""Configuration models for Streamlit Configure page validation."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any
import yaml


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
            "name": "",
            "description": "",
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

    date_column: str
    return_columns: List[str]
    benchmark_column: str | None
    risk_free_column: str | None
    column_display_names: Dict[str, str]
    column_tickers: Dict[str, str]

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


__all__ = [
    "PresetConfig",
    "ColumnMapping",
    "ConfigurationState",
    "load_preset",
    "list_available_presets",
]
