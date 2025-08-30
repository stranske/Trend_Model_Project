"""Configuration models for Streamlit Configure page validation."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
import yaml
import os


# Pydantic compatibility layer
if TYPE_CHECKING:  # pragma: no cover - mypy only

    class BaseModel:
        """Minimal subset of :class:`pydantic.BaseModel` for type checking."""

        def __init__(self, **data: Any) -> None: ...

        def model_dump_json(self) -> str: ...

else:  # pragma: no cover - fallback when pydantic isn't installed during CI
    try:  # pragma: no cover - runtime import
        from pydantic import BaseModel as BaseModel
    except Exception:  # pragma: no cover - simplified stub

        class BaseModel:
            """Runtime stub used when ``pydantic`` is unavailable."""

            def __init__(self, **data: Any) -> None:
                pass

            def model_dump_json(self) -> str:
                return "{}"


# Simple BaseModel that works without pydantic
class SimpleBaseModel:
    """Simple base model for configuration validation."""

    def __init__(self, **data):
        # Set defaults first
        for field_name, default_value in self._get_defaults().items():
            setattr(self, field_name, default_value)

        # Then set provided data
        for key, value in data.items():
            setattr(self, key, value)

        # Run validation
        self._validate()

    def _get_defaults(self) -> Dict[str, Any]:
        """Override in subclasses to provide defaults."""
        return {}

    def _validate(self) -> None:
        """Override in subclasses to add validation."""
        pass

    def dict(self) -> dict:
        """Return dictionary representation."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class Config(BaseModel):
    """Typed access to the YAML configuration."""

    version: str
    data: dict[str, Any]
    preprocessing: dict[str, Any]
    vol_adjust: dict[str, Any]
    sample_split: dict[str, Any]
    portfolio: dict[str, Any]
    benchmarks: dict[str, str] = {}
    metrics: dict[str, Any]
    export: dict[str, Any]
    output: dict[str, Any] | None = None
    run: dict[str, Any]
    multi_period: dict[str, Any] | None = None
    jobs: int | None = None
    checkpoint_dir: str | None = None
    random_seed: int | None = None

    def __init__(self, **data: Any) -> None:  # pragma: no cover - simple assign
        """Populate attributes from ``data`` regardless of ``BaseModel``."""
        # Initialize all typed attributes to ensure they exist
        # Set defaults for attributes that don't have them in class definition
        defaults = {
            'version': '',
            'data': {},
            'preprocessing': {},
            'vol_adjust': {},
            'sample_split': {},
            'portfolio': {},
            'metrics': {},
            'export': {},
            'run': {},
        }
        
        # Merge defaults with provided data (data takes precedence)
        complete_data = {**defaults, **data}
        
        # Call the parent __init__ with complete data
        super().__init__(**complete_data)
        
        # Also set any additional attributes from data that might not be declared
        for key, value in data.items():
            setattr(self, key, value)

    def model_dump_json(self) -> str:  # pragma: no cover - trivial
        import json

        return json.dumps(self.__dict__)

    # Provide a lightweight ``dict`` representation for tests.
    def model_dump(self) -> dict[str, Any]:  # pragma: no cover - trivial
        return dict(self.__dict__)


class PresetConfig(SimpleBaseModel):
    """Configuration preset with validation."""

    def _get_defaults(self) -> Dict[str, Any]:
        return {
            "name": "",
            "description": "",
            "lookback_months": 36,
            "rebalance_frequency": "monthly",
            "min_track_months": 24,
            "selection_count": 10,
            "risk_target": 0.10,
            "metrics": {},
            "portfolio": {},
            "vol_adjust": {},
            "sample_split": {},
        }

    def _validate(self) -> None:
        """Validate preset configuration."""
        # Validate metrics weights
        if self.metrics:
            total = sum(self.metrics.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Metric weights must sum to 1.0, got {total}")

        # Validate positive integers
        for field in ["selection_count", "lookback_months", "min_track_months"]:
            value = getattr(self, field, 0)
            if value <= 0:
                raise ValueError(f"{field} must be a positive integer")

        # Validate risk target
        if not 0.01 <= self.risk_target <= 0.50:
            raise ValueError("Risk target must be between 1% and 50%")

        # Validate rebalance frequency
        valid_frequencies = {"monthly", "quarterly", "annually"}
        if self.rebalance_frequency not in valid_frequencies:
            raise ValueError(f"Rebalance frequency must be one of: {valid_frequencies}")


class ColumnMapping(SimpleBaseModel):
    """Column mapping configuration for uploaded data."""

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

    def _get_defaults(self) -> Dict[str, Any]:
        return {
            "preset_name": None,
            "preset_config": None,
            "column_mapping": None,
            "custom_overrides": {},
            "validation_errors": [],
            "is_valid": False,
        }

    def to_trend_config(self) -> Dict[str, Any]:
        """Convert to trend analysis configuration format."""
        if not self.preset_config:
            raise ValueError("No preset configuration loaded")

        # Start with preset config
        config = {
            "lookback_months": self.preset_config.lookback_months,
            "rebalance_frequency": self.preset_config.rebalance_frequency,
            "min_track_months": self.preset_config.min_track_months,
            "selection_count": self.preset_config.selection_count,
            "risk_target": self.preset_config.risk_target,
            "metrics": self.preset_config.metrics,
            "portfolio": self.preset_config.portfolio,
            "vol_adjust": self.preset_config.vol_adjust,
            "sample_split": self.preset_config.sample_split,
        }

        # Apply custom overrides
        for key, value in self.custom_overrides.items():
            config[key] = value

        # Add column mapping if available
        if self.column_mapping:
            config["column_mapping"] = self.column_mapping.dict()

        return config


def load_preset(preset_name: str) -> PresetConfig:
    """Load a preset configuration from file."""
    preset_path = (
        Path(__file__).parent.parent.parent.parent
        / "config"
        / "presets"
        / f"{preset_name.lower()}.yml"
    )

    if not preset_path.exists():
        raise FileNotFoundError(f"Preset '{preset_name}' not found at {preset_path}")

    with preset_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid preset format in {preset_path}")

    return PresetConfig(**data)


def list_available_presets() -> List[str]:
    """List all available preset names."""
    presets_dir = Path(__file__).parent.parent.parent.parent / "config" / "presets"

    if not presets_dir.exists():
        return []

    presets = []
    for preset_file in presets_dir.glob("*.yml"):
        presets.append(preset_file.stem.title())

    return sorted(presets)


DEFAULTS = Path(__file__).resolve().parents[3] / "config" / "defaults.yml"


def load(path: str | Path | None = None) -> Config:
    """Load configuration from ``path`` or ``DEFAULTS``.

    If ``path`` is ``None``, the ``TREND_CFG`` environment variable is
    consulted before falling back to ``DEFAULTS``.
    """
    if path is None:
        env = os.environ.get("TREND_CFG")
        cfg_path = Path(env) if env else DEFAULTS
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
    "DEFAULTS",
    "PresetConfig",
    "ColumnMapping",
    "ConfigurationState",
    "load_preset",
    "list_available_presets",
]
