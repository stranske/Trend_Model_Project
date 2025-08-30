"""Configuration package initialization."""

# Import original config items from legacy module
from .legacy import Config, load

# Import new model classes
from .models import (
    PresetConfig,
    ColumnMapping,
    ConfigurationState,
    load_preset,
    list_available_presets,
)

__all__ = [
    "Config",
    "load",  # Original config items
    "PresetConfig", 
    "ColumnMapping",
    "ConfigurationState",
    "load_preset",
    "list_available_presets",
]