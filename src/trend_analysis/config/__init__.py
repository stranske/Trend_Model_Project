"""Configuration package initialization."""

# Re-export commonly used configuration models and helpers
from .models import (
    DEFAULTS,
    ColumnMapping,
    Config,
    ConfigType,
    ConfigurationState,
    PresetConfig,
    _find_config_directory,
    list_available_presets,
    load,
    load_config,
    load_preset,
)

# Removed import from .legacy as all symbols are available from .models

__all__ = [
    "PresetConfig",
    "ColumnMapping",
    "ConfigurationState",
    "load_preset",
    "list_available_presets",
    "load",
    "load_config",
    "Config",
    "ConfigType",
    "DEFAULTS",
    "_find_config_directory",
]
