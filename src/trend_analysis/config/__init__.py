"""Configuration package initialization."""

# Import the original config items first to maintain compatibility
try:
    from .. import config as _parent_config

    Config = _parent_config.Config
    load = _parent_config.load
except ImportError:
    # Fallback if parent config not available
    Config = None
    load = None

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
