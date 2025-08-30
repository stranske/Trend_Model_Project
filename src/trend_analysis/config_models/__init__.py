"""Configuration package initialization."""

# Note: Config and load are imported from the parent config.py module
# This creates a naming conflict that needs architectural resolution
# For now, we don't re-export Config and load to avoid circular imports

# Import new model classes from this package
from .models import (
    PresetConfig,
    ColumnMapping,
    ConfigurationState,
    load_preset,
    list_available_presets,
)

__all__ = [
    "PresetConfig",
    "ColumnMapping", 
    "ConfigurationState",
    "load_preset",
    "list_available_presets",
]
