"""Configuration package initialization."""

# The config package needs to expose the main Config and load from config.py 
# We import using importlib to avoid circular import issues

import importlib.util
import sys
from pathlib import Path

def _import_config_module():
    """Import config.py module avoiding circular imports."""
    config_path = Path(__file__).parent / "../config.py"
    config_path = config_path.resolve()
    
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load config module")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module

_config_mod = _import_config_module()
Config = _config_mod.Config  
load = _config_mod.load

from .models import (
    PresetConfig,
    ColumnMapping,
    ConfigurationState,
    load_preset,
    list_available_presets,
)

__all__ = [
    "Config",
    "load", 
    "PresetConfig",
    "ColumnMapping",
    "ConfigurationState",
    "load_preset",
    "list_available_presets",
]
