"""Configuration package initialization."""

# Re-export commonly used configuration models and helpers
from .model import TrendConfig, load_trend_config, validate_trend_config
from .models import (
    DEFAULTS,
    ColumnMapping,
    Config,
    ConfigType,
    ConfigurationState,
    PresetConfig,
    list_available_presets,
    load,
    load_config,
    load_preset,
)
from .patch import (
    ConfigPatch,
    PatchOperation,
    RiskFlag,
    apply_config_patch,
    apply_patch,
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
    "load_trend_config",
    "validate_trend_config",
    "Config",
    "ConfigType",
    "DEFAULTS",
    "TrendConfig",
    "ConfigPatch",
    "PatchOperation",
    "RiskFlag",
    "apply_patch",
    "apply_config_patch",
]
