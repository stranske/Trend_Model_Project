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
    apply_and_diff,
    apply_config_patch,
    apply_and_validate,
    apply_patch,
    diff_configs,
)
from .validation import ValidationError, ValidationResult, format_validation_messages, validate_config

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
    "apply_and_diff",
    "apply_and_validate",
    "apply_patch",
    "apply_config_patch",
    "diff_configs",
    "ValidationError",
    "ValidationResult",
    "validate_config",
    "format_validation_messages",
]
