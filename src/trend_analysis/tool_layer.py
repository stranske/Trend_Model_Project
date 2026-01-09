"""Tool layer wrappers for common config operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from trend_analysis.config.patch import ConfigPatch
from trend_analysis.config.patch import apply_patch as apply_config_patch
from trend_analysis.config.validation import ValidationResult, validate_config


@dataclass(slots=True)
class ToolLayer:
    """Provide a minimal interface for deterministic tool calls."""

    def apply_patch(
        self,
        config: Mapping[str, Any],
        patch: Mapping[str, Any] | ConfigPatch,
    ) -> dict[str, Any]:
        """Apply a config patch and return the updated config mapping."""

        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")

        if isinstance(patch, ConfigPatch):
            patch_obj = patch
        else:
            patch_obj = ConfigPatch.model_validate(patch)

        return apply_config_patch(dict(config), patch_obj)

    def validate_config(
        self,
        config: Mapping[str, Any],
        *,
        base_path: Path | None = None,
        strict: bool = False,
        skip_required_fields: bool = False,
    ) -> ValidationResult:
        """Validate a configuration payload and return structured results."""

        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")

        return validate_config(
            dict(config),
            base_path=base_path,
            strict=strict,
            skip_required_fields=skip_required_fields,
        )


__all__ = ["ToolLayer"]
