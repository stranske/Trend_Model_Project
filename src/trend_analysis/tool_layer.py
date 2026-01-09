"""Tool layer wrappers for common config operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trend_analysis.config.patch import ConfigPatch, apply_patch as apply_config_patch


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


__all__ = ["ToolLayer"]
