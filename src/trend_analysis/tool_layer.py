"""Tool layer wrappers for common config operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from trend_analysis import api
from trend_analysis.config import ConfigType, load_config
from trend_analysis.config.patch import ConfigPatch, diff_configs
from trend_analysis.config.patch import apply_patch as apply_config_patch
from trend_analysis.config.validation import ValidationResult, validate_config
from trend_analysis.data import load_csv


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

    def preview_diff(
        self,
        config: Mapping[str, Any],
        patch: Mapping[str, Any] | ConfigPatch,
    ) -> str:
        """Preview the unified diff for applying a patch to a config mapping."""

        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")

        if isinstance(patch, ConfigPatch):
            patch_obj = patch
        else:
            patch_obj = ConfigPatch.model_validate(patch)

        original = dict(config)
        updated = apply_config_patch(original, patch_obj)
        return diff_configs(original, updated)

    def run_analysis(
        self,
        config: Mapping[str, Any] | ConfigType,
        *,
        data: pd.DataFrame | None = None,
    ) -> Any:
        """Run the analysis pipeline using config and optional returns data."""

        if isinstance(config, Mapping):
            cfg_obj = load_config(dict(config))
        elif hasattr(config, "model_dump") and hasattr(config, "data"):
            cfg_obj = config
        else:
            raise TypeError("config must be a mapping or Config object")

        if data is None:
            data_settings = getattr(cfg_obj, "data", {}) or {}
            csv_path = data_settings.get("csv_path")
            if csv_path is None:
                raise KeyError("cfg.data['csv_path'] must be provided")

            missing_policy_cfg = data_settings.get("missing_policy")
            if missing_policy_cfg is None:
                missing_policy_cfg = data_settings.get("nan_policy")
            missing_limit_cfg = data_settings.get("missing_limit")
            if missing_limit_cfg is None:
                missing_limit_cfg = data_settings.get("nan_limit")

            data = load_csv(
                str(csv_path),
                errors="raise",
                missing_policy=missing_policy_cfg,
                missing_limit=missing_limit_cfg,
            )
            if data is None:
                raise FileNotFoundError(str(csv_path))
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        return api.run_simulation(cfg_obj, data)


__all__ = ["ToolLayer"]
