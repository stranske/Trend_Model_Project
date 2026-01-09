"""Tool layer wrappers for common config operations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
from pydantic import BaseModel, ConfigDict

from trend_analysis import api
from trend_analysis.config import ConfigType, load_config
from trend_analysis.config.patch import ConfigPatch, diff_configs
from trend_analysis.config.patch import apply_patch as apply_config_patch
from trend_analysis.config.validation import ValidationResult, validate_config
from trend_analysis.data import load_csv


@dataclass(slots=True)
class ToolLayer:
    """Provide a minimal interface for deterministic tool calls."""

    def _wrap_result(self, func: Callable[[], Any]) -> ToolResult:
        start = time.perf_counter()
        try:
            data = func()
            success = True
            error = None
        except Exception as exc:
            data = None
            success = False
            message = str(exc)
            error = f"{type(exc).__name__}: {message}" if message else type(exc).__name__

        elapsed_ms = (time.perf_counter() - start) * 1000
        return ToolResult(
            success=success,
            data=data,
            error=error,
            elapsed_ms=elapsed_ms,
        )

    def apply_patch(
        self,
        config: Mapping[str, Any],
        patch: Mapping[str, Any] | ConfigPatch,
    ) -> ToolResult:
        """Apply a config patch and return the updated config mapping."""

        def _execute() -> dict[str, Any]:
            if not isinstance(config, Mapping):
                raise TypeError("config must be a mapping")

            if isinstance(patch, ConfigPatch):
                patch_obj = patch
            else:
                patch_obj = ConfigPatch.model_validate(patch)

            return apply_config_patch(dict(config), patch_obj)

        return self._wrap_result(_execute)

    def validate_config(
        self,
        config: Mapping[str, Any],
        *,
        base_path: Path | None = None,
        strict: bool = False,
        skip_required_fields: bool = False,
    ) -> ToolResult:
        """Validate a configuration payload and return structured results."""

        def _execute() -> ValidationResult:
            if not isinstance(config, Mapping):
                raise TypeError("config must be a mapping")

            return validate_config(
                dict(config),
                base_path=base_path,
                strict=strict,
                skip_required_fields=skip_required_fields,
            )

        return self._wrap_result(_execute)

    def preview_diff(
        self,
        config: Mapping[str, Any],
        patch: Mapping[str, Any] | ConfigPatch,
    ) -> ToolResult:
        """Preview the unified diff for applying a patch to a config mapping."""

        def _execute() -> str:
            if not isinstance(config, Mapping):
                raise TypeError("config must be a mapping")

            if isinstance(patch, ConfigPatch):
                patch_obj = patch
            else:
                patch_obj = ConfigPatch.model_validate(patch)

            original = dict(config)
            updated = apply_config_patch(original, patch_obj)
            return diff_configs(original, updated)

        return self._wrap_result(_execute)

    def run_analysis(
        self,
        config: Mapping[str, Any] | ConfigType,
        *,
        data: pd.DataFrame | None = None,
    ) -> ToolResult:
        """Run the analysis pipeline using config and optional returns data."""

        def _execute() -> Any:
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

                data_frame = load_csv(
                    str(csv_path),
                    errors="raise",
                    missing_policy=missing_policy_cfg,
                    missing_limit=missing_limit_cfg,
                )
                if data_frame is None:
                    raise FileNotFoundError(str(csv_path))
            elif not isinstance(data, pd.DataFrame):
                raise TypeError("data must be a pandas DataFrame")
            else:
                data_frame = data

            return api.run_simulation(cfg_obj, data_frame)

        return self._wrap_result(_execute)


class ToolResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool
    data: Any | None
    error: str | None
    elapsed_ms: float


__all__ = ["ToolLayer", "ToolResult"]
