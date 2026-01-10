"""Tool layer wrappers for common config operations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict

from trend_analysis import api
from trend_analysis.config import ConfigType, load_config
from trend_analysis.config.patch import ConfigPatch, diff_configs
from trend_analysis.config.patch import apply_patch as apply_config_patch
from trend_analysis.config.validation import ValidationResult, validate_config
from trend_analysis.data import load_csv
from utils.paths import proj_path

_SANDBOX_DIRS = ("config", "data", "outputs")
_TOOL_LOG_PATH = Path("outputs") / "logs" / "tool_calls.jsonl"
_REDACTED_VALUE = "***redacted***"
_SENSITIVE_KEYS = (
    "password",
    "token",
    "secret",
    "credential",
    "api_key",
    "apikey",
    "access_key",
)
_DEFAULT_RATE_LIMIT = 100


def _is_relative_to(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _should_redact(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _SENSITIVE_KEYS)


def _summarize_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, pd.DataFrame):
        return f"DataFrame(rows={len(value)}, cols={len(value.columns)})"
    if isinstance(value, pd.Series):
        return f"Series(len={len(value)})"
    if isinstance(value, Mapping):
        return f"mapping(keys={len(value)})"
    if isinstance(value, (list, tuple, set)):
        return f"sequence(len={len(value)})"
    text = str(value)
    if len(text) > 120:
        return f"{type(value).__name__}(len={len(text)})"
    return text


def _sanitize_value(value: Any, *, key: str | None = None) -> Any:
    if key and _should_redact(key):
        return _REDACTED_VALUE
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return _sanitize_value(value.model_dump(exclude_none=True))
    if isinstance(value, pd.DataFrame):
        return {"type": "DataFrame", "rows": len(value), "cols": len(value.columns)}
    if isinstance(value, pd.Series):
        return {"type": "Series", "length": len(value)}
    if isinstance(value, Mapping):
        return {str(k): _sanitize_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, bytes):
        return f"bytes[{len(value)}]"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ToolLogEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: float
    request_id: str
    tool: str
    parameters: dict[str, Any]
    output_summary: str
    status: str


@dataclass(slots=True)
class ToolLayer:
    """Provide a minimal interface for deterministic tool calls."""

    rate_limits: Mapping[str, int] | None = None
    default_rate_limit: int = _DEFAULT_RATE_LIMIT
    log_path: Path | None = None
    _call_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def _allowed_roots(self) -> tuple[Path, ...]:
        base = proj_path()
        return tuple((base / name).resolve() for name in _SANDBOX_DIRS)

    def _sandbox_path(self, path: str | Path) -> Path:
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or Path")

        candidate = Path(path).expanduser()
        if any(part == ".." for part in candidate.parts):
            raise ValueError("path traversal is not allowed")

        if not candidate.is_absolute():
            candidate = proj_path() / candidate

        resolved = candidate.resolve(strict=False)
        if not any(_is_relative_to(resolved, root) for root in self._allowed_roots()):
            raise ValueError("path is outside the sandbox")

        return resolved

    def _allowed_limit(self, tool_name: str) -> int:
        if self.rate_limits and tool_name in self.rate_limits:
            return int(self.rate_limits[tool_name])
        return int(self.default_rate_limit)

    def _enforce_rate_limit(self, tool_name: str) -> None:
        limit = self._allowed_limit(tool_name)
        count = self._call_counts.get(tool_name, 0) + 1
        self._call_counts[tool_name] = count
        if limit > 0 and count > limit:
            raise RuntimeError(f"rate limit exceeded for tool '{tool_name}' ({limit})")

    def _log_tool_call(
        self,
        *,
        tool: str,
        request_id: str,
        parameters: Mapping[str, Any],
        result: "ToolResult",
        timestamp: float,
    ) -> None:
        log_path = self.log_path or _TOOL_LOG_PATH
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = ToolLogEntry(
            timestamp=timestamp,
            request_id=request_id,
            tool=tool,
            parameters=_sanitize_value(dict(parameters)),
            output_summary=_summarize_value(result.data if result.status == "success" else result.message),
            status=result.status,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(payload.model_dump_json() + "\n")

    def _wrap_result(self, tool_name: str, parameters: Mapping[str, Any], func: Callable[[], Any]) -> ToolResult:
        start = time.perf_counter()
        request_id = uuid4().hex
        try:
            self._enforce_rate_limit(tool_name)
            data = func()
            status = "success"
            message = None
        except Exception as exc:
            data = None
            status = "error"
            message = str(exc) or type(exc).__name__

        elapsed_ms = (time.perf_counter() - start) * 1000
        result = ToolResult(status=status, message=message, data=data)
        self._log_tool_call(
            tool=tool_name,
            request_id=request_id,
            parameters=parameters,
            result=result,
            timestamp=time.time(),
        )
        return result

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

        return self._wrap_result("apply_patch", {"config": config, "patch": patch}, _execute)

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
                base_path=self._sandbox_path(base_path) if base_path is not None else None,
                strict=strict,
                skip_required_fields=skip_required_fields,
            )

        return self._wrap_result(
            "validate_config",
            {
                "config": config,
                "base_path": base_path,
                "strict": strict,
                "skip_required_fields": skip_required_fields,
            },
            _execute,
        )

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

        return self._wrap_result("preview_diff", {"config": config, "patch": patch}, _execute)

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

                resolved_csv = self._sandbox_path(csv_path)
                data_frame = load_csv(
                    str(resolved_csv),
                    errors="raise",
                    missing_policy=missing_policy_cfg,
                    missing_limit=missing_limit_cfg,
                )
                if data_frame is None:
                    raise FileNotFoundError(str(resolved_csv))
            elif not isinstance(data, pd.DataFrame):
                raise TypeError("data must be a pandas DataFrame")
            else:
                data_frame = data

            return api.run_simulation(cfg_obj, data_frame)

        return self._wrap_result("run_analysis", {"config": config, "data": data}, _execute)


class ToolResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: str
    message: str | None
    data: Any | None


__all__ = ["ToolLayer", "ToolResult"]
