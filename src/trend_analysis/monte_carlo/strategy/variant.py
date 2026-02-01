"""Strategy variant overrides for Monte Carlo scenarios."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from trend_analysis.config.model import TrendConfig, validate_trend_config


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _ensure_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _coerce_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        return ()
    cleaned: list[str] = []
    for tag in values:
        label = str(tag).strip()
        if label:
            cleaned.append(label)
    return tuple(cleaned)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _deep_merge_overrides(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
    path: tuple[str, ...],
) -> dict[str, Any]:
    merged: dict[str, Any] = deepcopy(dict(base))
    for raw_key, override_value in overrides.items():
        key = raw_key if isinstance(raw_key, str) else str(raw_key)
        next_path = path + (key,)
        path_label = _format_path(next_path)
        if key not in merged:
            raise ValueError(f"override path '{path_label}' does not exist in base config")
        base_value = merged[key]
        if isinstance(override_value, Mapping):
            if not isinstance(base_value, Mapping):
                raise TypeError(
                    "override path '{path}' expects mapping, found {kind}".format(
                        path=path_label,
                        kind=type(base_value).__name__,
                    )
                )
            merged[key] = _deep_merge_overrides(base_value, override_value, next_path)
            continue

        if isinstance(base_value, Mapping):
            raise TypeError(
                "override path '{path}' expects mapping, got {kind}".format(
                    path=path_label,
                    kind=type(override_value).__name__,
                )
            )

        if isinstance(base_value, list):
            if not isinstance(override_value, list):
                raise TypeError(
                    "override path '{path}' expects list, got {kind}".format(
                        path=path_label,
                        kind=type(override_value).__name__,
                    )
                )
        elif isinstance(base_value, bool):
            if not isinstance(override_value, bool):
                raise TypeError(
                    "override path '{path}' expects bool, got {kind}".format(
                        path=path_label,
                        kind=type(override_value).__name__,
                    )
                )
        elif _is_number(base_value):
            if not _is_number(override_value):
                raise TypeError(
                    "override path '{path}' expects number, got {kind}".format(
                        path=path_label,
                        kind=type(override_value).__name__,
                    )
                )
        elif base_value is not None and not isinstance(override_value, type(base_value)):
            raise TypeError(
                "override path '{path}' expects {expected}, got {actual}".format(
                    path=path_label,
                    expected=type(base_value).__name__,
                    actual=type(override_value).__name__,
                )
            )

        merged[key] = deepcopy(override_value)
    return merged


@dataclass(frozen=True)
class StrategyVariant:
    """Strategy variant that overrides a base configuration."""

    name: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_non_empty_str(self.name, "name"))
        overrides = self.overrides or {}
        object.__setattr__(self, "overrides", _ensure_mapping(overrides, "overrides"))
        object.__setattr__(self, "tags", _coerce_tags(self.tags))

    def apply_to(self, base_config: Mapping[str, Any] | TrendConfig) -> dict[str, Any]:
        """Return the base config with overrides applied via deep merge."""

        base: Mapping[str, Any] | TrendConfig = base_config
        if isinstance(base, TrendConfig):
            base = base.model_dump()
        base = _ensure_mapping(base, "base_config")
        return _deep_merge_overrides(base, self.overrides, ())

    def to_trend_config(
        self, base_config: Mapping[str, Any] | TrendConfig, *, base_path: Path | str
    ) -> TrendConfig:
        """Return a validated TrendConfig derived from the merged configuration."""

        try:
            merged = self.apply_to(base_config)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Strategy '{self.name}' overrides invalid: {exc}") from exc

        resolved_base = base_path if isinstance(base_path, Path) else Path(str(base_path))
        try:
            return validate_trend_config(merged, base_path=resolved_base)
        except ValueError as exc:
            raise ValueError(f"Strategy '{self.name}' config invalid: {exc}") from exc


__all__ = ["StrategyVariant"]
