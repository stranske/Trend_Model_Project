"""Validation helpers for Monte Carlo strategy packs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from trend_analysis.config.model import TrendConfig, validate_trend_config
from trend_analysis.config.schema_validation import load_schema, validate_config_data
from trend_analysis.monte_carlo.strategy.variant import StrategyVariant

_ALLOWED_ENTRY_KEYS = {"name", "overrides", "tags"}


def _load_yaml_mapping(path: Path, label: str) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return payload


def validate_strategy_pack(
    path: Path, *, base_config_path: Path | None = None
) -> list[str]:
    """Validate a strategy-pack YAML file against the base config schema.

    Returns a list of validation errors. An empty list indicates success.
    """

    errors: list[str] = []
    base_path = base_config_path or Path("config/defaults.yml")

    try:
        base_config = _load_yaml_mapping(base_path, "base_config")
    except (OSError, ValueError) as exc:
        return [f"base_config: {exc}"]

    try:
        payload = _load_yaml_mapping(path, "strategy_pack")
    except (OSError, ValueError) as exc:
        return [f"strategy_pack: {exc}"]

    if "curated" not in payload:
        return ["strategy_pack.curated is required"]

    curated = payload.get("curated")
    if not isinstance(curated, list):
        return ["strategy_pack.curated must be a list"]

    schema = load_schema()

    for idx, entry in enumerate(curated):
        try:
            if isinstance(entry, str):
                variant = StrategyVariant(name=entry, curated=True)
            elif isinstance(entry, Mapping):
                if "name" not in entry:
                    raise ValueError("name is required")
                extra_keys = set(entry.keys()) - _ALLOWED_ENTRY_KEYS
                if extra_keys:
                    extra_label = ", ".join(sorted(str(key) for key in extra_keys))
                    raise ValueError(f"unsupported keys: {extra_label}")
                raw_tags = entry.get("tags", ())
                if isinstance(raw_tags, Mapping):
                    raise ValueError("tags must be a string or sequence of strings")
                raw_name = entry.get("name")
                if not isinstance(raw_name, str):
                    raise ValueError("name must be a string")
                variant = StrategyVariant(
                    name=raw_name,
                    overrides=entry.get("overrides", {}),
                    tags=entry.get("tags", ()),
                    curated=True,
                )
            else:
                raise ValueError("entry must be a mapping or string")
        except ValueError as exc:
            errors.append(f"strategy_pack.curated[{idx}] invalid: {exc}")
            continue

        base_snapshot = deepcopy(base_config)
        base_working = deepcopy(base_config)
        try:
            merged = variant.apply_to(base_working)
        except (TypeError, ValueError) as exc:
            errors.append(f"strategy_pack.curated[{idx}] invalid: {exc}")
            continue

        if base_working != base_snapshot:
            errors.append(
                "strategy_pack.curated[{idx}] invalid: base_config mutated during validation".format(
                    idx=idx
                )
            )

        schema_errors = validate_config_data(merged, schema)
        for issue in schema_errors:
            errors.append(f"strategy_pack.curated[{idx}] invalid: {issue}")

        try:
            validated = validate_trend_config(merged, base_path=base_path.parent)
        except ValueError as exc:
            errors.append(f"strategy_pack.curated[{idx}] invalid: {exc}")
            continue

        if not isinstance(validated, TrendConfig):
            errors.append(
                f"strategy_pack.curated[{idx}] invalid: expected TrendConfig, got {type(validated).__name__}"
            )

    return errors
