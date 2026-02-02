"""Validation helpers for Monte Carlo strategy packs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from trend_analysis.config.model import TrendConfig
from trend_analysis.monte_carlo.strategy.variant import StrategyVariant


def _load_yaml_mapping(path: Path, label: str) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return payload


def validate_strategy_pack(path: Path, *, base_config_path: Path | None = None) -> list[str]:
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

    for idx, entry in enumerate(curated):
        try:
            if isinstance(entry, str):
                variant = StrategyVariant(name=entry)
            elif isinstance(entry, Mapping):
                variant = StrategyVariant(
                    name=entry.get("name"),
                    overrides=entry.get("overrides", {}),
                    tags=entry.get("tags", ()),
                )
            else:
                raise ValueError("entry must be a mapping or string")
        except ValueError as exc:
            errors.append(f"strategy_pack.curated[{idx}] invalid: {exc}")
            continue

        try:
            validated = variant.to_trend_config(base_config, base_path=base_path.parent)
        except ValueError as exc:
            errors.append(f"strategy_pack.curated[{idx}] invalid: {exc}")
            continue

        if not isinstance(validated, TrendConfig):
            errors.append(
                f"strategy_pack.curated[{idx}] invalid: expected TrendConfig, got {type(validated).__name__}"
            )

    return errors
