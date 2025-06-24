"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Typed access to the YAML configuration."""

    version: str
    data: dict[str, Any]
    preprocessing: dict[str, Any]
    vol_adjust: dict[str, Any]
    sample_split: dict[str, Any]
    portfolio: dict[str, Any]
    metrics: dict[str, Any]
    export: dict[str, Any]
    run: dict[str, Any]


DEFAULTS = Path(__file__).resolve().parents[1] / "config" / "defaults.yml"


def load(path: str | Path | None = None) -> Config:
    """Load configuration from ``path`` or ``DEFAULTS``."""
    cfg_path: Path = Path(path) if path is not None else DEFAULTS
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise TypeError("Config file must contain a mapping")
    return Config(**data)


__all__ = ["Config", "load"]
