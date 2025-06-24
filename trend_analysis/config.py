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


def load(path: str | None = None) -> Config:
    """Load configuration from ``path`` or default ``config/defaults.yml``."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "defaults.yml"
    else:
        path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(**data)


__all__ = ["Config", "load"]
