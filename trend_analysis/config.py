"""Configuration loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar
from typing import Any

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Typed access to the YAML configuration."""

    defaults: ClassVar[Path] = (
        Path(__file__).resolve().parents[1] / "config" / "defaults.yml"
    )
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
    """Load configuration from ``path`` or the default location.

    Environment variable ``TREND_CFG`` overrides the builtin default when
    ``path`` is ``None``.
    """
    env_override = os.environ.get("TREND_CFG")
    if path is not None:
        cfg_path = Path(path)
    elif env_override is not None:
        cfg_path = Path(env_override)
    else:
        cfg_path = Config.defaults
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(**data)


__all__ = ["Config", "load"]
