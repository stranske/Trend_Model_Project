"""Configuration loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # pragma: no cover - mypy only

    class BaseModel:
        """Minimal subset of :class:`pydantic.BaseModel` for type checking."""

        def __init__(self, **data: Any) -> None:  # noqa: D401
            ...

        def model_dump_json(self) -> str:  # noqa: D401
            ...

else:  # pragma: no cover - fallback when pydantic isn't installed during CI
    try:  # pragma: no cover - runtime import
        from pydantic import BaseModel as BaseModel
    except Exception:  # pragma: no cover - simplified stub

        class BaseModel:
            """Runtime stub used when ``pydantic`` is unavailable."""

            def __init__(self, **data: Any) -> None:
                pass

            def model_dump_json(self) -> str:
                return "{}"


class Config(BaseModel):
    """Typed access to the YAML configuration."""

    version: str
    data: dict[str, Any]
    preprocessing: dict[str, Any]
    vol_adjust: dict[str, Any]
    sample_split: dict[str, Any]
    portfolio: dict[str, Any]
    benchmarks: dict[str, str] = {}
    signals: dict[str, Any] = {}
    performance: dict[str, Any] = {}
    metrics: dict[str, Any]
    export: dict[str, Any]
    output: dict[str, Any] | None = None
    run: dict[str, Any]
    multi_period: dict[str, Any] | None = None
    regime: dict[str, Any] | None = None
    robustness: dict[str, Any] | None = None
    jobs: int | None = None
    checkpoint_dir: str | None = None
    random_seed: int | None = None
    seed: int | None = None

    def __init__(self, **data: Any) -> None:  # pragma: no cover - simple assign
        """Populate attributes from ``data`` regardless of ``BaseModel``."""
        super().__init__(**data)
        for key, value in data.items():
            setattr(self, key, value)

    def model_dump_json(self) -> str:  # pragma: no cover - trivial
        import json

        return json.dumps(self.__dict__)

    # Provide a lightweight ``dict`` representation for tests.
    def model_dump(self) -> dict[str, Any]:  # pragma: no cover - trivial
        return dict(self.__dict__)


DEFAULTS = Path(__file__).resolve().parents[3] / "config" / "defaults.yml"


def load(path: str | Path | None = None) -> Config:
    """Load configuration from ``path`` or ``DEFAULTS``.

    If ``path`` is ``None``, the ``TREND_CFG`` environment variable is
    consulted before falling back to ``DEFAULTS``.
    """
    if path is None:
        env = os.environ.get("TREND_CFG")
        cfg_path = Path(env) if env else DEFAULTS
    else:
        cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise TypeError("Config file must contain a mapping")

    out_cfg = data.pop("output", None)
    if isinstance(out_cfg, dict):
        export_cfg = data.setdefault("export", {})
        fmt = out_cfg.get("format")
        if fmt:
            export_cfg["formats"] = [fmt] if isinstance(fmt, str) else list(fmt)
        path_val = out_cfg.get("path")
        if path_val:
            p = Path(path_val)
            export_cfg.setdefault("directory", str(p.parent) if p.parent else ".")
            export_cfg.setdefault("filename", p.name)

    return Config(**data)


__all__ = ["Config", "load"]
