"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING
import os

import yaml


def find_project_root(start_path: Path | None = None, markers: tuple[str, ...] | None = None) -> Path:
    """Find project root by searching for marker files up the directory tree.
    
    Args:
        start_path: Starting path for search (defaults to this file's directory)
        markers: File/directory names that indicate project root (defaults to common markers)
    
    Returns:
        Path to project root directory
        
    Raises:
        FileNotFoundError: If project root cannot be found
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent
    
    if markers is None:
        markers = ("pyproject.toml", "requirements.txt", ".git", "setup.py", "setup.cfg")
    
    # Try environment variable override first
    env_root = os.environ.get("TREND_PROJECT_ROOT")
    if env_root:
        root_path = Path(env_root)
        if root_path.is_dir():
            return root_path
    
    # Search up the directory tree
    current = start_path
    for _ in range(10):  # Safety limit to prevent infinite loops
        for marker in markers:
            if (current / marker).exists():
                return current
        
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    # Fallback: if we can't find markers, assume we're in a typical structure
    # and try to find a reasonable root based on known paths
    fallback_path = start_path
    while fallback_path != fallback_path.parent:
        # Look for config directory as a fallback indicator
        if (fallback_path / "config").is_dir() and (fallback_path / "src").is_dir():
            return fallback_path
        fallback_path = fallback_path.parent
    
    raise FileNotFoundError(
        f"Cannot find project root from {start_path}. "
        f"Tried markers: {markers}. "
        "Set TREND_PROJECT_ROOT environment variable as override."
    )


if TYPE_CHECKING:  # pragma: no cover - mypy only

    class BaseModel:
        """Minimal subset of :class:`pydantic.BaseModel` for type checking."""

        def __init__(self, **data: Any) -> None: ...

        def model_dump_json(self) -> str: ...

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
    metrics: dict[str, Any]
    export: dict[str, Any]
    output: dict[str, Any] | None = None
    run: dict[str, Any]
    multi_period: dict[str, Any] | None = None
    jobs: int | None = None
    checkpoint_dir: str | None = None
    random_seed: int | None = None

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


# Robust path resolution using project root detection
try:
    _PROJECT_ROOT = find_project_root()
    DEFAULTS = _PROJECT_ROOT / "config" / "defaults.yml"
except FileNotFoundError:
    # Fallback to original hardcoded path if root detection fails
    DEFAULTS = Path(__file__).resolve().parents[2] / "config" / "defaults.yml"


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


__all__ = ["Config", "load", "find_project_root", "DEFAULTS"]
