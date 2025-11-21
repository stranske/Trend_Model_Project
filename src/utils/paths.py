"""Project-level path helpers.

``proj_path`` resolves paths relative to the repository root regardless of the
current working directory.  The root may be overridden via the environment
variable ``TREND_REPO_ROOT`` to support containerised or embedded deployments
that relocate the source tree.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

_ENV_REPO_ROOT = "TREND_REPO_ROOT"


def _expand_root(raw: str | os.PathLike[str]) -> Path:
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def repo_root() -> Path:
    """Return the repository root, honouring ``TREND_REPO_ROOT`` when set."""

    env_root = os.environ.get(_ENV_REPO_ROOT)
    if env_root:
        return _expand_root(env_root)
    return Path(__file__).resolve().parents[2]


def proj_path(*parts: str | os.PathLike[str]) -> Path:
    """Join ``parts`` onto the repository root."""

    base = repo_root()
    return base.joinpath(*parts) if parts else base


__all__ = ["proj_path", "repo_root"]
