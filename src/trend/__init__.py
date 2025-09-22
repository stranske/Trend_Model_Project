"""Public interface for the lightweight ``trend`` package.

The package exposes a modern command-line interface that orchestrates the
volatility-adjusted trend pipeline implemented in :mod:`trend_analysis`.
External callers primarily rely on :mod:`trend.cli` which in turn delegates to
the underlying analysis APIs.
"""

from __future__ import annotations

from importlib import metadata as _metadata


def __getattr__(name: str) -> str:
    """Provide ``__version__`` dynamically without importing setuptools."""

    if name == "__version__":
        try:
            return _metadata.version("trend-model")
        except _metadata.PackageNotFoundError:  # pragma: no cover - dev mode
            return "0.0.dev0"
    raise AttributeError(name)


__all__ = ["__version__"]

