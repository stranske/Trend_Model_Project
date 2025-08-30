from __future__ import annotations

import asyncio
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any
import os


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


def debounce(wait_ms: int = 300) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that debounces async callbacks."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        last_call = 0.0

        handle: asyncio.Task[Any] | None = None

        async def fire(args: Any, kwargs: Any) -> None:
            await asyncio.sleep(wait_ms / 1000)
            if time.time() - last_call >= wait_ms / 1000:
                result = fn(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    await result

        async def wrapper(*args: Any, **kwargs: Any) -> None:
            nonlocal last_call, handle
            last_call = time.time()
            if handle is not None:
                handle.cancel()
            handle = asyncio.create_task(fire(args, kwargs))

        return wrapper

    return decorator


def list_builtin_cfgs() -> list[str]:
    """Return names of built-in YAML configs bundled with the package."""
    try:
        project_root = find_project_root()
        cfg_dir = project_root / "config"
    except FileNotFoundError:
        # Fallback to original hardcoded path if root detection fails
        cfg_dir = Path(__file__).resolve().parents[3] / "config"
    return sorted(p.stem for p in cfg_dir.glob("*.yml"))


__all__ = ["debounce", "list_builtin_cfgs"]
