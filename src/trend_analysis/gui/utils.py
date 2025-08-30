from __future__ import annotations

import asyncio
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any


def _find_config_directory() -> Path:
    """Find config directory by searching up from current file.
    
    This provides a more robust alternative to hardcoded parent navigation.
    Searches for a 'config' directory starting from the current file location
    and working up the directory tree.
    
    Returns:
        Path to the config directory
        
    Raises:
        FileNotFoundError: If config directory cannot be found
    """
    current = Path(__file__).resolve()
    
    # Search up the directory tree for a config directory
    for parent in current.parents:
        config_dir = parent / "config"
        if config_dir.is_dir() and (config_dir / "defaults.yml").exists():
            return config_dir
    
    # Fallback to the original hardcoded path for backward compatibility
    fallback_config = current.parents[3] / "config"
    if fallback_config.is_dir():
        return fallback_config
    
    raise FileNotFoundError(
        f"Could not find 'config' directory with defaults.yml in any parent of {current}. "
        "Please ensure the config directory exists in the project structure."
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
    cfg_dir = _find_config_directory()
    return sorted(p.stem for p in cfg_dir.glob("*.yml"))


__all__ = ["debounce", "list_builtin_cfgs"]
