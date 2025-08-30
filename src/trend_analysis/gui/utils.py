from __future__ import annotations

import asyncio
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

# Define CONFIG_DIR constant for maintainable path resolution
# Navigate from src/trend_analysis/gui/utils.py to config/ directory
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "config"


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
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yml"))


__all__ = ["debounce", "list_builtin_cfgs"]
