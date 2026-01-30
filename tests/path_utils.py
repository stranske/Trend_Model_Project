from __future__ import annotations

from pathlib import Path


def is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True
