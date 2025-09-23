"""Lightweight joblib stub for test execution."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def dump(obj: Any, filename: str | Path) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh)


def load(filename: str | Path) -> Any:
    with Path(filename).open("rb") as fh:
        return pickle.load(fh)
