"""Lightweight helper mirroring the tiny subset of :mod:`joblib` we rely on.

This module intentionally lives under the name ``joblib_shim.py``, so ``import joblib``
resolves to the real third-party package. Code that still needs the
lightweight pickle-based helpers should import them explicitly, e.g.
``from joblib_shim import dump``.
"""

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
