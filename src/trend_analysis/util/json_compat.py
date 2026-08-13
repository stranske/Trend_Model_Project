"""Small, explicit JSON-compatible primitives shared by runtime adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

JSON_UNSUPPORTED = object()


def json_primitive(value: Any) -> Any:
    """Return a safe JSON primitive or a private unsupported sentinel.

    Callers retain ownership of their container shape (for example, a Series
    can be a list in a JSONL record but a keyed mapping in a CLI response).
    This helper only centralises scalar and universally safe conversions.
    """

    if value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        scalar = value.item()
        return None if isinstance(scalar, float) and not math.isfinite(scalar) else scalar
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (set, tuple)):
        return list(value)
    return JSON_UNSUPPORTED


def json_compatible(value: Any) -> Any:
    """Recursively normalise values before serialising a known payload shape."""

    primitive = json_primitive(value)
    if primitive is not JSON_UNSUPPORTED:
        return json_compatible(primitive) if primitive is not value else primitive
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, pd.Series):
        return [json_compatible(item) for item in value.tolist()]
    if isinstance(value, pd.Index):
        return [json_compatible(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return json_compatible(value.to_dict())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_compatible(item) for item in value]
    raise TypeError(f"Object of type {type(value)!r} is not JSON serialisable")
