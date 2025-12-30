"""Hashing helpers.

This module centralises SHA256 hashing utilities so that hashing logic
is consistent across the project.  It deliberately avoids any project
specific dependencies so it can be imported from light‑weight contexts
like test helpers or build scripts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def sha256_bytes(data: bytes) -> str:
    """Return the SHA256 hex digest for *data*.

    Parameters
    ----------
    data:
        Raw byte string to hash.
    """
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA256 hex digest for *text*.

    The input is first encoded as UTF-8 prior to hashing.
    """
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: PathLike) -> str:
    """Return the SHA256 hex digest for the file at *path*."""
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def normalise_for_json(value: Any) -> Any:
    """Convert values to JSON-serialisable representations."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {k: normalise_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise_for_json(item) for item in value]
    if hasattr(value, "model_dump"):
        return normalise_for_json(value.model_dump())
    return value


def sha256_config(cfg: Mapping[str, Any] | Any) -> str:
    """Return a deterministic SHA256 digest for a config mapping.

    The mapping is serialised to JSON with sorted keys to ensure
    consistent hashing irrespective of key order.
    """
    normalised = normalise_for_json(cfg)
    text = json.dumps(normalised, sort_keys=True, separators=(",", ":"))
    return sha256_text(text)
