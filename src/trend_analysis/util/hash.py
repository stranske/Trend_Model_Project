"""Hashing helpers.

This module centralises SHA256 hashing utilities so that hashing logic
is consistent across the project.  It deliberately avoids any project
specific dependencies so it can be imported from light‑weight contexts
like test helpers or build scripts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Union

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


def content_run_id(
    input_sha256: str | None,
    config_sha256: str | None,
    seed: Any = None,
) -> str:
    """Return a deterministic, content-addressed run identifier.

    The available components (input-file digest, config digest and seed) are
    joined in a stable order and hashed. This is the single implementation
    shared by the reproducibility bundle and the working CLI so that identical
    inputs always produce the same ``run_id``.
    """
    run_id_src = "|".join(
        filter(
            None,
            [input_sha256, config_sha256, str(seed) if seed is not None else ""],
        )
    )
    return sha256_text(run_id_src)


def _config_payload(cfg: Any) -> Any:
    """Best-effort conversion of a config object to a hashable payload."""

    if hasattr(cfg, "model_dump"):
        try:
            return cfg.model_dump()
        except Exception:  # pragma: no cover - defensive for exotic configs
            pass
    if hasattr(cfg, "__dict__"):
        return dict(getattr(cfg, "__dict__"))
    return cfg


def working_run_id(cfg: Any, input_path: PathLike | None) -> str:
    """Resolve a deterministic working ``run_id`` for a CLI run.

    Returns ``cfg.run_id`` when it is already set; otherwise a
    content-addressed id derived from the input-file digest, config digest and
    seed when a source input file exists. Falls back to a random 12-char id for
    the genuinely-unknown case (in-memory/library callers without a source
    path), mirroring the historical ``uuid.uuid4().hex[:12]`` behaviour.
    """
    existing = getattr(cfg, "run_id", None)
    if existing:
        return str(existing)
    path = Path(input_path) if input_path else None
    if path is not None and path.exists():
        return content_run_id(
            sha256_file(path),
            sha256_config(_config_payload(cfg)),
            getattr(cfg, "seed", None),
        )
    import uuid

    return uuid.uuid4().hex[:12]
