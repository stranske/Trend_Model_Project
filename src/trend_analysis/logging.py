"""Lightweight structured logging helpers for CLI runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import DEFAULT_OUTPUT_DIRECTORY

# Track run identifiers to their active log file path.
_LOG_PATHS: dict[str, Path] = {}


def get_default_log_path(run_id: str) -> Path:
    """Return the default JSONL log file path for ``run_id``."""
    base = Path(DEFAULT_OUTPUT_DIRECTORY) / "logs"
    return base / f"run_{run_id}.jsonl"


def init_run_logger(run_id: str, log_path: Path | str | None = None) -> Path:
    """Create the structured log file and register it for ``run_id``."""
    path = Path(log_path) if log_path is not None else get_default_log_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    _LOG_PATHS[run_id] = path
    return path


def log_step(run_id: str, step: str, message: str, **metadata: Any) -> None:
    """Append a structured log record for ``run_id``."""
    path = _LOG_PATHS.get(run_id)
    if path is None:
        path = init_run_logger(run_id)
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_id": run_id,
        "step": step,
        "message": message,
        "metadata": metadata or {},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
