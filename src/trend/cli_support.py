"""Stable support seams for the public :mod:`trend.cli` entry point."""

from __future__ import annotations

import numbers
import platform
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trend_analysis import logging as run_logging
from trend_analysis.constants import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_OUTPUT_FORMATS
from trend_analysis.reporting.run_artifacts import find_existing_run

LOCK_PATH = Path(__file__).resolve().parents[2] / "requirements.lock"


def check_environment(lock_path: Path | None = None) -> int:
    """Print Python and package versions, reporting lockfile mismatches."""

    lock_file = lock_path or LOCK_PATH
    print(f"Python {platform.python_version()}")
    if not lock_file.exists():
        print(f"Lock file not found: {lock_file}")
        return 1
    mismatches: list[tuple[str, str | None, str]] = []
    for line in lock_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, expected = line.split("==", 1)
        name, expected = name.strip(), expected.split()[0]
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        print(f"{name} {installed or 'not installed'} (expected {expected})")
        if installed != expected:
            mismatches.append((name, installed, expected))
    if mismatches:
        print("Mismatches detected:")
        for name, installed, expected in mismatches:
            print(f"- {name}: installed {installed or 'none'}, expected {expected}")
        return 1
    print("All packages match lockfile.")
    return 0


def maybe_log_step(enabled: bool, run_id: str, event: str, message: str, **fields: Any) -> None:
    """Emit a structured run-log event only when requested."""

    if enabled:
        run_logging.log_step(run_id, event, message, **fields)


def extract_cache_stats(payload: object) -> dict[str, int] | None:
    """Return the last complete cache-statistics mapping in a result payload."""

    required = ("entries", "hits", "misses", "incremental_updates")
    found: list[dict[str, int]] = []

    def visit(obj: object) -> None:
        if isinstance(obj, (pd.Series, pd.DataFrame, np.ndarray)):
            return
        if isinstance(obj, Mapping):
            if all(key in obj for key in required):
                candidate: dict[str, int] = {}
                for key in required:
                    value = obj.get(key)
                    if isinstance(value, numbers.Integral):
                        candidate[key] = int(value)
                    elif isinstance(value, numbers.Real) and float(value).is_integer():
                        candidate[key] = int(float(value))
                    else:
                        break
                else:
                    found.append(candidate)
            for value in obj.values():
                visit(value)
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            for item in obj:
                visit(item)

    visit(payload)
    return found[-1] if found else None


def find_prior_run(cfg: Any, run_id: str) -> Path | None:
    """Return the manifest for an already-completed run with ``run_id``."""

    export_cfg = getattr(cfg, "export", None)
    out_dir = out_formats = None
    if isinstance(export_cfg, Mapping):
        out_dir, out_formats = export_cfg.get("directory"), export_cfg.get("formats")
    elif export_cfg is not None:
        out_dir, out_formats = getattr(export_cfg, "directory", None), getattr(
            export_cfg, "formats", None
        )
    if not out_dir and not out_formats:
        out_dir, out_formats = DEFAULT_OUTPUT_DIRECTORY, DEFAULT_OUTPUT_FORMATS
    return find_existing_run(Path(out_dir), run_id) if out_dir and out_formats else None
