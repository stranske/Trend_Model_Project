#!/usr/bin/env python3
"""DEPRECATED: Turnover cap demo script.

The functionality showcased here is (or will be) accessible via the unified
`trend` CLI and the underlying public APIs. This legacy demo is now a thin
wrapper to help users migrate their muscle memory.

Examples:
    trend run --config config/demo.yml --returns demo/demo_returns.csv
    trend report --out demo/exports

Removal Timeline:
    This file will be removed after the CLI has been broadly adopted.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import sys
import warnings

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "demo.yml"


def _warn() -> None:
    warnings.warn(
        "demo_turnover_cap.py is deprecated; use the unified `trend` CLI",
        DeprecationWarning,
        stacklevel=2,
    )


def main(argv: List[str] | None = None) -> int:
    _warn()
    try:
        from trend_analysis.cli import main as trend_main
    except Exception as exc:  # pragma: no cover
        print(f"Failed to import trend CLI: {exc}", file=sys.stderr)
        return 1
    # Delegate to generic run; specialized turnover demos should migrate
    # into documented examples or `trend stress` scenarios later.
    return trend_main(argv or ["run", "--config", str(CONFIG_PATH)])  # default


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
