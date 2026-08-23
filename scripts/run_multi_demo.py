#!/usr/bin/env python
"""Import-safe entry point for the multi-period demo checks.

The full demo remains in :mod:`scripts._run_multi_demo` because it deliberately
executes a broad end-to-end validation sequence.  Keeping that execution behind
``main()`` lets tooling inspect this module without generating data, launching
subprocesses, or running the repository test suite.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Sequence

FAST_SENTINEL = Path(__file__).resolve().parent.parent / "demo/.fast_demo_mode"


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete multi-period demo checks")
    parser.add_argument(
        "--release-verify",
        action="store_true",
        help="run the real demo even when a stale fast-test sentinel exists",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the existing multi-period demo sequence."""

    args = _parse_args(() if argv is None else argv)
    if args.release_verify:
        try:
            FAST_SENTINEL.unlink(missing_ok=True)
        except OSError as exc:
            raise SystemExit(
                f"cannot clear fast-demo sentinel for release verification: {exc}"
            ) from exc

    runpy.run_path(
        str(Path(__file__).with_name("_run_multi_demo.py")),
        run_name="__main__",
        init_globals={"RELEASE_VERIFY": args.release_verify},
    )


if __name__ == "__main__":
    main(sys.argv[1:])
