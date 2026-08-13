#!/usr/bin/env python
"""Import-safe entry point for the multi-period demo checks.

The full demo remains in :mod:`scripts._run_multi_demo` because it deliberately
executes a broad end-to-end validation sequence.  Keeping that execution behind
``main()`` lets tooling inspect this module without generating data, launching
subprocesses, or running the repository test suite.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    """Run the existing multi-period demo sequence."""
    runpy.run_path(str(Path(__file__).with_name("_run_multi_demo.py")), run_name="__main__")


if __name__ == "__main__":
    main()
