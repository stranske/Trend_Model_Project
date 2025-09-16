"""Ensure src/ is on sys.path for subprocesses spawned by tests.

Pytest spawns a subprocess for the Streamlit health wrapper smoke test using
`python -m trend_portfolio_app.health_wrapper`. That new interpreter will
search its working directory first; adding this ``sitecustomize`` inside the
``tests`` directory guarantees that when the test's CWD is the repo root (or
tests path is on PYTHONPATH) the ``src`` directory is still discoverable.
"""

from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "src"
if src.exists():
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
