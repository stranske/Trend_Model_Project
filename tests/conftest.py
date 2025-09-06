"""Pytest configuration.

This project historically relied on callers configuring ``PYTHONPATH`` so that
the ``src`` directory is importable.  Some environments (notably minimal CI
containers) invoke ``pytest`` without performing an editable install which
results in ``ModuleNotFoundError`` during collection.  To keep the tests
import‑safe we ensure the repository's ``src`` directory is on ``sys.path``
before any tests run.

The file also documents a quirk with dependencies like NumPy attempting to set
``PYTHONHASHSEED`` during test execution.  This has no effect because the
environment variable must be specified before the Python interpreter starts. To
reproduce hash behaviour set ``PYTHONHASHSEED=0`` in the environment prior to
running ``pytest``.
"""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

import pytest

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

# --- Ensure local ``src`` packages are importable ---------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_collection_modifyitems(config, items):
    qfile = pathlib.Path(__file__).with_name("quarantine.yml")
    if not qfile.exists() or yaml is None:
        return
    data = yaml.safe_load(qfile.read_text()) or {}
    bad = {t["id"] for t in data.get("tests", [])}
    for it in items:
        if it.nodeid in bad:
            it.add_marker(pytest.mark.quarantine(reason="repo quarantine list"))
            it.add_marker(pytest.mark.xfail(reason="quarantined", strict=False))
