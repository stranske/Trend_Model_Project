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
from typing import Iterator

import pytest

try:
    import yaml
except ImportError:
    yaml = None

from ._autofix_diag import DiagnosticsRecorder, get_recorder

# --- Ensure local ``src`` packages are importable ---------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session", autouse=True)
def _autofix_diagnostics_session() -> Iterator[DiagnosticsRecorder]:
    recorder = get_recorder()
    recorder.reset()
    yield recorder


@pytest.fixture()
def autofix_recorder() -> DiagnosticsRecorder:
    return get_recorder()


def pytest_collection_modifyitems(config, items):
    qfile = pathlib.Path(__file__).with_name("quarantine.yml")
    if qfile.exists() and yaml is not None:
        data = yaml.safe_load(qfile.read_text()) or {}
        bad = {t["id"] for t in data.get("tests", [])}
        for it in items:
            if it.nodeid in bad:
                it.add_marker(pytest.mark.quarantine(reason="repo quarantine list"))
                it.add_marker(pytest.mark.xfail(reason="quarantined", strict=False))

    for it in items:
        markers = sorted({marker.name for marker in it.iter_markers()})
        if not markers:
            continue
        existing_keys = {name for name, _ in getattr(it, "user_properties", [])}
        if "test_markers" in existing_keys:
            continue
        it.user_properties.append(("test_markers", ",".join(markers)))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    recorder = get_recorder()
    if not recorder.has_entries():
        return
    output_path = Path("ci/autofix/diagnostics.json")
    recorder.flush(output_path)
    setattr(session.config, "autofix_diagnostics_path", str(output_path))
