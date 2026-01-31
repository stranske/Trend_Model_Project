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

import importlib.util
import pathlib
import socket
import sys
from pathlib import Path
from typing import Iterator

import pytest

try:
    import yaml
except ImportError:
    yaml = None

from ._autofix_diag import DiagnosticsRecorder, get_recorder


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config) -> None:  # noqa: ARG001
    """Disable pytest-rerunfailures xdist sockets in restricted environments."""
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.close()
    except OSError:
        try:
            import pytest_rerunfailures
        except Exception:
            return
        pytest_rerunfailures.HAS_PYTEST_HANDLECRASHITEM = False


# --- Ensure local ``src`` packages are importable ---------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Pre-import modules with lazy loaders to avoid race conditions in xdist parallel execution.
# The trend_analysis package uses __getattr__ for lazy submodule loading, which can fail
# when multiple workers try to access submodules concurrently during import.
try:
    import trend_analysis.export.bundle  # noqa: F401
except ImportError:
    pass  # OK if dependencies not installed


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

    # Assign serial tests to a dedicated xdist worker group to prevent parallel
    # execution interference. Tests marked with @pytest.mark.serial will all run
    # on the same worker sequentially.
    for it in items:
        if it.get_closest_marker("serial"):
            it.add_marker(pytest.mark.xdist_group("serial"))

    # Performance tests are part of the runtime-critical suite even if they
    # don't explicitly carry the runtime marker.
    for it in items:
        if it.get_closest_marker("performance") and not it.get_closest_marker("runtime"):
            it.add_marker(pytest.mark.runtime)

    for it in items:
        markers = sorted({marker.name for marker in it.iter_markers()})
        if not markers:
            continue
        existing_keys = {name for name, _ in getattr(it, "user_properties", [])}
        if "test_markers" in existing_keys:
            continue
        it.user_properties.append(("test_markers", ",".join(markers)))


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def pytest_ignore_collect(collection_path: Path, config):  # noqa: ARG001
    repo_root = ROOT
    try:
        rel_path = collection_path.resolve().relative_to(repo_root)
    except Exception:
        return False

    if not _module_available("streamlit"):
        if rel_path.parts[:2] == ("tests", "app"):
            return True
        if rel_path.name.startswith(("test_streamlit_", "test_upload_")):
            return True

    if not _module_available("fastapi"):
        if rel_path.name.startswith("test_api_server"):
            return True

    return False


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    recorder = get_recorder()
    if not recorder.has_entries():
        return
    output_path = Path("ci/autofix/diagnostics.json")
    recorder.flush(output_path)
    setattr(session.config, "autofix_diagnostics_path", str(output_path))
