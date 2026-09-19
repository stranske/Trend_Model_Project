"""Regression for autouse eviction of stub-reloaded ``streamlit_app`` modules.

These tests live under ``tests/streamlit/`` so the autouse fixture in
``conftest.py`` is in scope. Baseline AppTest smoke tests intentionally call
``evict_streamlit_app_modules()`` directly and cannot prove the fixture works.
"""

from __future__ import annotations

import sys

import pytest

from tests.baseline.harness import REPO_ROOT
from tests.streamlit.test_mc_page import _load_page

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _streamlit_app_modules_loaded() -> bool:
    return any(m == "streamlit_app" or m.startswith("streamlit_app.") for m in sys.modules)


def test_stub_reload_pollutes_streamlit_app_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DummyStreamlit reload caches modules until the autouse fixture teardown runs."""
    _load_page(monkeypatch)
    monkeypatch.undo()
    assert _streamlit_app_modules_loaded(), "stub reload should cache streamlit_app modules"


def test_apptest_works_after_autouse_teardown() -> None:
    """Order regression: prior test's autouse teardown leaves AppTest a clean import path."""
    assert not _streamlit_app_modules_loaded(), "modules should be evicted before this test starts"
    at = AppTest.from_file(str(REPO_ROOT / "streamlit_app/pages/4_Help.py"), default_timeout=120)
    at.run()
    assert not at.exception, f"Help page raised: {[e.value for e in at.exception]}"
    produced = (
        len(at.number_input)
        + len(at.selectbox)
        + len(at.button)
        + len(at.radio)
        + len(at.checkbox)
        + len(at.error)
        + len(at.markdown)
        + len(at.title)
    )
    assert produced > 0, "Help page rendered nothing after autouse teardown (module leak)"
