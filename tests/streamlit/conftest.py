"""Shared pytest hooks for DummyStreamlit unit tests under ``tests/streamlit/``."""

from __future__ import annotations

import sys

import pytest


def evict_streamlit_app_modules() -> None:
    """Remove ``streamlit_app`` entries cached by ``importlib.reload`` stub tests."""
    for name in [
        m for m in list(sys.modules) if m == "streamlit_app" or m.startswith("streamlit_app.")
    ]:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def _evict_streamlit_app_modules_after_stub_tests() -> None:
    """Prevent stub-reloaded page modules from leaking into AppTest smoke tests."""
    yield
    evict_streamlit_app_modules()
