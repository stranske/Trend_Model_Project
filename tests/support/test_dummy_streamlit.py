"""Shared widget stub behavior used by page tests."""

import re
from pathlib import Path

import pytest

from tests.support.dummy_streamlit import DummyStreamlit

TESTS_DIR = Path(__file__).resolve().parent.parent

MIGRATED_MODULES = [
    "app/test_data_page.py",
    "app/test_results_page.py",
    "app/test_validation_page_renders.py",
    "streamlit/test_mc_page.py",
]


def test_only_one_dummy_streamlit_class_exists():
    """Guard against a page test re-introducing its own local copy.

    A stray ``class DummyStreamlit`` elsewhere would silently opt that page
    out of the shared stub's fixes, recreating the drift this module exists
    to prevent.
    """
    matches = [
        path
        for path in TESTS_DIR.rglob("*.py")
        if re.search(r"^class DummyStreamlit\b", path.read_text(), re.MULTILINE)
    ]
    assert matches == [TESTS_DIR / "support" / "dummy_streamlit.py"]


@pytest.mark.parametrize("relative_path", MIGRATED_MODULES)
def test_migrated_module_imports_shared_stub(relative_path):
    """Each previously-duplicated page test module imports the shared stub."""
    source = (TESTS_DIR / relative_path).read_text()
    assert "from tests.support.dummy_streamlit import" in source
    assert re.search(r"^class DummyStreamlit\b", source, re.MULTILINE) is None


@pytest.mark.parametrize("widget", ["radio", "selectbox"])
def test_empty_initial_selection(widget):
    stub = DummyStreamlit()
    assert getattr(stub, widget)("choice", ["one", "two"], index=None) is None


@pytest.mark.parametrize("widget", ["radio", "selectbox"])
def test_scripted_selection_overrides_empty_index(widget):
    stub = DummyStreamlit()
    setattr(stub, f"{widget}_value", "two")
    assert getattr(stub, widget)("choice", ["one", "two"], index=None) == "two"


@pytest.mark.parametrize("widget", ["radio", "selectbox"])
def test_numeric_index_selects_option(widget):
    stub = DummyStreamlit()
    assert getattr(stub, widget)("choice", ["one", "two"], index=1) == "two"
