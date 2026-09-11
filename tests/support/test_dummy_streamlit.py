"""Shared widget stub behavior used by page tests."""

import ast
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


SHARED_STUB_MODULE = "tests.support.dummy_streamlit"


def _defines_dummy_streamlit(path: Path) -> bool:
    """True if ``path`` declares a ``DummyStreamlit`` class at ANY nesting depth.

    Matching source text would only catch a definition at column zero, so a
    copy tucked inside a fixture or helper function would slip past the guard.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    return any(
        isinstance(node, ast.ClassDef) and node.name == "DummyStreamlit" for node in ast.walk(tree)
    )


def _imports_shared_stub(path: Path) -> bool:
    """True if ``path`` really imports ``DummyStreamlit`` from the shared module.

    Substring matching would be satisfied by a comment, a docstring or a
    string literal that merely mentions the import.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == SHARED_STUB_MODULE
        and any(alias.name == "DummyStreamlit" for alias in node.names)
        for node in ast.walk(tree)
    )


def test_only_one_dummy_streamlit_class_exists():
    """Guard against a page test re-introducing its own local copy.

    A stray ``class DummyStreamlit`` elsewhere would silently opt that page
    out of the shared stub's fixes, recreating the drift this module exists
    to prevent.
    """
    matches = [path for path in TESTS_DIR.rglob("*.py") if _defines_dummy_streamlit(path)]
    assert matches == [TESTS_DIR / "support" / "dummy_streamlit.py"]


@pytest.mark.parametrize("relative_path", MIGRATED_MODULES)
def test_migrated_module_imports_shared_stub(relative_path):
    """Each previously-duplicated page test module imports the shared stub."""
    path = TESTS_DIR / relative_path
    assert _imports_shared_stub(path)
    assert not _defines_dummy_streamlit(path)


def test_guard_detects_a_nested_class_and_a_mentioned_import(tmp_path):
    """The guards must see through indentation and through mere mentions.

    Both checks used to be textual: ``^class DummyStreamlit`` never matched an
    indented definition, and ``"from tests.support.dummy_streamlit import" in
    source`` was satisfied by a comment.
    """
    nested = tmp_path / "nested.py"
    nested.write_text("def fixture():\n    class DummyStreamlit:\n        pass\n")
    assert _defines_dummy_streamlit(nested)

    mentioned = tmp_path / "mentioned.py"
    mentioned.write_text("# from tests.support.dummy_streamlit import DummyStreamlit\n")
    assert not _imports_shared_stub(mentioned)

    real = tmp_path / "real.py"
    real.write_text(f"from {SHARED_STUB_MODULE} import DummyStreamlit\n")
    assert _imports_shared_stub(real)
    assert not _defines_dummy_streamlit(real)


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
