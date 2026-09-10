"""Shared widget stub behavior used by page tests."""

import pytest

from tests.support.dummy_streamlit import DummyStreamlit


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
