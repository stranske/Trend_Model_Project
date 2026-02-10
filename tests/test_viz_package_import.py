"""Tests for viz package import behavior."""

import sys


def test_viz_import_does_not_load_streamlit():
    before = set(sys.modules)

    import trend_analysis.viz  # noqa: F401

    added = set(sys.modules) - before
    assert "streamlit" not in added
