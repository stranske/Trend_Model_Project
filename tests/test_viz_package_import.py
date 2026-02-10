"""Tests for viz package import behavior."""

import sys


def test_viz_import_does_not_load_streamlit():
    import trend_analysis.viz  # noqa: F401

    assert "streamlit" not in sys.modules
