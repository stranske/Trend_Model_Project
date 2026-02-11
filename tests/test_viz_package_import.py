"""Tests for viz package import behavior."""

import sys


def test_viz_import_does_not_load_streamlit():
    streamlit_loaded = "streamlit" in sys.modules

    import trend_analysis.viz  # noqa: F401

    if not streamlit_loaded:
        assert "streamlit" not in sys.modules
