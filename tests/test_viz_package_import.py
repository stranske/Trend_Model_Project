"""Tests for viz package import behavior."""

import sys


def test_viz_import_does_not_load_streamlit():
    # Ensure prior tests don't influence this import check.
    for name in list(sys.modules):
        if name == "streamlit" or name.startswith("streamlit."):
            sys.modules.pop(name, None)
    import trend_analysis.viz  # noqa: F401

    assert "streamlit" not in sys.modules
