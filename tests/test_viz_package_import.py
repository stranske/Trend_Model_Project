"""Tests for viz package import behavior."""

import sys


def test_viz_import_does_not_load_streamlit():
    preexisting = {name for name in sys.modules if name.startswith("streamlit")}
    import trend_analysis.viz  # noqa: F401

    post_import = {name for name in sys.modules if name.startswith("streamlit")}
    assert post_import == preexisting
