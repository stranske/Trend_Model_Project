"""Tests for viz package import behavior."""

from __future__ import annotations

import importlib
import sys


def test_viz_import_does_not_load_streamlit(monkeypatch):
    """Importing viz should not pull in Streamlit when it's otherwise absent."""

    monkeypatch.delitem(sys.modules, "streamlit", raising=False)
    # Ensure a fresh import path for the viz module and its chart helper.
    monkeypatch.delitem(sys.modules, "trend_analysis.viz", raising=False)
    monkeypatch.delitem(sys.modules, "trend_analysis.viz.charts", raising=False)

    importlib.import_module("trend_analysis.viz")

    assert "streamlit" not in sys.modules


def test_lazy_attribute_access_no_streamlit_import(monkeypatch):
    """Lazy chart attribute access should not import Streamlit."""

    monkeypatch.delitem(sys.modules, "streamlit", raising=False)
    monkeypatch.delitem(sys.modules, "trend_analysis.viz", raising=False)
    monkeypatch.delitem(sys.modules, "trend_analysis.viz.charts", raising=False)

    viz = importlib.import_module("trend_analysis.viz")

    _ = viz.equity_curve

    assert "streamlit" not in sys.modules
