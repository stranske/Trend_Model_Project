"""Tests for viz package import behavior."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys


def test_viz_import_does_not_load_streamlit(monkeypatch):
    """Importing viz should not pull in Streamlit when it's otherwise absent."""

    monkeypatch.delitem(sys.modules, "streamlit", raising=False)
    # Ensure a fresh import path for the viz module and its chart helper.
    monkeypatch.delitem(sys.modules, "trend_analysis.viz", raising=False)
    monkeypatch.delitem(sys.modules, "trend_analysis.viz.charts", raising=False)

    importlib.import_module("trend_analysis.viz")

    assert "streamlit" not in sys.modules


def test_viz_import_fresh_interpreter_does_not_load_streamlit():
    """Importing viz in a fresh interpreter should not import Streamlit."""

    cmd = [
        sys.executable,
        "-c",
        (
            "import importlib, json, sys; "
            "importlib.import_module('trend_analysis.viz'); "
            "print(json.dumps('streamlit' in sys.modules))"
        ),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout.strip()) is False
