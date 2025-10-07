"""Compatibility shim for legacy results page numbering."""

from __future__ import annotations

import importlib

_results_module = importlib.import_module("streamlit_app.pages.4_Results")
_results_module = importlib.reload(_results_module)

analysis_runner = _results_module.analysis_runner
charts = _results_module.charts
render_results_page = _results_module.render_results_page

render_results_page()
