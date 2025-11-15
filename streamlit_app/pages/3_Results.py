"""Compatibility shim for legacy results page numbering."""

from __future__ import annotations

import importlib

_results_module = importlib.import_module("streamlit_app.pages.4_Results")
_results_module = importlib.reload(_results_module)

analysis_runner = _results_module.analysis_runner
charts = _results_module.charts


def render_results_page() -> None:
    """Delegate rendering to the canonical results page implementation."""

    _results_module.render_results_page()


def _should_auto_render() -> bool:
    """Return True when running inside a live Streamlit context."""

    try:  # pragma: no cover - runtime detection only
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


if _should_auto_render():  # pragma: no cover - Streamlit runtime only
    render_results_page()
