"""Numbered Streamlit entrypoint for the Monte Carlo page."""

from __future__ import annotations

from streamlit_app.monte_carlo_page import _should_auto_render, render


if _should_auto_render():
    render()
