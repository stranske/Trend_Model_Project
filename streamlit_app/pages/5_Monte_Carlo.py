"""Numbered Streamlit entrypoint for the Monte Carlo page."""

from __future__ import annotations

from streamlit_app.monte_carlo_page import render
from streamlit_app.theme import apply_ds_theme


apply_ds_theme()
render()
