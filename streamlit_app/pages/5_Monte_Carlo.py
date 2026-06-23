"""Numbered Streamlit entrypoint for the Monte Carlo page."""

from __future__ import annotations

import streamlit as st

from streamlit_app.theme import apply_ds_theme

apply_ds_theme()

# The Monte Carlo page renders Plotly charts and imports Plotly transitively
# (monte_carlo_page -> viz adapters / chart export bundle). Plotly is
# intentionally excluded from the lean offline demo bundle (the stlite/Pyodide
# vendoring ships numpy/pandas only), so importing the page there raises
# ModuleNotFoundError and the whole page crashes with a red traceback.
# Degrade gracefully: when Plotly is unavailable, show a clear message instead
# of crashing. When Plotly is present (the full app or a Plotly-enabled build)
# the page behaves exactly as before.
try:
    import plotly.graph_objects  # noqa: F401  (availability probe only)

    _PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    _PLOTLY_AVAILABLE = False

if _PLOTLY_AVAILABLE:
    from streamlit_app.monte_carlo_page import render

    render()
else:
    st.title("Monte Carlo Simulation")
    st.info(
        "The Monte Carlo page requires Plotly, which isn't included in this "
        "offline demo build. Run the full app (or a Plotly-enabled build) to "
        "use Monte Carlo simulations and charts."
    )
