"""Numbered Streamlit entrypoint for the Monte Carlo page."""

from __future__ import annotations

import streamlit as st

from streamlit_app import demo_profile
from streamlit_app.monte_carlo_page import render
from streamlit_app.theme import apply_density_compact, apply_ds_theme

apply_ds_theme()
apply_density_compact()
demo_profile.initialize_profile(st)
render()
