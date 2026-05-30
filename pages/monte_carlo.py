"""Root-entrypoint wrapper for the Streamlit Monte Carlo page."""

from __future__ import annotations

import runpy
from pathlib import Path

page = Path(__file__).resolve().parents[1] / "streamlit_app" / "pages" / "monte_carlo.py"
runpy.run_path(str(page), run_name="__main__")
