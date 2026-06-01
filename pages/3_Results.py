"""Root-entrypoint wrapper for the Streamlit Results page."""

from __future__ import annotations

import runpy
from pathlib import Path

page = Path(__file__).resolve().parents[1] / "streamlit_app" / "pages" / "3_Results.py"
runpy.run_path(str(page), run_name="__main__")
