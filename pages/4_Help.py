"""Root-entrypoint wrapper for the Streamlit Help page."""

from __future__ import annotations

import runpy
from pathlib import Path

page = Path(__file__).resolve().parents[1] / "streamlit_app" / "pages" / "4_Help.py"
runpy.run_path(str(page), run_name="__main__")
