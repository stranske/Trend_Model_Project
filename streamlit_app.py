"""Synthetic-only public Streamlit demo entrypoint."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

os.environ["TREND_DEMO_MODE"] = "1"


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "streamlit_app" / "app.py"), run_name="__main__")
