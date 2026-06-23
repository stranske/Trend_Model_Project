"""Shared design-system theme adapter for Streamlit entry points."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_design_system_on_path() -> None:
    design_system = Path(__file__).resolve().parents[1] / "design-system"
    design_system_path = str(design_system)
    if design_system_path not in sys.path:
        sys.path.insert(0, design_system_path)


def apply_ds_theme() -> None:
    """Apply the fleet Streamlit design-system theme."""

    _ensure_design_system_on_path()
    import ds_streamlit

    ds_streamlit.inject_theme()
