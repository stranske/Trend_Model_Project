"""Tests for the Streamlit one-click demo workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from streamlit_app.demo import DemoRunError, run_one_click_demo  # noqa: E402


def test_run_one_click_demo_populates_state():
    session: dict[str, object] = {}
    result = run_one_click_demo(session)

    assert "returns_df" in session
    assert isinstance(session["returns_df"], pd.DataFrame)
    assert "sim_results" in session
    assert hasattr(result, "portfolio_curve")
    curve = result.portfolio_curve()
    assert not curve.empty
    assert session["config_state"]["preset_name"] == "Balanced"  # type: ignore[index]
    assert session["sim_config"]["preset_name"] == "Balanced"  # type: ignore[index]
    assert "SPX" in session.get("benchmark_candidates", [])


def test_run_one_click_demo_missing_dataset(monkeypatch):
    session: dict[str, object] = {}
    def _raise():
        raise DemoRunError("missing")

    monkeypatch.setattr("streamlit_app.demo._resolve_demo_dataset", _raise)
    with pytest.raises(DemoRunError):
        run_one_click_demo(session)
