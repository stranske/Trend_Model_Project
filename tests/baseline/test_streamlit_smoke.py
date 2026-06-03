"""Tier 2 (UI layer): Streamlit AppTest smoke tests.

These drive the real Streamlit pages headlessly via ``st.testing.v1.AppTest`` --
no browser, no mocks. Unlike the legacy DummyStreamlit tests, AppTest runs the
genuine Streamlit runtime, so it catches rendering-rule violations (e.g. nested
expanders) that mocks silently allow.

The smoke contract for a page is simply: it renders **without an unhandled
exception**. A friendly ``st.error`` (e.g. "load data first") counts as wired,
correct behavior -- it's a guard, not a crash.
"""

from __future__ import annotations

import pytest

from .harness import REPO_ROOT

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

# Pages expected to render cleanly on a cold start (no data loaded).
# 3_Results renders a graceful "load data" guard -- that is a pass.
_CLEAN_PAGES = [
    "streamlit_app/app.py",
    "streamlit_app/pages/1_Data.py",
    "streamlit_app/pages/3_Results.py",
    "streamlit_app/pages/4_Help.py",
    "streamlit_app/pages/8_Validation.py",
    "streamlit_app/pages/5_Monte_Carlo.py",
]


def _run_page(rel_path: str) -> AppTest:
    at = AppTest.from_file(str(REPO_ROOT / rel_path), default_timeout=120)
    at.run()
    return at


@pytest.mark.parametrize("page", _CLEAN_PAGES, ids=[p.split("/")[-1] for p in _CLEAN_PAGES])
def test_page_renders_without_exception(page):
    """Wiring smoke: the page loads and does not raise."""
    at = _run_page(page)
    assert not at.exception, f"{page} raised: {[e.value for e in at.exception]}"
    # Sanity: the page produced *some* output (widget, error, or markdown).
    produced = (
        len(at.number_input)
        + len(at.selectbox)
        + len(at.button)
        + len(at.radio)
        + len(at.checkbox)
        + len(at.error)
        + len(at.markdown)
        + len(at.title)
    )
    assert produced > 0, f"{page} rendered nothing"


def test_model_page_renders_without_exception():
    # Regression guard for the nested-expander crash fixed 2026-05-30
    # (_render_config_chat_contents previously opened expanders inside the
    # "Config Chat" expander). Cold render shows the data guard, not an exception.
    at = _run_page("streamlit_app/pages/2_Model.py")
    assert not at.exception, f"2_Model raised: {[e.value for e in at.exception]}"


def test_demo_inputs_are_wired():
    """Setting a keyed demo input and re-running must not break the home page,
    and the value must round-trip into session_state (input -> state wiring)."""
    at = _run_page("streamlit_app/app.py")
    if not at.number_input:
        pytest.skip("home page exposed no number_input widgets")
    # The demo lookback input is keyed 'demo_lookback'.
    try:
        widget = at.number_input(key="demo_lookback")
    except Exception:
        pytest.skip("demo_lookback widget not present in this build")
    widget.set_value(48)
    at.run()
    assert not at.exception
    assert at.session_state["demo_lookback"] == 48
