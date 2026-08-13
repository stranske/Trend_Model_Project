"""Guard that developer surfaces stay hidden and on-page guidance matches the profile.

Issue #5816. Two defects observed by driving the live app:

1. ``streamlit_app/pages/1_Data.py`` rendered a ``Debug: Fund selection state`` expander
   unconditionally, dumping internal counters (run counter, ``data_loaded_key``,
   ``available_funds_count``, raw perf timings, ...) to end users. Issue #5411 gated the
   sibling perf *caption* behind ``show_perf_diagnostics`` but missed this expander, so the
   same information stayed visible. The fix puts the expander behind the same flag.

2. ``streamlit_app/app.py`` advertised "Use the **Custom Analysis** section to load your own
   data" unconditionally in Quick Start, while that section only renders when
   ``demo_profile.custom_analysis_enabled(...)`` is true. In the default ``presentation_safe``
   profile the instruction pointed at a section that is not on the page.

Deliberate-break gates: un-gating the Debug expander makes
``test_debug_surfaces_hidden_without_flag`` fail; making the Quick Start copy unconditional
again makes ``test_quick_start_matches_active_profile`` fail.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

# Reuse the established DummyStreamlit harness + page fixture from the sibling
# Data-page test module. ``data_page`` is a pytest fixture; importing it here
# registers it for this module's tests.
from tests.app.test_data_page import DummyStreamlit, data_page  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]

# The Debug expander body increments this session key and is the only writer of it,
# so its presence is a reliable signal that the expander rendered.
_DEBUG_RUN_KEY = "_debug_fund_run"


def _load_sample(monkeypatch: pytest.MonkeyPatch, page: ModuleType, stub: DummyStreamlit) -> None:
    """Drive the Data page into the loaded-dataset state that renders the
    fund-selection section (where the debug expander lives)."""
    stub.session_state.clear()
    stub.clear_calls = 0

    df = pd.DataFrame(
        {"FundA": [0.01, 0.02, -0.01], "SPX Index": [0.03, -0.02, 0.01]},
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )
    meta = {"validation": {"issues": [], "warnings": []}, "frequency_label": "monthly"}
    sample = page.data_cache.SampleDataset("demo.csv", Path("demo/demo_returns.csv"))

    monkeypatch.setattr(page.data_cache, "default_sample_dataset", lambda: sample)
    monkeypatch.setattr(page.data_cache, "dataset_choices", lambda: {sample.label: sample})
    monkeypatch.setattr(page.data_cache, "load_dataset_from_path", lambda path: (df, meta))
    stub.selectbox_map["Choose a sample"] = sample.label
    stub.selectbox_map["Benchmark column (optional)"] = "SPX Index"
    monkeypatch.setattr(page, "infer_benchmarks", lambda columns: ["SPX Index"])


def test_debug_surfaces_hidden_without_flag(
    monkeypatch: pytest.MonkeyPatch, data_page  # noqa: F811 - reused pytest fixture
) -> None:
    """With no debug flag set, the Data page renders no debug expander."""
    page, stub = data_page
    _load_sample(monkeypatch, page, stub)

    page.render_data_page()

    assert _DEBUG_RUN_KEY not in stub.session_state, (
        "Debug: Fund selection state expander rendered without show_perf_diagnostics; "
        "internal counters must not reach end users."
    )


def test_debug_surfaces_visible_with_flag(
    monkeypatch: pytest.MonkeyPatch, data_page  # noqa: F811 - reused pytest fixture
) -> None:
    """The expander is still available to developers behind the flag."""
    page, stub = data_page
    _load_sample(monkeypatch, page, stub)
    stub.session_state["show_perf_diagnostics"] = True

    page.render_data_page()

    assert stub.session_state.get(_DEBUG_RUN_KEY), (
        "show_perf_diagnostics was set but the debug expander did not render; "
        "the developer diagnostic must remain reachable."
    )


def test_quick_start_matches_active_profile() -> None:
    """Home-page Quick Start must not advertise a section this profile hides.

    ``app.py`` executes at import time, so this asserts on the page source in the same
    style as ``tests/test_demo_profile.py::test_gated_pages_render_profile_controls``.
    """
    source = (REPO_ROOT / "streamlit_app" / "app.py").read_text(encoding="utf-8")

    marker = "Use the **Custom Analysis** section"
    assert marker in source, "expected the Custom Analysis Quick Start bullet to still exist"

    gate = "demo_profile.custom_analysis_enabled(_active_profile)"
    assert gate in source, "Quick Start copy must be gated on custom_analysis_enabled"

    # The Custom Analysis bullet must appear *after* the profile gate, i.e. inside the
    # branch that only runs when the section is actually rendered.
    assert source.index(gate) < source.index(marker), (
        "The 'Use the Custom Analysis section' bullet appears before the "
        "custom_analysis_enabled gate, so presentation_safe users are told to use a "
        "section that is not rendered."
    )

    # And the presentation-safe branch must point at the control that actually exists.
    assert "switch **Demo mode**" in source or "Demo mode" in source, (
        "presentation_safe Quick Start should tell the user how to enable custom "
        "analysis (the Demo mode switcher), not reference a hidden section."
    )
