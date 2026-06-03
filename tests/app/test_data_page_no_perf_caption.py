"""Guard that developer perf diagnostics stay out of the production Data UI.

Issue #5411 (A31): ``streamlit_app/pages/1_Data.py`` rendered an always-visible
``st.caption`` exposing internal perf telemetry (raw ms timings, ``init_applied``,
``defaults_seeded`` counters, ...) to end users. The same data already lives in
the collapsed "Debug: Fund selection state" expander. The fix gates the caption
behind a ``show_perf_diagnostics`` session-state debug flag.

These tests pin that behaviour: the perf caption must be absent when the debug
flag is off, and present when it is on. Restoring the unconditional caption (the
deliberate-break gate in the issue) makes ``test_perf_caption_hidden_by_default``
fail.
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

_PERF_MARKER = "Perf: total"


def _load_sample(
    monkeypatch: pytest.MonkeyPatch, page: ModuleType, stub: DummyStreamlit
) -> None:
    """Drive the Data page into the loaded-dataset state that renders the
    fund-selection section (where the perf caption lives)."""
    stub.session_state.clear()
    stub.clear_calls = 0

    df = pd.DataFrame(
        {"FundA": [0.01, 0.02, -0.01], "SPX Index": [0.03, -0.02, 0.01]},
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )
    meta = {"validation": {"issues": [], "warnings": []}, "frequency_label": "monthly"}
    sample = page.data_cache.SampleDataset("demo.csv", Path("demo/demo_returns.csv"))

    monkeypatch.setattr(page.data_cache, "default_sample_dataset", lambda: sample)
    monkeypatch.setattr(
        page.data_cache, "dataset_choices", lambda: {sample.label: sample}
    )
    monkeypatch.setattr(
        page.data_cache, "load_dataset_from_path", lambda path: (df, meta)
    )
    stub.selectbox_map["Choose a sample"] = sample.label
    stub.selectbox_map["Benchmark column (optional)"] = "SPX Index"
    monkeypatch.setattr(page, "infer_benchmarks", lambda columns: ["SPX Index"])


def _perf_captions(stub: DummyStreamlit) -> list[str]:
    return [c for c in stub.captions if _PERF_MARKER in c]


def test_perf_caption_hidden_by_default(
    monkeypatch: pytest.MonkeyPatch, data_page  # noqa: F811 - reused pytest fixture
) -> None:
    page, stub = data_page
    _load_sample(monkeypatch, page, stub)

    page.render_data_page()

    # The fund-selection UI rendered (sanity: otherwise the assertion is vacuous).
    assert stub.dataframes
    # No developer perf caption leaks into the production UI when the flag is off.
    assert _perf_captions(stub) == []


def test_perf_caption_visible_with_debug_flag(
    monkeypatch: pytest.MonkeyPatch, data_page  # noqa: F811 - reused pytest fixture
) -> None:
    page, stub = data_page
    _load_sample(monkeypatch, page, stub)
    # Opt in to developer diagnostics; flag must be set after _load_sample clears
    # session_state.
    stub.session_state["show_perf_diagnostics"] = True

    page.render_data_page()

    perf = _perf_captions(stub)
    assert perf, "perf caption should render when show_perf_diagnostics is on"
