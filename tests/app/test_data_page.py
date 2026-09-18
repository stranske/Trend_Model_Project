from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pandas as pd
import pytest

from streamlit_app.components.upload_guard import GuardedUpload, hash_bytes
from tests.support.dummy_streamlit import DummyStreamlit
from trend_analysis.io.market_data import MarketDataValidationError


class DummyUpload:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._data

    def seek(self, _pos: int) -> None:
        return None


@pytest.fixture
def data_page(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, DummyStreamlit]:
    monkeypatch.setenv("TREND_DEMO_PROFILE", "public_llm_demo")
    stub = DummyStreamlit()
    module = ModuleType("streamlit")

    def bind(name: str) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return getattr(stub, name)(*args, **kwargs)

        return wrapper

    for attr in [
        "title",
        "write",
        "radio",
        "selectbox",
        "file_uploader",
        "button",
        "info",
        "success",
        "error",
        "warning",
        "caption",
        "code",
        "dataframe",
        "markdown",
        "expander",
        "container",
        "checkbox",
        "json",
        "cache_data",
        "subheader",
        "columns",
        "data_editor",
        "rerun",
        "metric",
    ]:
        setattr(module, attr, bind(attr))
    module.column_config = stub.column_config
    module.session_state = stub.session_state

    monkeypatch.setitem(sys.modules, "streamlit", module)

    stub.clear_calls = 0

    def mark_clear() -> None:
        stub.clear_calls += 1

    monkeypatch.setattr(
        "streamlit_app.components.analysis_runner.clear_cached_analysis",
        mark_clear,
    )

    from streamlit_app import state as app_state

    monkeypatch.setattr(app_state, "st", module)

    def fake_guard(uploaded: DummyUpload) -> GuardedUpload:
        data = uploaded.getvalue()
        return GuardedUpload(
            original_name=uploaded.name,
            stored_path=Path("/tmp") / uploaded.name,
            data=data,
            content_hash=hash_bytes(data),
            size=len(data),
        )

    page = importlib.reload(importlib.import_module("streamlit_app.pages.1_Data"))

    monkeypatch.setattr(page, "guard_and_buffer_upload", fake_guard)
    monkeypatch.setattr(page, "hash_path", lambda _p: "samplehash")
    monkeypatch.setattr(page, "validate_uploaded_csv", lambda *args, **kwargs: None)

    return page, stub


def test_data_page_autoloads_sample(monkeypatch: pytest.MonkeyPatch, data_page) -> None:
    page, stub = data_page

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
    stub.selectbox_map["Benchmark Column (optional)"] = "SPX Index"
    monkeypatch.setattr(page, "infer_benchmarks", lambda columns: ["SPX Index"])

    initial_clears = stub.clear_calls

    page.render_data_page()

    assert stub.success_messages
    assert stub.dataframes
    assert page.app_state.has_valid_upload()
    assert page.st.session_state["selected_benchmark"] == "SPX Index"
    assert page.st.session_state["data_loaded_key"].startswith("sample::")
    assert stub.clear_calls == initial_clears + 1
    for key in ["analysis_result", "analysis_result_key", "analysis_error"]:
        assert key not in page.st.session_state


def _load_sample_for_debug_gate(
    monkeypatch: pytest.MonkeyPatch, page: ModuleType, stub: DummyStreamlit
) -> None:
    """Drive the Data page into the loaded-dataset state that renders the
    fund-selection section, where the debug expander lives."""
    stub.session_state.clear()
    stub.expander_labels.clear()
    df = pd.DataFrame(
        {"FundA": [0.01, 0.02, -0.01], "SPX Index": [0.03, -0.02, 0.01]},
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )
    sample = page.data_cache.SampleDataset("demo.csv", Path("demo/demo_returns.csv"))
    monkeypatch.setattr(page.data_cache, "default_sample_dataset", lambda: sample)
    monkeypatch.setattr(page.data_cache, "dataset_choices", lambda: {sample.label: sample})
    monkeypatch.setattr(
        page.data_cache,
        "load_dataset_from_path",
        lambda _path: (df, {"validation": {"issues": [], "warnings": []}}),
    )
    monkeypatch.setattr(page, "infer_benchmarks", lambda _columns: ["SPX Index"])
    stub.selectbox_map["Choose a sample"] = sample.label
    stub.selectbox_map["Benchmark Column (optional)"] = "SPX Index"


def test_debug_surfaces_hidden_without_flag(monkeypatch: pytest.MonkeyPatch, data_page) -> None:
    """Issue #5816: internal selection counters/timings must not reach end users.

    The source-level assertion this test used to carry pinned exact indentation
    (``'            if st.session_state.get(...)'``), so any re-indent broke it without a
    behaviour change. The behavioural assertion below covers the same intent, so the
    brittle string match was dropped.
    """
    page, stub = data_page
    _load_sample_for_debug_gate(monkeypatch, page, stub)

    page.render_data_page()

    assert not any(label.startswith("Debug:") for label in stub.expander_labels)


def test_debug_surfaces_visible_with_flag(monkeypatch: pytest.MonkeyPatch, data_page) -> None:
    """The developer diagnostic must remain REACHABLE behind ``show_perf_diagnostics``.

    Without this counterpart, ``test_debug_surfaces_hidden_without_flag`` also passes if
    the expander is deleted outright, so the pair cannot distinguish "correctly gated"
    from "removed". This pins the gate rather than the removal.
    """
    page, stub = data_page
    _load_sample_for_debug_gate(monkeypatch, page, stub)
    stub.session_state["show_perf_diagnostics"] = True

    page.render_data_page()

    assert any(
        label.startswith("Debug:") for label in stub.expander_labels
    ), "show_perf_diagnostics was set but no Debug expander rendered"


def test_data_page_upload_failure(monkeypatch: pytest.MonkeyPatch, data_page) -> None:
    page, stub = data_page

    stub.session_state.clear()
    stub.clear_calls = 0

    stub.radio_value = "Upload your own"
    stub.uploaded = DummyUpload("bad.csv", b"bad,data")

    monkeypatch.setattr(page.data_cache, "default_sample_dataset", lambda: None)
    monkeypatch.setattr(page.data_cache, "dataset_choices", lambda: {})

    def raise_error(*_args: Any, **_kwargs: Any) -> None:
        raise MarketDataValidationError("validation failed", issues=["unsorted index"])

    monkeypatch.setattr(page.data_cache, "load_dataset_from_bytes", raise_error)

    initial_clears = stub.clear_calls

    page.render_data_page()

    assert stub.error_messages
    assert any("unsorted index" in call for call in stub.write_calls)
    assert page.st.session_state["upload_status"] == "error"
    assert page.st.session_state["returns_df"] is None
    assert page.st.session_state["validation_report"]["issues"] == ["unsorted index"]
    assert stub.clear_calls == initial_clears


def test_data_page_clamps_data_source_when_samples_are_missing(
    monkeypatch: pytest.MonkeyPatch, data_page
) -> None:
    page, stub = data_page

    stub.session_state.clear()
    stub.session_state["data_source"] = "Upload your own"
    stub.radio_value = "Upload your own"

    monkeypatch.setattr(page.data_cache, "default_sample_dataset", lambda: None)
    monkeypatch.setattr(page.data_cache, "dataset_choices", lambda: {})

    page.render_data_page()

    assert (
        stub.info_messages[-1]
        == "No dataset loaded yet. Switch to the sample tab for a quick start."
    )


def test_data_page_handles_generic_failure_with_plain_message(
    monkeypatch: pytest.MonkeyPatch, data_page
) -> None:
    page, stub = data_page

    stub.session_state.clear()
    stub.radio_value = "Upload your own"
    stub.uploaded = DummyUpload("bad.csv", b"bad,data")

    monkeypatch.setattr(page.data_cache, "default_sample_dataset", lambda: None)
    monkeypatch.setattr(page.data_cache, "dataset_choices", lambda: {})

    def raise_error(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("Traceback: raw parser failure")

    monkeypatch.setattr(page.data_cache, "load_dataset_from_bytes", raise_error)

    page.render_data_page()

    assert stub.error_messages[-1] == (
        "We couldn't process the file. Please confirm the format and try again."
    )
    assert stub.captions[-1] == "Traceback: raw parser failure"
    report = page.st.session_state["validation_report"]
    assert report["detail"] == "Traceback: raw parser failure"
