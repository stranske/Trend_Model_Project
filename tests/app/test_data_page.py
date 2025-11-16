from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pandas as pd
import pytest

from streamlit_app.components.upload_guard import GuardedUpload, hash_bytes
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


class DummyStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.radio_value = "Sample dataset"
        self.selectbox_value = None
        self.uploaded: Any = None
        self.title_calls: list[str] = []
        self.write_calls: list[str] = []
        self.info_messages: list[str] = []
        self.success_messages: list[str] = []
        self.error_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.captions: list[str] = []
        self.code_blocks: list[tuple[str, str | None]] = []
        self.dataframes: list[pd.DataFrame] = []
        self.selectbox_map: dict[str, Any] = {}

    def title(self, text: str) -> None:
        self.title_calls.append(text)

    def write(self, text: str) -> None:
        self.write_calls.append(str(text))

    def radio(self, *args: Any, **kwargs: Any) -> str:
        return self.radio_value

    def selectbox(self, *args: Any, **kwargs: Any) -> Any:
        label = args[0] if args else kwargs.get("label", "")
        if label in self.selectbox_map:
            return self.selectbox_map[label]
        if self.selectbox_value is not None:
            return self.selectbox_value
        options = kwargs.get("options") or args[1]
        return options[0]

    def file_uploader(self, *args: Any, **kwargs: Any) -> Any:
        return self.uploaded

    def button(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def info(self, message: str) -> None:
        self.info_messages.append(message)

    def success(self, message: str) -> None:
        self.success_messages.append(message)

    def error(self, message: str) -> None:
        self.error_messages.append(message)

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)

    def caption(self, text: str) -> None:
        self.captions.append(text)

    def code(self, text: str, *, language: str | None = None) -> None:
        self.code_blocks.append((str(text), language))

    def dataframe(self, df: pd.DataFrame) -> None:
        self.dataframes.append(df)

    def markdown(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        return None

    def cache_data(self, *args: Any, **kwargs: Any):
        def decorator(func):
            return func

        return decorator


@pytest.fixture
def data_page(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, DummyStreamlit]:
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
        "cache_data",
    ]:
        setattr(module, attr, bind(attr))
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
        index=pd.date_range("2024-01-31", periods=3, freq="M"),
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
