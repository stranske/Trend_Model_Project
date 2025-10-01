import importlib
import sys
from types import ModuleType
from typing import Any, Callable

import pandas as pd
import pytest

from trend_analysis.io.market_data import (
    MarketDataMetadata,
    MarketDataMode,
    MarketDataValidationError,
)


class DummyStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.uploaded: Any = None
        self.button_clicked = False
        self.title_calls: list[str] = []
        self.file_uploader_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.button_calls: list[str] = []
        self.success_messages: list[str] = []
        self.error_messages: list[str] = []
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.dataframes: list[pd.DataFrame] = []

    def title(self, text: str) -> None:
        self.title_calls.append(text)

    def file_uploader(self, *args: Any, **kwargs: Any) -> Any:
        self.file_uploader_calls.append((args, kwargs))
        return self.uploaded

    def button(self, label: str) -> bool:
        self.button_calls.append(label)
        return self.button_clicked

    def success(self, message: str) -> None:
        self.success_messages.append(message)

    def error(self, message: str) -> None:
        self.error_messages.append(message)

    def info(self, message: str) -> None:
        self.info_messages.append(message)

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)

    def dataframe(self, df: pd.DataFrame) -> None:
        self.dataframes.append(df)


@pytest.fixture
def upload_page(monkeypatch: pytest.MonkeyPatch):
    stub = DummyStreamlit()
    module = ModuleType("streamlit")

    def bind(name: str) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return getattr(stub, name)(*args, **kwargs)

        return wrapper

    for attr in [
        "title",
        "file_uploader",
        "button",
        "success",
        "error",
        "info",
        "warning",
        "dataframe",
    ]:
        setattr(module, attr, bind(attr))
    module.session_state = stub.session_state

    monkeypatch.setitem(sys.modules, "streamlit", module)

    from app.streamlit import state as app_state

    monkeypatch.setattr(app_state, "st", module)

    page = importlib.reload(importlib.import_module("streamlit_app.pages.1_Upload"))
    page.st = module

    stub.session_state.clear()
    stub.uploaded = None
    stub.button_clicked = False
    stub.title_calls.clear()
    stub.file_uploader_calls.clear()
    stub.button_calls.clear()
    stub.success_messages.clear()
    stub.error_messages.clear()
    stub.info_messages.clear()
    stub.warning_messages.clear()
    stub.dataframes.clear()
    module.session_state = stub.session_state

    return page, stub, module


def _build_meta(df: pd.DataFrame) -> dict[str, Any]:
    metadata = MarketDataMetadata(
        mode=MarketDataMode.RETURNS,
        frequency="M",
        frequency_label="monthly",
        start=df.index.min().to_pydatetime(),
        end=df.index.max().to_pydatetime(),
        rows=len(df),
        columns=list(df.columns),
    )
    warnings = []
    if len(df) < 12:
        warnings.append(
            f"Dataset is quite small ({len(df)} periods) – consider a longer history."
        )
    return {
        "metadata": metadata,
        "validation": {"issues": [], "warnings": warnings},
        "original_columns": list(df.columns),
        "symbols": list(df.columns),
        "n_rows": len(df),
        "mode": metadata.mode.value,
        "frequency": metadata.frequency_label,
        "frequency_code": metadata.frequency,
        "date_range": metadata.date_range,
        "start": metadata.start,
        "end": metadata.end,
    }


def test_render_upload_page_success(monkeypatch: pytest.MonkeyPatch, upload_page) -> None:
    page, stub, st_module = upload_page

    df = pd.DataFrame(
        {
            "FundA": [0.01, 0.02, -0.01],
            "SPX Index": [0.03, -0.02, 0.01],
        },
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )
    meta = _build_meta(df)

    stub.uploaded = object()
    monkeypatch.setattr(page, "load_and_validate_file", lambda handle: (df, meta))

    page.render_upload_page(st_module)

    assert stub.error_messages == []
    assert stub.success_messages
    assert "Loaded" in stub.success_messages[-1]
    assert any("Detected" in msg for msg in stub.info_messages)
    assert stub.dataframes and stub.dataframes[0].equals(df.head(12))
    assert st_module.session_state["upload_status"] == "success"
    assert st_module.session_state["schema_meta"] is meta
    assert st_module.session_state["returns_df"] is df
    assert (
        st_module.session_state["validation_report"] == meta["validation"]
    )
    assert st_module.session_state["benchmark_candidates"] == ["SPX Index"]


def test_render_upload_page_failure(monkeypatch: pytest.MonkeyPatch, upload_page) -> None:
    page, stub, st_module = upload_page

    stub.uploaded = object()

    def raise_validation(_: Any) -> tuple[Any, Any]:
        raise MarketDataValidationError(
            "Data validation failed:\n• unsorted index",
            issues=["unsorted index"],
        )

    monkeypatch.setattr(page, "load_and_validate_file", raise_validation)

    page.render_upload_page(st_module)

    assert len(stub.error_messages) == 1
    assert "unsorted index" in stub.error_messages[0]
    assert not stub.success_messages
    assert st_module.session_state["upload_status"] == "error"
    assert st_module.session_state["returns_df"] is None
    assert st_module.session_state["schema_meta"] is None
    assert st_module.session_state["validation_report"] == {
        "message": "Data validation failed:\n• unsorted index",
        "issues": ["unsorted index"],
    }
    assert st_module.session_state["benchmark_candidates"] == []
