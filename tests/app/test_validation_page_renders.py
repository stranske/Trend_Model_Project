from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any, Callable

import pandas as pd
import pytest


class RecordingSessionState(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__()
        self.get_keys: list[str] = []

    def get(self, key: str, default: Any = None) -> Any:
        self.get_keys.append(key)
        return super().get(key, default)


class DummyStreamlit:
    class _Context:
        def __init__(self, parent: "DummyStreamlit") -> None:
            self._parent = parent

        def __enter__(self) -> "DummyStreamlit":
            return self._parent

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def __getattr__(self, name: str) -> Any:
            return getattr(self._parent, name)

    def __init__(self) -> None:
        self.session_state = RecordingSessionState()
        self.page_config_calls: list[dict[str, Any]] = []
        self.title_calls: list[str] = []
        self.warning_messages: list[str] = []
        self.stop_called = False
        self.sidebar = DummyStreamlit._Context(self)

    def set_page_config(self, **kwargs: Any) -> None:
        self.page_config_calls.append(kwargs)

    def title(self, text: str) -> None:
        self.title_calls.append(text)

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)

    def stop(self) -> None:
        self.stop_called = True
        raise AssertionError("validation page should not stop when returns_df exists")

    def selectbox(self, *args: Any, **kwargs: Any) -> Any:
        options = kwargs.get("options") or args[1]
        return options[0]

    def columns(self, spec: int | list[int]) -> list["DummyStreamlit._Context"]:
        count = spec if isinstance(spec, int) else len(spec)
        return [DummyStreamlit._Context(self) for _ in range(count)]

    def expander(self, *args: Any, **kwargs: Any) -> "DummyStreamlit._Context":
        return DummyStreamlit._Context(self)

    def button(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def cache_data(
        self, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def markdown(self, *args: Any, **kwargs: Any) -> None:
        return None

    def header(self, *args: Any, **kwargs: Any) -> None:
        return None

    def subheader(self, *args: Any, **kwargs: Any) -> None:
        return None

    def code(self, *args: Any, **kwargs: Any) -> None:
        return None


def _install_streamlit_stub(
    monkeypatch: pytest.MonkeyPatch,
    stub: DummyStreamlit,
    *,
    active_context: bool,
) -> ModuleType:
    module = ModuleType("streamlit")
    module.__path__ = []  # mark as package so streamlit.runtime imports resolve

    for attr in (
        "set_page_config",
        "title",
        "warning",
        "stop",
        "selectbox",
        "columns",
        "expander",
        "button",
        "cache_data",
        "markdown",
        "header",
        "subheader",
        "code",
    ):
        setattr(module, attr, getattr(stub, attr))
    module.session_state = stub.session_state
    module.sidebar = stub.sidebar

    runtime = ModuleType("streamlit.runtime")
    scriptrunner = ModuleType("streamlit.runtime.scriptrunner")
    scriptrunner.get_script_run_ctx = lambda: object() if active_context else None

    monkeypatch.setitem(sys.modules, "streamlit", module)
    monkeypatch.setitem(sys.modules, "streamlit.runtime", runtime)
    monkeypatch.setitem(sys.modules, "streamlit.runtime.scriptrunner", scriptrunner)
    return module


def _reload_validation_page(monkeypatch: pytest.MonkeyPatch, streamlit_module: ModuleType):
    from streamlit_app import state as app_state

    monkeypatch.setattr(app_state, "st", streamlit_module)
    sys.modules.pop("streamlit_app.developer_settings_validation", None)
    importlib.invalidate_caches()
    return importlib.import_module("streamlit_app.developer_settings_validation")


def test_validation_page_auto_renders_with_uploaded_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = DummyStreamlit()
    stub.session_state.update(
        {
            "show_perf_diagnostics": True,
            "returns_df": pd.DataFrame(
                {"FundA": [0.01, -0.02], "FundB": [0.03, 0.01]},
                index=pd.date_range("2024-01-31", periods=2, freq="ME"),
            ),
            "schema_meta": {"validation": {"issues": [], "warnings": []}},
            "upload_status": "success",
        }
    )
    streamlit_module = _install_streamlit_stub(
        monkeypatch,
        stub,
        active_context=True,
    )

    _reload_validation_page(monkeypatch, streamlit_module)

    assert stub.page_config_calls
    assert any("Developer: Settings Validation" in call for call in stub.title_calls)
    assert "returns_df" in stub.session_state.get_keys
    assert "app_data" not in stub.session_state.get_keys
    assert not any("Please load data" in message for message in stub.warning_messages)
    assert not stub.stop_called


def test_run_test_analysis_uses_public_analysis_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = DummyStreamlit()
    streamlit_module = _install_streamlit_stub(
        monkeypatch,
        stub,
        active_context=False,
    )
    page = _reload_validation_page(monkeypatch, streamlit_module)

    calls: list[tuple[pd.DataFrame, dict[str, Any], str | None]] = []

    def fake_run_analysis(
        returns: pd.DataFrame,
        model_state: dict[str, Any],
        benchmark: str | None,
    ) -> str:
        calls.append((returns, model_state, benchmark))
        return "analysis-result"

    monkeypatch.setattr(page.analysis_runner, "run_analysis", fake_run_analysis)
    monkeypatch.setattr(
        page.analysis_runner,
        "_execute_analysis",
        lambda *_args, **_kwargs: pytest.fail("private analysis API should not be used"),
    )

    returns = pd.DataFrame({"FundA": [0.01, 0.02]})
    model_state = {"selection_count": 1}

    result = page.run_test_analysis(returns, model_state)

    assert result == {"status": "success", "result": "analysis-result"}
    assert calls == [(returns, model_state, None)]
