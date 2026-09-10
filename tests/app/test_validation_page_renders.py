from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from tests.support.dummy_streamlit import DummyStreamlit


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
    # This page must never reach st.stop() while returns_df exists; fail fast
    # at the offending call rather than only at the end-of-test assertion.
    stub.stop_error = "validation page should not stop when returns_df exists"
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
    # This page must never reach st.stop() while returns_df exists; fail fast
    # at the offending call rather than only at the end-of-test assertion.
    stub.stop_error = "validation page should not stop when returns_df exists"
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
