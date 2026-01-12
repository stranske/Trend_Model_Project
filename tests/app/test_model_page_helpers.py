from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture()
def model_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    def _noop(*_args, **_kwargs):
        return None

    def _passthrough_decorator(*args, **kwargs):
        """Decorator stub that returns the function unchanged."""

        def decorator(fn):
            return fn

        # Handle both @st.cache_data and @st.cache_data(...)
        if args and callable(args[0]):
            return args[0]
        return decorator

    class Context:
        def __enter__(self):
            return stub

        def __exit__(self, *_args):
            return False

    stub = SimpleNamespace()
    stub.session_state = {}
    stub.title = _noop
    stub.error = _noop
    stub.subheader = _noop
    stub.divider = _noop
    stub.info = _noop
    stub.success = _noop
    stub.warning = _noop
    stub.altair_chart = _noop
    stub.markdown = _noop
    stub.caption = _noop
    stub.metric = _noop
    stub.write = _noop
    stub.text_input = lambda _label, value="", **_kwargs: value
    stub.text_area = lambda _label, value="", **_kwargs: value
    stub.radio = lambda _label, options, index=0, **_kwargs: options[index]
    stub.date_input = lambda _label, value=None, **_kwargs: value
    stub.page_link = _noop
    stub.rerun = _noop
    stub.cache_data = _passthrough_decorator
    stub.cache_resource = _passthrough_decorator
    stub.expander = lambda *_args, **_kwargs: Context()
    stub.sidebar = Context()
    stub.form = lambda *_args, **_kwargs: Context()
    stub.form_submit_button = lambda *_args, **_kwargs: False
    stub.button = lambda *_args, **_kwargs: False
    stub.download_button = _noop
    stub.columns = lambda n: [Context() for _ in range(n)]
    stub.selectbox = lambda _label, options, index=0, **_kwargs: options[index]
    stub.number_input = lambda _label, **kwargs: kwargs.get("value", 0)
    stub.checkbox = lambda _label, value=False, **_kwargs: value
    stub.slider = lambda _label, **kwargs: kwargs.get("value", 0)

    monkeypatch.setitem(sys.modules, "streamlit", stub)

    stub.clear_calls = 0

    def mark_clear() -> None:
        stub.clear_calls += 1

    monkeypatch.setattr(
        "streamlit_app.components.analysis_runner.clear_cached_analysis",
        mark_clear,
    )

    from streamlit_app import state as app_state

    monkeypatch.setattr(app_state, "initialize_session_state", lambda: None)
    monkeypatch.setattr(app_state, "st", stub)
    monkeypatch.setattr(
        app_state,
        "get_uploaded_data",
        lambda: (
            pd.DataFrame({f"A{i}": [0.01 + i * 0.001, 0.02 + i * 0.001] for i in range(12)}),
            {},
        ),
    )

    module = importlib.reload(importlib.import_module("streamlit_app.pages.2_Model"))
    return module


def test_validate_model_catches_errors(model_module: ModuleType) -> None:
    """Test that _validate_model catches various validation errors."""
    # Test 1: Selection count exceeds column count
    values = {
        "lookback_periods": 36,
        "min_history_periods": 36,
        "selection_count": 15,  # Exceeds column_count=10
        "metric_weights": {"sharpe": 1.0},
    }
    errors = model_module._validate_model(values, column_count=10)
    assert any("Selection count" in err for err in errors)

    # Test 2: No positive metric weights
    values = {
        "lookback_periods": 36,
        "min_history_periods": 36,
        "selection_count": 5,
        "metric_weights": {"sharpe": 0.0, "return_ann": 0.0},
    }
    errors = model_module._validate_model(values, column_count=10)
    assert any("positive metric weight" in err for err in errors)

    # Test 3: Min history exceeds lookback
    values = {
        "lookback_periods": 24,
        "min_history_periods": 36,  # Exceeds lookback
        "selection_count": 5,
        "metric_weights": {"sharpe": 1.0},
    }
    errors = model_module._validate_model(values, column_count=10)
    assert any("Minimum history" in err for err in errors)


def test_render_model_page_clears_cached_results(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st

    stub.session_state.clear()
    stub.clear_calls = 0

    stub.form_submit_button = lambda *_args, **_kwargs: True

    stub.session_state.update(
        {
            "analysis_result": "cached",
            "analysis_result_key": "old",
            "analysis_error": {"message": "previous"},
        }
    )

    initial_clears = stub.clear_calls

    model_module.render_model_page()

    assert stub.clear_calls == initial_clears + 1
    for key in ["analysis_result", "analysis_result_key", "analysis_error"]:
        assert key not in stub.session_state


def test_render_config_chat_panel_stores_instruction(model_module: ModuleType) -> None:
    stub = model_module.st
    stub.session_state.clear()

    stub.text_area = lambda *_args, **_kwargs: "Increase lookback to 24"
    stub.button = lambda *_args, **_kwargs: True

    model_module.render_config_chat_panel()

    assert (
        stub.session_state.get("config_chat_last_instruction")
        == "Increase lookback to 24"
    )
