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

    class Placeholder:
        def progress(self, *_args, **_kwargs):
            return SimpleNamespace(progress=_noop)

        def empty(self):
            return None

    stub = SimpleNamespace()
    stub.session_state = {}
    stub.title = _noop
    stub.error = _noop
    stub.subheader = _noop
    stub.divider = _noop
    stub.info = _noop
    stub.success = _noop
    stub.warning = _noop
    stub.code = _noop
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
    stub.dialog = lambda *_args, **_kwargs: Context()
    stub.form = lambda *_args, **_kwargs: Context()
    stub.form_submit_button = lambda *_args, **_kwargs: False
    stub.button = lambda *_args, **_kwargs: False
    stub.download_button = _noop
    stub.tabs = lambda labels: [Context() for _ in labels]
    stub.columns = lambda n: [Context() for _ in range(n)]
    stub.selectbox = lambda _label, options, index=0, **_kwargs: options[index]
    stub.number_input = lambda _label, **kwargs: kwargs.get("value", 0)
    stub.checkbox = lambda _label, value=False, **_kwargs: value
    stub.slider = lambda _label, **kwargs: kwargs.get("value", 0)
    stub.empty = lambda: Placeholder()

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

    assert stub.session_state.get("config_chat_last_instruction") == "Increase lookback to 24"


def test_side_by_side_diff_renders_yaml(model_module: ModuleType) -> None:
    stub = model_module.st
    languages: list[str | None] = []

    def capture_code(_value: str, *, language: str | None = None, **_kwargs):
        languages.append(language)

    stub.code = capture_code

    model_module._render_side_by_side_diff({"lookback_periods": 12}, {"lookback_periods": 24})

    assert "yaml" in languages


def test_revert_uses_config_history_stack(model_module: ModuleType) -> None:
    stub = model_module.st
    stub.session_state.clear()

    initial_state = {"lookback_periods": 6, "min_history_periods": 6}
    stub.session_state["model_state"] = dict(initial_state)

    preview_one = {
        "before": dict(initial_state),
        "after": {"lookback_periods": 12, "min_history_periods": 6},
        "instruction": "Increase lookback",
        "diff": "--- before\n+++ after\n",
    }
    model_module._apply_preview_state(preview_one, run_analysis=False)

    preview_two = {
        "before": dict(preview_one["after"]),
        "after": {"lookback_periods": 12, "min_history_periods": 9},
        "instruction": "Increase min history",
        "diff": "--- before\n+++ after\n",
    }
    model_module._apply_preview_state(preview_two, run_analysis=False)

    history = model_module._get_config_change_history()
    assert len(history) == 2

    model_module._revert_last_config_change()
    assert stub.session_state.get("model_state") == preview_two["before"]
    assert len(model_module._get_config_change_history()) == 1

    model_module._revert_last_config_change()
    assert stub.session_state.get("model_state") == initial_state
    assert len(model_module._get_config_change_history()) == 0


def test_render_config_change_history_renders_entries(model_module: ModuleType) -> None:
    stub = model_module.st
    stub.session_state.clear()

    initial_state = {"lookback_periods": 6, "min_history_periods": 6}
    stub.session_state["model_state"] = dict(initial_state)

    preview = {
        "before": dict(initial_state),
        "after": {"lookback_periods": 12, "min_history_periods": 6},
        "instruction": "Increase lookback",
        "diff": "--- before\n+++ after\n",
    }
    model_module._apply_preview_state(preview, run_analysis=False)

    history_len = len(model_module._get_config_change_history())
    model_module._render_config_change_history()
    assert len(model_module._get_config_change_history()) == history_len


def test_risky_change_requires_confirmation(model_module: ModuleType) -> None:
    stub = model_module.st
    stub.session_state.clear()

    preview = {"after": {"lookback_periods": 12}, "risk_flags": ["constraints"]}
    assert model_module._requires_risky_confirmation(preview) is True

    model_module._queue_risky_apply(preview, run_analysis=False)
    pending = stub.session_state.get("config_chat_pending_apply")
    assert isinstance(pending, dict)
    assert pending.get("preview") == preview


def test_render_config_change_history_shows_tabs_and_entries(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()

    expander_labels: list[str] = []
    tab_sets: list[list[str]] = []
    unified_calls: list[str] = []
    side_by_side_calls: list[tuple[dict[str, int], dict[str, int]]] = []

    class DummyContext:
        def __enter__(self):
            return stub

        def __exit__(self, *_args):
            return False

    def record_expander(label: str, *args, **kwargs):
        expander_labels.append(label)
        return DummyContext()

    def record_tabs(labels: list[str]):
        tab_sets.append(list(labels))
        return [DummyContext() for _ in labels]

    monkeypatch.setattr(
        model_module, "_render_unified_diff", lambda diff: unified_calls.append(diff)
    )
    monkeypatch.setattr(
        model_module,
        "_render_side_by_side_diff",
        lambda before, after: side_by_side_calls.append((dict(before), dict(after))),
    )
    stub.expander = record_expander
    stub.tabs = record_tabs

    history = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "instruction": "Increase lookback",
            "before": {"lookback_periods": 6},
            "after": {"lookback_periods": 12},
            "diff": "--- before\n+++ after\n+  lookback_periods: 12\n",
        },
        {
            "timestamp": "2024-01-02T00:00:00Z",
            "instruction": "Increase min history",
            "before": {"min_history_periods": 6},
            "after": {"min_history_periods": 9},
            "diff": "--- before\n+++ after\n+  min_history_periods: 9\n",
        },
    ]
    stub.session_state[model_module._CONFIG_HISTORY_KEY] = history

    model_module._render_config_change_history()

    assert expander_labels == [
        "2024-01-02T00:00:00Z • Increase min history",
        "2024-01-01T00:00:00Z • Increase lookback",
    ]
    assert tab_sets == [["Unified diff", "Side-by-side"], ["Unified diff", "Side-by-side"]]
    assert unified_calls == [
        "--- before\n+++ after\n+  min_history_periods: 9\n",
        "--- before\n+++ after\n+  lookback_periods: 12\n",
    ]
    assert side_by_side_calls == [
        ({"min_history_periods": 6}, {"min_history_periods": 9}),
        ({"lookback_periods": 6}, {"lookback_periods": 12}),
    ]


def test_render_config_chat_revert_restores_previous_state(
    model_module: ModuleType,
) -> None:
    stub = model_module.st
    stub.session_state.clear()

    initial_state = {"lookback_periods": 6, "min_history_periods": 6}
    updated_state = {"lookback_periods": 12, "min_history_periods": 6}

    stub.session_state["model_state"] = dict(updated_state)
    stub.session_state["config_chat_preview"] = {"after": dict(updated_state)}
    stub.session_state[model_module._CONFIG_HISTORY_KEY] = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "instruction": "Increase lookback",
            "before": dict(initial_state),
            "after": dict(updated_state),
            "diff": "--- before\n+++ after\n",
        }
    ]

    def button_handler(label: str, *, key: str | None = None, **_kwargs):
        return key == "config_chat_revert_btn"

    class DummyContext:
        def __enter__(self):
            return stub

        def __exit__(self, *_args):
            return False

    stub.button = button_handler
    stub.expander = lambda *_args, **_kwargs: DummyContext()

    model_module.render_config_chat_panel(location="main", model_state=updated_state)

    assert stub.session_state.get("model_state") == initial_state
    assert stub.session_state.get("config_chat_preview") is None
    assert stub.session_state.get(model_module._CONFIG_HISTORY_KEY) == []


def test_risky_apply_requires_confirmation_dialog(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()

    model_state = {"lookback_periods": 6}
    preview = {"after": {"lookback_periods": 12}, "risk_flags": ["constraints"]}

    stub.session_state["model_state"] = dict(model_state)
    stub.session_state["config_chat_preview"] = dict(preview)

    dialog_titles: list[str] = []

    class DummyContext:
        def __enter__(self):
            return stub

        def __exit__(self, *_args):
            return False

    def record_dialog(title: str):
        dialog_titles.append(title)
        return DummyContext()

    def button_handler(label: str, *, key: str | None = None, **_kwargs):
        return key == "config_chat_apply_btn"

    applied: list[dict[str, object]] = []
    monkeypatch.setattr(
        model_module,
        "_apply_preview_state",
        lambda *args, **kwargs: applied.append({"args": args, "kwargs": kwargs}),
    )

    stub.dialog = record_dialog
    stub.button = button_handler

    model_module.render_config_chat_panel(location="main", model_state=model_state)

    assert dialog_titles == ["Confirm risky change"]
    assert applied == []
    assert "config_chat_pending_apply" in stub.session_state


def test_risky_confirmation_apply_uses_dialog_confirm(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()

    preview = {"after": {"lookback_periods": 12}, "risk_flags": ["constraints"]}
    stub.session_state["config_chat_pending_apply"] = {
        "preview": dict(preview),
        "run_analysis": False,
    }

    dialog_titles: list[str] = []

    class DummyContext:
        def __enter__(self):
            return stub

        def __exit__(self, *_args):
            return False

    def record_dialog(title: str):
        dialog_titles.append(title)
        return DummyContext()

    def button_handler(label: str, **_kwargs):
        return label == "Apply anyway"

    applied: list[dict[str, object]] = []
    monkeypatch.setattr(
        model_module,
        "_apply_preview_state",
        lambda *args, **kwargs: applied.append({"args": args, "kwargs": kwargs}),
    )

    stub.dialog = record_dialog
    stub.button = button_handler

    model_module._render_risky_change_dialog()

    assert dialog_titles == ["Confirm risky change"]
    assert applied == [{"args": (preview,), "kwargs": {"run_analysis": False}}]
    assert "config_chat_pending_apply" not in stub.session_state
