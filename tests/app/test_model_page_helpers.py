from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
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
    stub.dataframe = _noop
    stub.tabs = lambda labels: [Context() for _ in labels]
    stub.columns = lambda n: [Context() for _ in range(n)]
    stub.selectbox = lambda _label, options, index=0, **_kwargs: options[index]
    stub.number_input = lambda _label, **kwargs: kwargs.get("value", 0)
    stub.checkbox = lambda _label, value=False, **_kwargs: value
    stub.slider = lambda _label, **kwargs: kwargs.get("value", 0)
    stub.file_uploader = lambda *_args, **_kwargs: None
    stub.empty = lambda: Placeholder()

    monkeypatch.setitem(sys.modules, "streamlit", stub)
    runtime_module = ModuleType("streamlit.runtime")
    uploaded_module = ModuleType("streamlit.runtime.uploaded_file_manager")
    uploaded_module.UploadedFile = object
    runtime_module.uploaded_file_manager = uploaded_module
    monkeypatch.setitem(sys.modules, "streamlit.runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "streamlit.runtime.uploaded_file_manager", uploaded_module)

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


def test_current_run_key_changes_with_risk_free(model_module: ModuleType) -> None:
    stub = model_module.st
    stub.session_state.update(
        {
            "data_fingerprint": "abc123",
            "analysis_fund_columns": ["FundA"],
            "fund_columns": ["FundA"],
            "selected_risk_free": "RF1",
        }
    )
    model_state = {"trend_spec": {"window": 63, "lag": 1}}

    key_one = model_module._current_run_key(model_state, None)
    stub.session_state["selected_risk_free"] = "RF2"
    key_two = model_module._current_run_key(model_state, None)

    assert key_one != key_two


def test_render_config_chat_panel_stores_instruction(model_module: ModuleType) -> None:
    stub = model_module.st
    stub.session_state.clear()

    stub.text_area = lambda *_args, **_kwargs: "Increase lookback to 24"
    stub.button = lambda *_args, **_kwargs: True

    model_module.render_config_chat_panel()

    assert stub.session_state.get("config_chat_last_instruction") == "Increase lookback to 24"


def test_build_nl_chain_reuses_cached_chain_with_provider_config(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()
    cache: dict[str, object] = {}

    def fake_cached_config_patch_chain(
        _session_cache_key,
        chain_cache_key,
        _llm_cache_key,
        _api_key,
        _extra_payload,
    ):
        signature = model_module._chain_cache_signature(chain_cache_key)
        if signature not in cache:
            cache[signature] = object()
        return cache[signature]

    monkeypatch.setattr(model_module, "_cached_config_patch_chain", fake_cached_config_patch_chain)
    monkeypatch.setenv("TREND_LLM_TEMPERATURE", "0.2")

    chain_one, meta_one = model_module._build_nl_chain()
    chain_two, meta_two = model_module._build_nl_chain()

    assert chain_one is chain_two
    assert meta_one["chain_reused"] is False
    assert meta_two["chain_reused"] is True


def test_build_nl_chain_invalidates_on_model_change(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()
    cache: dict[str, object] = {}

    def fake_cached_config_patch_chain(
        _session_cache_key,
        chain_cache_key,
        _llm_cache_key,
        _api_key,
        _extra_payload,
    ):
        signature = model_module._chain_cache_signature(chain_cache_key)
        if signature not in cache:
            cache[signature] = object()
        return cache[signature]

    monkeypatch.setattr(model_module, "_cached_config_patch_chain", fake_cached_config_patch_chain)

    stub.session_state[model_module._LLM_PROVIDER_OVERRIDE_KEY] = "openai"
    stub.session_state[model_module._LLM_MODEL_OVERRIDE_KEY] = "gpt-4o-mini"
    stub.session_state["config_chat_preview"] = {"before": {}, "after": {}}
    stub.session_state["config_chat_last_instruction"] = "old"

    _chain, _meta = model_module._build_nl_chain()
    session_id_before = stub.session_state.get(model_module._CONFIG_CHAT_SESSION_KEY)

    stub.session_state[model_module._LLM_MODEL_OVERRIDE_KEY] = "gpt-4.1-mini"
    _chain, meta = model_module._build_nl_chain()

    session_id_after = stub.session_state.get(model_module._CONFIG_CHAT_SESSION_KEY)
    assert session_id_before != session_id_after
    assert meta["chain_cache_invalidation_fields"] == ["model"]
    assert meta["chain_cache_session_reset"] is True
    assert "config_chat_preview" not in stub.session_state
    assert "config_chat_last_instruction" not in stub.session_state


def test_build_nl_chain_invalidates_on_provider_change(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()
    cache: dict[str, object] = {}

    def fake_cached_config_patch_chain(
        _session_cache_key,
        chain_cache_key,
        _llm_cache_key,
        _api_key,
        _extra_payload,
    ):
        signature = model_module._chain_cache_signature(chain_cache_key)
        if signature not in cache:
            cache[signature] = object()
        return cache[signature]

    monkeypatch.setattr(model_module, "_cached_config_patch_chain", fake_cached_config_patch_chain)

    stub.session_state[model_module._LLM_PROVIDER_OVERRIDE_KEY] = "openai"
    stub.session_state[model_module._LLM_MODEL_OVERRIDE_KEY] = "gpt-4o-mini"
    stub.session_state["config_chat_preview"] = {"before": {}, "after": {}}
    stub.session_state["config_chat_last_instruction"] = "old"

    _chain, _meta = model_module._build_nl_chain()
    session_id_before = stub.session_state.get(model_module._CONFIG_CHAT_SESSION_KEY)

    stub.session_state[model_module._LLM_PROVIDER_OVERRIDE_KEY] = "anthropic"
    _chain, meta = model_module._build_nl_chain()

    session_id_after = stub.session_state.get(model_module._CONFIG_CHAT_SESSION_KEY)
    assert session_id_before != session_id_after
    assert meta["chain_cache_invalidation_fields"] == ["provider"]
    assert meta["chain_cache_session_reset"] is True
    assert "config_chat_preview" not in stub.session_state
    assert "config_chat_last_instruction" not in stub.session_state


def test_build_nl_chain_passes_api_key_to_cache(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()
    captured: dict[str, object] = {}

    def fake_cached_config_patch_chain(
        _session_cache_key,
        _chain_cache_key,
        _llm_cache_key,
        api_key_secret,
        _extra_payload,
    ):
        captured["api_key_secret"] = api_key_secret
        return object()

    monkeypatch.setattr(model_module, "_cached_config_patch_chain", fake_cached_config_patch_chain)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")

    model_module._build_nl_chain()

    secret = captured.get("api_key_secret")
    assert secret is not None
    assert secret.value == "test-key-123"
    assert secret.fingerprint == model_module._hash_api_key("test-key-123")


def test_build_nl_chain_invalidates_on_base_url_change(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()
    cache: dict[str, object] = {}

    def fake_cached_config_patch_chain(
        _session_cache_key,
        chain_cache_key,
        _llm_cache_key,
        _api_key,
        _extra_payload,
    ):
        signature = model_module._chain_cache_signature(chain_cache_key)
        if signature not in cache:
            cache[signature] = object()
        return cache[signature]

    monkeypatch.setattr(model_module, "_cached_config_patch_chain", fake_cached_config_patch_chain)

    stub.session_state[model_module._LLM_PROVIDER_OVERRIDE_KEY] = "openai"
    stub.session_state[model_module._LLM_MODEL_OVERRIDE_KEY] = "gpt-4o-mini"
    stub.session_state[model_module._LLM_BASE_URL_OVERRIDE_KEY] = "https://api.one"
    stub.session_state["config_chat_preview"] = {"before": {}, "after": {}}
    stub.session_state["config_chat_last_instruction"] = "old"

    _chain, _meta = model_module._build_nl_chain()
    session_id_before = stub.session_state.get(model_module._CONFIG_CHAT_SESSION_KEY)

    stub.session_state[model_module._LLM_BASE_URL_OVERRIDE_KEY] = "https://api.two"
    _chain, meta = model_module._build_nl_chain()

    session_id_after = stub.session_state.get(model_module._CONFIG_CHAT_SESSION_KEY)
    assert session_id_before != session_id_after
    assert meta["chain_cache_invalidation_fields"] == ["base_url"]
    assert meta["chain_cache_session_reset"] is True
    assert "config_chat_preview" not in stub.session_state
    assert "config_chat_last_instruction" not in stub.session_state


def test_build_nl_chain_invalidates_on_org_change(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()
    cache: dict[str, object] = {}

    def fake_cached_config_patch_chain(
        _session_cache_key,
        chain_cache_key,
        _llm_cache_key,
        _api_key,
        _extra_payload,
    ):
        signature = model_module._chain_cache_signature(chain_cache_key)
        if signature not in cache:
            cache[signature] = object()
        return cache[signature]

    monkeypatch.setattr(model_module, "_cached_config_patch_chain", fake_cached_config_patch_chain)

    stub.session_state[model_module._LLM_PROVIDER_OVERRIDE_KEY] = "openai"
    stub.session_state[model_module._LLM_MODEL_OVERRIDE_KEY] = "gpt-4o-mini"
    stub.session_state[model_module._LLM_ORG_OVERRIDE_KEY] = "org-one"
    stub.session_state["config_chat_preview"] = {"before": {}, "after": {}}
    stub.session_state["config_chat_last_instruction"] = "old"

    _chain, _meta = model_module._build_nl_chain()
    session_id_before = stub.session_state.get(model_module._CONFIG_CHAT_SESSION_KEY)

    stub.session_state[model_module._LLM_ORG_OVERRIDE_KEY] = "org-two"
    _chain, meta = model_module._build_nl_chain()

    session_id_after = stub.session_state.get(model_module._CONFIG_CHAT_SESSION_KEY)
    assert session_id_before != session_id_after
    assert meta["chain_cache_invalidation_fields"] == ["organization"]
    assert meta["chain_cache_session_reset"] is True
    assert "config_chat_preview" not in stub.session_state
    assert "config_chat_last_instruction" not in stub.session_state


def test_build_nl_chain_invalidates_on_temperature_change(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()
    cache: dict[str, object] = {}

    def fake_cached_config_patch_chain(
        _session_cache_key,
        chain_cache_key,
        _llm_cache_key,
        _api_key,
        _extra_payload,
    ):
        signature = model_module._chain_cache_signature(chain_cache_key)
        if signature not in cache:
            cache[signature] = object()
        return cache[signature]

    monkeypatch.setattr(model_module, "_cached_config_patch_chain", fake_cached_config_patch_chain)

    monkeypatch.setenv("TREND_LLM_TEMPERATURE", "0.1")
    _chain, _meta = model_module._build_nl_chain()
    session_id_before = stub.session_state.get(model_module._CONFIG_CHAT_SESSION_KEY)

    stub.session_state["config_chat_preview"] = {"before": {}, "after": {}}
    stub.session_state["config_chat_last_instruction"] = "old"
    monkeypatch.setenv("TREND_LLM_TEMPERATURE", "0.7")
    _chain, meta = model_module._build_nl_chain()

    session_id_after = stub.session_state.get(model_module._CONFIG_CHAT_SESSION_KEY)
    assert session_id_before != session_id_after
    assert meta["chain_cache_invalidation_fields"] == ["temperature"]
    assert meta["chain_cache_session_reset"] is True
    assert "config_chat_preview" not in stub.session_state
    assert "config_chat_last_instruction" not in stub.session_state


def test_current_chain_settings_snapshot_uses_resolved_env(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()

    monkeypatch.setenv("TREND_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("TREND_LLM_MODEL", "claude-3-haiku")
    monkeypatch.setenv("TREND_LLM_BASE_URL", "https://api.example")
    monkeypatch.setenv("TREND_LLM_ORG", "org-x")
    monkeypatch.setenv("TREND_LLM_TEMPERATURE", "0.35")

    snapshot = model_module._current_chain_settings_snapshot()

    assert snapshot["provider"] == "anthropic"
    assert snapshot["model"] == "claude-3-haiku"
    assert snapshot["base_url"] == "https://api.example"
    assert snapshot["organization"] == "org-x"
    assert snapshot["temperature"] == pytest.approx(0.35)


def test_record_preview_timing_tracks_chain_reuse(model_module: ModuleType) -> None:
    stub = model_module.st
    stub.session_state.clear()

    preview = {
        "instruction": "Increase lookback",
        "timings": {
            "chain_cache_key": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.2},
            "chain_cache_signature": "sig-123",
            "chain_resource_signature": "rsig-456",
            "chain_cache_summary": "openai:gpt-4o-mini@0.2",
            "chain_cache_miss_reason": None,
            "chain_cache_invalidation_fields": [],
            "chain_settings_changed": False,
            "chain_llm_changed_fields": [],
            "chain_cache_session_reset": False,
            "chain_build_seconds": 1.2,
            "chain_reused": True,
            "run_seconds": 2.3,
        },
    }

    model_module._record_preview_timing(preview, total_seconds=3.4)

    history = stub.session_state.get(model_module._CONFIG_PREVIEW_TIMINGS_KEY)
    assert isinstance(history, list)
    assert len(history) == 1
    entry = history[0]
    assert entry["chain_reused"] is True
    assert entry["total_seconds"] == pytest.approx(3.4)

    metrics = stub.session_state.get(model_module._CONFIG_CHAIN_METRICS_KEY)
    assert isinstance(metrics, dict)
    assert metrics["chain_reused"] is True


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


def test_unknown_key_review_requires_confirmation(model_module: ModuleType) -> None:
    preview = {"after": {"lookback_periods": 12}, "needs_review": True}
    assert model_module._requires_risky_confirmation(preview) is True


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
    assert tab_sets == [
        ["Unified diff", "Side-by-side"],
        ["Unified diff", "Side-by-side"],
    ]
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


def test_risky_confirmation_dialog_emits_warning_and_flags(
    model_module: ModuleType,
) -> None:
    stub = model_module.st
    stub.session_state.clear()

    preview = {"after": {"lookback_periods": 12}, "risk_flags": ["constraints"]}
    stub.session_state["config_chat_pending_apply"] = {
        "preview": dict(preview),
        "run_analysis": False,
    }

    warnings: list[str] = []
    captions: list[str] = []

    stub.warning = lambda message, **_kwargs: warnings.append(message)
    stub.caption = lambda message, **_kwargs: captions.append(message)

    class DummyContext:
        def __enter__(self):
            return stub

        def __exit__(self, *_args):
            return False

    stub.dialog = lambda *_args, **_kwargs: DummyContext()
    stub.button = lambda *_args, **_kwargs: False

    model_module._render_risky_change_dialog()

    assert any("sensitive configuration" in message.lower() for message in warnings)
    assert any("Flags:" in message for message in captions)


def test_risky_confirmation_dialog_emits_unknown_key_caption(
    model_module: ModuleType,
) -> None:
    stub = model_module.st
    stub.session_state.clear()

    preview = {
        "after": {"lookback_periods": 12},
        "needs_review": True,
        "risk_flags": [],
    }
    stub.session_state["config_chat_pending_apply"] = {
        "preview": dict(preview),
        "run_analysis": False,
    }

    warnings: list[str] = []
    captions: list[str] = []

    stub.warning = lambda message, **_kwargs: warnings.append(message)
    stub.caption = lambda message, **_kwargs: captions.append(message)

    class DummyContext:
        def __enter__(self):
            return stub

        def __exit__(self, *_args):
            return False

    stub.dialog = lambda *_args, **_kwargs: DummyContext()
    stub.button = lambda *_args, **_kwargs: False

    model_module._render_risky_change_dialog()

    assert any("sensitive configuration" in message.lower() for message in warnings)
    assert any("Unknown config keys detected" in message for message in captions)


def test_diff_text_to_html_snapshot(model_module: ModuleType) -> None:
    diff_text = (
        "--- before\n"
        "+++ after\n"
        "@@ -1,3 +1,3 @@\n"
        "-lookback_periods: 6\n"
        "+lookback_periods: 12\n"
        " min_history_periods: 6\n"
    )
    snapshot_path = Path(__file__).parents[1] / "fixtures" / "diff_preview_unified.html"
    expected = snapshot_path.read_text(encoding="utf-8")

    assert model_module._diff_text_to_html(diff_text) == expected


def test_render_side_by_side_diff_snapshot(model_module: ModuleType) -> None:
    stub = model_module.st
    markdown_calls: list[str] = []

    def capture_markdown(body: str, **_kwargs):
        markdown_calls.append(body)

    stub.markdown = capture_markdown

    model_module._render_side_by_side_diff(
        {"lookback_periods": 6, "min_history_periods": 6},
        {"lookback_periods": 12, "min_history_periods": 6},
    )

    snapshot_path = Path(__file__).parents[1] / "fixtures" / "diff_preview_side_by_side.html"
    expected = snapshot_path.read_text(encoding="utf-8")
    assert markdown_calls

    def normalize_ids(value: str) -> str:
        value = re.sub(r"difflib_chg_to\d+__", "difflib_chg_toX__", value)
        value = re.sub(r"id=\"from\d+_", 'id="fromX_', value)
        value = re.sub(r"id=\"to\d+_", 'id="toX_', value)
        value = re.sub(r"href=\"#difflib_chg_to\d+__", 'href="#difflib_chg_toX__', value)
        return value

    assert normalize_ids(markdown_calls[-1]) == normalize_ids(expected)


def test_render_config_diff_preview_renders_tabs_and_diff(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()

    tab_sets: list[list[str]] = []
    unified_calls: list[str] = []
    side_by_side_calls: list[tuple[dict[str, int], dict[str, int]]] = []

    class DummyContext:
        def __enter__(self):
            return stub

        def __exit__(self, *_args):
            return False

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
    stub.tabs = record_tabs

    preview = {
        "before": {"lookback_periods": 6, "min_history_periods": 6},
        "after": {"lookback_periods": 12, "min_history_periods": 6},
        "diff": "--- before\n+++ after\n+ lookback_periods: 12\n",
    }
    stub.session_state["config_chat_preview"] = preview

    model_module._render_config_diff_preview(model_state={"lookback_periods": 6})

    assert tab_sets == [["Unified diff", "Side-by-side"]]
    assert unified_calls == ["--- before\n+++ after\n+ lookback_periods: 12\n"]
    assert side_by_side_calls == [
        (
            {"lookback_periods": 6, "min_history_periods": 6},
            {"lookback_periods": 12, "min_history_periods": 6},
        )
    ]


def test_render_config_diff_preview_no_preview_shows_info(
    model_module: ModuleType,
) -> None:
    stub = model_module.st
    stub.session_state.clear()
    info_calls: list[str] = []

    def capture_info(message: str, **_kwargs):
        info_calls.append(message)

    stub.info = capture_info

    model_module._render_config_diff_preview(model_state={"lookback_periods": 6})

    assert info_calls == ["No preview available yet. Send an instruction to generate a diff."]


def test_record_preview_timing_stores_last_metrics(model_module: ModuleType) -> None:
    stub = model_module.st
    stub.session_state.clear()

    preview = {
        "instruction": "Increase lookback",
        "timings": {
            "chain_cache_key": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0},
            "chain_cache_signature": "abc123",
            "chain_build_seconds": 0.05,
            "chain_reused": True,
            "run_seconds": 1.2,
        },
    }

    model_module._record_preview_timing(preview, total_seconds=1.25)

    metrics = stub.session_state.get(model_module._CONFIG_CHAIN_METRICS_KEY)
    assert isinstance(metrics, dict)
    assert metrics["cache_signature"] == "abc123"
    assert metrics["chain_reused"] is True
    assert metrics["total_seconds"] == 1.25


def test_build_nl_chain_reuses_cached_chain(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()

    chain_obj = object()
    calls: list[dict[str, object]] = []

    def fake_cached_config_patch_chain(
        session_cache_key: str,
        chain_cache_key: dict[str, object],
        llm_cache_key: dict[str, object],
        api_key: object,
        extra_payload: str,
    ) -> object:
        calls.append(
            {
                "provider": chain_cache_key.get("provider"),
                "model": chain_cache_key.get("model"),
                "api_key": api_key,
                "temperature": chain_cache_key.get("temperature"),
            }
        )
        return chain_obj

    monkeypatch.setattr(
        model_module,
        "_resolve_llm_provider_config",
        lambda: model_module.LLMProviderConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
        ),
    )
    monkeypatch.setattr(model_module, "_resolve_llm_temperature", lambda: 0.1)
    monkeypatch.setattr(model_module, "_cached_config_patch_chain", fake_cached_config_patch_chain)

    chain_first, meta_first = model_module._build_nl_chain()
    chain_second, meta_second = model_module._build_nl_chain()

    assert chain_first is chain_obj
    assert chain_second is chain_obj
    assert meta_first["chain_reused"] is False
    assert meta_second["chain_reused"] is True
    assert meta_second["chain_cache_miss_reason"] is None
    api_key_secret = calls[0]["api_key"]
    assert isinstance(api_key_secret, model_module._ApiKeySecret)
    assert api_key_secret.value == "sk-test"


def test_build_nl_chain_invalidation_on_model_change(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()
    stub.session_state["config_chat_preview"] = {"after": {"lookback_periods": 6}}
    stub.session_state["config_chat_last_instruction"] = "Increase lookback"

    configs = iter(
        [
            model_module.LLMProviderConfig(
                provider="openai",
                model="gpt-4o-mini",
                api_key="sk-test",
            ),
            model_module.LLMProviderConfig(
                provider="openai",
                model="gpt-4o",
                api_key="sk-test",
            ),
        ]
    )

    monkeypatch.setattr(model_module, "_resolve_llm_provider_config", lambda: next(configs))
    monkeypatch.setattr(model_module, "_resolve_llm_temperature", lambda: 0.0)

    chain_one = object()
    chain_two = object()
    call_count = {"value": 0}

    def fake_cached_config_patch_chain(
        session_cache_key: str,
        chain_cache_key: dict[str, object],
        llm_cache_key: dict[str, object],
        api_key: str | None,
        extra_payload: str,
    ) -> object:
        call_count["value"] += 1
        return chain_one if call_count["value"] == 1 else chain_two

    monkeypatch.setattr(model_module, "_cached_config_patch_chain", fake_cached_config_patch_chain)

    _, meta_first = model_module._build_nl_chain()
    _, meta_second = model_module._build_nl_chain()

    assert meta_first["chain_reused"] is False
    assert "model" in (meta_second["chain_cache_invalidation_fields"] or [])
    assert "config_chat_preview" not in stub.session_state
    assert "config_chat_last_instruction" not in stub.session_state


def test_llm_session_overrides_win_over_env(
    monkeypatch: pytest.MonkeyPatch, model_module: ModuleType
) -> None:
    stub = model_module.st
    stub.session_state.clear()

    monkeypatch.setenv("TREND_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("TREND_LLM_MODEL", "claude-3")
    monkeypatch.setenv("TREND_LLM_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("TREND_LLM_ORG", "env-org")

    stub.session_state[model_module._LLM_PROVIDER_OVERRIDE_KEY] = "openai"
    stub.session_state[model_module._LLM_MODEL_OVERRIDE_KEY] = "gpt-4o-mini"
    stub.session_state[model_module._LLM_BASE_URL_OVERRIDE_KEY] = "https://override.example.com"
    stub.session_state[model_module._LLM_ORG_OVERRIDE_KEY] = "override-org"

    config = model_module._resolve_llm_provider_config()

    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"
    assert config.base_url == "https://override.example.com"
    assert config.organization == "override-org"


def test_llm_session_override_normalizes_whitespace(model_module: ModuleType) -> None:
    stub = model_module.st
    stub.session_state.clear()

    stub.session_state[model_module._LLM_PROVIDER_OVERRIDE_KEY] = "  OpenAI  "
    stub.session_state[model_module._LLM_MODEL_OVERRIDE_KEY] = "  gpt-4o-mini  "

    overrides = model_module._resolve_llm_session_overrides()

    assert overrides["provider"] == "openai"
    assert overrides["model"] == "gpt-4o-mini"
