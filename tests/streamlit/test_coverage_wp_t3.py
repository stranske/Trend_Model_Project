"""Coverage for the helper and panel half of the Model page (work package t3).

``streamlit_app/pages/2_Model.py`` is imported fresh for every test under the shared
``DummyStreamlit`` stub (extended with the few widgets and recorders the panels need), so each
helper runs against a real ``st.session_state`` and every rendered message is recorded.  Every
LLM, provider and analysis call is replaced by a local fake: nothing here reaches the network, a
process, or any path outside ``tmp_path``.
"""

from __future__ import annotations

import importlib
import logging
import operator
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from tests.support.dummy_streamlit import DummyStreamlit, Placeholder, StubContext
from trend_analysis.config.patch import ConfigPatch, PatchOperation
from trend_analysis.llm import ConfigPatchChain, ConfigPatchVariantsChain, LLMProviderConfig

_PAGE = "streamlit_app.pages.2_Model"

# Everything the page reads from the process environment; cleared so a developer machine and the
# CI runner see the same page.
_LLM_ENV_VARS = (
    "TREND_LLM_PROVIDER",
    "TREND_LLM_MODEL",
    "TREND_LLM_BASE_URL",
    "TREND_LLM_ORG",
    "TREND_LLM_TEMPERATURE",
    "TREND_LLM_ZONE",
    "TS_STREAMLIT_API_KEY",
    "TREND_LLM_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_STRANSKE",
    "TS_LLM_PROXY_URL",
    "TS_LLM_PROXY_TOKEN",
)


class _Slot(Placeholder):
    """``st.empty()`` stand-in that also records the status messages sent to it."""

    def __init__(self, stub: DummyStreamlit) -> None:
        super().__init__(stub)
        self.events: list[tuple[str, str]] = []

    def info(self, message: Any = "") -> None:
        self.events.append(("info", str(message)))

    def success(self, message: Any = "") -> None:
        self.events.append(("success", str(message)))

    def error(self, message: Any = "") -> None:
        self.events.append(("error", str(message)))


class ModelPageStreamlit(DummyStreamlit):
    """The shared stub plus the widgets and recorders the Model page panels use.

    Buttons report a click when their ``key`` or label is in ``clicked``; keyed text widgets
    read their value from ``session_state`` the way Streamlit does.
    """

    def __init__(self) -> None:
        super().__init__()
        self.clicked: set[str] = set()
        self.button_calls: list[dict[str, Any]] = []
        self.markdown_calls: list[tuple[str, dict[str, Any]]] = []
        self.dataframe_calls: list[tuple[Any, dict[str, Any]]] = []
        self.selectbox_kwargs: list[dict[str, Any]] = []
        self.spinner_messages: list[str] = []
        self.dialog_titles: list[str] = []
        self.slots: list[_Slot] = []

    def markdown(self, text: Any = "", *args: Any, **kwargs: Any) -> None:
        self.markdown_calls.append((str(text), dict(kwargs)))
        super().markdown(text, *args, **kwargs)

    def dataframe(self, df: Any, **kwargs: Any) -> None:
        self.dataframe_calls.append((df, dict(kwargs)))
        super().dataframe(df, **kwargs)

    def selectbox(self, *args: Any, **kwargs: Any) -> Any:
        self.selectbox_kwargs.append(dict(kwargs))
        return super().selectbox(*args, **kwargs)

    def button(self, label: str = "", *_args: Any, key: str | None = None, **kwargs: Any) -> bool:
        self.button_labels.append(label)
        self.button_calls.append({"label": label, "key": key, **kwargs})
        return label in self.clicked or key in self.clicked

    def text_area(
        self, label: str = "", value: str = "", *, key: str | None = None, **_kwargs: Any
    ) -> str:
        if key is not None:
            return str(self.session_state.get(key, value))
        return value

    def text_input(
        self, label: str = "", value: str = "", *, key: str | None = None, **_kwargs: Any
    ) -> str:
        if self.text_input_returns:
            return self.text_input_returns.pop(0)
        if key is not None:
            return str(self.session_state.get(key, value))
        return value

    def spinner(self, message: str = "", *_args: Any, **_kwargs: Any) -> StubContext:
        self.spinner_messages.append(message)
        return StubContext(self)

    def dialog(self, title: str, *_args: Any, **_kwargs: Any) -> StubContext:
        self.dialog_titles.append(title)
        return StubContext(self)

    def download_button(self, label: str = "", **kwargs: Any) -> None:
        self.downloads.append({"label": label, **kwargs})

    def empty(self) -> _Slot:
        slot = _Slot(self)
        self.slots.append(slot)
        return slot

    def cache_resource(self, *args: Any, **_kwargs: Any) -> Any:
        if args and callable(args[0]):
            return args[0]
        return lambda func: func


ModelPage = tuple[ModuleType, ModelPageStreamlit]


@pytest.fixture()
def model_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModelPage:
    """Import the Model page against a fresh stub with a clean environment."""

    for name in _LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    stub = ModelPageStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    for name in [m for m in sys.modules if m == "streamlit_app" or m.startswith("streamlit_app.")]:
        monkeypatch.delitem(sys.modules, name)
    page = importlib.import_module(_PAGE)
    # Importing the page injects the theme CSS; clear it so each test sees only its own output.
    stub.markdowns.clear()
    stub.markdown_calls.clear()
    return page, stub


# --- chain cache state and signatures -------------------------------------------------------


def test_chain_cache_state_repairs_a_corrupt_entries_map(model_page: ModelPage) -> None:
    page, st = model_page
    key = page._CONFIG_CHAIN_STATE_KEY

    st.session_state[key] = {"entries": {"sig": 7}, "last_signature": "sig"}
    assert page._get_chain_cache_state() == {"entries": {"sig": 7}, "last_signature": "sig"}

    corrupt = {"entries": ["not", "a", "dict"], "last_signature": "sig"}
    st.session_state[key] = corrupt
    repaired = page._get_chain_cache_state()
    assert repaired is corrupt
    assert repaired == {"entries": {}, "last_signature": "sig"}

    st.session_state[key] = "not a dict"
    fresh = page._get_chain_cache_state()
    assert fresh == {"entries": {}}
    assert st.session_state[key] is fresh


def test_chain_cache_signature_from_inputs_normalizes_before_hashing(
    model_page: ModelPage,
) -> None:
    page, _st = model_page

    assert page._chain_cache_signature_from_inputs(
        "  anthropic ", "claude-3", " https://llm.example ", "org-1", 0.25
    ) == ("511487fc2eb12449edc2b4e02bf111dfbd97a2a96f2f132c4dc164ed1cf60545")
    assert page._chain_cache_signature_from_inputs("openai", "   ", None, "  ", 0.123456) == (
        "47ba8325330ea2c93bc9512edb18930c1ea14789cc621c1fd4498c3e77159e2c"
    )


# --- config change history ------------------------------------------------------------------


def test_record_config_change_ignores_previews_without_both_snapshots(
    model_page: ModelPage,
) -> None:
    page, st = model_page

    page._record_config_change({"before": None, "after": {"a": 1}})
    page._record_config_change({"before": {"a": 1}, "after": ["not", "a", "mapping"]})
    page._record_config_change({"instruction": "no snapshots"})
    assert page._CONFIG_HISTORY_KEY not in st.session_state

    page._record_config_change({"before": {"a": 1}, "after": {"a": 2}, "instruction": "bump a"})
    history = st.session_state[page._CONFIG_HISTORY_KEY]
    assert [entry["instruction"] for entry in history] == ["bump a"]
    assert history[0]["before"] == {"a": 1}
    assert history[0]["after"] == {"a": 2}


def test_record_config_change_keeps_only_the_latest_twenty_entries(model_page: ModelPage) -> None:
    page, st = model_page

    def record(index: int) -> None:
        page._record_config_change(
            {"before": {"n": index}, "after": {"n": index + 1}, "instruction": f"change-{index}"}
        )

    for index in range(20):
        record(index)
    history = st.session_state[page._CONFIG_HISTORY_KEY]
    assert len(history) == 20
    assert history[0]["instruction"] == "change-0"

    record(20)
    history = st.session_state[page._CONFIG_HISTORY_KEY]
    assert [entry["instruction"] for entry in history] == [f"change-{i}" for i in range(1, 21)]


# --- preview timing -------------------------------------------------------------------------


def test_record_preview_timing_ignores_previews_without_timings(model_page: ModelPage) -> None:
    page, st = model_page

    page._record_preview_timing({"instruction": "no timings"}, total_seconds=1.0)
    page._record_preview_timing({"instruction": "text", "timings": "slow"}, total_seconds=1.0)

    assert page._CONFIG_PREVIEW_TIMINGS_KEY not in st.session_state
    assert page._CONFIG_CHAIN_METRICS_KEY not in st.session_state


def test_record_preview_timing_keeps_only_the_latest_twenty_entries(model_page: ModelPage) -> None:
    page, st = model_page

    def record(index: int) -> None:
        page._record_preview_timing(
            {"instruction": f"preview-{index}", "timings": {}}, total_seconds=float(index)
        )

    for index in range(20):
        record(index)
    history = st.session_state[page._CONFIG_PREVIEW_TIMINGS_KEY]
    assert len(history) == 20
    assert history[0]["instruction"] == "preview-0"

    record(20)
    history = st.session_state[page._CONFIG_PREVIEW_TIMINGS_KEY]
    assert [entry["instruction"] for entry in history] == [f"preview-{i}" for i in range(1, 21)]
    assert history[-1]["total_seconds"] == 20.0


def test_record_preview_timing_logs_a_summary_only_when_info_is_enabled(
    model_page: ModelPage, caplog: pytest.LogCaptureFixture
) -> None:
    page, _st = model_page
    preview = {
        "instruction": "Raise lookback",
        "timings": {
            "chain_cache_summary": "openai:gpt-4o-mini@0.200",
            "chain_cache_miss_reason": "settings_changed: model",
            "chain_cache_invalidation_fields": ["model", "base_url"],
            "chain_reused": False,
            "chain_build_seconds": 1.234,
            "run_seconds": 2.0,
        },
    }

    with caplog.at_level(logging.WARNING, logger=page._LOGGER.name):
        page._record_preview_timing(preview, total_seconds=3.5)
    assert caplog.records == []

    with caplog.at_level(logging.INFO, logger=page._LOGGER.name):
        page._record_preview_timing(preview, total_seconds=3.5)
    assert caplog.messages[0] == (
        "Config chat preview timing: reused=no build=1.23s run=2.00s total=3.50s "
        "cache=openai:gpt-4o-mini@0.200 miss=settings_changed: model "
        "invalidated_by=model, base_url"
    )


def test_record_chain_cache_stats_counts_hits_and_misses(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    monkeypatch.setattr(page, "_utc_timestamp", lambda: "2026-10-25T10:00:00Z")
    st.session_state[page._CONFIG_CHAIN_STATS_KEY] = {"hits": 2, "misses": 1}

    page._record_chain_cache_stats(
        {
            "chain_reused": True,
            "chain_cache_signature": "sig-a",
            "chain_cache_invalidation_fields": ["model"],
        },
        chain_build_seconds=0.5,
    )
    assert st.session_state[page._CONFIG_CHAIN_STATS_KEY] == {
        "hits": 3,
        "misses": 1,
        "last_build_seconds": 0.5,
        "last_reused": True,
        "last_miss_reason": None,
        "last_invalidation_fields": ["model"],
        "last_signature": "sig-a",
        "timestamp": "2026-10-25T10:00:00Z",
    }

    page._record_chain_cache_stats(
        {"chain_reused": False, "chain_cache_miss_reason": "first_build"}, chain_build_seconds=1.25
    )
    stats = st.session_state[page._CONFIG_CHAIN_STATS_KEY]
    assert (stats["hits"], stats["misses"], stats["last_reused"]) == (3, 2, False)
    assert stats["last_miss_reason"] == "first_build"
    assert stats["last_build_seconds"] == 1.25


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "—"),
        ("n/a", "—"),
        (object(), "—"),
        (0, "0.00s"),
        (1.234, "1.23s"),
        ("2.5", "2.50s"),
    ],
)
def test_format_seconds_dashes_missing_and_unparseable_values(
    model_page: ModelPage, value: object, expected: str
) -> None:
    page, _st = model_page

    assert page._format_seconds(value) == expected


def test_preview_timing_summary_without_usable_timings(model_page: ModelPage) -> None:
    page, _st = model_page

    assert page._preview_timing_summary(None) == "Preview timing unavailable."
    assert page._preview_timing_summary("slow", total_seconds=1.0) == "Preview timing unavailable."
    summary = page._preview_timing_summary(
        {
            "chain_reused": False,
            "chain_build_seconds": 0.5,
            "chain_lookup_seconds": None,
            "run_seconds": "n/a",
            "total_seconds": 2,
        }
    )
    assert summary == "Preview timing: cache miss | build 0.50s | lookup — | run — | total 2.00s"


def test_render_preview_timing_history_renders_latest_first_rows(model_page: ModelPage) -> None:
    page, st = model_page
    full = {
        "timestamp": "2026-10-25T10:00:00Z",
        "instruction": "Raise lookback",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.2,
        "cache_signature": "abcdef1234567890",
        "resource_signature": "0123456789abcdef",
        "cache_summary": "openai:gpt-4o-mini@0.200",
        "cache_miss_reason": "settings_changed: model",
        "cache_invalidation_fields": ["model", "temperature"],
        "cache_llm_changed_fields": ["timeout"],
        "cache_settings_changed": True,
        "cache_session_reset": True,
        "chain_build_seconds": 1.234,
        "chain_lookup_seconds": 0.25,
        "chain_reused": False,
        "run_seconds": 2,
        "total_seconds": 3.5,
    }
    steady = {
        "timestamp": "2026-10-25T10:05:00Z",
        "instruction": "Tighten max weight",
        "provider": "anthropic",
        "model": "claude-3-haiku",
        "temperature": 0.0,
        "cache_signature": "zzzzzzzzzz",
        "cache_invalidation_fields": [],
        "cache_llm_changed_fields": ["max_retries", "timeout"],
        "cache_settings_changed": False,
        "cache_session_reset": False,
        "chain_build_seconds": 0.0,
        "chain_reused": True,
        "run_seconds": "oops",
        "total_seconds": 1,
    }
    st.session_state[page._CONFIG_PREVIEW_TIMINGS_KEY] = [full, "junk", steady, {}]

    page._render_preview_timing_history()

    assert st.markdowns == ["**Preview timings**"]
    assert len(st.dataframe_calls) == 1
    rows, kwargs = st.dataframe_calls[0]
    assert kwargs == {"use_container_width": True, "hide_index": True}
    assert rows == [
        {
            "Timestamp": "Unknown time",
            "Instruction": "Preview",
            "Provider": "default",
            "Model": "default",
            "Temp": "—",
            "Settings changed": "—",
            "Session cache reset": "No",
            "Cache key": "—",
            "Cache sig": "—",
            "Resource sig": "—",
            "Cache miss": "—",
            "Cache invalidated by": "—",
            "LLM changed": "—",
            "Chain build": "—",
            "Chain lookup": "—",
            "Chain reused": "No",
            "Run": "—",
            "Total": "—",
        },
        {
            "Timestamp": "2026-10-25T10:05:00Z",
            "Instruction": "Tighten max weight",
            "Provider": "anthropic",
            "Model": "claude-3-haiku",
            "Temp": "0.000",
            "Settings changed": "No",
            "Session cache reset": "No",
            "Cache key": "—",
            "Cache sig": "zzzzzzzz",
            "Resource sig": "—",
            "Cache miss": "—",
            "Cache invalidated by": "",
            "LLM changed": "max_retries, timeout",
            "Chain build": "0.00s",
            "Chain lookup": "—",
            "Chain reused": "Yes",
            "Run": "—",
            "Total": "1.00s",
        },
        {
            "Timestamp": "2026-10-25T10:00:00Z",
            "Instruction": "Raise lookback",
            "Provider": "openai",
            "Model": "gpt-4o-mini",
            "Temp": "0.200",
            "Settings changed": "Yes",
            "Session cache reset": "Yes",
            "Cache key": "openai:gpt-4o-mini@0.200",
            "Cache sig": "abcdef12",
            "Resource sig": "01234567",
            "Cache miss": "settings_changed: model",
            "Cache invalidated by": "model, temperature",
            "LLM changed": "timeout",
            "Chain build": "1.23s",
            "Chain lookup": "0.25s",
            "Chain reused": "No",
            "Run": "2.00s",
            "Total": "3.50s",
        },
    ]


def test_render_preview_timing_history_reports_when_no_entry_is_usable(
    model_page: ModelPage,
) -> None:
    page, st = model_page
    st.session_state[page._CONFIG_PREVIEW_TIMINGS_KEY] = ["junk", 5, None]

    page._render_preview_timing_history()

    assert st.info_messages == ["No preview timing data yet."]
    assert st.dataframe_calls == []


def test_render_last_preview_metrics_reports_cache_and_hit_rate(model_page: ModelPage) -> None:
    page, st = model_page
    st.session_state[page._CONFIG_CHAIN_METRICS_KEY] = {
        "cache_signature": "sig1234567890",
        "resource_signature": "res9876543210",
        "cache_summary": "openai:gpt-4o-mini@0.200",
        "chain_reused": True,
        "cache_miss_reason": "first_build",
        "cache_invalidation_fields": ["provider"],
        "cache_llm_changed_fields": ["timeout", "max_retries"],
        "cache_settings_changed": False,
        "cache_session_reset": False,
        "chain_build_seconds": 0.5,
        "chain_lookup_seconds": 0.25,
        "run_seconds": 2.0,
        "total_seconds": 2.75,
    }
    st.session_state[page._CONFIG_CHAIN_STATS_KEY] = {
        "hits": 3,
        "misses": 1,
        "last_build_seconds": 0.5,
        "last_invalidation_fields": ["model", "base_url"],
    }

    page._render_last_preview_metrics()

    assert st.captions == [
        "Last preview cache — Key: openai:gpt-4o-mini@0.200 | Sig: sig12345 | "
        "Resource sig: res98765 | Chain reused: Yes | Cache miss: first_build | "
        "Invalidated by: provider | LLM changed: timeout, max_retries | "
        "Settings changed: No | Session reset: No | Build: 0.50s | Lookup: 0.25s | "
        "Run: 2.00s | Total: 2.75s",
        "Cache stats — Hits: 3 | Misses: 1 | Hit rate: 75% | Last build: 0.50s | "
        "Last invalidation: model, base_url",
    ]

    st.captions.clear()
    st.session_state[page._CONFIG_CHAIN_STATS_KEY] = {"hits": 2, "misses": 1}
    page._render_last_preview_metrics()
    assert st.captions[1].startswith("Cache stats — Hits: 2 | Misses: 1 | Hit rate: 67% | ")


def test_render_last_preview_metrics_with_sparse_data(model_page: ModelPage) -> None:
    page, st = model_page
    st.session_state[page._CONFIG_CHAIN_METRICS_KEY] = {}
    sparse_metrics = (
        "Last preview cache — Key: — | Sig: — | Resource sig: — | Chain reused: No | "
        "Cache miss: — | Invalidated by: — | LLM changed: — | Settings changed: — | "
        "Session reset: No | Build: — | Lookup: — | Run: — | Total: —"
    )

    page._render_last_preview_metrics()
    assert st.captions == [sparse_metrics]

    st.captions.clear()
    st.session_state[page._CONFIG_CHAIN_STATS_KEY] = {}
    page._render_last_preview_metrics()
    assert st.captions == [
        sparse_metrics,
        "Cache stats — Hits: 0 | Misses: 0 | Hit rate: — | Last build: — | Last invalidation: —",
    ]


# --- small formatters -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "—"),
        ("n/a", "—"),
        (object(), "—"),
        (0.1234, "12.3%"),
        ("0.5", "50.0%"),
        (0, "0.0%"),
        (1, "100.0%"),
    ],
)
def test_format_percent_scales_fractions_and_dashes_the_rest(
    model_page: ModelPage, value: object, expected: str
) -> None:
    page, _st = model_page

    assert page._format_percent(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("12.5", 12.5),
        (-3, 0.0),
        (0, 0.0),
        (100, 100.0),
        (150, 100.0),
        ("n/a", 0.0),
        (None, 0.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
    ],
)
def test_finite_nonnegative_float_clamps_into_range(
    model_page: ModelPage, value: object, expected: float
) -> None:
    page, _st = model_page

    assert page._finite_nonnegative_float(value, maximum=100.0) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "—"),
        (True, "Yes"),
        (False, "No"),
        (0.5, "0.500"),
        (3, "3"),
        ("x", "x"),
    ],
)
def test_format_value_renders_booleans_as_yes_no(
    model_page: ModelPage, value: object, expected: str
) -> None:
    page, _st = model_page

    assert page._format_value(value) == expected


def test_render_validation_error_styles_injects_scoped_css(model_page: ModelPage) -> None:
    page, st = model_page

    page._render_validation_error_styles()

    assert len(st.markdown_calls) == 1
    css, kwargs = st.markdown_calls[0]
    assert kwargs == {"unsafe_allow_html": True}
    assert css.startswith("\n<style>\n.validation-errors {\n  border: 1px solid #fecaca;")
    assert css.rstrip().endswith("</style>")


def test_render_validation_errors_lists_escaped_messages(model_page: ModelPage) -> None:
    page, st = model_page

    page._render_validation_errors([])
    assert st.markdown_calls == []

    page._render_validation_errors(
        ["Selection count (15) cannot exceed available assets (10).", "Use <b>bold</b> & more"]
    )
    styles, banner = st.markdown_calls
    assert ".validation-errors {" in styles[0]
    assert banner == (
        '<div class="validation-errors"><strong>Validation errors</strong><ul>'
        "<li>Selection count (15) cannot exceed available assets (10).</li>"
        "<li>Use &lt;b&gt;bold&lt;/b&gt; &amp; more</li></ul></div>",
        {"unsafe_allow_html": True},
    )


def test_config_status_label_reports_each_sync_state(model_page: ModelPage) -> None:
    page, st = model_page
    state = {"lookback_periods": 12, "selection_count": 10}

    st.session_state["active_saved_model_name"] = "Core"
    assert page._config_status_label(None) == ("Not loaded", "⚪")
    assert page._config_status_label({}) == ("Not loaded", "⚪")

    del st.session_state["active_saved_model_name"]
    assert page._config_status_label(state) == ("Unsaved", "🟡")

    st.session_state["active_saved_model_name"] = "Core"
    assert page._config_status_label(state) == ("Saved 'Core'", "🟢")
    st.session_state["last_loaded_model_state"] = "not a mapping"
    assert page._config_status_label(state) == ("Saved 'Core'", "🟢")

    st.session_state["last_loaded_model_state"] = dict(state)
    assert page._config_status_label(state) == ("In sync with 'Core'", "🟢")

    st.session_state["last_loaded_model_state"] = {"lookback_periods": 6, "selection_count": 10}
    assert page._config_status_label(state) == ("Modified (from 'Core')", "🟠")


def test_build_config_wrapper_copies_state_and_keeps_only_present_session_values(
    model_page: ModelPage,
) -> None:
    page, st = model_page
    model_state = {"lookback_periods": 12}
    st.session_state.update(
        {
            "analysis_fund_columns": [],
            "fund_columns": ["FundA", "FundB"],
            "data_fingerprint": "abc123",
            "uploaded_filename": "",
            "selected_benchmark": "SPX",
            "selected_risk_free": None,
        }
    )

    wrapper = page._build_config_wrapper(model_state)

    assert wrapper == {
        "model_state": {"lookback_periods": 12},
        "analysis_fund_columns": [],
        "fund_columns": ["FundA", "FundB"],
        "data_fingerprint": "abc123",
        "uploaded_filename": "",
        "selected_benchmark": "SPX",
    }
    assert wrapper["model_state"] is not model_state


# --- widget state ---------------------------------------------------------------------------


def test_reset_model_widget_state_clears_widget_and_metric_keys_only(
    model_page: ModelPage,
) -> None:
    page, st = model_page
    for key in page._MODEL_WIDGET_KEYS:
        st.session_state[key] = "stale"
    st.session_state.update(
        {
            "metric_sharpe": 1.0,
            "metric_custom": 0.5,
            "metrics_summary": "keep",
            "my_metric_x": "keep",
            "model_state": {"a": 1},
        }
    )
    st.session_state[7] = "integer key"

    page._reset_model_widget_state()

    assert dict(st.session_state) == {
        "metrics_summary": "keep",
        "my_metric_x": "keep",
        "model_state": {"a": 1},
        7: "integer key",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (date(2024, 1, 31), date(2024, 1, 31)),
        ("2024-01-31", date(2024, 1, 31)),
        ("2024-01-31T09:30:00", date(2024, 1, 31)),
        ("2024-02-30", None),
        ("31/01/2024", None),
        ("  2024-01-31", None),
        ("", None),
        (None, None),
        (20240131, None),
    ],
)
def test_parse_date_accepts_dates_and_iso_prefixes(
    model_page: ModelPage, value: object, expected: date | None
) -> None:
    page, _st = model_page

    assert page._parse_date(value) == expected


def test_sync_model_widgets_copies_every_supported_field(model_page: ModelPage) -> None:
    page, st = model_page

    page._sync_model_widgets_from_state(
        {
            "date_mode": "explicit",
            "start_date": "2023-01-31",
            "end_date": date(2024, 12, 31),
            "preset": "Aggressive",
            "weighting_name": "risk_parity",
            "inclusion_approach": "top_pct",
            "buy_hold_initial": "top_n",
            "rank_pct": 0.25,
            "mp_min_funds": 3.0,
            "mp_max_funds": 12,
            "info_ratio_benchmark": "SPX",
            "metric_weights": {"sharpe": 1.0, "return_ann": 0.5},
        }
    )

    assert dict(st.session_state) == {
        "date_mode_radio": "explicit",
        "sim_start_date": date(2023, 1, 31),
        "sim_end_date": date(2024, 12, 31),
        "preset_selector": "Aggressive",
        "weighting_name_selector": "risk_parity",
        "inclusion_approach_select": "top_pct",
        "buy_hold_initial_select": "top_n",
        "rank_pct_primary": 25,
        "mp_min_funds_input": 3,
        "mp_max_funds_input": 12,
        "benchmark_selector": "SPX",
        "metric_sharpe": 1.0,
        "metric_return_ann": 0.5,
    }
    assert isinstance(st.session_state["rank_pct_primary"], int)
    assert isinstance(st.session_state["mp_min_funds_input"], int)


def test_sync_model_widgets_ignores_missing_and_invalid_fields(model_page: ModelPage) -> None:
    page, st = model_page
    stale = {
        "date_mode_radio": "relative",
        "sim_start_date": date(2000, 1, 1),
        "preset_selector": "Old",
    }
    st.session_state.update(stale)

    page._sync_model_widgets_from_state(
        {
            "date_mode": "fortnightly",
            "start_date": None,
            "end_date": "not-a-date",
            "preset": "",
            "weighting_name": 5,
            "inclusion_approach": None,
            "buy_hold_initial": "",
            "rank_pct": "0.25",
            "mp_min_funds": "3",
            "mp_max_funds": None,
            "info_ratio_benchmark": "",
            "metric_weights": ["sharpe"],
        }
    )

    assert dict(st.session_state) == stale


@pytest.mark.xfail(
    strict=True,
    reason="_sync_model_widgets_from_state truncates instead of rounding: rank_pct 0.29 shows as 28",
)
def test_sync_model_widgets_round_trips_whole_percentages(model_page: ModelPage) -> None:
    page, st = model_page

    page._sync_model_widgets_from_state({"rank_pct": 0.29})

    assert st.session_state["rank_pct_primary"] == 29


def test_apply_config_wrapper_replaces_model_state_and_resyncs_widgets(
    model_page: ModelPage,
) -> None:
    page, st = model_page
    st.session_state.update(
        {
            "model_state": {"preset": "Old"},
            "preset_selector": "Old",
            "metric_stale": 9.0,
            "selected_risk_free": "RF1",
        }
    )
    loaded = {"preset": "Baseline", "metric_weights": {"sharpe": 1.0}}
    wrapper = {
        "model_state": loaded,
        "fund_columns": ["FundA"],
        "data_fingerprint": "fp-1",
        "selected_benchmark": "SPX",
        "unrelated": "ignored",
    }

    page._apply_config_wrapper(wrapper)

    assert st.session_state["model_state"] == loaded
    assert st.session_state["model_state"] is not loaded
    assert st.session_state["last_loaded_model_state"] == loaded
    assert st.session_state["last_loaded_model_state"] is not st.session_state["model_state"]
    assert st.session_state["preset_selector"] == "Baseline"
    assert "metric_stale" not in st.session_state
    assert st.session_state["metric_sharpe"] == 1.0
    assert st.session_state["fund_columns"] == ["FundA"]
    assert st.session_state["data_fingerprint"] == "fp-1"
    assert st.session_state["selected_benchmark"] == "SPX"
    assert st.session_state["selected_risk_free"] == "RF1"
    assert "analysis_fund_columns" not in st.session_state
    assert "unrelated" not in st.session_state


def test_apply_config_wrapper_without_model_state_only_restores_context(
    model_page: ModelPage,
) -> None:
    page, st = model_page
    st.session_state.update({"model_state": {"preset": "Old"}, "preset_selector": "Old"})

    page._apply_config_wrapper({"model_state": "not a mapping", "selected_risk_free": "RF2"})

    assert dict(st.session_state) == {
        "model_state": {"preset": "Old"},
        "preset_selector": "Old",
        "selected_risk_free": "RF2",
    }


# --- LLM settings, environment and provider configuration ------------------------------------


def test_resolve_llm_temperature_prefers_a_valid_override_then_env_then_zero(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    key = page._LLM_TEMPERATURE_OVERRIDE_KEY
    assert page._resolve_llm_temperature() == 0.0

    monkeypatch.setenv("TREND_LLM_TEMPERATURE", "0.7")
    assert page._resolve_llm_temperature() == 0.7
    st.session_state[key] = " 0.3 "
    assert page._resolve_llm_temperature() == 0.3
    st.session_state[key] = "not-a-number"
    assert page._resolve_llm_temperature() == 0.7
    st.session_state[key] = "   "
    assert page._resolve_llm_temperature() == 0.7

    monkeypatch.setenv("TREND_LLM_TEMPERATURE", "warm")
    assert page._resolve_llm_temperature() == 0.0


def test_hash_api_key_returns_a_sha256_digest_or_none(model_page: ModelPage) -> None:
    page, _st = model_page

    assert page._hash_api_key(None) is None
    assert page._hash_api_key("") is None
    assert page._hash_api_key("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_fallback_llm_provider_config_prefers_session_overrides(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    monkeypatch.setenv("TREND_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("TREND_LLM_MODEL", "env-model")
    monkeypatch.setenv("TREND_LLM_BASE_URL", "https://env.example")
    monkeypatch.setenv("TREND_LLM_ORG", "env-org")
    st.session_state.update(
        {
            page._LLM_PROVIDER_OVERRIDE_KEY: "  Anthropic ",
            page._LLM_MODEL_OVERRIDE_KEY: "claude-3-haiku",
            page._LLM_BASE_URL_OVERRIDE_KEY: "https://override.example",
            page._LLM_ORG_OVERRIDE_KEY: "override-org",
            page._LLM_API_KEY_OVERRIDE_KEY: "  sk-session  ",
        }
    )

    assert page._fallback_llm_provider_config() == LLMProviderConfig(
        provider="anthropic",
        model="claude-3-haiku",
        api_key="sk-session",
        base_url="https://override.example",
        organization="override-org",
    )


def test_fallback_llm_provider_config_reads_the_environment_and_defaults(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    assert page._fallback_llm_provider_config() == LLMProviderConfig(
        provider="openai", model="gpt-4o-mini"
    )

    monkeypatch.setenv("TREND_LLM_PROVIDER", "OLLAMA")
    monkeypatch.setenv("TREND_LLM_MODEL", "llama3")
    monkeypatch.setenv("TREND_LLM_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("TREND_LLM_ORG", "env-org")
    st.session_state[page._LLM_API_KEY_OVERRIDE_KEY] = "CHANGE_ME"
    assert page._fallback_llm_provider_config() == LLMProviderConfig(
        provider="ollama",
        model="llama3",
        base_url="http://localhost:11434",
        organization="env-org",
    )

    monkeypatch.setenv("TREND_LLM_PROVIDER", "mystery")
    assert page._fallback_llm_provider_config().provider == "openai"


def test_current_chain_settings_snapshot_falls_back_only_when_allowed(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, _st = model_page
    monkeypatch.setenv("TREND_LLM_PROVIDER", "mystery")

    with pytest.raises(
        ValueError,
        match=r"^Unknown LLM provider 'mystery'\. Expected one of: anthropic, ollama, openai\.$",
    ):
        page._current_chain_settings_snapshot(temperature=0.3)

    assert page._current_chain_settings_snapshot(temperature=0.3, allow_missing_api_key=True) == {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "base_url": None,
        "organization": None,
        "temperature": 0.3,
    }


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai", ["TS_STREAMLIT_API_KEY", "TREND_LLM_API_KEY", "OPENAI_API_KEY"]),
        ("anthropic", ["TS_STREAMLIT_API_KEY", "TREND_LLM_API_KEY", "CLAUDE_API_STRANSKE"]),
        ("ollama", ["TS_STREAMLIT_API_KEY", "TREND_LLM_API_KEY"]),
        ("mystery", None),
    ],
)
def test_llm_required_env_vars_lists_each_providers_variables(
    model_page: ModelPage, provider: str, expected: list[str] | None
) -> None:
    page, _st = model_page

    assert page._llm_required_env_vars(provider) == expected


def test_llm_env_var_present_accepts_either_anthropic_key_for_the_canonical_name(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, _st = model_page
    assert page._llm_env_var_present("CLAUDE_API_STRANSKE") is False
    assert page._llm_env_var_present("ANTHROPIC_API_KEY") is False

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fallback")
    assert page._llm_env_var_present("CLAUDE_API_STRANSKE") is True
    assert page._llm_env_var_present("ANTHROPIC_API_KEY") is True

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("CLAUDE_API_STRANSKE", "sk-ant-primary")
    assert page._llm_env_var_present("CLAUDE_API_STRANSKE") is True
    assert page._llm_env_var_present("ANTHROPIC_API_KEY") is False


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("TS_STREAMLIT_API_KEY", "ts-real", True),
        ("TS_STREAMLIT_API_KEY", "YOUR_KEY_HERE", False),
        ("TREND_LLM_API_KEY", "   ", False),
        ("OPENAI_API_KEY", "REPLACE_ME", False),
        ("OPENAI_API_KEY", "sk-real", True),
        ("OPENAI_API_KEY", None, False),
        ("UNLISTED_VARIABLE", "YOUR_PLACEHOLDER", True),
        ("UNLISTED_VARIABLE", "", False),
        ("UNLISTED_VARIABLE", None, False),
    ],
)
def test_llm_env_var_present_sanitizes_only_the_key_variables(
    model_page: ModelPage,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str | None,
    expected: bool,
) -> None:
    page, _st = model_page
    if value is not None:
        monkeypatch.setenv(name, value)

    assert page._llm_env_var_present(name) is expected


def test_llm_env_var_status_maps_each_required_name_to_its_presence(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, _st = model_page
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.setenv("TREND_LLM_API_KEY", "YOUR_KEY")

    assert page._llm_env_var_status("openai") == {
        "TS_STREAMLIT_API_KEY": False,
        "TREND_LLM_API_KEY": False,
        "OPENAI_API_KEY": True,
    }
    assert page._llm_env_var_status("mystery") == {}


def test_llm_status_panel_in_browser_runtime_reports_endpoint_and_session_key(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    monkeypatch.setattr(sys, "platform", "emscripten")
    st.session_state[page._LLM_BASE_URL_OVERRIDE_KEY] = " https://llm.example/v1 "
    st.session_state[page._LLM_API_KEY_OVERRIDE_KEY] = "sk-session-secret"

    page._render_llm_status_panel()

    assert st.info_messages == ["Active provider: OpenAI", "Active model: gpt-4o-mini"]
    assert st.captions == ["Browser endpoint: configured", "Session API key: present"]
    assert st.warning_messages == []
    assert st.write_calls == []

    st.captions.clear()
    st.session_state[page._LLM_BASE_URL_OVERRIDE_KEY] = "  "
    st.session_state[page._LLM_API_KEY_OVERRIDE_KEY] = "YOUR_API_KEY"
    page._render_llm_status_panel()
    assert st.captions == ["Browser endpoint: not configured", "Session API key: not entered"]


def test_llm_status_panel_warns_about_an_unknown_provider(model_page: ModelPage) -> None:
    page, st = model_page
    st.session_state["selected_provider"] = "Mystery"

    page._render_llm_status_panel()

    assert st.info_messages == ["Active provider: mystery", "Active model: gpt-4o-mini"]
    assert st.warning_messages == ["Unknown provider: mystery. Update your LLM settings."]
    assert st.write_calls == []


def test_llm_status_panel_reports_when_no_environment_variable_is_required(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    monkeypatch.setattr(page, "_llm_required_env_vars", lambda provider: [])

    page._render_llm_status_panel()

    assert st.captions == ["Expected environment variables: None required."]
    assert st.write_calls == []
    assert st.warning_messages == []


def test_sync_llm_selection_prefers_overrides_then_env_then_defaults(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page

    def selection() -> tuple[object, object]:
        page._sync_llm_selection_from_overrides()
        return st.session_state["selected_provider"], st.session_state["selected_model"]

    assert selection() == ("openai", "gpt-4o-mini")

    monkeypatch.setenv("TREND_LLM_PROVIDER", " OLLAMA ")
    monkeypatch.setenv("TREND_LLM_MODEL", " llama3 ")
    assert selection() == ("ollama", "llama3")

    st.session_state[page._LLM_PROVIDER_OVERRIDE_KEY] = "  Anthropic "
    st.session_state[page._LLM_MODEL_OVERRIDE_KEY] = "  claude-3-haiku "
    assert selection() == ("anthropic", "claude-3-haiku")


def test_llm_overrides_panel_resets_an_unknown_provider_to_the_env_default(
    model_page: ModelPage,
) -> None:
    page, st = model_page
    st.session_state[page._LLM_PROVIDER_OVERRIDE_KEY] = "mystery"

    page._render_llm_session_overrides_panel()

    assert st.expander_labels == ["LLM Settings (Session Only)"]
    assert st.selectbox_calls == [("Provider", [None, "openai", "anthropic", "ollama"])]
    kwargs = st.selectbox_kwargs[0]
    assert kwargs["index"] == 0
    assert kwargs["key"] == page._LLM_PROVIDER_OVERRIDE_KEY
    labels = [kwargs["format_func"](option) for option in (None, "openai", "anthropic", "ollama")]
    assert labels == ["Use env default", "OpenAI", "Anthropic", "Ollama"]

    st.selectbox_kwargs.clear()
    st.session_state[page._LLM_PROVIDER_OVERRIDE_KEY] = "anthropic"
    page._render_llm_session_overrides_panel()
    assert st.selectbox_kwargs[0]["index"] == 2


def test_llm_overrides_panel_in_browser_runtime_offers_only_the_openai_endpoint(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    monkeypatch.setattr(sys, "platform", "emscripten")

    page._render_llm_session_overrides_panel()

    assert st.selectbox_calls == [("Provider", ["openai"])]
    kwargs = st.selectbox_kwargs[0]
    assert kwargs["index"] == 0
    assert kwargs["format_func"]("openai") == "OpenAI-compatible endpoint"


def test_llm_overrides_panel_warns_about_a_non_numeric_temperature(model_page: ModelPage) -> None:
    page, st = model_page
    key = page._LLM_TEMPERATURE_OVERRIDE_KEY

    st.session_state[key] = "warm"
    page._render_llm_session_overrides_panel()
    assert st.warning_messages == ["Temperature override must be a number; using env default."]

    for text in ("0.4", ""):
        st.warning_messages.clear()
        st.session_state[key] = text
        page._render_llm_session_overrides_panel()
        assert st.warning_messages == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.123456, 0.1235),
        ("0.7", 0.7),
        (1, 1.0),
        ("warm", 0.0),
        (None, 0.0),
        (object(), 0.0),
    ],
)
def test_normalize_temperature_rounds_to_four_places_and_zeroes_garbage(
    model_page: ModelPage, value: object, expected: float
) -> None:
    page, _st = model_page

    assert page._normalize_temperature(value) == expected


def test_api_key_secret_equality_compares_value_and_fingerprint(model_page: ModelPage) -> None:
    page, _st = model_page
    secret = page._ApiKeySecret("sk-1", "fp-1")

    assert operator.eq(secret, page._ApiKeySecret("sk-1", "fp-1")) is True
    assert operator.eq(secret, page._ApiKeySecret("sk-1", "fp-2")) is False
    assert operator.eq(secret, page._ApiKeySecret("sk-2", "fp-1")) is False
    assert operator.eq(secret, "sk-1") is True
    assert operator.eq(secret, "sk-2") is False
    assert operator.eq(page._ApiKeySecret(None, None), None) is True
    assert operator.eq(secret, None) is False
    assert operator.eq(secret, 42) is False


# --- cached chain builders ------------------------------------------------------------------


def test_cached_compact_schema_loads_the_compact_schema(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, _st = model_page
    schema = {"title": "compact"}
    monkeypatch.setattr(page, "load_compact_schema", lambda: schema)

    assert page._cached_compact_schema() is schema


def test_cached_config_patch_chain_builds_the_chain_from_the_cached_parts(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, _st = model_page
    llm = object()
    schema = {"title": "compact"}
    seen: list[tuple[object, object, object]] = []

    def fake_client(llm_cache_key: object, api_key_secret: object, extra_payload: object) -> object:
        seen.append((llm_cache_key, api_key_secret, extra_payload))
        return llm

    monkeypatch.setattr(page, "_cached_llm_client", fake_client)
    monkeypatch.setattr(page, "_cached_compact_schema", lambda: schema)
    secret = page._ApiKeySecret("sk-1", "fp-1")
    llm_key = {"provider": "openai", "model": "gpt-4o-mini"}

    chain = page._cached_config_patch_chain(
        {"model": "gpt-4o-mini", "temperature": 0.3}, llm_key, secret, '{"a": 1}'
    )

    assert isinstance(chain, ConfigPatchChain)
    assert chain.llm is llm
    assert chain.schema is schema
    assert chain.prompt_builder is page.build_config_patch_prompt
    assert (chain.temperature, chain.model) == (0.3, "gpt-4o-mini")
    assert len(seen) == 1
    assert seen[0][0] is llm_key
    assert seen[0][1] is secret
    assert seen[0][2] == '{"a": 1}'
    assert page._cached_config_patch_chain({"model": "m"}, llm_key, secret, "").temperature == 0.0


def test_cached_variant_patch_chain_builds_the_chain_from_the_cached_parts(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, _st = model_page
    llm = object()
    schema = {"title": "compact"}
    seen: list[tuple[object, object, object]] = []

    def fake_client(llm_cache_key: object, api_key_secret: object, extra_payload: object) -> object:
        seen.append((llm_cache_key, api_key_secret, extra_payload))
        return llm

    monkeypatch.setattr(page, "_cached_llm_client", fake_client)
    monkeypatch.setattr(page, "_cached_compact_schema", lambda: schema)
    secret = page._ApiKeySecret("sk-1", "fp-1")
    llm_key = {"provider": "anthropic", "model": "claude-3-haiku"}

    chain = page._cached_variant_patch_chain(
        {"model": "claude-3-haiku", "temperature": 0.6}, llm_key, secret, "{}"
    )

    assert isinstance(chain, ConfigPatchVariantsChain)
    assert chain.llm is llm
    assert chain.schema is schema
    assert chain.prompt_builder is page.build_variant_patch_prompt
    assert (chain.temperature, chain.model) == (0.6, "claude-3-haiku")
    assert len(seen) == 1
    assert seen[0][0] is llm_key
    assert seen[0][1] is secret
    assert seen[0][2] == "{}"
    assert page._cached_variant_patch_chain({"model": "m"}, llm_key, secret, "").temperature == 0.0


def test_cached_nl_chain_unwraps_the_chain_from_a_tuple_result(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, _st = model_page
    chain = object()
    calls: list[dict[str, object]] = []

    def fake_build(**kwargs: object) -> tuple[object, dict[str, bool]]:
        calls.append(kwargs)
        return chain, {"chain_reused": False}

    monkeypatch.setattr(page, "_build_nl_chain", fake_build)

    assert page._cached_nl_chain("sig", "openai", "gpt-4o-mini", None, "org", 0.2) is chain
    assert calls == [
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "base_url": None,
            "organization": "org",
            "temperature": 0.2,
            "cache_signature": "sig",
        }
    ]


# --- natural-language chain construction ----------------------------------------------------


def _install_fake_chain_cache(
    page: ModuleType, monkeypatch: pytest.MonkeyPatch, *, fresh_chain_each_call: bool = False
) -> None:
    """Resolve a fixed provider config and fake the chain cache.

    By default one chain object is handed out per resource signature, like the real cache; with
    ``fresh_chain_each_call`` every lookup builds a new object.  Every object is kept alive so a
    recycled ``id()`` cannot read as a cache hit.
    """

    chains: dict[str, object] = {}
    created: list[object] = []

    def cached_chain(
        chain_cache_key: dict[str, Any], llm_cache_key: dict[str, Any], _key: object, _extra: str
    ) -> object:
        signature = page._chain_resource_signature(chain_cache_key, llm_cache_key)
        if fresh_chain_each_call or signature not in chains:
            chain = object()
            created.append(chain)
            chains[signature] = chain
        return chains[signature]

    monkeypatch.setattr(page, "_cached_config_patch_chain", cached_chain)
    monkeypatch.setattr(
        page,
        "_resolve_llm_provider_config",
        lambda: LLMProviderConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test"),
    )


def test_build_nl_chain_resets_the_chat_when_an_explicit_setting_changes(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    _install_fake_chain_cache(page, monkeypatch)

    first_chain, first_meta = page._build_nl_chain(model="gpt-4o-mini", temperature=0.0)
    assert first_meta["chain_cache_invalidation_fields"] == []
    assert first_meta["chain_cache_session_reset"] is False
    assert first_meta["chain_cache_miss_reason"] == "first_build"

    st.session_state["config_chat_preview"] = {"after": {}}
    st.session_state["config_chat_last_instruction"] = "old instruction"
    second_chain, second_meta = page._build_nl_chain(model="gpt-4.1-mini", temperature=0.0)

    assert second_chain is not first_chain
    assert second_meta["chain_settings_changed"] is True
    assert second_meta["chain_cache_invalidation_fields"] == ["model"]
    assert second_meta["chain_cache_session_reset"] is True
    assert second_meta["chain_cache_miss_reason"] == "settings_changed: model"
    assert "config_chat_preview" not in st.session_state
    assert "config_chat_last_instruction" not in st.session_state
    assert st.session_state[page._CONFIG_CHAIN_STATE_KEY]["last_invalidation_fields"] == ["model"]


def test_build_nl_chain_miss_reason_for_a_new_cache_signature(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, _st = model_page
    _install_fake_chain_cache(page, monkeypatch)

    page._build_nl_chain(temperature=0.0, cache_signature="sig-a")
    _chain, meta = page._build_nl_chain(temperature=0.0, cache_signature="sig-b")

    assert meta["chain_reused"] is False
    assert meta["chain_settings_changed"] is True
    assert meta["chain_cache_invalidation_fields"] == []
    assert meta["chain_cache_miss_reason"] == "settings_changed"


def test_build_nl_chain_miss_reason_for_a_changed_llm_field(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, _st = model_page
    _install_fake_chain_cache(page, monkeypatch)

    page._build_nl_chain(temperature=0.0, timeout=10.0)
    _chain, meta = page._build_nl_chain(temperature=0.0, timeout=20.0)

    assert meta["chain_reused"] is False
    assert meta["chain_settings_changed"] is False
    assert meta["chain_llm_changed_fields"] == ["timeout"]
    assert meta["chain_cache_miss_reason"] == "llm_settings_changed: timeout"


def test_build_nl_chain_miss_reason_when_only_the_resource_signature_drifted(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    _install_fake_chain_cache(page, monkeypatch, fresh_chain_each_call=True)

    page._build_nl_chain(temperature=0.0)
    st.session_state[page._CONFIG_CHAIN_STATE_KEY]["last_resource_signature"] = "stale"
    _chain, meta = page._build_nl_chain(temperature=0.0)

    assert meta["chain_reused"] is False
    assert meta["chain_settings_changed"] is False
    assert meta["chain_llm_changed_fields"] == []
    assert meta["chain_cache_miss_reason"] == "llm_settings_changed"


def test_build_nl_chain_miss_reason_when_only_the_core_fields_drifted(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    _install_fake_chain_cache(page, monkeypatch)

    page._build_nl_chain(temperature=0.0, cache_signature="sig-a")
    state = st.session_state[page._CONFIG_CHAIN_STATE_KEY]
    state["last_cache_key"] = {**state["last_cache_key"], "provider": "stale"}
    _chain, meta = page._build_nl_chain(temperature=0.0, cache_signature="sig-b")

    assert meta["chain_cache_invalidation_fields"] == []
    assert meta["chain_cache_miss_reason"] == "settings_changed: provider"


def test_build_nl_chain_logs_the_lookup_when_info_is_enabled(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    page, _st = model_page
    _install_fake_chain_cache(page, monkeypatch)
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(page, "monotonic", lambda: next(ticks))

    with caplog.at_level(logging.INFO, logger=page._LOGGER.name):
        _chain, meta = page._build_nl_chain(temperature=0.0)

    assert meta["chain_lookup_seconds"] == 0.25
    assert (
        "Config chat chain: reused=no cache=openai:gpt-4o-mini@0.000 lookup_s=0.250 "
        "miss=first_build"
    ) in caplog.messages


def test_get_variant_chain_builds_the_cache_context_for_the_variant_chain(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, _st = model_page
    config = LLMProviderConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="abc",
        base_url="https://api.example",
        organization="org-1",
        timeout=30.0,
        max_retries=2,
        extra={"a": 1},
    )
    monkeypatch.setattr(page, "_resolve_llm_provider_config", lambda: config)
    monkeypatch.setattr(page, "_resolve_llm_temperature", lambda: 0.25)
    chain = object()
    captured: list[tuple[Any, ...]] = []

    def fake_variant_chain(*args: Any) -> object:
        captured.append(args)
        return chain

    monkeypatch.setattr(page, "_cached_variant_patch_chain", fake_variant_chain)

    result_chain, context = page._get_variant_chain()

    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert result_chain is chain
    chain_key, llm_key, secret, payload = captured[0]
    assert chain_key == {
        "cache_version": "v1",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.25,
    }
    assert llm_key == {
        "cache_version": "v1",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "base_url": "https://api.example",
        "organization": "org-1",
        "timeout": 30.0,
        "max_retries": 2,
        "extra_payload_hash": "f9d86028c6e0",
        "api_key_fingerprint": digest,
    }
    assert (secret.value, secret.fingerprint) == ("abc", digest)
    assert payload == '{"a": 1}'
    assert context["cache_key"] is chain_key
    assert context["llm_cache_key"] is llm_key


# --- metric extraction ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.5", 1.5),
        (2, 2.0),
        ("n/a", None),
        (None, None),
        (float("nan"), None),
        (pd.NA, None),
    ],
)
def test_coerce_metric_value_returns_floats_or_none(
    model_page: ModelPage, value: object, expected: float | None
) -> None:
    page, _st = model_page

    assert page._coerce_metric_value(value) == expected


def test_extract_metric_map_prefers_the_public_attributes_of_user_stats(
    model_page: ModelPage,
) -> None:
    page, _st = model_page
    stats = SimpleNamespace(sharpe=1.2, _private=9, vol=float("nan"), label="x")
    result = SimpleNamespace(
        details={"out_user_stats": stats}, metrics=pd.DataFrame([{"sharpe": 0.1}])
    )

    assert page._extract_metric_map(result) == {"sharpe": 1.2, "vol": None, "label": None}


def test_extract_metric_map_reads_mapping_user_stats(model_page: ModelPage) -> None:
    page, _st = model_page
    result = SimpleNamespace(details={"out_user_stats": {"sharpe": "1.1", "n": 3}})

    assert page._extract_metric_map(result) == {"sharpe": 1.1, "n": 3.0}


def test_extract_metric_map_falls_back_to_the_metrics_frame(model_page: ModelPage) -> None:
    page, _st = model_page
    result = SimpleNamespace(
        details={"out_user_stats": {}, "out_sample_stats": {"sortino": 0.9}},
        metrics=pd.DataFrame([{"sharpe": 2.0, "vol": "n/a"}]),
    )

    assert page._extract_metric_map(result) == {"sharpe": 2.0, "vol": None}


def test_extract_metric_map_reads_sample_stats_as_a_mapping_or_an_object(
    model_page: ModelPage,
) -> None:
    page, _st = model_page
    as_mapping = SimpleNamespace(
        details={"out_user_stats": None, "out_sample_stats": {"sortino": "0.9", "bad": "x"}},
        metrics=pd.DataFrame(),
    )
    as_object = SimpleNamespace(
        details={"out_sample_stats": SimpleNamespace(vol=0.1, label="x")}, metrics=None
    )

    assert page._extract_metric_map(as_mapping) == {"sortino": 0.9, "bad": None}
    assert page._extract_metric_map(as_object) == {"vol": 0.1, "label": None}
    assert page._extract_metric_map(SimpleNamespace()) == {}


# --- preview generation ---------------------------------------------------------------------


class _FakeNlChain:
    """Chain double that records its calls and returns a prepared ``ConfigPatch``."""

    def __init__(self, patch: ConfigPatch) -> None:
        self.patch = patch
        self.calls: list[tuple[dict[str, Any], str]] = []

    def run(self, *, current_config: dict[str, Any], instruction: str) -> ConfigPatch:
        self.calls.append((current_config, instruction))
        return self.patch


def test_generate_config_preview_applies_the_patch_and_reports_timings(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    patch = ConfigPatch(
        operations=[PatchOperation(op="set", path="signals.window", value=63)],
        summary="Longer trend window",
    )
    chain = _FakeNlChain(patch)
    meta = {
        "chain_reused": True,
        "chain_cache_key": {"provider": "openai"},
        "chain_cache_signature": "sig-1",
        "chain_resource_signature": "rsig-1",
        "chain_cache_summary": "openai:gpt-4o-mini@0.000",
        "chain_cache_miss_reason": None,
        "chain_cache_invalidation_fields": [],
        "chain_settings_changed": False,
        "chain_llm_changed_fields": [],
        "chain_cache_session_reset": False,
        "chain_lookup_seconds": 0.01,
    }
    monkeypatch.setattr(page, "_get_nl_chain", lambda: (chain, meta))
    ticks = iter([100.0, 100.5, 101.0, 104.0])
    monkeypatch.setattr(page, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(page, "_utc_timestamp", lambda: "2026-10-25T10:00:00Z")
    model_state = {"signals": {"window": 21}, "selection_count": 10}

    preview = page._generate_config_preview(model_state, "Use a 63 day trend window")

    assert chain.calls == [
        ({"signals": {"window": 21}, "selection_count": 10}, "Use a 63 day trend window")
    ]
    assert chain.calls[0][0] is not model_state
    assert model_state == {"signals": {"window": 21}, "selection_count": 10}
    assert preview["instruction"] == "Use a 63 day trend window"
    assert preview["before"] == {"signals": {"window": 21}, "selection_count": 10}
    assert preview["before"] is not model_state
    assert preview["after"] == {"signals": {"window": 63}, "selection_count": 10}
    assert preview["diff"] == (
        "--- before\n+++ after\n@@ -1,3 +1,3 @@\n signals:\n-  window: 21\n+  window: 63\n"
        " selection_count: 10\n"
    )
    assert preview["summary"] == "Longer trend window"
    assert preview["risk_flags"] == []
    assert preview["needs_review"] is False
    assert preview["patch"] == {
        "operations": [{"op": "set", "path": "signals.window", "value": 63, "rationale": None}],
        "needs_review": False,
        "risk_flags": [],
        "summary": "Longer trend window",
    }
    assert preview["timings"] == {
        "chain_build_seconds": 0.5,
        "chain_reused": True,
        "chain_cache_key": {"provider": "openai"},
        "chain_cache_signature": "sig-1",
        "chain_resource_signature": "rsig-1",
        "chain_cache_summary": "openai:gpt-4o-mini@0.000",
        "chain_cache_miss_reason": None,
        "chain_cache_invalidation_fields": [],
        "chain_settings_changed": False,
        "chain_llm_changed_fields": [],
        "chain_cache_session_reset": False,
        "chain_lookup_seconds": 0.01,
        "run_seconds": 3.0,
    }
    assert st.session_state[page._CONFIG_CHAIN_STATS_KEY] == {
        "hits": 1,
        "misses": 0,
        "last_build_seconds": 0.5,
        "last_reused": True,
        "last_miss_reason": None,
        "last_invalidation_fields": [],
        "last_signature": "sig-1",
        "timestamp": "2026-10-25T10:00:00Z",
    }


def test_generate_config_preview_surfaces_risk_flags_and_review_needs(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, _st = model_page
    patch = ConfigPatch(
        operations=[PatchOperation(op="remove", path="portfolio.max_turnover")],
        summary="Drop the turnover cap",
        needs_review=True,
    )
    monkeypatch.setattr(page, "_get_nl_chain", lambda: (_FakeNlChain(patch), {}))

    preview = page._generate_config_preview(
        {"portfolio": {"max_turnover": 0.5, "max_weight": 0.2}}, "Remove the turnover limit"
    )

    assert preview["risk_flags"] == ["REMOVES_CONSTRAINT"]
    assert preview["needs_review"] is True
    assert preview["after"] == {"portfolio": {"max_weight": 0.2}}


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (None, 25.0),
        (3, 5.0),
        (40, 40.0),
        (200, 90.0),
        (0, 25.0),
        (-4, 25.0),
        ("slow", 25.0),
    ],
)
def test_estimate_llm_seconds_clamps_the_stored_estimate(
    model_page: ModelPage, stored: object, expected: float
) -> None:
    page, st = model_page
    if stored is not None:
        st.session_state["config_chat_llm_seconds"] = stored

    assert page._estimate_llm_seconds() == expected


def test_record_llm_seconds_blends_new_durations_and_ignores_non_positive_ones(
    model_page: ModelPage,
) -> None:
    page, st = model_page
    key = "config_chat_llm_seconds"

    page._record_llm_seconds(0.0)
    assert key not in st.session_state

    page._record_llm_seconds(12.5)
    assert st.session_state[key] == 12.5

    st.session_state[key] = 10.0
    page._record_llm_seconds(20.0)
    assert st.session_state[key] == pytest.approx(14.0)

    page._record_llm_seconds(-1.0)
    assert st.session_state[key] == pytest.approx(14.0)


def _drive_preview_progress(
    page: ModuleType, monkeypatch: pytest.MonkeyPatch, preview: dict[str, Any], clock: list[float]
) -> dict[str, list[Any]]:
    """Run ``_generate_preview_with_progress`` on a gated worker thread and a scripted clock.

    The worker blocks until the page's first ``sleep``; the fake ``sleep`` then releases it and
    waits for it to finish, so the polling loop runs exactly once.
    """

    release = threading.Event()
    futures: list[Future[Any]] = []
    record: dict[str, list[Any]] = {"sleeps": [], "calls": []}

    class RecordingExecutor(ThreadPoolExecutor):
        def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
            future = super().submit(fn, *args, **kwargs)
            futures.append(future)
            return future

    def fake_generate(model_state: Any, instruction: str) -> dict[str, Any]:
        assert release.wait(timeout=10)
        record["calls"].append((model_state, instruction))
        return preview

    def fake_sleep(seconds: float) -> None:
        record["sleeps"].append(seconds)
        release.set()
        futures[0].result(timeout=10)

    ticks = iter(clock)
    monkeypatch.setattr(page, "ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(page, "_generate_config_preview", fake_generate)
    monkeypatch.setattr(page, "sleep", fake_sleep)
    monkeypatch.setattr(page, "monotonic", lambda: next(ticks))
    return record


def test_generate_preview_with_progress_reports_progress_and_records_timing(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    preview = {
        "after": {"lookback_periods": 12},
        "timings": {"chain_reused": True, "chain_build_seconds": 0.5, "run_seconds": 2.0},
    }
    record = _drive_preview_progress(page, monkeypatch, preview, [100.0, 110.0, 112.5])
    model_state = {"lookback_periods": 6}

    result = page._generate_preview_with_progress(model_state, "Raise lookback")

    assert result is preview
    assert record["calls"] == [(model_state, "Raise lookback")]
    assert record["sleeps"] == [0.25]
    assert st.progress_calls == [
        (0.0, "Preparing preview..."),
        (0.4, "Generating preview... ~15s remaining"),
        (1.0, "Preview ready."),
    ]
    assert preview["timings"]["total_seconds"] == 12.5
    assert st.session_state["config_chat_llm_seconds"] == 12.5
    history = st.session_state[page._CONFIG_PREVIEW_TIMINGS_KEY]
    assert [(entry["chain_reused"], entry["total_seconds"]) for entry in history] == [(True, 12.5)]
    assert st.session_state["config_chat_last_preview_summary"] == (
        "Preview timing: cache hit | build 0.50s | lookup — | run 2.00s | total 12.50s"
    )


def test_generate_preview_with_progress_without_timings_and_with_a_stored_estimate(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    st.session_state["config_chat_llm_seconds"] = 40.0
    preview = {"after": {"lookback_periods": 12}}
    _drive_preview_progress(page, monkeypatch, preview, [100.0, 110.0, 112.5])

    result = page._generate_preview_with_progress({"lookback_periods": 6}, "Raise lookback")

    assert result is preview
    assert "timings" not in preview
    assert st.progress_calls == [
        (0.0, "Preparing preview..."),
        (0.25, "Generating preview... ~30s remaining"),
        (1.0, "Preview ready."),
    ]
    assert st.session_state["config_chat_llm_seconds"] == pytest.approx(29.0)
    assert page._CONFIG_PREVIEW_TIMINGS_KEY not in st.session_state
    assert st.session_state["config_chat_last_preview_summary"] == "Preview timing unavailable."


# --- run key and applying a preview ---------------------------------------------------------


def test_current_run_key_hashes_the_fund_list_without_the_excluded_columns(
    model_page: ModelPage,
) -> None:
    page, st = model_page
    st.session_state.update(
        {
            "data_fingerprint": "fp1",
            "selected_risk_free": "RF",
            "fund_columns": ["A", "SPX", "RF", "B"],
        }
    )
    model_state = {"lookback_periods": 12}

    assert page._current_run_key(model_state, "SPX").split(":", 5) == [
        "fp1",
        "SPX",
        "RF",
        "f2314439597c",
        "1434fd84d531",
        '{"lookback_periods": 12}',
    ]

    st.session_state["analysis_fund_columns"] = ["A"]
    assert page._current_run_key(model_state, "SPX").split(":", 5)[3] == "d2a4b37cac0a"

    st.session_state["analysis_fund_columns"] = None
    assert page._current_run_key(model_state, "SPX").split(":", 5)[3] == "f2314439597c"

    del st.session_state["fund_columns"]
    assert page._current_run_key(model_state, "SPX").split(":", 5)[3] == "4f53cda18c2b"


def test_current_run_key_excludes_the_regime_proxy_only_when_regime_is_enabled(
    model_page: ModelPage,
) -> None:
    page, st = model_page
    st.session_state.update({"data_fingerprint": "fp1", "analysis_fund_columns": ["A", "REGIME"]})
    enabled = {"regime_enabled": True, "regime_proxy": "REGIME"}
    disabled = {"regime_enabled": False, "regime_proxy": "REGIME"}

    enabled_parts = page._current_run_key(enabled, None).split(":", 5)
    disabled_parts = page._current_run_key(disabled, None).split(":", 5)

    assert enabled_parts[1:3] == ["__none__", "__none__"]
    assert enabled_parts[3] == "d2a4b37cac0a"
    assert disabled_parts[3] == "ee7af0844924"


def _analysis_preview() -> dict[str, Any]:
    return {
        "instruction": "Raise lookback",
        "before": {"lookback_periods": 6},
        "after": {"lookback_periods": 12},
    }


def test_apply_preview_state_warns_when_the_preview_has_no_after_state(
    model_page: ModelPage,
) -> None:
    page, st = model_page
    st.session_state["model_state"] = {"lookback_periods": 6}

    page._apply_preview_state(
        {"before": {"lookback_periods": 6}, "after": "oops"}, run_analysis=True
    )

    assert st.warning_messages == ["Preview is missing updated configuration data."]
    assert st.session_state["model_state"] == {"lookback_periods": 6}
    assert page._CONFIG_HISTORY_KEY not in st.session_state
    assert st.success_messages == []


def test_apply_preview_state_requires_loaded_data_before_running_analysis(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    runs: list[tuple[Any, ...]] = []
    monkeypatch.setattr(page.analysis_runner, "run_analysis", lambda *args, **kw: runs.append(args))
    st.session_state["model_state"] = {"lookback_periods": 6}

    page._apply_preview_state(_analysis_preview(), run_analysis=True)

    assert st.error_messages == ["Load data before running analysis."]
    assert runs == []
    assert st.session_state["model_state"] == {"lookback_periods": 12}
    assert "analysis_result" not in st.session_state
    assert st.success_messages == []


def test_apply_preview_state_runs_the_analysis_with_the_risk_free_column(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    frame = pd.DataFrame({"A": [0.01, 0.02]})
    result = object()
    calls: list[tuple[Any, ...]] = []
    cleared: list[str] = []

    def fake_run(df: Any, model_state: Any, benchmark: Any, *, data_hash: Any = None) -> object:
        calls.append((df, dict(model_state), benchmark, data_hash))
        return result

    monkeypatch.setattr(page.analysis_runner, "run_analysis", fake_run)
    monkeypatch.setattr(page.analysis_runner, "clear_cached_analysis", lambda: cleared.append("c"))
    st.session_state.update(
        {
            "returns_df": frame,
            "selected_benchmark": "SPX",
            "selected_risk_free": "RF",
            "data_hash": "hash-1",
            "data_fingerprint": "fp1",
            "model_state": {"lookback_periods": 6},
            "analysis_error": {"message": "stale"},
        }
    )

    page._apply_preview_state(_analysis_preview(), run_analysis=True)

    assert len(calls) == 1
    assert calls[0][0] is frame
    assert calls[0][1:] == ({"lookback_periods": 12, "risk_free_column": "RF"}, "SPX", "hash-1")
    assert st.spinner_messages == ["Running analysis..."]
    assert st.session_state["model_state"] == {"lookback_periods": 12}
    assert st.session_state["analysis_result"] is result
    assert st.session_state["analysis_result_key"] == page._current_run_key(
        {"lookback_periods": 12}, "SPX"
    )
    assert "analysis_error" not in st.session_state
    assert cleared == ["c"]
    assert st.success_messages == ["Applied config changes and ran analysis."]
    assert st.error_messages == []


def test_apply_preview_state_records_a_failed_analysis(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    seen_states: list[dict[str, Any]] = []

    def failing_run(df: Any, model_state: Any, benchmark: Any, *, data_hash: Any = None) -> None:
        seen_states.append(dict(model_state))
        raise RuntimeError("boom")

    monkeypatch.setattr(page.analysis_runner, "run_analysis", failing_run)
    st.session_state.update(
        {"returns_df": pd.DataFrame({"A": [0.01]}), "model_state": {"lookback_periods": 6}}
    )

    page._apply_preview_state(_analysis_preview(), run_analysis=True)

    assert seen_states == [{"lookback_periods": 12}]
    assert st.error_messages == ["Analysis failed: boom"]
    assert st.success_messages == []
    assert st.session_state["analysis_result"] is None
    assert st.session_state["analysis_result_key"] is None
    assert st.session_state["analysis_error"] == {"message": "Analysis failed.", "detail": "boom"}
    assert st.session_state["model_state"] == {"lookback_periods": 12}


# --- risky change dialog and revert ---------------------------------------------------------


def _pending(preview: object, *, run_analysis: bool = False) -> dict[str, object]:
    return {"preview": preview, "run_analysis": run_analysis}


def test_risky_dialog_discards_a_pending_apply_without_a_preview(model_page: ModelPage) -> None:
    page, st = model_page
    st.session_state["config_chat_pending_apply"] = _pending("not a mapping", run_analysis=True)

    page._render_risky_change_dialog()

    assert "config_chat_pending_apply" not in st.session_state
    assert st.dialog_titles == []


def test_risky_dialog_discards_a_pending_apply_that_is_not_risky(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    applied: list[object] = []
    monkeypatch.setattr(page, "_apply_preview_state", lambda *args, **kw: applied.append(args))
    st.session_state["config_chat_pending_apply"] = _pending(
        {"after": {"a": 1}, "risk_flags": [], "needs_review": False}
    )

    page._render_risky_change_dialog()

    assert "config_chat_pending_apply" not in st.session_state
    assert st.dialog_titles == []
    assert applied == []


def test_risky_dialog_reports_when_streamlit_has_no_dialog_support(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    monkeypatch.setattr(st, "dialog", None)
    pending = _pending({"after": {"a": 1}, "risk_flags": ["BROAD_SCOPE"]})
    st.session_state["config_chat_pending_apply"] = pending

    page._render_risky_change_dialog()

    assert st.error_messages == ["RISKY CHANGE: confirmation dialog unavailable."]
    assert st.session_state["config_chat_pending_apply"] is pending
    assert st.warning_messages == []


def test_risky_dialog_cancel_discards_the_pending_apply_without_applying(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    applied: list[object] = []
    monkeypatch.setattr(page, "_apply_preview_state", lambda *args, **kw: applied.append(args))
    pending = _pending({"after": {"a": 1}, "risk_flags": ["BROAD_SCOPE"]})

    st.session_state["config_chat_pending_apply"] = pending
    page._render_risky_change_dialog()
    assert st.dialog_titles == ["Confirm risky change"]
    assert st.session_state["config_chat_pending_apply"] is pending

    st.clicked.add("Cancel")
    page._render_risky_change_dialog()
    assert "config_chat_pending_apply" not in st.session_state
    assert applied == []


def test_revert_without_history_warns_and_keeps_the_state(model_page: ModelPage) -> None:
    page, st = model_page
    st.session_state["model_state"] = {"lookback_periods": 12}

    page._revert_last_config_change()

    assert st.warning_messages == ["No prior config change to revert."]
    assert st.session_state["model_state"] == {"lookback_periods": 12}
    assert st.success_messages == []


def test_revert_drops_a_malformed_last_entry_and_reports_the_failure(
    model_page: ModelPage,
) -> None:
    page, st = model_page
    key = page._CONFIG_HISTORY_KEY
    st.session_state["model_state"] = {"lookback_periods": 12}
    st.session_state[key] = [
        {"before": {"lookback_periods": 6}, "after": {"lookback_periods": 12}},
        {"instruction": "broken", "before": "oops"},
    ]

    page._revert_last_config_change()

    message = "Revert failed: missing previous configuration."
    assert st.error_messages == [message]
    assert st.session_state["model_state"] == {"lookback_periods": 12}
    assert len(st.session_state[key]) == 1
    assert st.success_messages == []

    st.session_state[key] = ["not an entry"]
    st.error_messages.clear()
    page._revert_last_config_change()
    assert st.error_messages == [message]
    assert st.session_state[key] == []


# --- diff and history panels ----------------------------------------------------------------


def test_render_unified_diff_reports_blank_diffs(model_page: ModelPage) -> None:
    page, st = model_page

    page._render_unified_diff("")
    page._render_unified_diff("  \n\t")
    assert st.info_messages == ["No differences found.", "No differences found."]
    assert st.markdown_calls == []

    page._render_unified_diff("--- before\n+++ after\n-a\n+b\n")
    assert st.info_messages == ["No differences found.", "No differences found."]
    styles, body = st.markdown_calls
    assert ".config-diff" in styles[0]
    assert body == (
        '<div class="config-diff">'
        '<div class="diff-line diff-header">--- before</div>'
        '<div class="diff-line diff-header">+++ after</div>'
        '<div class="diff-line diff-remove">-a</div>'
        '<div class="diff-line diff-add">+b</div>'
        "</div>",
        {"unsafe_allow_html": True},
    )


def _stub_diff_renderers(
    page: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], list[tuple[Any, Any]]]:
    unified: list[str] = []
    side_by_side: list[tuple[Any, Any]] = []
    monkeypatch.setattr(page, "_render_unified_diff", lambda diff: unified.append(diff))
    monkeypatch.setattr(
        page,
        "_render_side_by_side_diff",
        lambda before, after: side_by_side.append((before, after)),
    )
    return unified, side_by_side


def test_render_config_diff_preview_warns_about_an_incomplete_preview(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    unified, side_by_side = _stub_diff_renderers(page, monkeypatch)
    st.session_state["config_chat_preview"] = {"before": {"a": 1}, "after": None}

    page._render_config_diff_preview(model_state={"a": 1})

    assert st.warning_messages == ["Preview data is incomplete. Try generating a new diff."]
    assert st.tab_groups == []
    assert (unified, side_by_side) == ([], [])


def test_render_config_diff_preview_summarizes_the_preview_timings(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    _stub_diff_renderers(page, monkeypatch)
    preview: dict[str, Any] = {"before": {"a": 1}, "after": {"a": 2}, "diff": "d"}
    st.session_state["config_chat_preview"] = preview

    def render(timings: object) -> list[str]:
        preview["timings"] = timings
        st.captions.clear()
        page._render_config_diff_preview(model_state=None)
        return list(st.captions)

    assert render(
        {
            "chain_reused": True,
            "chain_cache_signature": "abcdef1234",
            "chain_settings_changed": False,
            "chain_build_seconds": 0.5,
            "chain_lookup_seconds": 0.25,
            "run_seconds": 2.0,
            "total_seconds": 2.75,
            "chain_cache_miss_reason": "first_build",
            "chain_cache_invalidation_fields": ["model", "temperature"],
        }
    ) == [
        "Preview timing — Chain reused: Yes | Cache sig: abcdef12 | Settings changed: No | "
        "Chain build: 0.50s | Chain lookup: 0.25s | Run: 2.00s | Total: 2.75s | "
        "Cache miss: first_build | Invalidated by: model, temperature"
    ]
    assert render({"chain_settings_changed": True, "chain_cache_invalidation_fields": []}) == [
        "Preview timing — Chain reused: No | Cache sig: — | Settings changed: Yes | "
        "Chain build: — | Chain lookup: — | Run: — | Total: —"
    ]
    assert render({}) == [
        "Preview timing — Chain reused: No | Cache sig: — | Settings changed: — | "
        "Chain build: — | Chain lookup: — | Run: — | Total: —"
    ]
    assert render("slow") == []


def test_render_config_change_history_handles_every_entry_shape(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    unified, side_by_side = _stub_diff_renderers(page, monkeypatch)
    st.session_state[page._CONFIG_HISTORY_KEY] = [
        {
            "timestamp": "2026-10-25T10:00:00Z",
            "instruction": "Raise lookback",
            "summary": "Lookback 6 to 12",
            "risk_flags": ["REMOVES_CONSTRAINT", "BROAD_SCOPE"],
            "before": {"lookback_periods": 6},
            "after": {"lookback_periods": 12},
            "diff": "--- before\n+++ after\n",
        },
        "junk",
        {"instruction": None, "before": {"a": 1}, "after": "oops"},
        {"before": {"lookback_periods": 6}, "after": {"lookback_periods": 12}, "diff": None},
    ]

    page._render_config_change_history()

    assert st.markdowns == ["**Change history**"]
    assert st.expander_labels == [
        "Unknown time • Config change",
        "Unknown time • Config change",
        "2026-10-25T10:00:00Z • Raise lookback",
    ]
    assert st.captions == ["Lookback 6 to 12", "Risk flags: REMOVES_CONSTRAINT, BROAD_SCOPE"]
    assert st.warning_messages == ["History entry is missing configuration data."]
    assert st.tab_groups == [["Unified diff", "Side-by-side"], ["Unified diff", "Side-by-side"]]
    assert unified == [
        "--- before\n+++ after\n@@ -1 +1 @@\n-lookback_periods: 6\n+lookback_periods: 12\n",
        "--- before\n+++ after\n",
    ]
    assert side_by_side == [
        ({"lookback_periods": 6}, {"lookback_periods": 12}),
        ({"lookback_periods": 6}, {"lookback_periods": 12}),
    ]


# --- config chat contents and panel ---------------------------------------------------------


def _stub_chat_sections(page: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the sub-panels of the chat contents with recorders; return their call log."""

    calls: list[str] = []
    for name in (
        "_render_last_preview_metrics",
        "_render_risky_change_dialog",
        "_render_config_change_history",
        "_render_preview_timing_history",
    ):
        monkeypatch.setattr(page, name, lambda _name=name: calls.append(_name))
    for name in ("_render_config_summary", "_render_config_diff_preview"):
        monkeypatch.setattr(page, name, lambda state, _name=name: calls.append(_name))
    monkeypatch.setattr(
        page.nl_operation_viewer,
        "render_nl_operation_viewer",
        lambda: calls.append("render_nl_operation_viewer"),
    )
    return calls


def test_config_chat_contents_shows_the_last_preview_summary_and_cache_key(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    _stub_chat_sections(page, monkeypatch)
    intro = "Describe the configuration change you want to try."
    summary = "Preview timing: cache hit | build 0.50s | lookup — | run 2.00s | total 3.00s"

    st.session_state["config_chat_last_preview_summary"] = 5
    page._render_config_chat_contents({"lookback_periods": 6})
    assert st.captions[0] == intro
    assert len(st.captions) == 2
    assert st.captions[1].startswith("Current cache key: openai:gpt-4o-mini@0.000 | Sig: ")

    st.captions.clear()
    st.session_state["config_chat_last_preview_summary"] = summary
    page._render_config_chat_contents({"lookback_periods": 6})
    assert st.captions[:2] == [intro, summary]
    assert st.captions[2].startswith("Current cache key: openai:gpt-4o-mini@0.000 | Sig: ")


def test_config_chat_contents_reports_an_unavailable_cache_key(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    _stub_chat_sections(page, monkeypatch)

    def failing_context() -> dict[str, Any]:
        raise ValueError("Missing API key for openai")

    monkeypatch.setattr(page, "_build_chain_cache_context", failing_context)

    page._render_config_chat_contents(None)

    assert st.captions == [
        "Describe the configuration change you want to try.",
        "Cache key unavailable: Missing API key for openai",
    ]


def test_config_chat_contents_signature_label_distinguishes_cache_keys(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    _stub_chat_sections(page, monkeypatch)
    labels: list[str] = []
    for provider, model in (("openai", "gpt-4o-mini"), ("anthropic", "claude-3-haiku")):
        cache_key = page._build_chain_cache_key(provider=provider, model=model, temperature=0.0)
        monkeypatch.setattr(
            page,
            "_build_chain_cache_context",
            lambda key=cache_key: {"cache_key": key, "llm_cache_key": {}},
        )
        st.captions.clear()
        page._render_config_chat_contents(None)
        labels.append(st.captions[-1].split("| Sig: ")[1])

    assert labels[0] != labels[1]


def test_config_chat_send_requires_an_instruction(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    _stub_chat_sections(page, monkeypatch)
    st.clicked.add("config_chat_send")

    st.session_state["config_chat_instruction"] = "   "
    page._render_config_chat_contents({"a": 1})
    assert st.warning_messages == ["Enter an instruction before sending."]
    assert "config_chat_last_instruction" not in st.session_state
    assert st.success_messages == []

    st.warning_messages.clear()
    st.session_state["config_chat_instruction"] = "  Raise lookback  "
    page._render_config_chat_contents({"a": 1})
    assert st.warning_messages == []
    assert st.session_state["config_chat_last_instruction"] == "Raise lookback"
    assert st.success_messages == ["Instruction captured. Preview coming next."]


@pytest.mark.parametrize(
    ("instruction", "model_state", "kind", "message"),
    [
        ("   ", {"a": 1}, "warning_messages", "Enter an instruction before previewing."),
        (
            "Raise lookback",
            None,
            "error_messages",
            "No configuration is loaded to preview against.",
        ),
    ],
)
def test_config_chat_preview_needs_an_instruction_and_a_loaded_configuration(
    model_page: ModelPage,
    monkeypatch: pytest.MonkeyPatch,
    instruction: str,
    model_state: dict[str, int] | None,
    kind: str,
    message: str,
) -> None:
    page, st = model_page
    _stub_chat_sections(page, monkeypatch)
    generated: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        page, "_generate_preview_with_progress", lambda *args: generated.append(args)
    )
    st.clicked.add("config_chat_preview_btn")
    st.session_state["config_chat_instruction"] = instruction

    page._render_config_chat_contents(model_state)

    assert getattr(st, kind) == [message]
    assert generated == []
    assert "config_chat_preview" not in st.session_state


def test_config_chat_preview_reports_failures_and_stores_successful_previews(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    _stub_chat_sections(page, monkeypatch)
    model_state = {"lookback_periods": 6}
    preview = {"after": {"lookback_periods": 12}}
    generated: list[tuple[Any, str]] = []

    def fake_generate(state: Any, instruction: str) -> dict[str, Any]:
        generated.append((state, instruction))
        if len(generated) == 1:
            raise RuntimeError("provider down")
        return preview

    monkeypatch.setattr(page, "_generate_preview_with_progress", fake_generate)
    st.clicked.add("config_chat_preview_btn")
    st.session_state["config_chat_instruction"] = "  Raise lookback  "

    page._render_config_chat_contents(model_state)
    assert st.error_messages == ["Preview failed: provider down"]
    assert "config_chat_preview" not in st.session_state
    assert "config_chat_last_instruction" not in st.session_state
    assert st.success_messages == []

    st.error_messages.clear()
    page._render_config_chat_contents(model_state)
    assert generated == [(model_state, "Raise lookback"), (model_state, "Raise lookback")]
    assert st.session_state["config_chat_preview"] is preview
    assert st.session_state["config_chat_last_instruction"] == "Raise lookback"
    assert st.success_messages == ["Preview ready. Review the diff below."]
    assert st.error_messages == []


@pytest.mark.parametrize(
    ("button", "risky", "run_analysis"),
    [
        ("config_chat_apply_btn", False, False),
        ("config_chat_apply_run_btn", False, True),
        ("config_chat_apply_btn", True, False),
        ("config_chat_apply_run_btn", True, True),
    ],
)
def test_config_chat_apply_buttons_apply_or_queue_the_preview(
    model_page: ModelPage,
    monkeypatch: pytest.MonkeyPatch,
    button: str,
    risky: bool,
    run_analysis: bool,
) -> None:
    page, st = model_page
    _stub_chat_sections(page, monkeypatch)
    applied: list[tuple[object, bool]] = []
    monkeypatch.setattr(
        page,
        "_apply_preview_state",
        lambda preview, *, run_analysis: applied.append((preview, run_analysis)),
    )
    preview = {"after": {"lookback_periods": 12}, "risk_flags": ["BROAD_SCOPE"] if risky else []}
    st.session_state["config_chat_preview"] = preview
    st.clicked.add(button)

    page._render_config_chat_contents({"lookback_periods": 6})

    if risky:
        assert applied == []
        assert st.session_state["config_chat_pending_apply"] == {
            "preview": preview,
            "run_analysis": run_analysis,
        }
    else:
        assert applied == [(preview, run_analysis)]
        assert "config_chat_pending_apply" not in st.session_state


def test_config_chat_panel_is_replaced_by_a_notice_when_the_zone_disables_llm(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    rendered: list[str] = []
    monkeypatch.setattr(page, "_render_llm_status_panel", lambda: rendered.append("status"))
    monkeypatch.setattr(
        page, "_render_llm_session_overrides_panel", lambda: rendered.append("keys")
    )
    monkeypatch.setattr(
        page, "_render_config_chat_contents", lambda state: rendered.append(f"contents:{state}")
    )

    monkeypatch.setenv("TREND_LLM_ZONE", "Disabled")
    page.render_config_chat_panel(model_state={"a": 1})
    assert st.info_messages == [
        "Natural-language config editing is disabled for this deployment zone "
        "(TREND_LLM_ZONE=disabled)."
    ]
    assert rendered == []
    assert st.expander_labels == []

    monkeypatch.delenv("TREND_LLM_ZONE")
    page.render_config_chat_panel(model_state={"a": 1})
    assert rendered == ["status", "keys", "contents:{'a': 1}"]
    assert st.expander_labels == ["💬 Config Chat"]


def test_config_chat_panel_falls_back_to_a_null_context_without_a_sidebar_context(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    rendered: list[str] = []
    monkeypatch.setattr(page, "_render_llm_status_panel", lambda: rendered.append("status"))
    monkeypatch.setattr(
        page, "_render_llm_session_overrides_panel", lambda: rendered.append("keys")
    )
    monkeypatch.setattr(
        page, "_render_config_chat_contents", lambda state: rendered.append(f"contents:{state}")
    )
    monkeypatch.setattr(st, "sidebar", SimpleNamespace())

    page.render_config_chat_panel(location="sidebar", model_state={"a": 1})

    assert rendered == ["status", "keys", "contents:{'a': 1}"]
    assert st.expander_labels == ["💬 Config Chat"]


# --- what-if panel --------------------------------------------------------------------------


def test_what_if_panel_is_replaced_by_a_notice_when_the_zone_disables_llm(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    monkeypatch.setenv("TREND_LLM_ZONE", "disabled")

    page._render_variant_what_if_panel(model_state={"a": 1}, returns=None, benchmark=None)

    assert st.subheaders == ["🔀 What-if (3 variants)"]
    assert st.info_messages == [
        "LLM what-if generation is disabled for this deployment zone (TREND_LLM_ZONE=disabled)."
    ]
    assert st.captions == []
    assert st.button_calls == []


@pytest.mark.parametrize(
    ("instruction", "has_state", "has_returns", "kind", "message"),
    [
        ("   ", True, True, "warning_messages", "Enter an instruction to generate variants."),
        (
            "Tighten weights",
            False,
            True,
            "error_messages",
            "No configuration is loaded to generate variants.",
        ),
        (
            "Tighten weights",
            True,
            False,
            "error_messages",
            "Load data on the Data page before running What-if analysis.",
        ),
    ],
)
def test_what_if_panel_validates_its_inputs_before_generating(
    model_page: ModelPage,
    monkeypatch: pytest.MonkeyPatch,
    instruction: str,
    has_state: bool,
    has_returns: bool,
    kind: str,
    message: str,
) -> None:
    page, st = model_page
    chains: list[str] = []
    monkeypatch.setattr(page, "_get_variant_chain", lambda: chains.append("chain"))
    st.clicked.add("what_if_run")
    stale_keys = ("what_if_variant_results", "what_if_variant_metrics", "what_if_variant_error")
    st.session_state["what_if_instruction"] = instruction
    for key in stale_keys:
        st.session_state[key] = "stale"

    page._render_variant_what_if_panel(
        model_state={"a": 1} if has_state else None,
        returns=pd.DataFrame({"A": [0.1]}) if has_returns else None,
        benchmark=None,
    )

    assert getattr(st, kind) == [message]
    assert chains == []
    for key in stale_keys:
        assert key not in st.session_state


def test_what_if_panel_runs_the_three_variants_and_stores_the_results(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    returns = pd.DataFrame({"A": [0.01, 0.02]})
    model_state = {"lookback_periods": 6}
    patches = ["conservative-patch", "baseline-patch", "aggressive-patch"]
    results = {
        "conservative": SimpleNamespace(
            metrics=pd.DataFrame([{"sharpe": 1.1, "return_ann": 0.08}])
        ),
        "baseline": SimpleNamespace(metrics=pd.DataFrame([{"sharpe": 1.0, "return_ann": 0.07}])),
        "aggressive": SimpleNamespace(metrics=pd.DataFrame([{"sharpe": 0.9, "return_ann": 0.06}])),
    }
    seen: dict[str, Any] = {}

    class FakeChain:
        def run(self, *, current_config: dict[str, Any], instruction: str) -> SimpleNamespace:
            seen["chain"] = (current_config, instruction)
            return SimpleNamespace(variants=patches)

    def fake_variant_analysis(
        frame: Any, state: Any, benchmark: Any, variants: Any, *, data_hash: Any, progress: Any
    ) -> dict[str, Any]:
        seen["analysis"] = (frame, state, benchmark, variants, data_hash)
        for index, label in enumerate(("conservative", "baseline", "aggressive"), start=1):
            progress(label, index, 3)
        return results

    monkeypatch.setattr(page, "_get_variant_chain", lambda: (FakeChain(), {}))
    monkeypatch.setattr(page.analysis_runner, "run_variant_analysis", fake_variant_analysis)
    st.clicked.add("what_if_run")
    st.session_state.update(
        {
            "what_if_instruction": "  Reduce turnover  ",
            "data_hash": "hash-1",
            "what_if_variant_error": "stale error",
        }
    )

    page._render_variant_what_if_panel(model_state=model_state, returns=returns, benchmark="SPX")

    assert seen["chain"] == ({"lookback_periods": 6}, "Reduce turnover")
    frame, state, benchmark, variants, data_hash = seen["analysis"]
    assert frame is returns
    assert state is model_state
    assert benchmark == "SPX"
    assert variants is patches
    assert data_hash == "hash-1"
    assert st.progress_calls == [
        (0.0, "Generating variants..."),
        (1 / 3, "Running conservative (1/3)"),
        (2 / 3, "Running baseline (2/3)"),
        (1.0, "Running aggressive (3/3)"),
        (1.0, "All variants complete."),
    ]
    _progress_slot, status_slot = st.slots
    assert status_slot.events == [
        ("info", "Running conservative variant (1/3)…"),
        ("info", "Running baseline variant (2/3)…"),
        ("info", "Running aggressive variant (3/3)…"),
        ("success", "What-if analysis complete."),
    ]
    assert st.session_state["what_if_variant_results"] is results
    assert "what_if_variant_error" not in st.session_state
    pd.testing.assert_frame_equal(
        st.session_state["what_if_variant_metrics"],
        pd.DataFrame(
            [
                {
                    "Metric": "return_ann",
                    "conservative": 0.08,
                    "baseline": 0.07,
                    "aggressive": 0.06,
                },
                {"Metric": "sharpe", "conservative": 1.1, "baseline": 1.0, "aggressive": 0.9},
            ]
        ),
    )
    assert st.error_messages == []


def test_what_if_panel_reports_a_variant_run_error(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page
    error = page.analysis_runner.VariantRunError(
        label="aggressive", stage="analysis run", detail="boom"
    )

    def failing_analysis(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise error

    chain = SimpleNamespace(run=lambda **_kwargs: SimpleNamespace(variants=[]))
    monkeypatch.setattr(page, "_get_variant_chain", lambda: (chain, {}))
    monkeypatch.setattr(page.analysis_runner, "run_variant_analysis", failing_analysis)
    st.clicked.add("what_if_run")
    st.session_state["what_if_instruction"] = "Tighten weights"

    page._render_variant_what_if_panel(
        model_state={"a": 1}, returns=pd.DataFrame({"A": [0.1]}), benchmark=None
    )

    message = "Variant 'aggressive' failed during analysis run: boom"
    assert st.session_state["what_if_variant_error"] == message
    assert st.slots[1].events == [("error", message)]
    assert st.error_messages == [message]
    assert "what_if_variant_results" not in st.session_state


def test_what_if_panel_reports_an_unexpected_failure(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, st = model_page

    def missing_key() -> None:
        raise ValueError("Missing API key for openai")

    monkeypatch.setattr(page, "_get_variant_chain", missing_key)
    st.clicked.add("what_if_run")
    st.session_state["what_if_instruction"] = "Tighten weights"

    page._render_variant_what_if_panel(
        model_state={"a": 1}, returns=pd.DataFrame({"A": [0.1]}), benchmark=None
    )

    message = "What-if analysis failed: Missing API key for openai"
    assert st.session_state["what_if_variant_error"] == message
    assert st.slots[1].events == [("error", message)]
    assert st.error_messages == [message]


def test_what_if_panel_warns_when_the_variant_results_have_no_metrics(
    model_page: ModelPage,
) -> None:
    page, st = model_page
    st.session_state["what_if_variant_metrics"] = pd.DataFrame()

    page._render_variant_what_if_panel(model_state={"a": 1}, returns=None, benchmark=None)

    assert st.warning_messages == ["No metrics were detected in the variant results."]
    assert st.dataframe_calls == []
    assert st.downloads == []


def test_what_if_panel_renders_the_metrics_table_with_csv_and_json_downloads(
    model_page: ModelPage,
) -> None:
    page, st = model_page
    table = pd.DataFrame(
        [
            {"Metric": "return_ann", "conservative": 0.08, "baseline": 0.07, "aggressive": 0.06},
            {"Metric": "sharpe", "conservative": 1.1, "baseline": 1.0, "aggressive": 0.9},
        ]
    )
    st.session_state["what_if_variant_metrics"] = table

    page._render_variant_what_if_panel(model_state={"a": 1}, returns=None, benchmark=None)

    assert len(st.dataframe_calls) == 1
    assert st.dataframe_calls[0][0] is table
    assert st.dataframe_calls[0][1] == {"use_container_width": True, "hide_index": True}
    assert st.downloads == [
        {
            "label": "Download Metrics (CSV)",
            "data": b"Metric,conservative,baseline,aggressive\n"
            b"return_ann,0.08,0.07,0.06\nsharpe,1.1,1.0,0.9\n",
            "file_name": "what_if_variants_metrics.csv",
            "mime": "text/csv",
        },
        {
            "label": "Download Metrics (JSON)",
            "data": b'[{"Metric":"return_ann","conservative":0.08,"baseline":0.07,'
            b'"aggressive":0.06},{"Metric":"sharpe","conservative":1.1,"baseline":1.0,'
            b'"aggressive":0.9}]',
            "file_name": "what_if_variants_metrics.json",
            "mime": "application/json",
        },
    ]
    assert st.warning_messages == []


# --- small pure helpers and the auto-render gate --------------------------------------------


def test_normalize_weights_scales_to_one_and_leaves_non_positive_totals_alone(
    model_page: ModelPage,
) -> None:
    page, _st = model_page

    assert page._normalize_weights({"a": 1, "b": 3}) == {"a": 0.25, "b": 0.75}
    assert page._normalize_weights({"a": 1.0, "b": 2.0}) == {"a": 0.3333, "b": 0.6667}
    zero = {"a": 0.0, "b": 0}
    assert page._normalize_weights(zero) is zero
    negative = {"a": -1.0, "b": 0.5}
    assert page._normalize_weights(negative) is negative


def test_get_benchmark_columns_lists_known_benchmarks_first(model_page: ModelPage) -> None:
    page, _st = model_page
    frame = pd.DataFrame(columns=["Date", "FundA", "spx", "index", "ACWI", "FundB"])

    assert page._get_benchmark_columns(frame) == ["spx", "ACWI", "FundA", "FundB"]
    assert page._get_benchmark_columns(None) == []


def test_validate_model_requires_a_benchmark_for_a_positive_information_ratio_weight(
    model_page: ModelPage,
) -> None:
    page, _st = model_page
    values: dict[str, Any] = {
        "lookback_periods": 12,
        "min_history_periods": 12,
        "selection_count": 5,
        "metric_weights": {"sharpe": 1.0, "info_ratio": 0.5},
    }

    assert page._validate_model(values, column_count=10) == [
        "Select a benchmark for Information Ratio metric."
    ]
    assert page._validate_model({**values, "info_ratio_benchmark": "SPX"}, column_count=10) == []
    zero_weight = {**values, "metric_weights": {"sharpe": 1.0, "info_ratio": 0.0}}
    assert page._validate_model(zero_weight, column_count=10) == []


def test_should_auto_render_follows_the_script_run_context(
    model_page: ModelPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    page, _st = model_page
    monkeypatch.delitem(sys.modules, "streamlit.runtime", raising=False)
    monkeypatch.delitem(sys.modules, "streamlit.runtime.scriptrunner", raising=False)
    assert page._should_auto_render() is False

    scriptrunner = ModuleType("streamlit.runtime.scriptrunner")
    monkeypatch.setitem(sys.modules, "streamlit.runtime.scriptrunner", scriptrunner)
    scriptrunner.get_script_run_ctx = lambda: None
    assert page._should_auto_render() is False
    scriptrunner.get_script_run_ctx = lambda: object()
    assert page._should_auto_render() is True
