"""Tests for work package t6: developer_settings_validation and comparison_llm coverage.

Covers uncovered statements in:
- streamlit_app/developer_settings_validation.py (200 statements)
- streamlit_app/components/comparison_llm.py (119 statements)

All external seams (providers, proxy, network) are stubbed.
Tests use the existing DummyStreamlit infrastructure.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests.support.dummy_streamlit import DummyStreamlit, Placeholder

# =============================================================================
# Helper stubs
# =============================================================================


class EnhancedPlaceholder(Placeholder):
    """Enhanced Placeholder that supports text, write, markdown, etc. methods."""

    def text(self, msg: str) -> None:
        self._stub.write_calls.append(msg)

    def write(self, msg: str) -> None:
        self._stub.write_calls.append(msg)

    def markdown(self, msg: str) -> None:
        self._stub.markdowns.append(msg)

    def info(self, msg: str) -> None:
        self._stub.info_messages.append(msg)

    def warning(self, msg: str) -> None:
        self._stub.warning_messages.append(msg)

    def error(self, msg: str) -> None:
        self._stub.error_messages.append(msg)

    def success(self, msg: str) -> None:
        self._stub.success_messages.append(msg)


class CustomDummyStreamlit(DummyStreamlit):
    """Keep the t6 placeholder extensions local to this test module."""

    def empty(self) -> EnhancedPlaceholder:
        return EnhancedPlaceholder(self)


@dataclass
class MockRunResult:
    """Mock RunResult for testing extract_key_metrics."""

    weights: pd.Series | None = None
    period_results: list[dict[str, Any]] | None = None
    period_count: int | None = None
    metrics: pd.DataFrame | None = None
    costs: dict[str, float] | None = None
    turnover: pd.Series | None = None


@dataclass
class StubResult:
    """Stub result for comparison_llm tests."""

    details: dict[str, Any]


def _enable_widget_call_recording(stub: DummyStreamlit) -> None:
    """Record text_input/text_area keys and button keys for comparison_llm widget tests."""
    stub.text_input_calls: list[tuple[str, str | None, str]] = []
    stub.button_key_calls: list[str | None] = []
    stub.selectbox_index_calls: list[int | None] = []

    def recording_text_input(label: str = "", value: str = "", **kwargs: Any) -> str:
        key = kwargs.get("key")
        resolved = str(kwargs.get("value", value))
        stub.text_input_calls.append((label, key, resolved))
        if key is not None and key not in stub.session_state:
            stub.session_state[key] = resolved
        return resolved

    real_button = stub.button

    def recording_button(label: str = "", *args: Any, **kwargs: Any) -> bool:
        stub.button_key_calls.append(kwargs.get("key"))
        return real_button(label, *args, **kwargs)

    real_selectbox = stub.selectbox

    def recording_selectbox(*args: Any, **kwargs: Any) -> Any:
        stub.selectbox_index_calls.append(kwargs.get("index"))
        return real_selectbox(*args, **kwargs)

    stub.text_input = recording_text_input
    stub.button = recording_button
    stub.selectbox = recording_selectbox


def _install_streamlit_stub(
    monkeypatch: pytest.MonkeyPatch,
    stub: DummyStreamlit,
    *,
    active_context: bool,
    use_enhanced_placeholder: bool = False,
) -> ModuleType:
    """Install a stub streamlit module."""
    module = ModuleType("streamlit")
    module.__path__ = []

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
        "info",
        "success",
        "error",
        "caption",
        "text_input",
        "spinner",
        "progress",
        "empty",
        "dataframe",
        "download_button",
        "session_state",
        "sidebar",
    ):
        setattr(module, attr, getattr(stub, attr))

    # Add text_area as an alias to text_input
    module.text_area = stub.text_input

    module.session_state = stub.session_state
    module.sidebar = stub.sidebar

    runtime = ModuleType("streamlit.runtime")
    scriptrunner = ModuleType("streamlit.runtime.scriptrunner")
    scriptrunner.get_script_run_ctx = lambda: object() if active_context else None

    monkeypatch.setitem(sys.modules, "streamlit", module)
    monkeypatch.setitem(sys.modules, "streamlit.runtime", runtime)
    monkeypatch.setitem(sys.modules, "streamlit.runtime.scriptrunner", scriptrunner)
    return module


def _reload_module(
    monkeypatch: pytest.MonkeyPatch, module_name: str, *, auto_render: bool = False
) -> Any:
    """Import a module fresh against the stubbed streamlit.

    The page module renders itself at import when it runs inside a Streamlit script context. That
    first render would consume the queued ``button_responses`` and call the real analysis runner
    before a test has installed its mocks, so the import runs with no script context unless the
    test is about the import-time render (``auto_render=True``).
    """
    for name in [
        m for m in list(sys.modules) if m == module_name or m.startswith(f"{module_name}.")
    ]:
        del sys.modules[name]
    importlib.invalidate_caches()
    scriptrunner = sys.modules.get("streamlit.runtime.scriptrunner")
    original = getattr(scriptrunner, "get_script_run_ctx", None)
    if scriptrunner is not None and not auto_render:
        scriptrunner.get_script_run_ctx = lambda: None
    try:
        return importlib.import_module(module_name)
    finally:
        if scriptrunner is not None and original is not None:
            scriptrunner.get_script_run_ctx = original


def _mock_demo_profile_and_state(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool = True,
    profile_name: str = "public_llm_demo",
    returns_data: pd.DataFrame | None = None,
) -> None:
    """Mock demo_profile and state modules."""
    package = importlib.import_module("streamlit_app")
    mock_profile = MagicMock()
    mock_profile.custom_analysis_enabled.return_value = enabled
    mock_profile.get_active_profile.return_value = profile_name
    mock_profile.initialize_profile = MagicMock()

    monkeypatch.setitem(sys.modules, "streamlit_app.demo_profile", mock_profile)
    monkeypatch.setattr(package, "demo_profile", mock_profile, raising=False)
    monkeypatch.setattr(
        sys.modules["streamlit_app.demo_profile"],
        "custom_analysis_enabled",
        lambda _: enabled,
    )
    monkeypatch.setattr(
        sys.modules["streamlit_app.demo_profile"],
        "get_active_profile",
        lambda: profile_name,
    )
    monkeypatch.setattr(
        sys.modules["streamlit_app.demo_profile"],
        "initialize_profile",
        lambda _: None,
    )

    def fake_get_uploaded_data() -> tuple[Any, Any]:
        if returns_data is not None:
            return (returns_data, {"validation": {"issues": [], "warnings": []}})
        return (None, {})

    mock_app_state = MagicMock()
    mock_app_state.get_uploaded_data = fake_get_uploaded_data
    mock_app_state.initialize_session_state = MagicMock()

    monkeypatch.setitem(sys.modules, "streamlit_app.state", mock_app_state)
    monkeypatch.setattr(package, "state", mock_app_state, raising=False)
    monkeypatch.setattr(
        sys.modules["streamlit_app.state"], "get_uploaded_data", fake_get_uploaded_data
    )
    monkeypatch.setattr(
        sys.modules["streamlit_app.state"], "initialize_session_state", lambda: None
    )


# =============================================================================
# developer_settings_validation.py - get_baseline_state (line 346)
# =============================================================================


def test_get_baseline_state_returns_expected_structure() -> None:
    """Covers line 346."""
    from streamlit_app.developer_settings_validation import get_baseline_state

    state = get_baseline_state()
    assert isinstance(state, dict)
    assert state["preset"] == "Test"
    assert state["lookback_periods"] == 3
    assert state["selection_count"] == 10
    assert state["inclusion_approach"] == "threshold"
    assert state["rank_pct"] == 0.10
    assert state["random_seed"] == 42
    assert state["weighting_name"] == "equal"


# =============================================================================
# developer_settings_validation.py - format_value (lines 528, 530-538)
# =============================================================================


def test_format_value_with_none() -> None:
    """Covers line 528."""
    from streamlit_app.developer_settings_validation import format_value

    assert format_value(None) == "None"


def test_format_value_with_small_float() -> None:
    """Covers lines 530-532."""
    from streamlit_app.developer_settings_validation import format_value

    assert format_value(0.001) == "0.0010"
    assert format_value(-0.001) == "-0.0010"


def test_format_value_with_regular_float() -> None:
    """Covers line 533."""
    from streamlit_app.developer_settings_validation import format_value

    assert format_value(0.5) == "0.50"


def test_format_value_with_dict() -> None:
    """Covers line 534."""
    from streamlit_app.developer_settings_validation import format_value

    result = format_value({"a": 1})
    assert "{\n " in result


def test_format_value_with_list_short() -> None:
    """Covers line 535."""
    from streamlit_app.developer_settings_validation import format_value

    assert format_value([1, 2]) == "[1, 2]"


def test_format_value_with_list_long() -> None:
    """Covers lines 536-538."""
    from streamlit_app.developer_settings_validation import format_value

    assert format_value([1, 2, 3, 4, 5]) == "[1, 2, 3]... (5 items)"


def test_format_value_with_other_types() -> None:
    """Covers line 539."""
    from streamlit_app.developer_settings_validation import format_value

    assert format_value("test") == "test"


# =============================================================================
# developer_settings_validation.py - extract_key_metrics
# (lines 430, 434-458, 461-472, 475-478, 481-482, 486, 488-490, 492, 495-496, 498-499, 501)
# =============================================================================


def test_extract_key_metrics_with_weights() -> None:
    """Covers lines 434-440."""
    from streamlit_app.developer_settings_validation import extract_key_metrics

    result = MockRunResult(
        weights=pd.Series({"A": 0.5, "B": 0.3, "C": 0.2, "D": 0.0}),
    )
    metrics = extract_key_metrics(result)
    assert metrics["num_funds"] == 3
    assert metrics["max_weight"] == 0.5
    assert "weight_std" in metrics


def test_extract_key_metrics_with_period_results() -> None:
    """Covers lines 443-454."""
    from streamlit_app.developer_settings_validation import extract_key_metrics

    result = MockRunResult(
        period_results=[
            {"weights": {"A": 0.5, "B": 0.5}},
            {"weights": {"A": 0.4, "B": 0.3, "C": 0.3}},
        ],
    )
    metrics = extract_key_metrics(result)
    assert metrics["num_periods"] == 2
    assert metrics["avg_funds_per_period"] == 2.5
    assert metrics["min_funds_period"] == 2
    assert metrics["max_funds_period"] == 3


def test_extract_key_metrics_period_weights_positive_count() -> None:
    """Pins line 449: period_results weights dict drives fund counts."""
    from streamlit_app.developer_settings_validation import extract_key_metrics

    result = MockRunResult(
        period_results=[
            {"weights": {"X": 0.5, "Y": 0.5, "Z": 0.0}},
            {"weights": {"P": 1.0, "Q": 1.0, "R": 1.0}},
        ],
    )
    metrics = extract_key_metrics(result)
    assert metrics["num_periods"] == 2
    assert metrics["avg_funds_per_period"] == 2.5
    assert metrics["min_funds_period"] == 2
    assert metrics["max_funds_period"] == 3


def test_extract_key_metrics_period_count_fallback() -> None:
    """Covers lines 457-458."""
    from streamlit_app.developer_settings_validation import extract_key_metrics

    result = MockRunResult(period_count=5)
    metrics = extract_key_metrics(result)
    assert metrics["num_periods"] == 5


def test_extract_key_metrics_with_summary_metrics() -> None:
    """Covers lines 461-472."""
    from streamlit_app.developer_settings_validation import extract_key_metrics

    result = MockRunResult(
        metrics=pd.DataFrame({"CAGR": [0.15], "Sharpe": [1.5], "MaxDrawdown": [-0.10]}),
    )
    metrics = extract_key_metrics(result)
    assert metrics["portfolio_CAGR"] == 0.15
    assert metrics["portfolio_Sharpe"] == 1.5


def test_extract_key_metrics_with_sortino_and_maxdrawdown() -> None:
    """Covers lines 466, 470."""
    from streamlit_app.developer_settings_validation import extract_key_metrics

    result = MockRunResult(
        metrics=pd.DataFrame({"Sortino": [2.0], "MaxDrawdown": [-0.15]}),
    )
    metrics = extract_key_metrics(result)
    assert metrics["portfolio_Sortino"] == 2.0
    assert metrics["portfolio_MaxDrawdown"] == -0.15


def test_extract_key_metrics_with_costs() -> None:
    """Covers lines 475-478."""
    from streamlit_app.developer_settings_validation import extract_key_metrics

    result = MockRunResult(costs={"total": 0.05, "transaction": 0.03})
    metrics = extract_key_metrics(result)
    assert metrics["cost_total"] == 0.05


def test_extract_key_metrics_with_turnover() -> None:
    """Covers lines 481-482."""
    from streamlit_app.developer_settings_validation import extract_key_metrics

    result = MockRunResult(turnover=pd.Series([0.1, 0.2, 0.3]))
    metrics = extract_key_metrics(result)
    assert metrics["avg_turnover"] == pytest.approx(0.2, rel=1e-5)


def test_extract_key_metrics_hash_success() -> None:
    """Covers lines 486, 488-490."""
    from streamlit_app.developer_settings_validation import extract_key_metrics

    result = MockRunResult()
    metrics = extract_key_metrics(result)
    assert "_result_hash" in metrics
    assert metrics["_result_hash"] == "14fe006b"


def test_extract_key_metrics_hash_typeerror(capsys: pytest.CaptureFixture[str]) -> None:
    """Covers lines 492, 495-496."""
    from streamlit_app.developer_settings_validation import extract_key_metrics

    with patch("streamlit_app.developer_settings_validation.json.dumps", side_effect=TypeError()):
        result = MockRunResult()
        metrics = extract_key_metrics(result)
        assert metrics.get("_result_hash") == "error"
        captured = capsys.readouterr()
        assert "serialization issue" in captured.out
        assert "TypeError" in captured.out


def test_extract_key_metrics_hash_valueerror(capsys: pytest.CaptureFixture[str]) -> None:
    """Covers lines 492, 495-496."""
    from streamlit_app.developer_settings_validation import extract_key_metrics

    with patch("streamlit_app.developer_settings_validation.json.dumps", side_effect=ValueError()):
        result = MockRunResult()
        metrics = extract_key_metrics(result)
        assert metrics.get("_result_hash") == "error"
        captured = capsys.readouterr()
        assert "serialization issue" in captured.out
        assert "ValueError" in captured.out


def test_extract_key_metrics_hash_unexpected_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Covers lines 498-499, 501."""
    from streamlit_app.developer_settings_validation import extract_key_metrics

    with patch(
        "streamlit_app.developer_settings_validation.json.dumps", side_effect=RuntimeError("boom")
    ):
        result = MockRunResult()
        metrics = extract_key_metrics(result)
        assert metrics.get("_result_hash") == "error"
        captured = capsys.readouterr()
        assert "Unexpected error computing _result_hash" in captured.out
        assert "boom" in captured.out


# =============================================================================
# developer_settings_validation.py - run_test_analysis (lines 521-522)
# =============================================================================


def test_run_test_analysis_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers line 521."""
    from streamlit_app.developer_settings_validation import run_test_analysis

    mock_result = object()
    with patch(
        "streamlit_app.developer_settings_validation.analysis_runner.run_analysis",
        return_value=mock_result,
    ):
        with patch("streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"):
            result = run_test_analysis(pd.DataFrame({"A": [1]}), {})
            assert result == {"status": "success", "result": mock_result}


def test_run_test_analysis_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers line 522."""
    from streamlit_app.developer_settings_validation import run_test_analysis

    with patch(
        "streamlit_app.developer_settings_validation.analysis_runner.run_analysis",
        side_effect=ValueError("error"),
    ):
        with patch("streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"):
            result = run_test_analysis(pd.DataFrame({"A": [1]}), {})
            assert result == {"status": "error", "error": "error"}


# =============================================================================
# developer_settings_validation.py - render_validation_page (lines 572-573, 628-730, ...)
# =============================================================================


def test_render_validation_page_stops_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 551-559."""
    stub = CustomDummyStreamlit()
    stub.stop_error = "disabled"
    _install_streamlit_stub(monkeypatch, stub, active_context=False)
    _mock_demo_profile_and_state(monkeypatch, enabled=False)
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    with pytest.raises(AssertionError, match="disabled"):
        page.render_validation_page()
    assert any("disabled" in msg.lower() for msg in stub.warning_messages)


def test_render_validation_page_stops_without_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 570-573."""
    stub = CustomDummyStreamlit()
    stub.stop_error = "no data"
    _install_streamlit_stub(monkeypatch, stub, active_context=False)
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=None)
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    with pytest.raises(AssertionError, match="no data"):
        page.render_validation_page()
    assert any("Please load data" in msg for msg in stub.warning_messages)


@pytest.mark.parametrize("preload_modules", [False, True])
def test_render_validation_page_renders(
    monkeypatch: pytest.MonkeyPatch, preload_modules: bool
) -> None:
    """Covers lines 561-627."""
    stub = CustomDummyStreamlit()
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    if preload_modules:
        importlib.import_module("streamlit_app.demo_profile")
        importlib.import_module("streamlit_app.state")
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    page.render_validation_page()
    assert not stub.stop_called
    assert any("Settings Validation" in msg for msg in stub.title_calls)


def test_render_validation_page_sidebar_widgets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 583-601."""
    stub = CustomDummyStreamlit()
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    page.render_validation_page()
    assert any("Category" in label for label, _ in stub.selectbox_calls)


def test_render_validation_page_baseline_test_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 610-625."""
    stub = CustomDummyStreamlit()
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    page.render_validation_page()
    assert any("Baseline Configuration" in msg for msg in stub.subheaders)
    assert any("Test Configuration" in msg for msg in stub.subheaders)


def test_render_validation_page_button(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers line 627."""
    stub = CustomDummyStreamlit()
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    page.render_validation_page()
    assert any("Run Comparison Test" in label for label in stub.button_labels)


def test_render_validation_page_button_click_baseline_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 628-651."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [True]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    mock_result = MockRunResult()
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    with patch.object(page.analysis_runner, "run_analysis", return_value=mock_result):
        with patch("streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"):
            page.render_validation_page()
    # Baseline ran successfully


def test_render_validation_page_hash_computation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 669-670."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [True]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    mock_result = MockRunResult()
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    with patch(
        "streamlit_app.developer_settings_validation.json.dumps",
        side_effect=TypeError("serialize"),
    ):
        with patch.object(
            page.analysis_runner, "run_analysis", return_value=mock_result
        ) as run_mock:
            with patch(
                "streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"
            ):
                page.render_validation_page()
    assert run_mock.call_count == 2
    assert any("HASH COMPUTATION FAILED" in msg for msg in stub.warning_messages)
    assert any("`selection_count`" in msg for msg in stub.warning_messages)
    last_test = stub.session_state["last_validation_test"]
    assert last_test["setting"] == "selection_count"
    assert last_test["baseline_value"] == 10
    assert last_test["test_value"] == 5
    assert last_test["changed"] is False
    assert last_test["baseline_metrics"]["_result_hash"] == "error"
    assert last_test["test_metrics"]["_result_hash"] == "error"


def test_render_validation_page_setting_not_wired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 674-675."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [True]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    mock_result = MockRunResult()
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    with patch.object(page.analysis_runner, "run_analysis", return_value=mock_result) as run_mock:
        with patch("streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"):
            page.render_validation_page()
    assert run_mock.call_count == 2
    assert any("SETTING NOT WIRED" in msg for msg in stub.error_messages)
    assert any("`selection_count`" in msg for msg in stub.error_messages)
    last_test = stub.session_state["last_validation_test"]
    assert last_test["setting"] == "selection_count"
    assert last_test["baseline_value"] == 10
    assert last_test["test_value"] == 5
    assert last_test["changed"] is False


def test_render_validation_page_setting_is_wired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 686-689."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [True]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    # Create different results for baseline and test
    call_count = [0]

    def fake_run(*_args: Any, **_kwargs: Any) -> MockRunResult:
        call_count[0] += 1
        w = pd.Series({"A": 0.5 + call_count[0] * 0.1})  # Different weights
        return MockRunResult(
            weights=w, metrics=pd.DataFrame({"CAGR": [0.1 + call_count[0] * 0.01]})
        )

    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    with patch.object(page.analysis_runner, "run_analysis", side_effect=fake_run) as run_mock:
        with patch("streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"):
            page.render_validation_page()
    assert run_mock.call_count == 2
    assert call_count[0] == 2
    assert any("SETTING IS WIRED" in msg for msg in stub.success_messages)
    assert any("`selection_count`" in msg for msg in stub.success_messages)
    last_test = stub.session_state["last_validation_test"]
    assert last_test["setting"] == "selection_count"
    assert last_test["baseline_value"] == 10
    assert last_test["test_value"] == 5
    assert last_test["changed"] is True
    assert (
        last_test["baseline_metrics"]["_result_hash"] != last_test["test_metrics"]["_result_hash"]
    )


def test_render_validation_page_results_no_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 664-684."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [True]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    mock_result = MockRunResult()
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    with patch.object(page.analysis_runner, "run_analysis", return_value=mock_result) as run_mock:
        with patch("streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"):
            page.render_validation_page()
    assert run_mock.call_count == 2
    assert any("SETTING NOT WIRED" in msg for msg in stub.error_messages)
    assert not any("🔄" in msg for msg in stub.markdowns)
    last_test = stub.session_state["last_validation_test"]
    assert last_test["changed"] is False
    assert (
        last_test["baseline_metrics"]["_result_hash"] == last_test["test_metrics"]["_result_hash"]
    )


def test_render_validation_page_metric_display_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 691-723 (metric display with changes)."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [True]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    # Create different results with metrics
    call_count = [0]

    def fake_run(*_args: Any, **_kwargs: Any) -> MockRunResult:
        call_count[0] += 1
        w = pd.Series({"A": 0.5 + call_count[0] * 0.1})  # Different weights
        return MockRunResult(
            weights=w,
            metrics=pd.DataFrame(
                {
                    "CAGR": [0.1 + call_count[0] * 0.01],
                    "Sharpe": [1.0 + call_count[0] * 0.1],
                    "Volatility": [0.15 - call_count[0] * 0.01],
                }
            ),
        )

    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    with patch.object(page.analysis_runner, "run_analysis", side_effect=fake_run):
        with patch("streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"):
            page.render_validation_page()
    assert call_count[0] >= 2
    assert not any("`_result_hash`" in msg for msg in stub.markdowns)
    assert any("`portfolio_CAGR`" in msg for msg in stub.markdowns)
    assert any("🔄" in msg for msg in stub.markdowns)


def test_render_validation_page_results_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 661-684."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [True]  # Click the Run Comparison Test button
    stub.selectbox_returns = [
        "Selection",
        "Selection Count",
    ]  # Select first category and first setting
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    # Different results for baseline and test
    call_count = [0]

    def fake_run(*_args: Any, **_kwargs: Any) -> MockRunResult:
        call_count[0] += 1
        w = pd.Series({"A": 0.5 + call_count[0] * 0.1})  # Different weights
        return MockRunResult(
            weights=w, metrics=pd.DataFrame({"CAGR": [0.1 + call_count[0] * 0.01]})
        )

    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    with patch.object(page.analysis_runner, "run_analysis", side_effect=fake_run) as run_mock:
        with patch("streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"):
            page.render_validation_page()
    assert run_mock.call_count == 2
    assert call_count[0] == 2
    assert any("SETTING IS WIRED" in msg for msg in stub.success_messages)
    assert any("`portfolio_CAGR`" in msg for msg in stub.markdowns)
    assert any("🔄" in msg for msg in stub.markdowns)
    last_test = stub.session_state["last_validation_test"]
    assert last_test["setting"] == "selection_count"
    assert last_test["baseline_value"] == 10
    assert last_test["test_value"] == 5
    assert last_test["changed"] is True


def test_render_validation_page_batch_section(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 733-737."""
    stub = CustomDummyStreamlit()
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    page.render_validation_page()
    assert any("Batch Validation" in msg for msg in stub.subheaders)


def test_render_validation_page_batch_test_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 738-760, 763-774, 776-780 (batch test execution)."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [
        False,
        True,
    ]  # First button is "Run Comparison Test", second is "Test All"
    stub.selectbox_returns = ["Selection"]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    mock_result = MockRunResult()
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    # Mock TESTABLE_SETTINGS
    mock_setting = MagicMock()
    mock_setting.key = "selection_count"
    mock_setting.baseline = 10
    mock_setting.test_values = [20]
    mock_setting.category = "Selection"
    mock_setting.label = "selection_count"
    mock_setting.expected_effect = "More funds"
    run_mock = MagicMock(return_value=mock_result)
    with patch("streamlit_app.developer_settings_validation.TESTABLE_SETTINGS", [mock_setting]):
        with patch.object(page.analysis_runner, "run_analysis", run_mock):
            with patch(
                "streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"
            ):
                page.render_validation_page()
    assert run_mock.call_count == 2
    assert stub.progress_calls[0] == (0.0, None)
    assert stub.progress_calls[-1] == (1.0, None)
    assert any(msg == "Testing: selection_count..." for msg in stub.write_calls)
    assert any(msg == "Complete!" for msg in stub.write_calls)


def test_render_validation_page_batch_results_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 800-806 (batch results display)."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [False, True]
    stub.selectbox_returns = ["Selection"]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    mock_result = MockRunResult()
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    mock_setting = MagicMock()
    mock_setting.key = "selection_count"
    mock_setting.baseline = 10
    mock_setting.test_values = [20]
    mock_setting.category = "Selection"
    mock_setting.label = "selection_count"
    mock_setting.expected_effect = "More funds"
    with patch("streamlit_app.developer_settings_validation.TESTABLE_SETTINGS", [mock_setting]):
        with patch.object(page.analysis_runner, "run_analysis", return_value=mock_result):
            with patch(
                "streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"
            ):
                page.render_validation_page()
    # Check that results dataframe was displayed
    assert any("Setting" in df.columns for df in stub.dataframes)


def test_render_validation_page_batch_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 808-826 (batch summary and download)."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [False, True]
    stub.selectbox_returns = ["Selection"]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    mock_result = MockRunResult()
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    mock_setting = MagicMock()
    mock_setting.key = "selection_count"
    mock_setting.baseline = 10
    mock_setting.test_values = [20]
    mock_setting.category = "Selection"
    mock_setting.label = "selection_count"
    mock_setting.expected_effect = "More funds"
    with patch("streamlit_app.developer_settings_validation.TESTABLE_SETTINGS", [mock_setting]):
        with patch.object(
            page.analysis_runner, "run_analysis", return_value=mock_result
        ) as run_mock:
            with patch(
                "streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"
            ):
                page.render_validation_page()
    assert run_mock.call_count == 2
    batch_df = next(
        df
        for df in stub.dataframes
        if list(df.columns) == ["Setting", "Key", "Baseline", "Test", "Status", "Expected Effect"]
    )
    assert batch_df.iloc[0].to_dict() == {
        "Setting": "selection_count",
        "Key": "selection_count",
        "Baseline": "10",
        "Test": "20",
        "Status": "❌ NOT WIRED",
        "Expected Effect": "More funds",
    }
    assert any(
        "**Summary for Selection:**" in msg and "Not Wired: 1" in msg for msg in stub.markdowns
    )
    assert len(stub.downloads) == 1
    assert stub.downloads[0]["label"] == "📥 Download Results CSV"
    assert stub.downloads[0]["mime"] == "text/csv"
    assert stub.downloads[0]["data"].startswith("Setting,Key,Baseline,Test,Status,Expected Effect")


def test_render_validation_page_batch_button(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 737-827."""
    stub = CustomDummyStreamlit()
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    page.render_validation_page()
    assert any("Test All" in label and "Settings" in label for label in stub.button_labels)


def test_apply_inclusion_approach_special_handling_random(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 753-755 (random inclusion approach)."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [False, True]
    stub.selectbox_returns = ["Selection"]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    mock_result = MockRunResult()
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    original_baseline = page.get_baseline_state

    def baseline_without_random_seed() -> dict[str, Any]:
        state = dict(original_baseline())
        state.pop("random_seed", None)
        return state

    monkeypatch.setattr(page, "get_baseline_state", baseline_without_random_seed)
    mock_setting = page.SettingDef(
        key="inclusion_approach",
        label="inclusion_approach",
        category="Selection",
        baseline="random",
        test_values=["threshold"],
        expected_effect="Random selection",
        economic_intuition="Random changes selection",
    )
    captured_states: list[dict[str, Any]] = []

    def capture_test_analysis(
        _returns: pd.DataFrame, model_state: dict[str, Any]
    ) -> dict[str, Any]:
        captured_states.append(dict(model_state))
        return {"status": "success", "result": mock_result}

    monkeypatch.setattr(page, "run_test_analysis", capture_test_analysis)
    with patch("streamlit_app.developer_settings_validation.TESTABLE_SETTINGS", [mock_setting]):
        page.render_validation_page()
    assert len(captured_states) == 2
    assert any(
        state.get("inclusion_approach") == "random" and state.get("random_seed") == 42
        for state in captured_states
    )


def test_apply_inclusion_approach_special_handling_top_pct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 756-757 (top_pct inclusion approach)."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [False, True]
    stub.selectbox_returns = ["Selection"]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    mock_result = MockRunResult()
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    original_baseline = page.get_baseline_state

    def baseline_without_rank_pct() -> dict[str, Any]:
        state = dict(original_baseline())
        state.pop("rank_pct", None)
        return state

    monkeypatch.setattr(page, "get_baseline_state", baseline_without_rank_pct)
    mock_setting = page.SettingDef(
        key="inclusion_approach",
        label="inclusion_approach",
        category="Selection",
        baseline="threshold",
        test_values=["top_pct"],
        expected_effect="Top percentile selection",
        economic_intuition="Top pct changes selection",
    )
    captured_states: list[dict[str, Any]] = []

    def capture_test_analysis(
        _returns: pd.DataFrame, model_state: dict[str, Any]
    ) -> dict[str, Any]:
        captured_states.append(dict(model_state))
        return {"status": "success", "result": mock_result}

    monkeypatch.setattr(page, "run_test_analysis", capture_test_analysis)
    with patch("streamlit_app.developer_settings_validation.TESTABLE_SETTINGS", [mock_setting]):
        page.render_validation_page()
    assert len(captured_states) == 2
    assert any(
        state.get("inclusion_approach") == "top_pct" and state.get("rank_pct") == 0.20
        for state in captured_states
    )
    assert any(msg == "Testing: inclusion_approach..." for msg in stub.write_calls)


def test_render_validation_page_full_validation_section(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 828-915."""
    stub = CustomDummyStreamlit()
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    page.render_validation_page()
    assert any("Full Validation" in msg for msg in stub.expander_labels)


def test_render_validation_page_full_validation_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 833-860 (full validation execution)."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [False, False, True]  # Skip first two, click third (Full Validation)
    stub.selectbox_returns = ["Selection"]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    mock_result = MockRunResult()
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    mock_setting = MagicMock()
    mock_setting.key = "selection_count"
    mock_setting.baseline = 10
    mock_setting.test_values = [20]
    mock_setting.category = "Selection"
    mock_setting.label = "selection_count"
    mock_setting.expected_effect = "More funds"
    run_mock = MagicMock(return_value=mock_result)
    with patch("streamlit_app.developer_settings_validation.TESTABLE_SETTINGS", [mock_setting]):
        with patch.object(page.analysis_runner, "run_analysis", run_mock):
            with patch(
                "streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"
            ):
                page.render_validation_page()
    assert run_mock.call_count == 2
    assert stub.progress_calls[0] == (0.0, None)
    assert stub.progress_calls[-1] == (1.0, None)
    assert any(msg == "Complete!" for msg in stub.write_calls)


def test_render_validation_page_full_validation_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 889-915 (full validation results display and download)."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [False, False, True]
    stub.selectbox_returns = ["Selection"]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    mock_result = MockRunResult()
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    mock_setting = MagicMock()
    mock_setting.key = "selection_count"
    mock_setting.baseline = 10
    mock_setting.test_values = [20]
    mock_setting.category = "Selection"
    mock_setting.label = "selection_count"
    mock_setting.expected_effect = "More funds"
    with patch("streamlit_app.developer_settings_validation.TESTABLE_SETTINGS", [mock_setting]):
        with patch.object(
            page.analysis_runner, "run_analysis", return_value=mock_result
        ) as run_mock:
            with patch(
                "streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"
            ):
                page.render_validation_page()
    assert run_mock.call_count == 2
    full_df = next(
        df
        for df in stub.dataframes
        if list(df.columns)
        == ["Category", "Setting", "Key", "Baseline", "Test", "Status", "Expected Effect"]
    )
    assert full_df.iloc[0].to_dict() == {
        "Category": "Selection",
        "Setting": "selection_count",
        "Key": "selection_count",
        "Baseline": "10",
        "Test": "20",
        "Status": "❌ NOT WIRED",
        "Expected Effect": "More funds",
    }
    assert any(msg == "Summary by Category" for msg in stub.subheaders)
    assert any("**Selection**: 0✅ / 1❌ / 0⚠️" in msg for msg in stub.error_messages)
    assert len(stub.downloads) == 1
    assert stub.downloads[0]["label"] == "📥 Download Full Results CSV"
    assert stub.downloads[0]["mime"] == "text/csv"
    assert stub.downloads[0]["data"].startswith(
        "Category,Setting,Key,Baseline,Test,Status,Expected Effect"
    )


def test_render_validation_page_full_validation_category_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 893-906 (full validation category summary)."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [False, False, True]
    stub.selectbox_returns = ["Selection"]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    mock_setting = MagicMock()
    mock_setting.key = "selection_count"
    mock_setting.baseline = 10
    mock_setting.test_values = [20]
    mock_setting.category = "Selection"
    mock_setting.label = "selection_count"
    mock_setting.expected_effect = "More funds"
    run_mock = MagicMock(side_effect=RuntimeError("analysis failed"))
    with patch("streamlit_app.developer_settings_validation.TESTABLE_SETTINGS", [mock_setting]):
        with patch.object(page.analysis_runner, "run_analysis", run_mock):
            with patch(
                "streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"
            ):
                page.render_validation_page()
    assert run_mock.call_count >= 1
    assert any("**Selection**: 0✅ / 0❌ / 1⚠️" in msg for msg in stub.warning_messages)


def test_render_validation_page_full_validation_inclusion_approach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 852-856 (inclusion_approach special handling in full validation)."""
    stub = CustomDummyStreamlit()
    stub.button_responses = [False, False, True]
    stub.selectbox_returns = ["Selection"]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    returns_df = pd.DataFrame(
        {"FundA": [0.01]}, index=pd.date_range("2024-01-31", periods=1, freq="ME")
    )
    _mock_demo_profile_and_state(monkeypatch, enabled=True, returns_data=returns_df)
    mock_result = MockRunResult()
    page = _reload_module(monkeypatch, "streamlit_app.developer_settings_validation")
    # Mock TESTABLE_SETTINGS with inclusion_approach
    mock_setting = MagicMock()
    mock_setting.key = "inclusion_approach"
    mock_setting.baseline = "threshold"
    mock_setting.test_values = ["random"]
    mock_setting.category = "Selection"
    mock_setting.label = "inclusion_approach"
    mock_setting.expected_effect = "Random selection"
    with patch("streamlit_app.developer_settings_validation.TESTABLE_SETTINGS", [mock_setting]):
        with patch.object(page.analysis_runner, "run_analysis", return_value=mock_result):
            with patch(
                "streamlit_app.developer_settings_validation.analysis_runner.AnalysisPayload"
            ):
                page.render_validation_page()


# =============================================================================
# developer_settings_validation.py - _should_auto_render (lines 922-923)
# =============================================================================


def test_should_auto_render_with_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers line 921."""
    from streamlit_app.developer_settings_validation import _should_auto_render

    scriptrunner = MagicMock()
    scriptrunner.get_script_run_ctx.return_value = object()
    monkeypatch.setitem(sys.modules, "streamlit.runtime.scriptrunner", scriptrunner)
    assert _should_auto_render() is True


def test_should_auto_render_without_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers line 922."""
    from streamlit_app.developer_settings_validation import _should_auto_render

    scriptrunner = MagicMock()
    scriptrunner.get_script_run_ctx.return_value = None
    monkeypatch.setitem(sys.modules, "streamlit.runtime.scriptrunner", scriptrunner)
    assert _should_auto_render() is False


def test_should_auto_render_with_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers line 923."""
    from streamlit_app.developer_settings_validation import _should_auto_render

    monkeypatch.delitem(sys.modules, "streamlit.runtime.scriptrunner", raising=False)
    assert _should_auto_render() is False


# =============================================================================
# comparison_llm.py - _cache_bucket (lines 57-61)
# =============================================================================


def test_cache_bucket_creates_new_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 57-61."""
    import streamlit as st

    monkeypatch.setattr(st, "session_state", {})
    from streamlit_app.components.comparison_llm import _cache_bucket

    cache = _cache_bucket()
    assert isinstance(cache, dict)
    assert "comparison_llm_cache" in st.session_state


# =============================================================================
# comparison_llm.py - _render_analysis_output (lines 65-80)
# =============================================================================


def test_render_analysis_output_with_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 65-80."""
    from streamlit_app.components.comparison_llm import _render_analysis_output

    with patch(
        "trend_analysis.export.summary_frame_from_result", return_value=pd.DataFrame({"a": [1]})
    ):
        result = _render_analysis_output({}, "A")
        assert result == "Simulation A summary:\n a\n 1"


def test_render_analysis_output_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 65-80."""
    from streamlit_app.components.comparison_llm import _render_analysis_output

    with patch("trend_analysis.export.summary_frame_from_result", side_effect=Exception()):
        result = _render_analysis_output({"key": "val"}, "A")
        assert "Summary table unavailable" in result
        assert "Available sections: key" in result


# =============================================================================
# comparison_llm.py - _prefix_metric_entries (lines 84-93)
# =============================================================================


def test_prefix_metric_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 84-93."""
    from streamlit_app.components.comparison_llm import _prefix_metric_entries
    from trend_analysis.llm.result_metrics import MetricEntry

    entries = [MetricEntry(path="a.b", value=1.0, source="test")]
    prefixed = _prefix_metric_entries(entries, "A")
    assert prefixed[0].path == "A.a.b"
    assert prefixed[0].source == "A:test"


# =============================================================================
# comparison_llm.py - _build_comparison_chain (lines 104, 111-112)
# =============================================================================


def test_build_comparison_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 104, 111-112."""
    from trend_analysis.llm import build_comparison_prompt

    from streamlit_app.components.comparison_llm import _build_comparison_chain

    mock_config = MagicMock()
    mock_config.model = "test-model"
    mock_llm = MagicMock()
    with patch(
        "streamlit_app.components.comparison_llm.resolve_llm_provider_config",
        return_value=mock_config,
    ) as mock_resolve:
        with patch(
            "streamlit_app.components.comparison_llm.create_llm", return_value=mock_llm
        ) as mock_create_llm:
            with patch(
                "streamlit_app.components.comparison_llm.ResultSummaryChain.from_env",
                return_value=MagicMock(),
            ) as mock_from_env:
                chain = _build_comparison_chain(
                    "openai",
                    api_key="key",
                    model="model",
                    base_url="http://url",
                    organization="org",
                )
                mock_resolve.assert_called_once_with(
                    "openai",
                    api_key="key",
                    model="model",
                    base_url="http://url",
                    organization="org",
                )
                mock_create_llm.assert_called_once_with(mock_config)
                mock_from_env.assert_called_once_with(
                    llm=mock_llm,
                    prompt_builder=build_comparison_prompt,
                    model="test-model",
                )
                assert chain is mock_from_env.return_value


# =============================================================================
# comparison_llm.py - generate_comparison_explanation (lines 133-185)
# =============================================================================


def test_generate_comparison_explanation_no_metrics() -> None:
    """Covers lines 133-144."""
    from streamlit_app.components.comparison_llm import (
        ComparisonExplanationResult,
        generate_comparison_explanation,
    )

    result = generate_comparison_explanation(
        {}, {}, config_diff="diff", questions=None, label_a="A", label_b="B"
    )
    assert isinstance(result, ComparisonExplanationResult)
    assert result.text == "No metrics were detected in either simulation output."
    assert result.trace_url is None
    assert result.claim_issues == []
    assert result.metric_count == 0


def test_generate_comparison_explanation_with_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 146-150, 152-153, 155, 161, 163, 170, 180, 185."""
    from streamlit_app.components.comparison_llm import generate_comparison_explanation

    details_a = {"in_sample_stats": {"pf": {"cagr": 0.15, "sharpe": 1.5}}}
    details_b = {"in_sample_stats": {"pf": {"cagr": 0.10, "sharpe": 1.2}}}
    mock_chain = MagicMock()
    mock_chain.run.return_value = MagicMock(text="Analysis complete", trace_url="http://trace")
    with patch(
        "streamlit_app.components.comparison_llm._build_comparison_chain",
        return_value=mock_chain,
    ):
        result = generate_comparison_explanation(
            details_a,
            details_b,
            config_diff="diff",
            questions="Test question",
            label_a="A",
            label_b="B",
        )
    assert result.metric_count == 4
    assert result.trace_url == "http://trace"
    assert result.text.startswith("Analysis complete")
    run_kwargs = mock_chain.run.call_args.kwargs
    assert run_kwargs["questions"] == "Test question"
    assert len(run_kwargs["metric_entries"]) == 4
    entry_paths = [entry.path for entry in run_kwargs["metric_entries"]]
    assert sum(path.startswith("A.") for path in entry_paths) == 2
    assert sum(path.startswith("B.") for path in entry_paths) == 2
    assert all(path.startswith("A.") for path in entry_paths[:2])
    assert all(path.startswith("B.") for path in entry_paths[2:])
    assert {entry.source for entry in run_kwargs["metric_entries"]} == {
        "A:in_sample_stats",
        "B:in_sample_stats",
    }
    catalog_a, catalog_b = run_kwargs["metric_catalog"]
    assert catalog_a.startswith("Metrics A (A):")
    assert catalog_b.startswith("Metrics B (B):")
    assert "A.in_sample_stats.pf.cagr" in catalog_a
    assert "B.in_sample_stats.pf.cagr" in catalog_b


def test_generate_comparison_explanation_no_metrics_b_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins line 153: empty B catalog uses the no-metrics placeholder."""
    from streamlit_app.components.comparison_llm import generate_comparison_explanation

    details_a = {"in_sample_stats": {"pf": {"cagr": 0.15}}}
    details_b = {}
    mock_chain = MagicMock()
    mock_chain.run.return_value = MagicMock(text="Analysis", trace_url=None)
    with patch(
        "streamlit_app.components.comparison_llm._build_comparison_chain",
        return_value=mock_chain,
    ):
        result = generate_comparison_explanation(
            details_a,
            details_b,
            config_diff="",
            questions=None,
            label_a="A",
            label_b="B",
        )
    assert result.metric_count == 1
    catalog_a, catalog_b = mock_chain.run.call_args.kwargs["metric_catalog"]
    assert catalog_a == ("Metrics A (A):\nA.in_sample_stats.pf.cagr: 0.15 [from A:in_sample_stats]")
    assert catalog_b == "Metrics B (B):\nNo metrics available."


def test_generate_comparison_explanation_no_metrics_a_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers line 152."""
    from streamlit_app.components.comparison_llm import generate_comparison_explanation

    details_a = {}
    details_b = {"in_sample_stats": {"pf": {"cagr": 0.10}}}
    mock_chain = MagicMock()
    mock_chain.run.return_value = MagicMock(text="Analysis", trace_url=None)
    with patch(
        "streamlit_app.components.comparison_llm._build_comparison_chain",
        return_value=mock_chain,
    ):
        result = generate_comparison_explanation(
            details_a,
            details_b,
            config_diff="",
            questions=None,
            label_a="A",
            label_b="B",
        )
    assert result.metric_count == 1
    assert result.trace_url is None
    assert result.text.startswith("Analysis")
    catalog_a, catalog_b = mock_chain.run.call_args.kwargs["metric_catalog"]
    assert catalog_a == "Metrics A (A):\nNo metrics available."
    assert catalog_b == ("Metrics B (B):\nB.in_sample_stats.pf.cagr: 0.1 [from B:in_sample_stats]")


# =============================================================================
# comparison_llm.py - render_comparison_llm (lines 211-384)
# =============================================================================


def test_render_comparison_llm_zone_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 203-210."""
    stub = CustomDummyStreamlit()
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    with patch("streamlit_app.components.comparison_llm.llm_zone_disabled", return_value=True):
        from streamlit_app.components.comparison_llm import render_comparison_llm

        render_comparison_llm(
            result_a=StubResult(details={}),
            result_b=StubResult(details={}),
            label_a="A",
            label_b="B",
            config_diff="",
            run_key="k",
        )
        assert any("disabled" in msg.lower() for msg in stub.info_messages)


def test_render_comparison_llm_missing_details(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 216-220."""
    stub = CustomDummyStreamlit()
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    with patch("streamlit_app.components.comparison_llm.llm_zone_disabled", return_value=False):
        from streamlit_app.components.comparison_llm import render_comparison_llm

        ra = MagicMock(spec=["other_attr"])
        rb = MagicMock(spec=["other_attr"])
        render_comparison_llm(
            result_a=ra, result_b=rb, label_a="A", label_b="B", config_diff="", run_key="k"
        )
        assert any("unavailable" in msg.lower() for msg in stub.info_messages)


def test_render_comparison_llm_questions_textarea(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 222-228 (questions textarea)."""
    stub = CustomDummyStreamlit()
    stub.session_state = {}
    _enable_widget_call_recording(stub)
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    with patch("streamlit_app.components.comparison_llm.llm_zone_disabled", return_value=False):
        with patch(
            "streamlit_app.components.comparison_llm.browser_llm_runtime", return_value=False
        ):
            with patch(
                "streamlit_app.components.comparison_llm.llm_provider_options",
                return_value=["openai"],
            ):
                with patch(
                    "streamlit_app.components.comparison_llm.default_api_key", return_value=None
                ):
                    with patch(
                        "streamlit_app.components.comparison_llm.sanitize_api_key",
                        return_value=True,
                    ):
                        with patch(
                            "streamlit_app.components.comparison_llm.resolve_api_key_input",
                            return_value=None,
                        ):
                            from streamlit_app.components.comparison_llm import (
                                render_comparison_llm,
                            )

                            render_comparison_llm(
                                result_a=StubResult(details={}),
                                result_b=StubResult(details={}),
                                label_a="A",
                                label_b="B",
                                config_diff="",
                                run_key="k",
                            )
                            assert (
                                "Questions (optional)",
                                "comparison_llm_questions::k",
                            ) in [(label, key) for label, key, _ in stub.text_input_calls]


def test_render_comparison_llm_provider_selectbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 247-248 (provider selectbox index calculation)."""
    stub = CustomDummyStreamlit()
    stub.session_state = {}
    monkeypatch.setenv("TREND_LLM_PROVIDER", "anthropic")
    _enable_widget_call_recording(stub)
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    with patch("streamlit_app.components.comparison_llm.llm_zone_disabled", return_value=False):
        with patch(
            "streamlit_app.components.comparison_llm.browser_llm_runtime", return_value=False
        ):
            with patch(
                "streamlit_app.components.comparison_llm.llm_provider_options",
                return_value=["openai", "anthropic"],
            ):
                with patch(
                    "streamlit_app.components.comparison_llm.default_api_key", return_value=None
                ):
                    with patch(
                        "streamlit_app.components.comparison_llm.sanitize_api_key",
                        return_value=True,
                    ):
                        with patch(
                            "streamlit_app.components.comparison_llm.resolve_api_key_input",
                            return_value=None,
                        ):
                            from streamlit_app.components.comparison_llm import (
                                render_comparison_llm,
                            )

                            render_comparison_llm(
                                result_a=StubResult(details={}),
                                result_b=StubResult(details={}),
                                label_a="A",
                                label_b="B",
                                config_diff="",
                                run_key="k",
                            )
                            assert stub.selectbox_index_calls == [1]


def test_render_comparison_llm_api_key_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers line 261 (API key fallback from env) and line 314 (default key on click)."""
    stub = CustomDummyStreamlit()
    stub.session_state = {}
    stub.button_responses = [True]

    def recording_download(*args: Any, **kwargs: Any) -> None:
        stub.downloads.append({"label": args[0] if args else None, **kwargs})

    stub.download_button = recording_download
    _enable_widget_call_recording(stub)
    _install_streamlit_stub(monkeypatch, stub, active_context=True)

    from streamlit_app.components.comparison_llm import ComparisonExplanationResult

    explanation = ComparisonExplanationResult(
        text="ok",
        trace_url=None,
        claim_issues=[],
        metric_count=0,
        created_at="2024-01-01T00:00:00Z",
    )
    with patch("streamlit_app.components.comparison_llm.llm_zone_disabled", return_value=False):
        with patch(
            "streamlit_app.components.comparison_llm.browser_llm_runtime", return_value=False
        ):
            with patch(
                "streamlit_app.components.comparison_llm.llm_provider_options",
                return_value=["openai"],
            ):
                with patch(
                    "streamlit_app.components.comparison_llm.default_api_key",
                    return_value="env_key",
                ) as default_key_mock:
                    with patch(
                        "streamlit_app.components.comparison_llm.sanitize_api_key",
                        return_value=True,
                    ):
                        with patch(
                            "streamlit_app.components.comparison_llm.resolve_api_key_input",
                            return_value=None,
                        ):
                            with patch(
                                "streamlit_app.components.comparison_llm.generate_comparison_explanation",
                                return_value=explanation,
                            ) as generate_mock:
                                from streamlit_app.components.comparison_llm import (
                                    render_comparison_llm,
                                )

                                render_comparison_llm(
                                    result_a=StubResult(details={"in_sample_stats": {}}),
                                    result_b=StubResult(details={"in_sample_stats": {}}),
                                    label_a="A",
                                    label_b="B",
                                    config_diff="",
                                    run_key="k",
                                )
                            assert stub.session_state["comparison_llm_api_key::k"] == "env_key"
                            assert default_key_mock.call_args_list[-1].args == ("openai",)
                            generate_mock.assert_called_once()
                            assert generate_mock.call_args.kwargs["api_key"] == "env_key"


def test_render_comparison_llm_api_key_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers line 276 (API key configured caption)."""
    stub = CustomDummyStreamlit()
    stub.session_state = {"comparison_llm_api_key::k": "test_key"}
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    with patch("streamlit_app.components.comparison_llm.llm_zone_disabled", return_value=False):
        with patch(
            "streamlit_app.components.comparison_llm.browser_llm_runtime", return_value=False
        ):
            with patch(
                "streamlit_app.components.comparison_llm.llm_provider_options",
                return_value=["openai"],
            ):
                with patch(
                    "streamlit_app.components.comparison_llm.default_api_key", return_value=None
                ):
                    with patch(
                        "streamlit_app.components.comparison_llm.sanitize_api_key",
                        return_value=True,
                    ):
                        with patch(
                            "streamlit_app.components.comparison_llm.resolve_api_key_input",
                            return_value="test_key",
                        ):
                            from streamlit_app.components.comparison_llm import (
                                render_comparison_llm,
                            )

                            render_comparison_llm(
                                result_a=StubResult(details={}),
                                result_b=StubResult(details={}),
                                label_a="A",
                                label_b="B",
                                config_diff="",
                                run_key="k",
                            )
                            assert any(
                                "✓ API key configured" in msg for msg in stub.caption_messages
                            )


def test_render_comparison_llm_button_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers line 304-305 (button key generation)."""
    stub = CustomDummyStreamlit()
    stub.session_state = {}
    _enable_widget_call_recording(stub)
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    with patch("streamlit_app.components.comparison_llm.llm_zone_disabled", return_value=False):
        with patch(
            "streamlit_app.components.comparison_llm.browser_llm_runtime", return_value=False
        ):
            with patch(
                "streamlit_app.components.comparison_llm.llm_provider_options",
                return_value=["openai"],
            ):
                with patch(
                    "streamlit_app.components.comparison_llm.default_api_key", return_value=None
                ):
                    with patch(
                        "streamlit_app.components.comparison_llm.sanitize_api_key",
                        return_value=True,
                    ):
                        with patch(
                            "streamlit_app.components.comparison_llm.resolve_api_key_input",
                            return_value=None,
                        ):
                            from streamlit_app.components.comparison_llm import (
                                render_comparison_llm,
                            )

                            render_comparison_llm(
                                result_a=StubResult(details={}),
                                result_b=StubResult(details={}),
                                label_a="A",
                                label_b="B",
                                config_diff="",
                                run_key="test_key_12345",
                            )
                            assert stub.button_key_calls == ["btn_compare_llm_45af89b510a3"]
                            assert stub.button_labels == ["Explain comparison"]


def test_render_comparison_llm_exception_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers lines 330-333 (exception handling in generate_comparison_explanation)."""
    stub = CustomDummyStreamlit()
    stub.session_state = {}
    stub.button_responses = [True]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    with patch("streamlit_app.components.comparison_llm.llm_zone_disabled", return_value=False):
        with patch(
            "streamlit_app.components.comparison_llm.browser_llm_runtime", return_value=False
        ):
            with patch(
                "streamlit_app.components.comparison_llm.llm_provider_options",
                return_value=["openai"],
            ):
                with patch(
                    "streamlit_app.components.comparison_llm.default_api_key", return_value=None
                ):
                    with patch(
                        "streamlit_app.components.comparison_llm.sanitize_api_key",
                        return_value=True,
                    ):
                        with patch(
                            "streamlit_app.components.comparison_llm.resolve_api_key_input",
                            return_value="key",
                        ):
                            with patch(
                                "streamlit_app.components.comparison_llm.generate_comparison_explanation",
                                side_effect=Exception("Test error"),
                            ):
                                from streamlit_app.components.comparison_llm import (
                                    render_comparison_llm,
                                )

                                render_comparison_llm(
                                    result_a=StubResult(details={}),
                                    result_b=StubResult(details={}),
                                    label_a="A",
                                    label_b="B",
                                    config_diff="",
                                    run_key="k",
                                )
                                assert any(
                                    "could not generate" in msg.lower()
                                    for msg in stub.error_messages
                                )


def test_render_comparison_llm_llm_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 230-302."""
    stub = CustomDummyStreamlit()
    stub.session_state = {}
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    with patch("streamlit_app.components.comparison_llm.llm_zone_disabled", return_value=False):
        with patch(
            "streamlit_app.components.comparison_llm.browser_llm_runtime", return_value=False
        ):
            with patch(
                "streamlit_app.components.comparison_llm.llm_provider_options",
                return_value=["openai"],
            ):
                with patch(
                    "streamlit_app.components.comparison_llm.default_api_key", return_value=None
                ):
                    with patch(
                        "streamlit_app.components.comparison_llm.sanitize_api_key",
                        return_value=True,
                    ):
                        with patch(
                            "streamlit_app.components.comparison_llm.resolve_api_key_input",
                            return_value=None,
                        ):
                            from streamlit_app.components.comparison_llm import (
                                render_comparison_llm,
                            )

                            render_comparison_llm(
                                result_a=StubResult(details={}),
                                result_b=StubResult(details={}),
                                label_a="A",
                                label_b="B",
                                config_diff="",
                                run_key="k",
                            )
                            assert any("LLM Settings" in msg for msg in stub.markdowns)


def test_render_comparison_llm_button_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers lines 304-333."""
    stub = CustomDummyStreamlit()
    stub.session_state = {}
    stub.button_responses = [True]
    _install_streamlit_stub(monkeypatch, stub, active_context=True)
    with patch("streamlit_app.components.comparison_llm.llm_zone_disabled", return_value=False):
        with patch(
            "streamlit_app.components.comparison_llm.browser_llm_runtime", return_value=False
        ):
            with patch(
                "streamlit_app.components.comparison_llm.llm_provider_options",
                return_value=["openai"],
            ):
                with patch(
                    "streamlit_app.components.comparison_llm.default_api_key", return_value=None
                ):
                    with patch(
                        "streamlit_app.components.comparison_llm.sanitize_api_key",
                        return_value=True,
                    ):
                        with patch(
                            "streamlit_app.components.comparison_llm.resolve_api_key_input",
                            return_value="key",
                        ):
                            with patch(
                                "streamlit_app.components.comparison_llm.generate_comparison_explanation",
                                return_value=None,
                            ):
                                from streamlit_app.components.comparison_llm import (
                                    render_comparison_llm,
                                )

                                render_comparison_llm(
                                    result_a=StubResult(details={}),
                                    result_b=StubResult(details={}),
                                    label_a="A",
                                    label_b="B",
                                    config_diff="",
                                    run_key="k",
                                )
                                assert any(
                                    "Click Explain comparison" in msg for msg in stub.info_messages
                                )
