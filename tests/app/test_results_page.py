import importlib
import sys
import warnings
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from trend_analysis.diagnostics import PipelineReasonCode, pipeline_failure


class _ContextManager:
    def __enter__(self):
        return None

    def __exit__(self, *_exc):
        return False


class DummyStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.button_responses: list[bool] = []
        self.error_messages: list[str] = []
        self.caption_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.info_messages: list[str] = []
        self.success_messages: list[str] = []
        self.markdown_messages: list[str] = []
        self.subheaders: list[str] = []
        self.altair_payloads: list[object] = []
        self.dataframes: list[pd.DataFrame] = []
        self.metrics: list[tuple[str, object]] = []
        self.checkbox_labels: list[str] = []
        self.button_labels: list[str] = []
        self.tab_groups: list[list[str]] = []

    # Basic UI primitives -------------------------------------------------
    def title(self, _text: str) -> None:  # pragma: no cover - trivial
        return None

    def header(self, _text: str) -> None:  # pragma: no cover - trivial
        return None

    def markdown(self, text: str, *_args, **_kwargs) -> None:
        self.markdown_messages.append(text)

    def button(self, label: str, *_args, **_kwargs) -> bool:
        self.button_labels.append(label)
        if self.button_responses:
            return self.button_responses.pop(0)
        return False

    def spinner(self, *_args, **_kwargs) -> _ContextManager:
        return _ContextManager()

    def columns(self, count) -> list["ColumnContext"]:
        # count can be an int or a list of column weights
        n = count if isinstance(count, int) else len(count)
        return [ColumnContext(self) for _ in range(n)]

    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def success(self, text: str) -> None:
        self.success_messages.append(text)

    def divider(self) -> None:  # pragma: no cover - trivial
        return None

    def altair_chart(self, payload, **_kwargs) -> None:
        self.altair_payloads.append(payload)

    def dataframe(self, df: pd.DataFrame, **_: object) -> None:
        self.dataframes.append(df)

    def error(self, message: str) -> None:
        self.error_messages.append(message)

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)

    def info(self, message: str) -> None:
        self.info_messages.append(message)

    def caption(self, message: str) -> None:
        self.caption_messages.append(message)

    def tabs(self, labels: list[str]):
        self.tab_groups.append(list(labels))
        return [ColumnContext(self) for _ in labels]

    def metric(self, label: str, value: object) -> None:
        self.metrics.append((label, value))

    def expander(self, *_args, **_kwargs) -> "ColumnContext":
        return ColumnContext(self)

    def checkbox(
        self, label: str, value: bool = False, key: str | None = None, **_kwargs
    ) -> bool:
        self.checkbox_labels.append(label)
        if key is None:
            return bool(value)
        if key not in self.session_state:
            self.session_state[key] = value
        return bool(self.session_state.get(key))

    def cache_data(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


class ColumnContext:
    def __init__(self, stub: DummyStreamlit) -> None:
        self._stub = stub

    def __enter__(self) -> DummyStreamlit:
        return self._stub

    def __exit__(self, *_exc) -> bool:
        return False


@pytest.fixture()
def results_page(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, DummyStreamlit]:
    monkeypatch.setenv("TREND_DEMO_PROFILE", "public_llm_demo")
    stub = DummyStreamlit()

    module = ModuleType("streamlit")

    def bind(name: str):
        def wrapper(*args, **kwargs):
            return getattr(stub, name)(*args, **kwargs)

        return wrapper

    for attr in [
        "title",
        "header",
        "markdown",
        "button",
        "spinner",
        "columns",
        "subheader",
        "success",
        "divider",
        "altair_chart",
        "dataframe",
        "error",
        "warning",
        "info",
        "caption",
        "cache_data",
        "tabs",
        "metric",
        "expander",
        "checkbox",
    ]:
        setattr(module, attr, bind(attr))

    def __getattr__(name: str):  # pragma: no cover - fallback
        def _noop(*_args, **_kwargs):
            return None

        return _noop

    module.__getattr__ = __getattr__

    module.session_state = stub.session_state

    altair_stub = ModuleType("altair")

    def _altair_noop(*_args, **_kwargs):
        return altair_stub

    altair_stub.Chart = _altair_noop
    altair_stub.X = _altair_noop
    altair_stub.Y = _altair_noop
    altair_stub.Axis = _altair_noop
    altair_stub.Tooltip = _altair_noop
    altair_stub.Color = _altair_noop
    altair_stub.Scale = _altair_noop

    monkeypatch.setitem(sys.modules, "streamlit", module)
    monkeypatch.setitem(sys.modules, "altair", altair_stub)

    from streamlit_app import state as app_state

    monkeypatch.setattr(app_state, "st", module)
    monkeypatch.setattr(app_state, "initialize_session_state", lambda: None)

    page = importlib.reload(importlib.import_module("streamlit_app.pages.3_Results"))

    return page, stub


def _sample_returns() -> pd.DataFrame:
    data = {
        "FundA": [0.01, -0.005, 0.012],
        "FundB": [0.008, 0.007, -0.002],
    }
    index = pd.date_range("2023-01-31", periods=3, freq="ME")
    return pd.DataFrame(data, index=index)


def test_results_page_recomputes_when_benchmark_changes(
    monkeypatch: pytest.MonkeyPatch, results_page
) -> None:
    page, stub = results_page
    returns = _sample_returns()

    stub.session_state.update(
        {
            "model_state": {
                "trend_spec": {"window": 63, "lag": 1},
                "metric_weights": {"sharpe": 1.0},
            },
            "selected_benchmark": "BenchA",
            "data_fingerprint": "abc123",
            "returns_df": returns,
            "schema_meta": {},
            "upload_status": "success",
        }
    )

    run_calls: list[str | None] = []

    def fake_run(
        df: pd.DataFrame,
        model_state: dict,
        benchmark: str | None,
        **_kwargs,
    ):
        run_calls.append(benchmark)
        return SimpleNamespace(
            metrics=pd.DataFrame({"Sharpe": [1.23]}),
            details={
                "portfolio_equal_weight_combined": df["FundA"],
                "risk_diagnostics": {
                    "turnover": pd.Series([0.1, 0.2], index=returns.index[:2]),
                    "final_weights": pd.Series({"FundA": 0.6, "FundB": 0.4}),
                },
            },
            fallback_info=None,
        )

    for chart in [
        "equity_chart",
        "drawdown_chart",
        "rolling_sharpe_chart",
        "turnover_chart",
        "exposure_chart",
    ]:
        monkeypatch.setattr(
            getattr(page, "charts"), chart, lambda *_args, chart_name=chart: chart_name
        )

    monkeypatch.setattr(page.analysis_runner, "run_analysis", fake_run)

    stub.button_responses = [True]
    stub.error_messages.clear()
    stub.caption_messages.clear()

    page.render_results_page()
    assert run_calls == ["BenchA"]
    assert stub.session_state.get("analysis_result_key")
    assert "Generate Summary" in stub.checkbox_labels

    # Changing benchmark should invalidate cached result and trigger a new run.
    stub.session_state["selected_benchmark"] = "BenchB"
    page.render_results_page()

    assert run_calls == ["BenchA", "BenchB"]


def test_weight_pivot_formatter_avoids_applymap_futurewarning(results_page) -> None:
    page, _stub = results_page
    pivot = pd.DataFrame({"FundA": [0.1234], "FundB": [0.0]}, index=["2024-01-31"])

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        formatted = page._format_weights_pivot(pivot)

    assert formatted.loc["2024-01-31", "FundA"] == "12.3%"
    assert formatted.loc["2024-01-31", "FundB"] == "0.0%"


def test_results_page_includes_regime_proxy_in_analysis_input(
    monkeypatch: pytest.MonkeyPatch, results_page
) -> None:
    page, stub = results_page
    returns = pd.DataFrame(
        {
            "FundA": [0.01, 0.02, 0.0],
            "FundB": [0.03, -0.01, 0.01],
            "SPX": [-0.02, -0.01, 0.015],
            "RF": [0.001, 0.001, 0.001],
        },
        index=pd.date_range("2023-01-31", periods=3, freq="ME"),
    )

    stub.session_state.update(
        {
            "model_state": {
                "trend_spec": {"window": 63, "lag": 1},
                "metric_weights": {"sharpe": 1.0},
                "regime_enabled": True,
                "regime_proxy": "SPX",
            },
            "selected_benchmark": None,
            "selected_risk_free": "RF",
            "data_fingerprint": "abc123",
            "returns_df": returns,
            "schema_meta": {},
            "upload_status": "success",
            "analysis_fund_columns": ["FundA", "FundB"],
            "fund_columns": list(returns.columns),
        }
    )

    seen_columns: list[str] = []

    def fake_run(
        df: pd.DataFrame,
        model_state: dict,
        benchmark: str | None,
        **_kwargs,
    ):
        seen_columns.extend(list(df.columns))
        return SimpleNamespace(
            metrics=pd.DataFrame({"Sharpe": [1.23]}),
            details={
                "portfolio_equal_weight_combined": df["FundA"],
                "risk_diagnostics": {
                    "turnover": pd.Series([0.1, 0.2], index=returns.index[:2]),
                    "final_weights": pd.Series({"FundA": 0.6, "FundB": 0.4}),
                },
            },
            fallback_info=None,
        )

    for chart in [
        "equity_chart",
        "drawdown_chart",
        "rolling_sharpe_chart",
        "turnover_chart",
        "exposure_chart",
    ]:
        monkeypatch.setattr(
            getattr(page, "charts"), chart, lambda *_args, chart_name=chart: chart_name
        )

    monkeypatch.setattr(page.analysis_runner, "run_analysis", fake_run)

    page.render_results_page()

    assert "SPX" in seen_columns
    assert "RF" in seen_columns


def test_results_page_reports_plain_language_error(
    monkeypatch: pytest.MonkeyPatch, results_page
) -> None:
    page, stub = results_page
    returns = _sample_returns()

    stub.session_state.update(
        {
            "model_state": {"trend_spec": {"window": 63, "lag": 1}},
            "selected_benchmark": None,
            "data_fingerprint": "xyz789",
            "returns_df": returns,
            "schema_meta": {},
            "upload_status": "success",
        }
    )

    def raise_error(*_args, **_kwargs):
        raise ValueError("No returns available after filtering")

    monkeypatch.setattr(page.analysis_runner, "run_analysis", raise_error)

    stub.button_responses = [True]
    stub.error_messages.clear()
    stub.caption_messages.clear()

    page.render_results_page()

    assert stub.error_messages == [
        "We couldn't run the analysis with the current data or settings. Please review the configuration and try again."
    ]
    assert "No returns available after filtering" in stub.caption_messages
    assert stub.session_state.get("analysis_result") is None
    assert stub.session_state.get("analysis_error") == {
        "message": "We couldn't run the analysis with the current data or settings. Please review the configuration and try again.",
        "detail": "No returns available after filtering",
    }


def test_results_error_summary_hides_reason_code(results_page) -> None:
    page, _stub = results_page
    result = pipeline_failure(PipelineReasonCode.NO_FUNDS_SELECTED)

    summary, detail = page._diagnostic_message(result)

    assert summary == "Analysis did not produce results."
    assert "PIPELINE_NO_FUNDS_SELECTED" not in summary
    assert detail is not None
    assert "No investable funds satisfy the selection filters." in detail
    assert "Try another preset, or adjust Customize Demo Settings." in detail


def test_results_failed_result_is_not_marked_complete(results_page) -> None:
    page, stub = results_page
    returns = _sample_returns()
    result = pipeline_failure(PipelineReasonCode.NO_FUNDS_SELECTED)

    stub.session_state.update(
        {
            "model_state": {
                "trend_spec": {"window": 63, "lag": 1},
                "metric_weights": {"sharpe": 1.0},
            },
            "analysis_fund_columns": ["FundA", "FundB"],
            "fund_columns": ["FundA", "FundB"],
            "selected_benchmark": None,
            "data_fingerprint": "abc123",
            "returns_df": returns,
            "schema_meta": {},
            "upload_status": "success",
            "demo_preset": "Balanced",
        }
    )
    stub.session_state["analysis_result"] = result
    stub.session_state["analysis_result_key"] = page._current_run_key(
        stub.session_state["model_state"], stub.session_state["selected_benchmark"]
    )

    page.render_results_page()

    assert not stub.success_messages
    assert stub.error_messages == ["Analysis did not produce results."]
    assert any(
        "No investable funds satisfy the selection filters." in msg
        for msg in stub.caption_messages
    )
    assert "Run analysis" in stub.button_labels
    assert "Re-run with custom settings" not in stub.button_labels


def test_results_empty_diagnostic_summary_is_not_marked_complete(
    monkeypatch: pytest.MonkeyPatch, results_page
) -> None:
    page, stub = results_page
    returns = _sample_returns()
    result = pipeline_failure(PipelineReasonCode.NO_FUNDS_SELECTED)

    stub.session_state.update(
        {
            "model_state": {
                "trend_spec": {"window": 63, "lag": 1},
                "metric_weights": {"sharpe": 1.0},
            },
            "analysis_fund_columns": ["FundA", "FundB"],
            "fund_columns": ["FundA", "FundB"],
            "selected_benchmark": None,
            "data_fingerprint": "abc123",
            "returns_df": returns,
            "schema_meta": {},
            "upload_status": "success",
            "demo_preset": "Balanced",
        }
    )
    stub.session_state["analysis_result"] = result
    stub.session_state["analysis_result_key"] = page._current_run_key(
        stub.session_state["model_state"], stub.session_state["selected_benchmark"]
    )
    monkeypatch.setattr(
        page, "_diagnostic_message", lambda _result: ("", "empty detail")
    )

    page.render_results_page()

    assert not stub.success_messages
    assert stub.error_messages == [""]
    assert "empty detail" in stub.caption_messages
    assert "Run analysis" in stub.button_labels
    assert "Re-run with custom settings" not in stub.button_labels


def test_results_page_renders_explain_results(
    monkeypatch: pytest.MonkeyPatch, results_page
) -> None:
    page, stub = results_page
    returns = _sample_returns()

    stub.session_state.update(
        {
            "model_state": {
                "trend_spec": {"window": 63, "lag": 1},
                "metric_weights": {"sharpe": 1.0},
            },
            "selected_benchmark": "BenchA",
            "selected_risk_free": None,
            "data_fingerprint": "abc123",
            "returns_df": returns,
            "schema_meta": {},
            "upload_status": "success",
            "analysis_fund_columns": ["FundA", "FundB"],
            "fund_columns": ["FundA", "FundB"],
        }
    )

    called: dict[str, object] = {}

    def fake_render(result, *, run_key: str, provider: str | None = None) -> None:
        called["result"] = result
        called["run_key"] = run_key
        called["provider"] = provider

    monkeypatch.setattr(page.explain_results, "render_explain_results", fake_render)

    def fake_run(
        df: pd.DataFrame,
        model_state: dict,
        benchmark: str | None,
        **_kwargs,
    ):
        return SimpleNamespace(
            metrics=pd.DataFrame({"Sharpe": [1.23]}),
            details={
                "portfolio_equal_weight_combined": df["FundA"],
                "risk_diagnostics": {
                    "turnover": pd.Series([0.1, 0.2], index=returns.index[:2]),
                    "final_weights": pd.Series({"FundA": 0.6, "FundB": 0.4}),
                },
            },
            fallback_info=None,
        )

    for chart in [
        "equity_chart",
        "drawdown_chart",
        "rolling_sharpe_chart",
        "turnover_chart",
        "exposure_chart",
    ]:
        monkeypatch.setattr(
            getattr(page, "charts"), chart, lambda *_args, chart_name=chart: chart_name
        )

    monkeypatch.setattr(page.analysis_runner, "run_analysis", fake_run)

    stub.button_responses = [True]

    expected_run_key = page._current_run_key(
        stub.session_state["model_state"], stub.session_state["selected_benchmark"]
    )

    page.render_results_page()

    assert called["result"] is not None
    assert called["run_key"] == expected_run_key


def test_current_run_key_changes_with_risk_free(results_page) -> None:
    page, stub = results_page
    stub.session_state.update(
        {
            "data_fingerprint": "abc123",
            "analysis_fund_columns": ["FundA"],
            "fund_columns": ["FundA"],
            "selected_risk_free": "RF1",
        }
    )
    model_state = {"trend_spec": {"window": 63, "lag": 1}}

    key_one = page._current_run_key(model_state, None)
    stub.session_state["selected_risk_free"] = "RF2"
    key_two = page._current_run_key(model_state, None)

    assert key_one != key_two


def test_results_hides_empty_state_when_result_present(
    monkeypatch: pytest.MonkeyPatch, results_page
) -> None:
    page, stub = results_page
    returns = _sample_returns()
    result = SimpleNamespace(
        metrics=pd.DataFrame({"Sharpe": [1.23]}),
        details={
            "out_user_stats": {"sharpe": 2.34},
            "portfolio_equal_weight_combined": returns["FundA"],
            "risk_diagnostics": {
                "turnover": pd.Series([0.1, 0.2], index=returns.index[:2]),
                "final_weights": pd.Series({"FundA": 0.6, "FundB": 0.4}),
            },
        },
        fallback_info=None,
        portfolio=returns["FundA"],
        weights=pd.Series({"FundA": 0.6, "FundB": 0.4}),
    )

    stub.session_state.update(
        {
            "model_state": {
                "trend_spec": {"window": 63, "lag": 1},
                "metric_weights": {"sharpe": 1.0},
            },
            "analysis_fund_columns": ["FundA", "FundB"],
            "fund_columns": ["FundA", "FundB"],
            "selected_benchmark": None,
            "data_fingerprint": "abc123",
            "returns_df": returns,
            "schema_meta": {},
            "upload_status": "success",
            "demo_preset": "Balanced",
        }
    )
    stub.session_state["analysis_result"] = result
    stub.session_state["analysis_result_key"] = page._current_run_key(
        stub.session_state["model_state"], stub.session_state["selected_benchmark"]
    )

    for chart in [
        "equity_chart",
        "drawdown_chart",
        "rolling_sharpe_chart",
        "turnover_chart",
        "exposure_chart",
    ]:
        monkeypatch.setattr(
            getattr(page, "charts"), chart, lambda *_args, chart_name=chart: chart_name
        )
    monkeypatch.setattr(
        page.explain_results, "render_explain_results", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        page.analysis_runner,
        "run_analysis",
        lambda *_args, **_kwargs: pytest.fail("cached result should render directly"),
    )

    page.render_results_page()

    assert not any(
        "Run the analysis to generate performance and risk diagnostics." in msg
        for msg in stub.markdown_messages
    )
    assert "Run analysis" not in stub.button_labels
    assert "Re-run with custom settings" in stub.button_labels
    assert any(
        "Demo results loaded — 2 funds — Sharpe 2.34." in msg
        for msg in stub.success_messages
    )


def test_results_completed_state_skips_non_finite_fallback_sharpe(
    monkeypatch: pytest.MonkeyPatch, results_page
) -> None:
    page, stub = results_page
    returns = _sample_returns()
    result = SimpleNamespace(
        metrics=pd.DataFrame({"Sharpe": [float("inf"), 1.23]}),
        details={"portfolio_equal_weight_combined": returns["FundA"]},
        fallback_info=None,
        portfolio=returns["FundA"],
        weights=pd.Series({"FundA": 0.6, "FundB": 0.4}),
    )
    stub.session_state.update(
        {
            "model_state": {
                "trend_spec": {"window": 63, "lag": 1},
                "metric_weights": {"sharpe": 1.0},
            },
            "analysis_fund_columns": ["FundA", "FundB"],
            "fund_columns": ["FundA", "FundB"],
            "selected_benchmark": None,
            "data_fingerprint": "abc123",
            "returns_df": returns,
            "schema_meta": {},
            "upload_status": "success",
            "demo_preset": "Balanced",
        }
    )
    stub.session_state["analysis_result"] = result
    stub.session_state["analysis_result_key"] = page._current_run_key(
        stub.session_state["model_state"], stub.session_state["selected_benchmark"]
    )

    for chart in [
        "equity_chart",
        "drawdown_chart",
        "rolling_sharpe_chart",
        "turnover_chart",
        "exposure_chart",
    ]:
        monkeypatch.setattr(
            getattr(page, "charts"), chart, lambda *_args, chart_name=chart: chart_name
        )
    monkeypatch.setattr(
        page.explain_results, "render_explain_results", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        page.analysis_runner,
        "run_analysis",
        lambda *_args, **_kwargs: pytest.fail("cached result should render directly"),
    )

    page.render_results_page()

    assert any(
        "Demo results loaded — 2 funds — Sharpe 1.23." in msg
        for msg in stub.success_messages
    )
    assert not any("Sharpe —" in msg for msg in stub.success_messages)


def test_demo_run_renders_results_end_to_end(
    monkeypatch: pytest.MonkeyPatch, results_page
) -> None:
    page, stub = results_page
    returns = _sample_returns()
    returns["RF"] = [0.001, 0.001, 0.001]
    result = SimpleNamespace(
        metrics=pd.DataFrame({"Sharpe": [1.23]}),
        details={
            "portfolio_equal_weight_combined": returns["FundA"],
            "risk_diagnostics": {
                "turnover": pd.Series([0.1, 0.2], index=returns.index[:2]),
                "final_weights": pd.Series({"FundA": 0.6, "FundB": 0.4}),
            },
        },
        fallback_info=None,
        portfolio=returns["FundA"],
        weights=pd.Series({"FundA": 0.6, "FundB": 0.4}),
    )

    from streamlit_app.components import demo_runner

    setup = demo_runner.DemoSetup(
        config_state={"preset_name": "Balanced"},
        sim_config={"preset_name": "Balanced"},
        pipeline_config=SimpleNamespace(),
        benchmark=None,
    )
    demo_runner._update_session_state(stub, setup, returns, {})
    demo_runner._store_demo_result_state(stub, setup, returns, result)

    for chart in [
        "equity_chart",
        "drawdown_chart",
        "rolling_sharpe_chart",
        "turnover_chart",
        "exposure_chart",
    ]:
        monkeypatch.setattr(
            getattr(page, "charts"), chart, lambda *_args, chart_name=chart: chart_name
        )
    monkeypatch.setattr(
        page.explain_results, "render_explain_results", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(stub, "button", lambda *_args, **_kwargs: False)

    recompute_calls: list[tuple[object, ...]] = []

    def fail_recompute(*args, **_kwargs):
        recompute_calls.append(args)
        raise AssertionError("demo result should be reused without recomputing")

    monkeypatch.setattr(page.analysis_runner, "run_analysis", fail_recompute)

    original_current_run_key = page._current_run_key
    render_run_keys: list[str] = []

    def recording_run_key(model_state, benchmark):
        key = original_current_run_key(model_state, benchmark)
        render_run_keys.append(key)
        return key

    monkeypatch.setattr(page, "_current_run_key", recording_run_key)

    expected_run_key = page._current_run_key(
        stub.session_state["model_state"], stub.session_state["selected_benchmark"]
    )
    assert stub.session_state["analysis_result_key"] == expected_run_key

    page.render_results_page()

    assert render_run_keys[-1] == stub.session_state["analysis_result_key"]
    assert not recompute_calls
    assert stub.session_state["analysis_fund_columns"] == ["FundA", "FundB"]
    assert stub.session_state["selected_risk_free"] == "RF"
    assert stub.session_state["analysis_result"] is result
    assert stub.session_state["analysis_result_key"] == expected_run_key
    assert any("Using 2 selected funds" in msg for msg in stub.caption_messages)
    assert not stub.error_messages

    rerun_columns: list[str] = []
    rerun_model_states: list[dict] = []

    def record_rerun(df, model_state, _benchmark, **_kwargs):
        rerun_columns.extend(list(df.columns))
        rerun_model_states.append(dict(model_state))
        return result

    monkeypatch.setattr(page.analysis_runner, "run_analysis", record_rerun)
    monkeypatch.setattr(stub, "button", lambda *_args, **_kwargs: True)

    page.render_results_page()

    assert rerun_columns == ["FundA", "FundB", "RF"]
    assert rerun_model_states[-1]["risk_free_column"] == "RF"


def test_demo_run_marks_or_hides_inapplicable_tabs(
    monkeypatch: pytest.MonkeyPatch, results_page
) -> None:
    page, stub = results_page
    returns = _sample_returns()
    result = SimpleNamespace(
        metrics=pd.DataFrame({"Sharpe": [1.23]}),
        details={
            "period_count": 1,
            "period_results": [
                {
                    "period": ("2024-01-01", "2024-01-31", "2024-02-01", "2024-02-29"),
                    "in_sample_scaled": returns.iloc[:2],
                    "out_sample_scaled": returns.iloc[2:],
                    "ew_weights": {"FundA": 0.5, "FundB": 0.5},
                    "fund_weights": {"FundA": 0.6, "FundB": 0.4},
                }
            ],
            "portfolio_equal_weight_combined": returns["FundA"],
            "risk_diagnostics": {
                "turnover": pd.Series([0.1, 0.2], index=returns.index[:2]),
                "final_weights": pd.Series({"FundA": 0.6, "FundB": 0.4}),
            },
        },
        period_count=1,
        fallback_info=None,
        portfolio=returns["FundA"],
        weights=pd.Series({"FundA": 0.6, "FundB": 0.4}),
    )

    from streamlit_app.components import demo_runner

    setup = demo_runner.DemoSetup(
        config_state={"preset_name": "Balanced"},
        sim_config={"preset_name": "Balanced"},
        pipeline_config=SimpleNamespace(),
        benchmark=None,
    )
    demo_runner._update_session_state(stub, setup, returns, {})
    demo_runner._store_demo_result_state(stub, setup, returns, result)

    for chart in [
        "equity_chart",
        "drawdown_chart",
        "rolling_sharpe_chart",
        "turnover_chart",
        "exposure_chart",
    ]:
        monkeypatch.setattr(
            getattr(page, "charts"), chart, lambda *_args, chart_name=chart: chart_name
        )
    monkeypatch.setattr(
        page.explain_results, "render_explain_results", lambda *_args, **_kwargs: None
    )
    download_calls: list[tuple[object, dict[str, object]]] = []

    def render_download_spy(*args, **kwargs):
        download_calls.append((args, kwargs))

    monkeypatch.setattr(page, "_render_download_section", render_download_spy)
    monkeypatch.setattr(stub, "button", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        page.analysis_runner,
        "run_analysis",
        lambda *_args, **_kwargs: pytest.fail("demo result should be reused"),
    )

    page.render_results_page()

    labels = stub.tab_groups[-1]
    assert any("Summary" in label for label in labels)
    assert any("Visualizations" in label for label in labels)
    assert any(
        "Period Analysis" in label and "multi-period only" in label for label in labels
    )
    assert any(
        "Fund Details" in label and "multi-period/custom only" in label
        for label in labels
    )
    assert any("Export" in label and "multi-period only" in label for label in labels)
    assert any(
        "Compare" in label and "needs saved configs" in label for label in labels
    )
    assert any(
        "Run a multi-period analysis to enable" in msg for msg in stub.info_messages
    )
    assert any(
        "Export requires a multi-period analysis" in msg for msg in stub.info_messages
    )
    assert not download_calls


def test_period_count_uses_period_results_when_explicit_count_missing(
    results_page,
) -> None:
    page, _stub = results_page

    result = SimpleNamespace(
        details={
            "period_count": 0,
            "period_results": [
                {"period": ("2024-01-01", "2024-01-31", "2024-02-01", "2024-02-29")},
                {"period": ("2024-02-01", "2024-02-29", "2024-03-01", "2024-03-31")},
            ],
        },
        period_count=0,
    )

    assert page._result_period_count(result, result.details) == 2
