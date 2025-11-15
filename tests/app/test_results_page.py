import importlib
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest


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
        self.subheaders: list[str] = []
        self.altair_payloads: list[object] = []
        self.dataframes: list[pd.DataFrame] = []

    # Basic UI primitives -------------------------------------------------
    def title(self, _text: str) -> None:  # pragma: no cover - trivial
        return None

    def markdown(self, *_args, **_kwargs) -> None:  # pragma: no cover - trivial
        return None

    def button(self, *_args, **_kwargs) -> bool:
        if self.button_responses:
            return self.button_responses.pop(0)
        return False

    def spinner(self, *_args, **_kwargs) -> _ContextManager:
        return _ContextManager()

    def columns(self, count: int) -> list["ColumnContext"]:
        return [ColumnContext(self) for _ in range(count)]

    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def altair_chart(self, payload, **_kwargs) -> None:
        self.altair_payloads.append(payload)

    def dataframe(self, df: pd.DataFrame) -> None:
        self.dataframes.append(df)

    def error(self, message: str) -> None:
        self.error_messages.append(message)

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)

    def info(self, message: str) -> None:
        self.info_messages.append(message)

    def caption(self, message: str) -> None:
        self.caption_messages.append(message)

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
    stub = DummyStreamlit()

    module = ModuleType("streamlit")

    def bind(name: str):
        def wrapper(*args, **kwargs):
            return getattr(stub, name)(*args, **kwargs)

        return wrapper

    for attr in [
        "title",
        "markdown",
        "button",
        "spinner",
        "columns",
        "subheader",
        "altair_chart",
        "dataframe",
        "error",
        "warning",
        "info",
        "caption",
        "cache_data",
    ]:
        setattr(module, attr, bind(attr))

    module.session_state = stub.session_state

    monkeypatch.setitem(sys.modules, "streamlit", module)

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
    index = pd.date_range("2023-01-31", periods=3, freq="M")
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

    def fake_run(df: pd.DataFrame, model_state: dict, benchmark: str | None):
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

    # Changing benchmark should invalidate cached result and trigger a new run.
    stub.session_state["selected_benchmark"] = "BenchB"
    page.render_results_page()

    assert run_calls == ["BenchA", "BenchB"]


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
    assert stub.caption_messages == ["No returns available after filtering"]
    assert stub.session_state.get("analysis_result") is None
    assert stub.session_state.get("analysis_error") == {
        "message": "We couldn't run the analysis with the current data or settings. Please review the configuration and try again.",
        "detail": "No returns available after filtering",
    }
