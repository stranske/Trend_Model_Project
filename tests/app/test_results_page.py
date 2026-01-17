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
        self.metrics: list[tuple[str, object]] = []
        self.checkbox_labels: list[str] = []

    # Basic UI primitives -------------------------------------------------
    def title(self, _text: str) -> None:  # pragma: no cover - trivial
        return None

    def header(self, _text: str) -> None:  # pragma: no cover - trivial
        return None

    def markdown(self, *_args, **_kwargs) -> None:  # pragma: no cover - trivial
        return None

    def button(self, *_args, **_kwargs) -> bool:
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

    def success(self, _text: str) -> None:  # pragma: no cover - trivial
        return None

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
        return [ColumnContext(self) for _ in labels]

    def metric(self, label: str, value: object) -> None:
        self.metrics.append((label, value))

    def expander(self, *_args, **_kwargs) -> "ColumnContext":
        return ColumnContext(self)

    def checkbox(self, label: str, value: bool = False, key: str | None = None, **_kwargs) -> bool:
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
