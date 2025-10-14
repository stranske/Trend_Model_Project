"""Tests for the Streamlit-facing portfolio app module."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock

import pandas as pd
import pytest


class _ContextManager:
    """Simple context manager used by the fake Streamlit shim."""

    def __init__(self, value_factory: Callable[[], Any] | None = None) -> None:
        self._factory = value_factory or (lambda: self)

    def __enter__(self) -> Any:
        return self._factory()

    def __exit__(self, *_exc: object) -> bool:
        return False


class SessionStateDict(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - attribute passthrough
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class FakeStreamlit:
    """Minimal Streamlit replacement covering the behaviours the app uses."""

    def __init__(self) -> None:
        self.page_configs: list[dict[str, Any]] = []
        self.titles: list[str] = []
        self.headers: list[str] = []
        self.subheaders: list[str] = []
        self.button_calls: list[tuple[str, bool]] = []
        self.button_responses: dict[str, list[bool]] = {}
        self.text_inputs: list[tuple[str, str, str]] = []
        self.downloads: list[tuple[str, str]] = []
        self.spinner_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.success_messages: list[str] = []
        self.info_messages: list[str] = []
        self.dataframes: list[pd.DataFrame] = []
        self.captions: list[str] = []
        self.line_charts: list[pd.DataFrame] = []
        self.bar_charts: list[pd.DataFrame] = []
        self.columns_specs: list[Any] = []
        self.empty_calls = 0
        self.session_state: SessionStateDict = SessionStateDict()
        self.sidebar = _ContextManager(lambda: self)

    def set_page_config(self, **kwargs: Any) -> None:
        self.page_configs.append(kwargs)

    def title(self, value: str) -> None:
        self.titles.append(value)

    def header(self, value: str) -> None:
        self.headers.append(value)

    def subheader(self, value: str) -> None:
        self.subheaders.append(value)

    def button(self, label: str, **_: Any) -> bool:
        responses = self.button_responses.setdefault(label, [])
        value = responses.pop(0) if responses else False
        self.button_calls.append((label, value))
        return value

    def text_input(self, label: str, *, key: str, value: str = "", **_: Any) -> str:
        self.text_inputs.append((label, key, value))
        self.session_state[key] = value
        return value

    def download_button(self, label: str, *, file_name: str, **_: Any) -> None:
        self.downloads.append((label, file_name))

    def columns(self, spec: Any) -> list[_ContextManager]:
        self.columns_specs.append(spec)
        if isinstance(spec, int):
            expected = spec
        elif isinstance(spec, (list, tuple)):
            expected = len(spec)
        else:
            expected = 1
        return [_ContextManager() for _ in range(max(expected, 1))]

    def empty(self) -> _ContextManager:
        self.empty_calls += 1
        return _ContextManager()

    def spinner(self, message: str) -> _ContextManager:
        self.spinner_messages.append(message)
        return _ContextManager()

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)

    def success(self, message: str) -> None:
        self.success_messages.append(message)

    def info(self, message: str) -> None:
        self.info_messages.append(message)

    def error(self, message: str) -> None:
        self.warning_messages.append(f"error:{message}")

    def dataframe(self, frame: pd.DataFrame, **_: Any) -> None:
        self.dataframes.append(frame.copy())

    def caption(self, message: str) -> None:
        self.captions.append(message)

    def line_chart(self, frame: pd.DataFrame) -> None:
        self.line_charts.append(frame.copy())

    def bar_chart(self, frame: pd.DataFrame) -> None:
        self.bar_charts.append(frame.copy())


class _IterablePeriod:
    def __init__(self, values: list[str]) -> None:
        self._values = values

    def __iter__(self):  # pragma: no cover - trivial iterator
        return iter(self._values)


@pytest.fixture
def load_app(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import ``trend_portfolio_app.app`` with a mocked Streamlit module."""

    module_name = "trend_portfolio_app.app"
    if module_name in sys.modules:
        del sys.modules[module_name]

    fake_streamlit = MagicMock(name="streamlit")
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    module = importlib.import_module(module_name)
    return module


def test_app_import_skips_render_with_magicmock(load_app: ModuleType) -> None:
    """Importing with a MagicMock should not trigger the auto-render."""

    module = load_app
    assert module.page_config_calls == []
    assert module.titles == []


def test_render_app_single_period_success(monkeypatch: pytest.MonkeyPatch, load_app: ModuleType) -> None:
    """Running the Streamlit app renders the single-period success path."""

    module = load_app
    fake_st = FakeStreamlit()
    fake_st.button_responses = {
        "Reset to defaults": [False],
        "Run Single Period": [True],
        "Run Multi-Period": [False],
    }
    fake_st.session_state["config_dict"] = {
        "data": {"csv_path": "demo.csv"},
        "portfolio": {"policy": ""},
    }
    fake_st.session_state["data.csv_path"] = "demo.csv"
    fake_st.session_state["portfolio.rebalance._months"] = "6"

    module.page_config_calls.clear()
    module.titles.clear()
    module.st = fake_st

    monkeypatch.setattr(module, "validate_trend_config", lambda cfg, base_path: None)

    class DummyConfig(dict):
        pass

    monkeypatch.setattr(module, "Config", DummyConfig)

    pipeline_stub = SimpleNamespace(
        run=lambda cfg: pd.DataFrame({"metric": [1.2345, 2.5], "text": ["a", "b"]}),
        run_full=lambda cfg: {
            "out_sample_stats": {
                "summary": {"sharpe": 1.0, "cagr": 0.25},
                "object_stats": SimpleNamespace(sharpe=0.75, cagr=0.12),
            },
            "benchmark_ir": {"component": {"AssetA": 0.8, "equal_weight": 0.4}},
            "risk_diagnostics": {
                "asset_volatility": pd.DataFrame({"asset": [0.1, 0.2]}),
                "portfolio_volatility": pd.Series({"portfolio": 0.3}),
                "turnover": pd.Series({"2024-01": 0.4}),
                "turnover_value": 0.1234,
            },
        },
    )
    monkeypatch.setattr(module, "pipeline", pipeline_stub)
    monkeypatch.setattr(module, "run_multi", lambda cfg: [])

    module._render_app()

    assert module.page_config_calls == [True]
    assert module.titles == ["Trend Portfolio App"]
    assert any(msg.startswith("Completed.") for msg in fake_st.success_messages)
    assert ("Download CSV", "single_period_summary.csv") in fake_st.downloads
    # Converted month helper should update nested configuration length in days.
    cfg_dict = fake_st.session_state["config_dict"]
    assert cfg_dict["portfolio"]["rebalance"]["length"] == 126
    # Risk diagnostics should render charts or dataframes.
    assert fake_st.captions  # includes at least the realised volatility caption
    assert fake_st.line_charts or fake_st.bar_charts


def test_render_run_section_multi_period(monkeypatch: pytest.MonkeyPatch, load_app: ModuleType) -> None:
    """Multi-period execution renders summary downloads and success state."""

    module = load_app
    fake_st = FakeStreamlit()
    fake_st.button_responses = {
        "Run Single Period": [False],
        "Run Multi-Period": [True],
    }
    fake_st.session_state["config_dict"] = {
        "data": {"csv_path": "sample.csv"},
        "portfolio": {},
    }
    fake_st.session_state["metrics.alpha"] = 0.5

    module.st = fake_st
    monkeypatch.setattr(module, "validate_trend_config", lambda cfg, base_path: None)

    class DummyConfig(dict):
        pass

    monkeypatch.setattr(module, "Config", DummyConfig)
    monkeypatch.setattr(module, "pipeline", SimpleNamespace(run=lambda cfg: None, run_full=lambda cfg: {}))

    run_multi_result = [
        {
            "period": ("2020-01", "2020-06", "2020-07", "2020-12"),
            "out_ew_stats": {"sharpe": 0.5, "cagr": 0.12},
            "out_user_stats": {"sharpe": 0.7, "cagr": 0.18},
        },
        {
            "period": ["2021-01", "2021-06", "2021-07", "2021-12"],
            "out_ew_stats": SimpleNamespace(sharpe=0.65, cagr=0.2),
            "out_user_stats": {},
        },
        {
            "period": None,
            "out_ew_stats": None,
            "out_user_stats": None,
        },
    ]
    monkeypatch.setattr(module, "run_multi", lambda cfg: run_multi_result)

    module._render_run_section(fake_st.session_state["config_dict"])

    assert any("Periods: 3" in msg for msg in fake_st.success_messages)
    assert ("Download periods CSV", "multi_period_summary.csv") in fake_st.downloads
    assert ("Download raw JSON", "multi_period_raw.json") in fake_st.downloads
    assert not fake_st.warning_messages
    # Summary dataframe should round metrics to four decimals.
    assert not fake_st.dataframes[-1].empty
    assert "ew_sharpe" in fake_st.dataframes[-1]


def test_app_helper_utilities(monkeypatch: pytest.MonkeyPatch, load_app: ModuleType) -> None:
    """Exercise lower-level helper functions for additional coverage."""

    module = load_app
    fake_st = FakeStreamlit()
    module.st = fake_st

    merged = module._merge_update({"a": {"b": 1}}, {"a": {"c": 2}, "d": 3})
    assert merged["a"]["b"] == 1 and merged["a"]["c"] == 2 and merged["d"] == 3

    assert module._expected_columns(3) == 3
    assert module._expected_columns([1, 2]) == 2
    assert module._expected_columns(object()) == 1

    cols = module._normalize_columns(None, 2)
    assert len(cols) == 2
    assert fake_st.empty_calls == 1

    fillers = module._normalize_columns([object()], 3)
    assert len(fillers) == 3

    module._columns(2)
    assert fake_st.columns_specs[-1] == 2

    df = pd.DataFrame({"a": [1.234567, None], "b": ["x", "y"]})
    summary_df = module._summarise_run_df(df)
    assert summary_df["a"].iloc[0] == pytest.approx(1.2346)
    assert module._summarise_run_df(None).empty

    result_summary = module._build_summary_from_result(
        {
            "out_sample_stats": {"portfolio": {"value": 1}},
            "benchmark_ir": {"demo": {"Asset": 0.4, "equal_weight": 0.3}},
        }
    )
    assert "ir_demo" in result_summary.columns
    assert module._build_summary_from_result(None).empty
    assert module._build_summary_from_result({"out_sample_stats": {}}).empty

    multi_summary = module._summarise_multi(
        [
            {
                "period": ("2020-01", "2020-06", "2020-07", "2020-12"),
                "out_ew_stats": {"sharpe": 0.5, "cagr": 0.1},
                "out_user_stats": None,
            },
            {"period": [], "out_ew_stats": {}, "out_user_stats": {}},
            {"period": _IterablePeriod(["2022-01", "2022-06"])},
        ]
    )
    assert "ew_sharpe" in multi_summary.columns

    fake_st.session_state.update(
        {
            "metrics.window._months": "3",
            "metrics.alpha": 0.7,
            "metrics.window.bad._months": "bad",
            "other": "skip",
            "data.csv_path": "user.csv",
        }
    )
    cfg_dict: dict[str, Any] = {"metrics": {}, "data": {}}
    module._apply_session_state(cfg_dict)
    assert cfg_dict["metrics"]["window"]["length"] == 63
    assert cfg_dict["metrics"]["alpha"] == 0.7
    assert cfg_dict["data"]["csv_path"] == "user.csv"


def test_read_defaults_includes_demo_path(load_app: ModuleType) -> None:
    module = load_app
    defaults = module._read_defaults()
    assert defaults["data"]["csv_path"].endswith("demo/demo_returns.csv")
    assert defaults["portfolio"]["policy"] == ""


def test_render_sidebar_reset_restores_defaults(monkeypatch: pytest.MonkeyPatch, load_app: ModuleType) -> None:
    module = load_app
    fake_st = FakeStreamlit()
    fake_st.button_responses = {"Reset to defaults": [True]}
    fake_st.session_state["config_dict"] = {"data": {"csv_path": "custom.csv"}}
    module.st = fake_st

    defaults = {"data": {"csv_path": "demo.csv"}, "portfolio": {"policy": ""}}
    monkeypatch.setattr(module, "_read_defaults", lambda: defaults)

    module._render_sidebar(fake_st.session_state["config_dict"])

    assert fake_st.session_state["config_dict"] == defaults
    assert ("Download YAML", "config.yml") in fake_st.downloads


def test_render_run_section_warns_when_no_results(monkeypatch: pytest.MonkeyPatch, load_app: ModuleType) -> None:
    module = load_app
    fake_st = FakeStreamlit()
    fake_st.button_responses = {"Run Single Period": [True], "Run Multi-Period": [False]}
    fake_st.session_state["config_dict"] = {"data": {}, "portfolio": {}}
    module.st = fake_st

    monkeypatch.setattr(module, "validate_trend_config", lambda cfg, base_path: None)
    monkeypatch.setattr(module, "Config", dict)
    monkeypatch.setattr(module, "pipeline", SimpleNamespace(run=lambda cfg: "not_df", run_full=lambda cfg: {}))
    monkeypatch.setattr(module, "_summarise_run_df", lambda value: [])

    module._render_run_section(fake_st.session_state["config_dict"])

    assert any("Analysis failed" in msg for msg in fake_st.warning_messages)


def test_render_run_section_reports_partial_results(monkeypatch: pytest.MonkeyPatch, load_app: ModuleType) -> None:
    module = load_app
    fake_st = FakeStreamlit()
    fake_st.button_responses = {"Run Single Period": [True], "Run Multi-Period": [False]}
    fake_st.session_state["config_dict"] = {"data": {}, "portfolio": {}}
    module.st = fake_st

    monkeypatch.setattr(module, "validate_trend_config", lambda cfg, base_path: None)
    monkeypatch.setattr(module, "Config", dict)

    summary_df = pd.DataFrame({"metric": [1.0]})

    def raising_run_full(cfg: dict[str, Any]) -> dict[str, Any]:
        raise FileNotFoundError("missing diagnostics")

    monkeypatch.setattr(module, "pipeline", SimpleNamespace(run=lambda cfg: summary_df, run_full=raising_run_full))

    module._render_run_section(fake_st.session_state["config_dict"])

    assert any("Completed" in msg for msg in fake_st.success_messages)
    assert any("Partial results" in msg for msg in fake_st.info_messages)


def test_render_run_section_builds_summary_from_full_result(monkeypatch: pytest.MonkeyPatch, load_app: ModuleType) -> None:
    module = load_app
    fake_st = FakeStreamlit()
    fake_st.button_responses = {"Run Single Period": [True], "Run Multi-Period": [False]}
    fake_st.session_state["config_dict"] = {"data": {}, "portfolio": {}}
    module.st = fake_st

    monkeypatch.setattr(module, "validate_trend_config", lambda cfg, base_path: None)
    monkeypatch.setattr(module, "Config", dict)

    empty_df = pd.DataFrame({"metric": []})
    full_result = {
        "out_sample_stats": {"portfolio": {"sharpe": 0.5, "cagr": 0.1}},
        "benchmark_ir": {"component": {"Asset": 0.2}},
        "risk_diagnostics": {},
    }

    monkeypatch.setattr(module, "pipeline", SimpleNamespace(run=lambda cfg: empty_df, run_full=lambda cfg: full_result))

    module._render_run_section(fake_st.session_state["config_dict"])

    assert any(frame.equals(frame.copy()) for frame in fake_st.dataframes)


def test_module_auto_renders_with_realistic_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "trend_portfolio_app.app"
    if module_name in sys.modules:
        del sys.modules[module_name]

    fake = FakeStreamlit()
    streamlit_module = ModuleType("streamlit")

    def _wrap(method: str):
        def _call(*args: Any, **kwargs: Any) -> Any:
            return getattr(fake, method)(*args, **kwargs)

        return _call

    for name in [
        "set_page_config",
        "title",
        "header",
        "subheader",
        "button",
        "text_input",
        "download_button",
        "columns",
        "empty",
        "spinner",
        "warning",
        "success",
        "info",
        "error",
        "dataframe",
        "caption",
        "line_chart",
        "bar_chart",
    ]:
        setattr(streamlit_module, name, _wrap(name))

    streamlit_module.session_state = fake.session_state
    streamlit_module.sidebar = fake.sidebar

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)

    module = importlib.import_module(module_name)

    assert module.page_config_calls == [True]
    assert module.titles == ["Trend Portfolio App"]
