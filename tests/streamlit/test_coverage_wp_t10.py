"""Coverage tests for work package t10 (Monte Carlo page, plots, NL viewer, demo, state)."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import plotly.graph_objects as go
import pytest

from streamlit_app import state as state_mod
from streamlit_app.components import demo_runner, mc_plots
from tests.support.dummy_streamlit import DummyStreamlit
from trend_analysis.config.patch import ConfigPatch, PatchOperation, RiskFlag
from trend_analysis.llm.nl_logging import NLOperationLog
from trend_analysis.monte_carlo.registry import ScenarioRegistryEntry
from trend_analysis.monte_carlo.scenario import MonteCarloScenario, MonteCarloSettings

_REAL_STREAMLIT = sys.modules.get("streamlit") or importlib.import_module("streamlit")
_RELOADED: dict[str, ModuleType] = {}


def _reload(name: str) -> ModuleType:
    """Reload a page module under the stub and remember it so the fixture below can rebind it."""
    module = importlib.reload(importlib.import_module(name))
    _RELOADED[name] = module
    return module


@pytest.fixture(autouse=True)
def _rebind_reloaded_modules_to_real_streamlit() -> Iterator[None]:
    """Reloading a page module under the DummyStreamlit stub rebinds its module-level ``st`` in place,
    so a module object other test files imported at collection time would keep the stub. After every
    test put the real streamlit back into each module this file reloaded. It does not depend on the
    order of the conftest's eviction fixture or of monkeypatch's teardown: the real module is swapped
    in for the reload itself, and a module that was already evicted is re-registered first."""
    yield
    saved = sys.modules.get("streamlit")
    sys.modules["streamlit"] = _REAL_STREAMLIT
    try:
        for name, module in list(_RELOADED.items()):
            sys.modules.setdefault(name, module)
            importlib.reload(module)
    finally:
        if saved is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = saved
        _RELOADED.clear()


def _install_stub(monkeypatch: pytest.MonkeyPatch) -> DummyStreamlit:
    stub = DummyStreamlit()
    module = ModuleType("streamlit")
    for name in dir(stub):
        if not name.startswith("__"):
            setattr(module, name, getattr(stub, name))
    module.session_state = stub.session_state
    monkeypatch.setitem(sys.modules, "streamlit", module)
    return stub


def _load_page(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, DummyStreamlit]:
    stub = _install_stub(monkeypatch)
    _reload("streamlit_app.components.mc_tables")
    _reload("streamlit_app.components.mc_plots")
    page = _reload("streamlit_app.monte_carlo_page")
    return page, stub


def _scenario(name: str = "macro") -> MonteCarloScenario:
    return MonteCarloScenario(
        name=name,
        description="Desc",
        version="1.0",
        base_config=Path("config/defaults.yml"),
        monte_carlo=MonteCarloSettings(
            mode="mixture", n_paths=500, horizon_years=10, frequency="M", seed=42, jobs=4
        ),
    )


def _entry(**kwargs: Any) -> NLOperationLog:
    defaults: dict[str, Any] = dict(
        request_id="req-1",
        timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        operation="nl_to_patch",
        input_hash="hash-1",
        prompt_template="Prompt",
        prompt_variables={},
        model_output="output",
        parsed_patch=None,
        validation_result=None,
        error=None,
        duration_ms=12.5,
        model_name="gpt-4o-mini",
        temperature=0.2,
        token_usage=None,
        trace_url=None,
    )
    defaults.update(kwargs)
    return NLOperationLog(**defaults)


def _load_nl(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    st_stub = MagicMock()
    st_stub.session_state = {}
    from contextlib import nullcontext

    st_stub.expander.side_effect = lambda *_a, **_k: nullcontext()
    st_stub.spinner.side_effect = lambda *_a, **_k: nullcontext()
    st_stub.container.side_effect = lambda *_a, **_k: nullcontext()
    monkeypatch.setitem(sys.modules, "streamlit", st_stub)
    module = _reload("streamlit_app.components.nl_operation_viewer")
    return module, st_stub


# --- monte_carlo_page -------------------------------------------------------


def test_should_auto_render_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    page, _stub = _load_page(monkeypatch)
    fake = ModuleType("streamlit.runtime.scriptrunner")
    fake.get_script_run_ctx = lambda: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "streamlit.runtime.scriptrunner", fake)
    assert page._should_auto_render() is True
    fake.get_script_run_ctx = lambda: None  # type: ignore[attr-defined]
    assert page._should_auto_render() is False
    monkeypatch.setitem(sys.modules, "streamlit.runtime.scriptrunner", None)
    assert page._should_auto_render() is False


def test_session_frequency_and_csv_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    page, stub = _load_page(monkeypatch)
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"A": [0.1, 0.2, 0.3]}, index=idx)
    assert page._session_frequency(df) == "D"
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("A\n1\n", encoding="utf-8")
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("x", encoding="utf-8")
    stub.session_state["data_saved_path"] = str(txt_file)
    stub.session_state["uploaded_file_path"] = str(csv_file)
    assert page._session_csv_path() == str(csv_file)
    stub.session_state["uploaded_file_path"] = str(tmp_path / "missing.csv")
    assert page._session_csv_path() is None


def test_analysis_frame_regime_and_price_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)
    df = pd.DataFrame(
        {"FundA": [0.1, 0.2], "PROXY": [0.0, 0.0], "Bench": [0.1, 0.1]},
    )
    stub.session_state["analysis_fund_columns"] = ["FundA", "PROXY"]
    out = page._analysis_frame_from_session(
        df, {"regime_enabled": True, "regime_proxy": "PROXY"}, "Bench"
    )
    assert list(out.columns) == ["FundA", "Bench", "PROXY"]
    # regime proxy excluded from funds but kept as extra; fund list is exact above
    with pytest.raises(ValueError, match="session returns contain no numeric columns"):
        page._returns_to_price_history(pd.DataFrame({"x": ["a", "b"]}))
    with pytest.raises(ValueError, match="session returns contain values <= -1"):
        page._returns_to_price_history(pd.DataFrame({"x": [0.1, -1.0]}))
    prices = page._returns_to_price_history(pd.DataFrame({"x": [0.1, 0.1]}))
    assert prices["x"].tolist() == [110.00000000000001, 121.00000000000001]


def test_session_runner_kwargs_risk_free_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)
    captured: dict[str, Any] = {}

    class FakeCfg:
        portfolio: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> FakeCfg:
        captured.update(kwargs)
        return FakeCfg()

    monkeypatch.setattr(page, "build_config_from_ui_state", fake_build)
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    stub.session_state["returns_df"] = pd.DataFrame({"FundA": [-1.5, 0.1]}, index=idx)
    stub.session_state["model_state"] = {"foo": 1}
    stub.session_state["selected_risk_free"] = "FundA"
    kwargs, err = page._session_runner_kwargs()
    assert kwargs == {}
    assert err == "session returns contain values <= -1"
    assert captured["model_state"]["risk_free_column"] == "FundA"


def test_cache_data_identity_and_clamp_and_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    page, _stub = _load_page(monkeypatch)
    fake_st = SimpleNamespace(session_state={})
    monkeypatch.setattr(page, "st", fake_st)

    def fn(x: int) -> int:
        return x + 1

    assert page._cache_data()(fn) is fn
    assert page._cache_data(show_spinner=False)(fn) is fn
    assert page._clamp_int(None, 5, 10) == 5
    assert page._clamp_int(99, 5, 10) == 10
    assert page._coerce_seed("   ") == (None, None)
    assert page._coerce_seed("-3") == (None, "Random seed must be a non-negative integer.")
    assert page._coerce_seed("7") == (7, None)


def test_build_override_type_error_and_fold_options(monkeypatch: pytest.MonkeyPatch) -> None:
    page, _stub = _load_page(monkeypatch)
    bad = _scenario()
    bad.monte_carlo = "not-settings"  # type: ignore[assignment]
    with pytest.raises(TypeError, match="monte_carlo settings are not resolved"):
        page._build_override_scenario(bad, n_paths=100, horizon_years=5, seed=None, jobs=1)
    disabled = _scenario()
    disabled.enable_fold_runs = False
    assert page._fold_options(disabled) == []
    enabled = _scenario()
    enabled.enable_fold_runs = True
    enabled.folds = "not-a-mapping"  # type: ignore[assignment]
    assert page._fold_options(enabled) == []
    enabled.folds = {"enabled": False, "n_folds": 3}
    assert page._fold_options(enabled) == []
    enabled.folds = {"n_folds": 3}
    assert page._fold_options(enabled) == ["All folds", "Fold 1", "Fold 2", "Fold 3"]
    enabled.folds = {"n_folds": "oops"}
    assert page._fold_options(enabled) == ["All folds"]
    enabled.folds = {"fold_starts": ["2020-01", "2021-01"]}
    assert page._fold_options(enabled) == ["All folds", "Fold 1", "Fold 2"]
    enabled.folds = {"fold_starts": "notalist"}
    assert page._fold_options(enabled) == ["All folds"]


def test_filter_and_extract_and_fold_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    page, _stub = _load_page(monkeypatch)
    df = pd.DataFrame({"fold_id": [1, 2, 2], "v": [10, 20, 30]})
    assert page._filter_results_by_fold(df, "Fold 2")["v"].tolist() == [20, 30]
    labelled = pd.DataFrame({"fold_label": ["Fold 1", "Fold 2"], "v": [1, 2]})
    assert page._filter_results_by_fold(labelled, "Fold 2")["v"].tolist() == [2]
    plain = pd.DataFrame({"v": [1, 2]})
    assert page._filter_results_by_fold(plain, "Fold 2") is plain
    nav = pd.DataFrame({"p": [100.0, 101.0]})
    nav2 = pd.DataFrame({"p": [200.0, 201.0]})

    class WithMeta:
        metadata = {"nav_paths_by_fold": {2: nav2}, "nav_paths": nav}

    assert page._extract_nav_paths(WithMeta(), fold_id=2) is nav2
    assert page._extract_nav_paths(WithMeta(), fold_id=None) is nav
    assert (
        page._extract_nav_paths({"nav_paths_by_fold": {1: nav}, "nav_paths": nav2}, fold_id=1)
        is nav
    )
    assert page._extract_nav_paths({"nav_paths": nav2}) is nav2
    assert page._extract_nav_paths(object()).empty
    assert page._extract_nav_paths(SimpleNamespace(nav_paths="nope", metadata={})).empty
    assert page._fold_selection_for_adapters("Fold 3") == 3
    assert page._fold_selection_for_adapters("Custom fold") == "Custom fold"
    assert page._fold_selection_for_adapters("All folds") is None


def test_render_diagnostic_empty_and_sharpe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)
    assert page._render_diagnostic_charts(pd.DataFrame(), pd.DataFrame({"a": [1.0]})) == {}
    assert stub.warning_messages[-1] == "Diagnostics unavailable: summary frame is empty."
    summary = pd.DataFrame({"sharpe": [1.0]})
    assert page._render_diagnostic_charts(summary, pd.DataFrame()) == {}
    assert stub.warning_messages[-1] == "Diagnostics unavailable: canonical paths are empty."
    # Sharpe failure branch: fake viz modules where sharpe make raises
    import trend_analysis.viz as viz_pkg  # noqa: F401

    sharpe_mod = ModuleType("trend_analysis.viz.sharpe_ladder")

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("no sharpe")

    sharpe_mod.make = boom  # type: ignore[attr-defined]
    charts_mod = ModuleType("trend_analysis.viz.charts")
    charts_mod.corr_heatmap = SimpleNamespace(build_figure=lambda *_a, **_k: go.Figure())  # type: ignore[attr-defined]
    charts_mod.rolling_panel = SimpleNamespace(build_figure=lambda *_a, **_k: go.Figure())  # type: ignore[attr-defined]
    charts_mod.seasonality_heatmap = SimpleNamespace(build_figure=lambda *_a, **_k: go.Figure())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trend_analysis.viz.sharpe_ladder", sharpe_mod)
    monkeypatch.setitem(sys.modules, "trend_analysis.viz.charts", charts_mod)
    paths = pd.DataFrame({"p": [100.0, 101.0]}, index=pd.date_range("2024-01-01", periods=2))
    out = page._render_diagnostic_charts(summary, paths)
    assert list(out) == [
        "Sharpe Ladder",
        "Correlation Heatmap",
        "Rolling Diagnostics",
        "Seasonality Heatmap",
    ]
    assert "Sharpe ladder unavailable" in stub.warning_messages[-1]


def test_progress_callback_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)
    calls: list[Any] = []

    class Bar:
        def progress(self, value: float, text: str | None = None) -> None:
            calls.append((value, text))
            if text is not None:
                raise RuntimeError("no text support")

    class Slot:
        def __init__(self) -> None:
            self.metrics: list[Any] = []

        def metric(self, label: str, value: Any) -> None:
            self.metrics.append((label, value))

    elapsed_slot, eta_slot = Slot(), Slot()
    cb = page._progress_callback_factory(
        progress_bar=Bar(), elapsed_slot=elapsed_slot, eta_slot=eta_slot, start_time=0.0
    )
    monkeypatch.setattr(page, "monotonic", lambda: 5000.0)
    cb({"completed": 1, "total": 2})
    assert calls == [(0.5, "Running paths (1/2)"), (0.5, None)]
    assert elapsed_slot.metrics[0][0] == "Elapsed"
    assert eta_slot.metrics[0][0] == "ETA"


def test_build_chart_bundle_empty_and_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    page, _stub = _load_page(monkeypatch)
    assert page._build_chart_bundle_payload({}) == (None, [])
    bundle_file = tmp_path / "bundle.zip"
    import zipfile

    with zipfile.ZipFile(bundle_file, "w") as zf:
        zf.writestr("a.txt", "hi")
    monkeypatch.setattr(page, "save_chart_bundle", lambda *a, **k: bundle_file)
    payload, warnings = page._build_chart_bundle_payload({"C": go.Figure()}, provenance="prov-x")
    assert warnings == []
    assert payload is not None
    assert payload["label"] == "Download charts bundle"
    with zipfile.ZipFile(payload["data"]) as zf:
        assert zf.read("README.txt").decode() == "prov-x"


def test_render_results_mapping_and_no_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)
    page._render_results(object(), fold_selection=None)
    assert stub.warning_messages[-1] == "No results frame available to display."
    frame = pd.DataFrame({"strategy": ["A", "A"], "sharpe": [1.0, 2.0], "fold_id": [1, 2]})
    nav = pd.DataFrame({"p": [100.0]}, index=pd.date_range("2024-01-01", periods=1))
    monkeypatch.setattr(page, "_cached_make_summary", lambda *a, **k: frame)
    monkeypatch.setattr(page, "_cached_make_paths", lambda df: df)
    monkeypatch.setattr(
        page, "_build_download_payloads", lambda *a, **k: [{"label": "x", "data": "y"}]
    )
    monkeypatch.setattr(page, "_build_chart_bundle_payload", lambda *a, **k: (None, []))
    plots = sys.modules["streamlit_app.components.mc_plots"]
    tables = sys.modules["streamlit_app.components.mc_tables"]
    monkeypatch.setattr(tables, "render_summary_table", lambda df: df)
    for name in (
        "render_sharpe_histogram",
        "render_fan_chart",
        "render_path_distribution_chart",
        "render_risk_return_chart",
        "render_box_plot",
        "render_cdf_plot",
    ):
        monkeypatch.setattr(plots, name, lambda *_a, **_k: go.Figure())
    monkeypatch.setattr(page, "_render_diagnostic_charts", lambda *a, **k: {})
    results = {"results_frame": frame, "nav_paths": nav}
    page._render_results(results, fold_selection="Fold 2")
    assert "Summary" in stub.subheaders
    assert "Charts" in stub.subheaders
    # no-nav warning branch
    stub.warning_messages.clear()
    page._render_results({"results_frame": frame}, fold_selection=None)
    assert "No NAV paths available for the selected fold." in stub.warning_messages


def _setup_render(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, DummyStreamlit]:
    page, stub = _load_page(monkeypatch)
    entry = ScenarioRegistryEntry(
        name="macro", path=Path("x.yml"), description="Macro scenario", tags=("macro",)
    )
    monkeypatch.setattr(page, "list_scenarios", lambda **_k: [entry])
    monkeypatch.setattr(page, "_session_runner_kwargs", lambda: ({}, None))
    return page, stub


def test_render_no_scenarios_and_load_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)
    monkeypatch.setattr(page, "list_scenarios", lambda **_k: [])
    page.render()
    assert stub.warning_messages == ["No scenarios available for the selected filters."]
    page2, stub2 = _setup_render(monkeypatch)

    def boom(_name: str) -> Any:
        raise RuntimeError("bad yaml")

    monkeypatch.setattr(page2, "load_scenario", boom)
    page2.render()
    assert stub2.error_messages == ["Unable to load the selected scenario."]
    assert stub2.write_calls[-1] == "bad yaml"
    page3, stub3 = _setup_render(monkeypatch)
    monkeypatch.setattr(page3, "load_scenario", lambda _n: None)
    page3.render()
    assert stub3.error_messages == []
    page4, stub4 = _setup_render(monkeypatch)
    bad = _scenario()
    bad.monte_carlo = None  # type: ignore[assignment]
    monkeypatch.setattr(page4, "load_scenario", lambda _n: bad)
    page4.render()
    assert stub4.error_messages == ["Scenario settings are not resolved."]


def test_render_session_error_and_fold_select(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _setup_render(monkeypatch)
    monkeypatch.setattr(page, "load_scenario", lambda _n: _scenario())
    monkeypatch.setattr(page, "_session_runner_kwargs", lambda: ({}, "detail-boom"))
    page.render()
    assert "Current Data/Model state could not be applied to Monte Carlo." in stub.warning_messages
    assert stub.write_calls[-1] == "detail-boom"
    page2, stub2 = _setup_render(monkeypatch)
    scen = _scenario()
    scen.enable_fold_runs = True
    scen.folds = {"n_folds": 2}
    monkeypatch.setattr(page2, "load_scenario", lambda _n: scen)
    page2.render()
    fold_calls = [c for c in stub2.selectbox_calls if c[0] == "Fold selection"]
    assert fold_calls == [("Fold selection", ["All folds", "Fold 1", "Fold 2"])]


def test_render_validate_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    # issues branch
    page, stub = _setup_render(monkeypatch)
    monkeypatch.setattr(page, "load_scenario", lambda _n: _scenario())

    class RunnerIssues:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def validate(self) -> list[str]:
            return ["issue-one"]

    monkeypatch.setattr(page, "MonteCarloRunner", RunnerIssues)
    stub.button_responses = [False, True]
    page.render()
    assert stub.warning_messages[-1] == "Validation completed with issues."
    assert stub.info_messages[-1] == "issue-one"
    # exception branch
    page2, stub2 = _setup_render(monkeypatch)
    monkeypatch.setattr(page2, "load_scenario", lambda _n: _scenario())

    class RunnerBoom:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def validate(self) -> list[str]:
            raise RuntimeError("validate-boom")

    monkeypatch.setattr(page2, "MonteCarloRunner", RunnerBoom)
    stub2.button_responses = [False, True]
    page2.render()
    assert stub2.error_messages[-1] == "Validation failed."
    assert stub2.write_calls[-1] == "validate-boom"
    # no validate entrypoint
    page3, stub3 = _setup_render(monkeypatch)
    monkeypatch.setattr(page3, "load_scenario", lambda _n: _scenario())

    class RunnerPlain:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

    monkeypatch.setattr(page3, "MonteCarloRunner", RunnerPlain)
    stub3.button_responses = [False, True]
    page3.render()
    assert stub3.info_messages[-1] == "Validation entrypoint is not available for this runner."


def test_render_run_failure_and_provenance_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _setup_render(monkeypatch)
    monkeypatch.setattr(page, "load_scenario", lambda _n: _scenario())

    class RunnerFail:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def run(self, **_k: Any) -> Any:
            raise RuntimeError("run-boom")

    monkeypatch.setattr(page, "MonteCarloRunner", RunnerFail)
    stub.button_responses = [True, False]
    stub.session_state[page.MC_RESULTS_KEY] = {"results_frame": pd.DataFrame({"a": [1]})}
    captured: dict[str, Any] = {}
    monkeypatch.setattr(page, "_render_results", lambda *a, **k: captured.update(kwargs=k))
    page.render()
    assert stub.error_messages[-1] == "Simulation failed."
    assert stub.session_state[page.MC_LAST_ERROR_KEY] == "run-boom"
    assert stub.write_calls[-1] == "run-boom"
    # provenance fallback: stored results without provenance string uses local provenance
    assert captured["kwargs"]["provenance"].startswith("Monte Carlo provenance:")


# --- mc_plots ---------------------------------------------------------------


class _StubSt:
    def __init__(self) -> None:
        self.warning_messages: list[str] = []
        self.plotly_calls: list[Any] = []

    def warning(self, message: Any) -> None:
        self.warning_messages.append(str(message))

    def plotly_chart(self, *args: Any, **kwargs: Any) -> None:
        self.plotly_calls.append((args, kwargs))


def test_mc_extract_and_metric_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"strategy": ["A"], "sharpe": [1.0]})

    class Holder:
        results_frame = df

    assert mc_plots._extract_results_frame(Holder()) is df
    assert mc_plots._extract_results_frame({"results": df}) is df
    assert mc_plots._extract_results_frame({"frame": df}) is df
    assert mc_plots._extract_results_frame(Holder()).shape == (1, 2)
    assert mc_plots._extract_results_frame(object()).empty
    assert mc_plots._metric_frame(pd.DataFrame(), None).columns.tolist() == ["Strategy", "Value"]
    no_strat = pd.DataFrame({"sharpe": [1.0]})
    assert mc_plots._metric_frame(no_strat, None).empty
    assert mc_plots._metric_frame(df, "missing_col").empty
    out = mc_plots._metric_frame(pd.DataFrame({"strategy": ["A", "B"], "sharpe": [1.5, "x"]}), None)
    assert out.to_dict("list") == {"Strategy": ["A"], "Value": [1.5]}


def test_mc_select_metric_name_branches() -> None:
    df = pd.DataFrame({"strategy": ["A"], "sharpe": [1.0]})
    assert mc_plots._select_metric_name(df, "missing") is None
    assert mc_plots._select_metric_name(df, "sharpe") == "sharpe"
    dd = pd.DataFrame({"strategy": ["A"], "max_dd": [0.1], "other": [2.0]})
    # canonical max_dd -> maxdd? alias maxdd matches "max_dd" canonical maxdd yes
    assert mc_plots._select_metric_name(dd, None) == "max_dd"
    term = pd.DataFrame({"strategy": ["A"], "terminal_wealth": [100.0], "other": [2.0]})
    assert mc_plots._select_metric_name(term, None) == "terminal_wealth"
    numeric = pd.DataFrame({"strategy": ["A"], "alpha": [3.0], "beta": [4.0]})
    assert mc_plots._select_metric_name(numeric, None) == "alpha"
    text_only = pd.DataFrame({"strategy": ["A"], "label": ["x"]})
    assert mc_plots._select_metric_name(text_only, None) is None


def test_mc_metric_warning_messages() -> None:
    assert (
        mc_plots._metric_warning_message(pd.DataFrame(), None, "Label")
        == "Label unavailable: results frame is missing."
    )
    assert (
        mc_plots._metric_warning_message(pd.DataFrame({"x": [1]}), None, "Label")
        == "Label unavailable: required column 'strategy' is missing."
    )
    df = pd.DataFrame({"strategy": ["A"], "label": ["x"]})
    assert (
        mc_plots._metric_warning_message(df, "sharpe", "Label")
        == "Label unavailable: required column 'sharpe' is missing."
    )
    assert (
        mc_plots._metric_warning_message(df, None, "Label")
        == "Label unavailable: no numeric metric columns were found."
    )
    nan_df = pd.DataFrame({"strategy": ["A"], "sharpe": ["nope"]})
    assert (
        mc_plots._metric_warning_message(nan_df, None, "Label")
        == "Label unavailable: column 'sharpe' has no numeric values."
    )
    good = pd.DataFrame({"strategy": ["A"], "sharpe": [1.0]})
    assert mc_plots._metric_warning_message(good, None, "Label") is None
    assert (
        mc_plots._metric_warning_message_for_aliases(pd.DataFrame(), ("sharpe",), "L")
        == "L unavailable: results frame is missing."
    )
    assert mc_plots._metric_warning_message_for_aliases(good, ("sharpe",), "L") is None


def test_mc_nav_long_frame_variants() -> None:
    assert mc_plots._nav_long_frame(pd.DataFrame()).columns.tolist() == ["Date", "Path", "NAV"]
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    cols = pd.MultiIndex.from_tuples(
        [("p1", "NAV"), ("p2", "NAV"), ("p1", "other"), ("p2", "other")],
        names=["path", "asset"],
    )
    frame = pd.DataFrame(
        [[100.0, 200.0, 1.0, 2.0], [101.0, 201.0, 3.0, 4.0]], index=idx, columns=cols
    )
    out = mc_plots._nav_long_frame(frame)
    assert out["NAV"].tolist() == [100.0, 101.0, 200.0, 201.0]
    cols2 = pd.MultiIndex.from_tuples([("p1", "zzz"), ("p2", "zzz")], names=["path", "asset"])
    frame2 = pd.DataFrame([[10.0, 20.0], [11.0, 21.0]], index=idx, columns=cols2)
    assert mc_plots._nav_long_frame(frame2)["NAV"].tolist() == [10.0, 11.0, 20.0, 21.0]
    cols3 = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y")], names=["u", "v"])
    frame3 = pd.DataFrame([[5.0, 6.0], [7.0, 8.0]], index=idx, columns=cols3)
    out3 = mc_plots._nav_long_frame(frame3)
    assert sorted(out3["Path"].unique().tolist()) == ["a_x", "b_y"]
    many = pd.DataFrame({f"p{i}": [float(i), float(i)] for i in range(5)}, index=idx)
    out4 = mc_plots._nav_long_frame(many, max_paths=2)
    assert sorted(out4["Path"].unique().tolist()) == ["p0", "p1"]
    messy = pd.DataFrame({"p": [100.0, None]}, index=["2024-01-01", "not-a-date"])
    out5 = mc_plots._nav_long_frame(messy)
    assert out5["NAV"].tolist() == [100.0]
    assert str(out5["Date"].iloc[0].date()) == "2024-01-01"


def test_mc_nav_wide_frame_variants() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    cols = pd.MultiIndex.from_tuples([("p1", "wealth"), ("p2", "wealth")], names=["path", "asset"])
    frame = pd.DataFrame([[100.0, 200.0], [101.0, 201.0]], index=idx, columns=cols)
    out = mc_plots._nav_wide_frame(frame)
    assert out.values.tolist() == [[100.0, 200.0], [101.0, 201.0]]
    cols2 = pd.MultiIndex.from_tuples([("p1", "zzz"), ("p2", "zzz")], names=["path", "asset"])
    frame2 = pd.DataFrame([[10.0, 20.0], [11.0, 21.0]], index=idx, columns=cols2)
    assert mc_plots._nav_wide_frame(frame2).values.tolist() == [[10.0, 20.0], [11.0, 21.0]]
    cols3 = pd.MultiIndex.from_tuples([("a", "x")], names=["u", "v"])
    frame3 = pd.DataFrame([[5.0], [7.0]], index=idx, columns=cols3)
    assert list(mc_plots._nav_wide_frame(frame3).columns) == ["a_x"]
    many = pd.DataFrame({f"p{i}": [1.0, 2.0] for i in range(4)}, index=idx)
    assert mc_plots._nav_wide_frame(many, max_paths=2).shape == (2, 2)


def test_mc_render_warnings_and_empty_charts(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubSt()
    monkeypatch.setattr(mc_plots, "st", stub)
    empty = pd.DataFrame()
    fig = mc_plots.render_sharpe_histogram(empty)
    assert len(fig.data) == 0
    assert stub.warning_messages[-1] == (
        "Sharpe distribution chart unavailable: results frame is missing."
    )
    fig2 = mc_plots.render_box_plot(empty)
    assert len(fig2.data) == 0
    assert (
        stub.warning_messages[-1]
        == "Metric comparison chart unavailable: results frame is missing."
    )
    fig3 = mc_plots.render_cdf_plot(empty)
    assert len(fig3.data) == 0
    assert stub.warning_messages[-1] == "Outcome CDF chart unavailable: results frame is missing."
    assert len(mc_plots.sharpe_histogram(empty).data) == 0
    assert len(mc_plots.box_plot(empty).data) == 0
    assert len(mc_plots.cdf_plot(empty).data) == 0
    assert len(mc_plots.fan_chart("not-a-frame").data) == 0  # type: ignore[arg-type]
    no_term = pd.DataFrame({"strategy": ["A"], "notes": ["x"]})
    assert len(mc_plots.path_distribution_chart(no_term).data) == 0


def test_mc_risk_return_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubSt()
    monkeypatch.setattr(mc_plots, "st", stub)
    assert len(mc_plots.risk_return_chart(pd.DataFrame({"x": [1]})).data) == 0
    # numeric fallback for return metric (no alias columns)
    df = pd.DataFrame(
        {
            "strategy": ["A", "A", "B", "B"],
            "alpha": [1.0, 2.0, 3.0, 4.0],
            "beta": [0.1, 0.2, 0.3, 0.4],
        }
    )
    fig = mc_plots.risk_return_chart(df)
    assert len(fig.data) >= 1
    assert fig.layout.xaxis.title.text == "Beta"
    assert fig.layout.yaxis.title.text == "Alpha"
    # dropna empties frame
    bad = pd.DataFrame({"strategy": ["A"], "terminal_wealth": [None], "volatility": [None]})
    assert len(mc_plots.risk_return_chart(bad).data) == 0
    # render: first warning None (return alias present) then risk warning also None -> fallback msg
    stub.warning_messages.clear()
    single = pd.DataFrame({"strategy": ["A", "B"], "terminal_wealth": [100.0, 110.0]})
    fig2 = mc_plots.render_risk_return_chart(single)
    assert len(fig2.data) == 0
    assert stub.warning_messages[-1] == "Risk-return chart unavailable: no data to display."


# --- nl_operation_viewer ----------------------------------------------------


def test_nl_list_and_load_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module, _st = _load_nl(monkeypatch)
    assert module._list_log_files(tmp_path / "missing") == []
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "nl_ops_a.jsonl").write_text("", encoding="utf-8")
    assert [p.name for p in module._list_log_files(log_dir)] == ["nl_ops_a.jsonl"]
    log = tmp_path / "nl_ops_2026.jsonl"
    good = _entry()
    log.write_text(
        json.dumps({"bad": 1}) + "\n" + json.dumps(good.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    entries = module._load_log_entries(log)
    assert len(entries) == 1
    assert entries[0][0] == 2
    assert entries[0][1].request_id == "req-1"


def test_nl_contains_sensitive_and_redact_url(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _st = _load_nl(monkeypatch)
    assert module._redact_text(None) == ""
    assert module._redact_text("") == ""
    assert module._contains_sensitive("x", key="api_key") is True
    assert module._contains_sensitive({"nested": {"token": "abc"}}) is True
    assert module._contains_sensitive(["plain", {"password": "x"}]) is True
    assert module._contains_sensitive("sk-" + "A" * 25) is True
    assert module._contains_sensitive("hello world") is False
    assert module._contains_sensitive(42) is False
    assert module._redact_url("http://[::1") == "http://[::1"
    assert module._redact_url("not a url") == "not a url"
    assert (
        module._redact_url("https://user:pw@example.com/path?q=secret")
        == "https://example.com/path"
    )
    assert (
        module._entry_has_sensitive_prompt(_entry(prompt_template="token sk-" + "B" * 25)) is True
    )
    assert module._entry_has_sensitive_prompt(_entry(prompt_variables={"api_key": "x"})) is True


def test_nl_sanitize_patch_and_format_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _st = _load_nl(monkeypatch)
    patch = ConfigPatch(
        summary="Adjust things",
        operations=[PatchOperation(op="set", path="risk.vol", value=0.2)],
    )
    payload = module._sanitize_patch_payload(patch)
    raw_patch = MagicMock()
    raw_patch.model_dump.return_value = {"summary": "ok", "operations": ["raw-op"]}
    assert module._sanitize_patch_payload(raw_patch)["operations"] == ["raw-op"]
    # inject non-dict operation after dump to cover guard
    payload2 = {"operations": ["raw-op", {"op": "set", "path": "a.b", "value": 3}]}
    assert module._build_diff_summary(payload2) == ["set a.b -> 3"]
    assert payload["summary"] == "Adjust things"

    class BadTs:
        def isoformat(self, **_k: Any) -> str:
            raise RuntimeError("bad ts")

    fake_entry = SimpleNamespace(timestamp=BadTs(), duration_ms="oops")
    assert module._format_timestamp(fake_entry) == str(fake_entry.timestamp)
    assert module._format_duration(fake_entry) == "—"
    assert module._format_duration(_entry(duration_ms=12.4)) == "12ms"
    assert module._build_entry_id(7) == "7"
    assert module._build_entry_id(7, log_path=Path("/tmp/nl_ops_x.jsonl")) == "nl_ops_x_7"


def test_nl_render_patch_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    module, st_stub = _load_nl(monkeypatch)
    patch = ConfigPatch(
        summary="My summary",
        needs_review=True,
        risk_flags=[RiskFlag.INCREASES_LEVERAGE],
        operations=[PatchOperation(op="set", path="risk.vol", value=0.2)],
    )
    entry = _entry(parsed_patch=patch)
    module._render_patch_summary(entry)
    markdowns = [str(c) for c in st_stub.markdown.call_args_list]
    assert "**Patch summary**" in markdowns[0]
    captions = [str(c) for c in st_stub.caption.call_args_list]
    assert any("My summary" in c for c in captions)
    assert any("Needs review: Yes" in c for c in captions)
    assert any("Risk flags:" in c and "INCREASES_LEVERAGE" in c for c in captions)
    codes = [str(c) for c in st_stub.code.call_args_list]
    assert any("set risk.vol" in c for c in codes)


def test_nl_render_replay_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    module, st_stub = _load_nl(monkeypatch)
    st_stub.selectbox.return_value = "openai"
    st_stub.text_input.return_value = "gpt-x"
    st_stub.slider.return_value = 0.1
    monkeypatch.setattr(module, "browser_llm_runtime", lambda: True)
    monkeypatch.setattr(module, "llm_provider_options", lambda: ["openai"])
    monkeypatch.setattr(module, "resolve_api_key_input", lambda v: "key-123")
    st_stub.text_input.side_effect = ["key-123", "https://example.com", "gpt-x"]

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("provider down sk-" + "C" * 25)

    monkeypatch.setattr(module, "replay_nl_entry", boom)
    entry = _entry(prompt_template="Authorization: Bearer abcdefghij")
    module._render_replay(entry, entry_id="e1", run_replay=True)
    warnings = [str(c) for c in st_stub.warning.call_args_list]
    assert any("Sensitive data detected" in w for w in warnings)
    errors = [str(c) for c in st_stub.error.call_args_list]
    assert any("Replay failed" in e for e in errors)


def test_nl_render_viewer_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module, st_stub = _load_nl(monkeypatch)
    # no readable entries
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "nl_ops_a.jsonl").write_text("", encoding="utf-8")
    st_stub.selectbox.return_value = "nl_ops_a.jsonl"
    module.render_nl_operation_viewer(base_dir=log_dir)
    infos = [str(c) for c in st_stub.info.call_args_list]
    assert any("no readable entries" in i for i in infos)
    # empty dir -> no log files
    st_stub.info.reset_mock()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    module.render_nl_operation_viewer(base_dir=empty_dir)
    infos2 = [str(c) for c in st_stub.info.call_args_list]
    assert any("No NL operation logs found" in i for i in infos2)
    # entry with error, trace, variables
    st_stub.selectbox.side_effect = ["nl_ops_b.jsonl", "label-1", "openai"]
    st_stub.button.return_value = False
    entry = _entry(
        error="failed sk-" + "D" * 25,
        trace_url="https://user:pw@example.com/trace?q=1",
        prompt_variables={"fund": "A"},
        model_output="hello",
    )
    log2 = log_dir / "nl_ops_b.jsonl"
    log2.write_text(json.dumps(entry.model_dump(mode="json")) + "\n", encoding="utf-8")
    # _build_choices label must match selectbox second return; patch to return known label
    orig_choices = module._build_choices

    def fake_choices(entries: Any) -> Any:
        choices = orig_choices(entries)
        return [type(choices[0])(label="label-1", entry=choices[0].entry, index=choices[0].index)]

    monkeypatch.setattr(module, "_build_choices", fake_choices)
    module.render_nl_operation_viewer(base_dir=log_dir)
    warnings = [str(c) for c in st_stub.warning.call_args_list]
    assert any("Error:" in w for w in warnings)
    captions = [str(c) for c in st_stub.caption.call_args_list]
    assert any("Trace URL: https://example.com/trace" in c for c in captions)
    markdowns = [str(c) for c in st_stub.markdown.call_args_list]
    assert any("Prompt variables" in m for m in markdowns)


# --- demo_runner ------------------------------------------------------------


class _FakeSt:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.errors: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)


def _demo_df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=48, freq="ME")
    return pd.DataFrame(
        {"FundA": [0.01] * 48, "FundB": [0.02] * 48, "SPX": [0.015] * 48}, index=idx
    )


def test_demo_run_with_overrides_happy_and_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _demo_df()
    meta = demo_runner.SchemaMeta(n_rows=48)
    fake_st = _FakeSt()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(demo_runner, "_load_demo_returns", lambda: (df, meta))
    monkeypatch.setattr(
        demo_runner,
        "_load_preset",
        lambda _n: {
            "metrics": {"sharpe": 1.0},
            "lookback_periods": 12,
            "portfolio": {"max_weight": 0.5, "cooldown_periods": 3},
            "risk_target": 0.12,
        },
    )
    monkeypatch.setattr(demo_runner, "run_simulation", lambda *_a, **_k: "RESULT")
    monkeypatch.setattr(demo_runner, "_update_session_state", lambda *a: captured.update(update=a))
    monkeypatch.setattr(
        demo_runner, "_store_demo_result_state", lambda *a: captured.update(store=a)
    )
    ok = demo_runner.run_demo_with_overrides(
        "Balanced", overrides={"portfolio": {"max_weight": 0.4}}, st_module=fake_st
    )
    assert ok is True
    first_policy = captured["store"][1].sim_config["policy"]
    assert first_policy["max_weight"] == 0.4
    assert first_policy["cooldown_periods"] == 3
    ok_lb = demo_runner.run_demo_with_overrides(
        "Balanced", overrides={"lookback_periods": 24}, st_module=fake_st
    )
    assert ok_lb is True
    assert captured["store"][1].sim_config["lookback_periods"] == 24
    assert captured["store"][3] == "RESULT"
    assert captured["store"][1].benchmark == "SPX"
    assert captured["store"][1].sim_config["portfolio"]["weighting"]["name"] == "equal"
    # load failure
    fake_st2 = _FakeSt()

    def boom_load() -> Any:
        raise FileNotFoundError("missing demo")

    monkeypatch.setattr(demo_runner, "_load_demo_returns", boom_load)
    assert demo_runner.run_demo_with_overrides(st_module=fake_st2) is False
    assert fake_st2.errors == ["Unable to load demo returns data: missing demo"]
    # simulation failure
    fake_st3 = _FakeSt()
    monkeypatch.setattr(demo_runner, "_load_demo_returns", lambda: (df, meta))

    def boom_run(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("sim-boom")

    monkeypatch.setattr(demo_runner, "run_simulation", boom_run)
    assert demo_runner.run_demo_with_overrides(st_module=fake_st3) is False
    assert fake_st3.errors == ["Demo simulation failed: sim-boom"]


def test_demo_run_with_overrides_default_st(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSt()
    monkeypatch.setitem(sys.modules, "streamlit", fake)  # type: ignore[arg-type]
    df = _demo_df()
    meta = demo_runner.SchemaMeta(n_rows=48)
    monkeypatch.setattr(demo_runner, "_load_demo_returns", lambda: (df, meta))
    monkeypatch.setattr(demo_runner, "_load_preset", lambda _n: {})
    monkeypatch.setattr(demo_runner, "run_simulation", lambda *_a, **_k: "R2")
    monkeypatch.setattr(demo_runner, "_update_session_state", lambda *_a: None)
    monkeypatch.setattr(demo_runner, "_store_demo_result_state", lambda *_a: None)
    assert demo_runner.run_demo_with_overrides() is True


def test_demo_analysis_run_key_fallbacks() -> None:
    model = {"a": 1}
    key1 = demo_runner._analysis_run_key(
        {
            "data_fingerprint": "fp",
            "analysis_fund_columns": "notalist",
            "fund_columns": ["F1", "F2"],
        },
        model,
        None,
    )
    key2 = demo_runner._analysis_run_key(
        {"data_fingerprint": "fp", "analysis_fund_columns": ["F1"]}, model, None
    )
    assert key1 != key2
    key_expected = demo_runner._analysis_run_key(
        {"data_fingerprint": "fp", "analysis_fund_columns": ["F1", "F2"]}, model, None
    )
    assert key1 == key_expected
    key3 = demo_runner._analysis_run_key(
        {"data_fingerprint": "fp", "fund_columns": "bad"}, model, None
    )
    key4 = demo_runner._analysis_run_key({"data_fingerprint": "fp"}, model, None)
    assert key3 == key4


# --- state ------------------------------------------------------------------


@pytest.fixture
def state_store(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    store: dict[str, Any] = {}
    monkeypatch.setattr(state_mod.st, "session_state", store)
    return store


def test_state_config_wrappers(state_store: dict[str, Any]) -> None:
    state_store["saved_config_wrappers"] = "broken"
    assert state_mod.get_saved_config_wrappers() == {}
    assert state_store["saved_config_wrappers"] == {}
    with pytest.raises(
        ValueError, match="A non-empty name is required to save a model configuration."
    ):
        state_mod.save_config_wrapper("  ", {"a": 1})
    state_mod.save_config_wrapper("  MyCfg ", {"x": [1, 2]})
    assert state_store["saved_config_wrappers"] == {"MyCfg": {"x": [1, 2]}}
    assert state_mod.load_saved_config_wrapper("missing") is None
    loaded = state_mod.load_saved_config_wrapper("MyCfg")
    assert loaded == {"x": [1, 2]}
    loaded["x"].append(3)
    assert state_store["saved_config_wrappers"]["MyCfg"] == {"x": [1, 2]}


def test_state_rename_and_import_export(state_store: dict[str, Any]) -> None:
    state_mod.save_model_state("Old", {"a": 1})
    state_mod.save_config_wrapper("Old", {"model_state": {"a": 1}, "extra": 2})
    with pytest.raises(ValueError, match="Provide a new name to rename the configuration."):
        state_mod.rename_saved_model_state("Old", "   ")
    state_mod.rename_saved_model_state("Old", "New")
    assert state_mod.load_saved_model_state("New") == {"a": 1}
    assert state_mod.load_saved_config_wrapper("New") == {"model_state": {"a": 1}, "extra": 2}
    assert state_mod.export_model_state("New") == '{"extra": 2, "model_state": {"a": 1}}'
    with pytest.raises(ValueError, match="Provide a name for the imported configuration."):
        state_mod.import_model_state(" ", "{}")
    out = state_mod.import_model_state("Imported", '{"model_state": {"b": 2}, "meta": 1}')
    assert out == {"b": 2}
    assert state_mod.load_saved_config_wrapper("Imported") == {"meta": 1, "model_state": {"b": 2}}


def test_state_values_equal_and_stringify() -> None:
    assert state_mod._stringify_value({1: "a", "b": 2}) in (
        '{"1": "a", "b": 2}',
        "{1: 'a', 'b': 2}",
    )
    assert state_mod._values_equal({"a": 1}, {"a": 1, "b": 2}, 1e-9) is False
    assert state_mod._values_equal({"a": [1, 2]}, {"a": [1, 2]}, 1e-9) is True
    assert state_mod._values_equal([1, 2], [1], 1e-9) is False
    assert state_mod._values_equal([1, 2], [1, 3], 1e-9) is False
    assert state_mod._values_equal("same", "same", 1e-9) is True
    assert state_mod._values_equal(1.0, 1, 1e-9) is True


def test_state_diff_sequence_branches() -> None:
    diffs = state_mod.diff_model_states({"a": [1, 2]}, {"a": (1, 2, 3)})
    assert len(diffs) == 1
    assert diffs[0].path == "a"
    assert diffs[0].type_changed is True
    assert diffs[0].change_type == "changed"
    added = state_mod.diff_model_states({"a": [1]}, {"a": [1, 9]})
    assert [(d.path, d.change_type, d.right) for d in added] == [("a[1]", "added", 9)]
    removed = state_mod.diff_model_states({"a": [1, 9]}, {"a": [1]})
    assert [(d.path, d.change_type, d.left) for d in removed] == [("a[1]", "removed", 9)]
    assert state_mod.format_model_state_diff([]) == "No differences found."
    assert state_mod.format_model_state_diff(added) == "+ a[1]: (B) 9"
    removed_text = state_mod.format_model_state_diff(removed)
    assert removed_text == "- a[1]: (A) 9"
