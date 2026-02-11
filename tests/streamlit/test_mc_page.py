from __future__ import annotations

import importlib
import sys
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pandas as pd
import pytest

from trend_analysis.monte_carlo.registry import ScenarioRegistryEntry
from trend_analysis.monte_carlo.scenario import MonteCarloScenario, MonteCarloSettings


class _Context:
    def __init__(self, stub: "DummyStreamlit") -> None:
        self._stub = stub

    def __enter__(self) -> "DummyStreamlit":
        return self._stub

    def __exit__(self, *_exc: object) -> bool:
        return False


class _Placeholder:
    def __init__(self, stub: "DummyStreamlit") -> None:
        self._stub = stub

    def progress(self, value: float, text: str | None = None) -> "_ProgressBar":
        bar = _ProgressBar(self._stub)
        bar.progress(value, text=text)
        return bar

    def metric(self, label: str, value: str) -> None:
        self._stub.metric_calls.append((label, value))

    def empty(self) -> None:
        return None


class _ProgressBar:
    def __init__(self, stub: "DummyStreamlit") -> None:
        self._stub = stub

    def progress(self, value: float, text: str | None = None) -> None:
        self._stub.progress_calls.append((value, text))


class DummyStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.button_responses: list[bool] = []
        self.selectbox_returns: list[str] = []
        self.multiselect_returns: list[list[str]] = []
        self.slider_returns: list[int] = []
        self.text_input_returns: list[str] = []
        self.downloads: list[dict[str, Any]] = []
        self.plotly_calls: list[dict[str, Any]] = []
        self.dataframes: list[pd.DataFrame] = []
        self.error_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.info_messages: list[str] = []
        self.caption_messages: list[str] = []
        self.write_messages: list[str] = []
        self.progress_calls: list[tuple[float, str | None]] = []
        self.metric_calls: list[tuple[str, str]] = []
        self.selectbox_calls: list[tuple[str, list[str]]] = []
        self.multiselect_calls: list[tuple[str, list[str]]] = []
        self.slider_calls: list[tuple[str, dict[str, Any]]] = []
        self.tabs_calls: list[list[str]] = []

    def title(self, _text: str) -> None:
        return None

    def subheader(self, _text: str) -> None:
        return None

    def write(self, message: str) -> None:
        self.write_messages.append(message)

    def caption(self, message: str) -> None:
        self.caption_messages.append(message)

    def error(self, message: str) -> None:
        self.error_messages.append(message)

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)

    def info(self, message: str) -> None:
        self.info_messages.append(message)

    def success(self, _message: str) -> None:
        return None

    def divider(self) -> None:
        return None

    def altair_chart(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def plotly_chart(self, *_args: Any, **kwargs: Any) -> None:
        self.plotly_calls.append(dict(kwargs))

    def multiselect(self, label: str, options: list[str], **_kwargs: Any) -> list[str]:
        self.multiselect_calls.append((label, options))
        if self.multiselect_returns:
            return self.multiselect_returns.pop(0)
        return []

    def selectbox(self, label: str, options: list[str], index: int = 0, **_kwargs: Any) -> str:
        self.selectbox_calls.append((label, list(options)))
        if self.selectbox_returns:
            return self.selectbox_returns.pop(0)
        return options[index]

    def slider(self, label: str, **kwargs: Any) -> int:
        self.slider_calls.append((label, dict(kwargs)))
        if self.slider_returns:
            return int(self.slider_returns.pop(0))
        return int(kwargs.get("value", 0))

    def text_input(self, _label: str, value: str = "", **_kwargs: Any) -> str:
        if self.text_input_returns:
            return self.text_input_returns.pop(0)
        return value

    def button(self, *_args: Any, **_kwargs: Any) -> bool:
        if self.button_responses:
            return self.button_responses.pop(0)
        return False

    def columns(self, count: int | list[int]) -> list[_Context]:
        size = count if isinstance(count, int) else len(count)
        return [_Context(self) for _ in range(size)]

    def empty(self) -> _Placeholder:
        return _Placeholder(self)

    def dataframe(self, df: pd.DataFrame, **_kwargs: Any) -> None:
        self.dataframes.append(df)

    def download_button(self, **kwargs: Any) -> None:
        self.downloads.append(kwargs)

    def expander(self, *_args: Any, **_kwargs: Any) -> _Context:
        return _Context(self)

    def progress(self, value: float, text: str | None = None) -> _ProgressBar:
        bar = _ProgressBar(self)
        bar.progress(value, text=text)
        return bar

    def tabs(self, labels: list[str]) -> list[_Context]:
        self.tabs_calls.append(list(labels))
        return [_Context(self) for _ in labels]

    def cache_data(self, *_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return _decorator


@dataclass
class DummyResults:
    results_frame: pd.DataFrame
    nav_paths: pd.DataFrame | None = None
    metadata: dict[str, Any] | None = None


def _altair_stub() -> ModuleType:
    class _Chart:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def mark_line(self, **_kwargs: Any) -> "_Chart":
            return self

        def mark_bar(self, **_kwargs: Any) -> "_Chart":
            return self

        def mark_boxplot(self, **_kwargs: Any) -> "_Chart":
            return self

        def encode(self, **_kwargs: Any) -> "_Chart":
            return self

        def properties(self, **_kwargs: Any) -> "_Chart":
            return self

    module = ModuleType("altair")
    module.Chart = _Chart
    module.X = lambda *args, **_kwargs: args
    module.Y = lambda *args, **_kwargs: args
    module.Axis = lambda *args, **_kwargs: args
    module.Tooltip = lambda *args, **_kwargs: args
    module.Color = lambda *args, **_kwargs: args
    module.Scale = lambda *args, **_kwargs: args
    module.Bin = lambda *args, **_kwargs: args
    return module


def _install_streamlit_stub(monkeypatch: pytest.MonkeyPatch) -> DummyStreamlit:
    stub = DummyStreamlit()
    module = ModuleType("streamlit")
    for name in dir(stub):
        if name.startswith("__"):
            continue
        setattr(module, name, getattr(stub, name))
    module.session_state = stub.session_state

    monkeypatch.setitem(sys.modules, "streamlit", module)
    monkeypatch.setitem(sys.modules, "altair", _altair_stub())
    return stub


def _load_page(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, DummyStreamlit]:
    stub = _install_streamlit_stub(monkeypatch)
    importlib.reload(importlib.import_module("streamlit_app.components.mc_tables"))
    importlib.reload(importlib.import_module("streamlit_app.components.mc_plots"))
    page = importlib.reload(importlib.import_module("streamlit_app.pages.monte_carlo"))
    return page, stub


def _make_scenario(name: str) -> MonteCarloScenario:
    settings = MonteCarloSettings(
        mode="mixture",
        n_paths=500,
        horizon_years=10,
        frequency="M",
        seed=42,
        jobs=4,
    )
    return MonteCarloScenario(
        name=name,
        description="Scenario description",
        version="1.0",
        base_config=Path("config/defaults.yml"),
        monte_carlo=settings,
    )


def _sample_results() -> DummyResults:
    results_frame = pd.DataFrame(
        {
            "strategy": ["A", "A", "B", "B"],
            "path_id": [1, 2, 1, 2],
            "sharpe": [1.2, 1.1, 0.9, 0.8],
            "max_drawdown": [-0.2, -0.25, -0.3, -0.28],
            "terminal_wealth": [120.0, 118.0, 110.0, 112.0],
        }
    )
    nav_paths = pd.DataFrame(
        {"path_1": [100.0, 102.0, 105.0], "path_2": [100.0, 99.0, 101.0]},
        index=pd.date_range("2023-01-01", periods=3, freq="D"),
    )
    return DummyResults(results_frame=results_frame, nav_paths=nav_paths)


def test_scenario_picker_and_tag_filtering(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)

    scenarios = [
        ScenarioRegistryEntry(
            name="macro",
            path=Path("config/scenarios/monte_carlo/example.yml"),
            description="Macro scenario",
            tags=("macro",),
        ),
        ScenarioRegistryEntry(
            name="credit",
            path=Path("config/scenarios/monte_carlo/example.yml"),
            description="Credit scenario",
            tags=("credit",),
        ),
    ]

    calls: list[dict[str, Any]] = []

    def fake_list_scenarios(*, tags: list[str] | None = None) -> list[ScenarioRegistryEntry]:
        calls.append({"tags": tags})
        if tags:
            return [entry for entry in scenarios if set(tags) & set(entry.tags)]
        return scenarios

    monkeypatch.setattr(page, "list_scenarios", fake_list_scenarios)
    monkeypatch.setattr(page, "load_scenario", lambda name: _make_scenario(name))

    stub.multiselect_returns = [["macro"]]

    page.render()

    assert stub.multiselect_calls
    assert stub.multiselect_calls[0][1] == ["credit", "macro"]
    assert calls[0]["tags"] is None
    assert calls[1]["tags"] == ["macro"]
    assert stub.selectbox_calls[0][1] == ["macro"]


def test_runtime_override_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)

    monkeypatch.setattr(
        page,
        "list_scenarios",
        lambda **_kwargs: [
            ScenarioRegistryEntry(
                name="macro",
                path=Path("config/scenarios/monte_carlo/example.yml"),
                description="Macro scenario",
                tags=("macro",),
            )
        ],
    )
    monkeypatch.setattr(page, "load_scenario", lambda name: _make_scenario(name))

    stub.slider_returns = [50, 3, 20]
    stub.text_input_returns = ["abc"]

    page.render()

    assert any("Number of paths" in message for message in stub.error_messages)
    assert any("Horizon years" in message for message in stub.error_messages)
    assert any("Parallel jobs" in message for message in stub.error_messages)
    assert any("Random seed" in message for message in stub.error_messages)


def test_override_slider_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)

    monkeypatch.setattr(
        page,
        "list_scenarios",
        lambda **_kwargs: [
            ScenarioRegistryEntry(
                name="macro",
                path=Path("config/scenarios/monte_carlo/example.yml"),
                description="Macro scenario",
                tags=("macro",),
            )
        ],
    )
    monkeypatch.setattr(page, "load_scenario", lambda name: _make_scenario(name))

    page.render()

    slider_map = {label: kwargs for label, kwargs in stub.slider_calls}
    assert slider_map["Number of paths"]["min_value"] == 100
    assert slider_map["Number of paths"]["max_value"] == 5000
    assert slider_map["Horizon (years)"]["min_value"] == 5
    assert slider_map["Horizon (years)"]["max_value"] == 50
    assert slider_map["Parallel jobs"]["min_value"] == 1
    assert slider_map["Parallel jobs"]["max_value"] == 16


def test_run_button_flow_with_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)

    monkeypatch.setattr(
        page,
        "list_scenarios",
        lambda **_kwargs: [
            ScenarioRegistryEntry(
                name="macro",
                path=Path("config/scenarios/monte_carlo/example.yml"),
                description="Macro scenario",
                tags=("macro",),
            )
        ],
    )
    monkeypatch.setattr(page, "load_scenario", lambda name: _make_scenario(name))

    captured: dict[str, Any] = {}

    class FakeRunner:
        def __init__(self, scenario: MonteCarloScenario) -> None:
            captured["scenario"] = scenario

        def run(
            self,
            *,
            progress_callback: Callable[[dict[str, Any]], None] | None = None,
            jobs: int | None = None,
        ):
            captured["jobs"] = jobs
            if progress_callback:
                progress_callback({"completed": 1, "total": 2, "path_id": 0})
                progress_callback({"completed": 2, "total": 2, "path_id": 1})
            return _sample_results()

    monkeypatch.setattr(page, "MonteCarloRunner", FakeRunner)

    stub.button_responses = [True, False]
    stub.slider_returns = [200, 15, 4]
    stub.text_input_returns = ["123"]

    page.render()

    scenario = captured["scenario"]
    settings = scenario.monte_carlo
    assert settings.n_paths == 200
    assert settings.horizon_years == 15.0
    assert settings.seed == 123
    assert settings.jobs == 4
    assert captured["jobs"] == 4
    assert stub.progress_calls
    assert len(stub.plotly_calls) >= 7
    assert ["Core", "Diagnostics"] in stub.tabs_calls
    assert all(call.get("use_container_width") is True for call in stub.plotly_calls)
    assert stub.dataframes

    columns = list(stub.dataframes[0].columns)
    assert columns == [
        "Strategy",
        "Sharpe (median)",
        "Sharpe (5th)",
        "Max DD (median)",
        "Max DD (5th)",
        "Terminal Wealth",
    ]


def test_validate_button_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)

    monkeypatch.setattr(
        page,
        "list_scenarios",
        lambda **_kwargs: [
            ScenarioRegistryEntry(
                name="macro",
                path=Path("config/scenarios/monte_carlo/example.yml"),
                description="Macro scenario",
                tags=("macro",),
            )
        ],
    )
    monkeypatch.setattr(page, "load_scenario", lambda name: _make_scenario(name))

    class FakeRunner:
        def __init__(self, scenario: MonteCarloScenario) -> None:
            self.scenario = scenario
            self.validate_called = False

        def validate(self):
            self.validate_called = True
            return []

    runner = FakeRunner(_make_scenario("macro"))

    def fake_runner_factory(_scenario: MonteCarloScenario) -> FakeRunner:
        return runner

    monkeypatch.setattr(page, "MonteCarloRunner", fake_runner_factory)

    stub.button_responses = [False, True]

    page.render()

    assert runner.validate_called


def test_cancel_button_sets_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)

    monkeypatch.setattr(
        page,
        "list_scenarios",
        lambda **_kwargs: [
            ScenarioRegistryEntry(
                name="macro",
                path=Path("config/scenarios/monte_carlo/example.yml"),
                description="Macro scenario",
                tags=("macro",),
            )
        ],
    )
    monkeypatch.setattr(page, "load_scenario", lambda name: _make_scenario(name))

    stub.session_state[page.MC_RUNNING_KEY] = True
    stub.session_state[page.MC_CANCEL_KEY] = False
    stub.button_responses = [True]

    page.render()

    assert stub.session_state[page.MC_CANCEL_KEY] is True


def test_cancel_interrupts_run(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)

    monkeypatch.setattr(
        page,
        "list_scenarios",
        lambda **_kwargs: [
            ScenarioRegistryEntry(
                name="macro",
                path=Path("config/scenarios/monte_carlo/example.yml"),
                description="Macro scenario",
                tags=("macro",),
            )
        ],
    )
    monkeypatch.setattr(page, "load_scenario", lambda name: _make_scenario(name))

    class FakeRunner:
        def __init__(self, scenario: MonteCarloScenario) -> None:
            self.scenario = scenario

        def run(
            self,
            *,
            progress_callback: Callable[[dict[str, Any]], None] | None = None,
            jobs: int | None = None,
        ):
            stub.session_state[page.MC_CANCEL_KEY] = True
            if progress_callback:
                progress_callback({"completed": 1, "total": 2, "path_id": 0})
            return _sample_results()

    monkeypatch.setattr(page, "MonteCarloRunner", FakeRunner)

    stub.button_responses = [True, False]

    page.render()

    assert any("cancel" in message.lower() for message in stub.warning_messages)


def test_progress_updates_throttled(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)

    bar = _ProgressBar(stub)
    elapsed_slot = _Placeholder(stub)
    eta_slot = _Placeholder(stub)

    times = iter([0.0, 0.5, 1.1, 2.2])

    monkeypatch.setattr(page, "monotonic", lambda: next(times))

    callback = page._progress_callback_factory(
        progress_bar=bar,
        elapsed_slot=elapsed_slot,
        eta_slot=eta_slot,
        start_time=0.0,
    )

    callback({"completed": 1, "total": 4, "path_id": 0})
    callback({"completed": 2, "total": 4, "path_id": 1})
    callback({"completed": 3, "total": 4, "path_id": 2})
    callback({"completed": 4, "total": 4, "path_id": 3})

    assert len(stub.progress_calls) >= 2


def test_download_link_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)

    results = _sample_results()

    page._render_results(results, fold_selection=None)

    assert len(stub.downloads) == 3
    filenames = [entry.get("file_name") for entry in stub.downloads]
    assert any(name and name.endswith(".csv") for name in filenames)
    assert any(name and name.endswith(".parquet") for name in filenames)
    assert any(name and name.endswith(".zip") for name in filenames)
    mimes = [entry.get("mime") for entry in stub.downloads]
    assert "text/csv" in mimes
    assert "application/x-parquet" in mimes
    assert "application/zip" in mimes

    csv_entry = next(entry for entry in stub.downloads if entry.get("mime") == "text/csv")
    assert isinstance(csv_entry.get("data"), str)
    assert "Strategy" in csv_entry.get("data")

    parquet_entry = next(
        entry for entry in stub.downloads if entry.get("mime") == "application/x-parquet"
    )
    parquet_data = parquet_entry.get("data")
    assert hasattr(parquet_data, "getvalue")
    parquet_bytes = parquet_data.getvalue()
    assert parquet_bytes[:4] == b"PAR1"

    zip_entry = next(entry for entry in stub.downloads if entry.get("mime") == "application/zip")
    zip_data = zip_entry.get("data")
    assert hasattr(zip_data, "getvalue")
    if hasattr(zip_data, "seek"):
        zip_data.seek(0)
    with zipfile.ZipFile(zip_data) as bundle:
        assert set(bundle.namelist()) == {"summary.csv", "representative_paths.parquet"}
        summary_text = bundle.read("summary.csv").decode("utf-8")
        assert "Strategy" in summary_text


def test_download_payload_file_names_use_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    page, _stub = _load_page(monkeypatch)

    from datetime import datetime as real_datetime

    class _FixedDatetime:
        @classmethod
        def now(cls) -> real_datetime:
            return real_datetime(2024, 1, 2, 3, 4, 5)

    monkeypatch.setattr(page, "datetime", _FixedDatetime)

    summary_table = pd.DataFrame(
        {
            "Strategy": ["A"],
            "Sharpe (median)": [1.1],
            "Sharpe (5th)": [0.5],
            "Max DD (median)": [-0.2],
            "Max DD (5th)": [-0.3],
            "Terminal Wealth": [120.0],
        }
    )
    filtered_results = _sample_results().results_frame

    payloads = page._build_download_payloads(summary_table, filtered_results)

    filenames = {payload["file_name"] for payload in payloads}
    assert filenames == {
        "mc_summary_20240102_030405.csv",
        "mc_representative_paths_20240102_030405.parquet",
        "mc_bundle_20240102_030405.zip",
    }


def test_build_download_payloads_csv_content(monkeypatch: pytest.MonkeyPatch) -> None:
    page, _stub = _load_page(monkeypatch)

    summary_table = pd.DataFrame(
        {
            "Strategy": ["A", "B"],
            "Sharpe (median)": [1.1, 0.9],
        }
    )
    filtered_results = _sample_results().results_frame

    payloads = page._build_download_payloads(summary_table, filtered_results)
    csv_payload = next(payload for payload in payloads if payload["mime"] == "text/csv")

    assert csv_payload["label"] == "Download summary CSV"
    assert isinstance(csv_payload["data"], str)
    assert "Strategy,Sharpe (median)" in csv_payload["data"]
    assert "A,1.1" in csv_payload["data"]


def test_build_download_payloads_parquet_content(monkeypatch: pytest.MonkeyPatch) -> None:
    page, _stub = _load_page(monkeypatch)

    summary_table = pd.DataFrame({"Strategy": ["A"], "Sharpe (median)": [1.1]})
    filtered_results = _sample_results().results_frame

    payloads = page._build_download_payloads(summary_table, filtered_results)
    parquet_payload = next(
        payload for payload in payloads if payload["mime"] == "application/x-parquet"
    )
    parquet_buffer = parquet_payload["data"]

    assert parquet_payload["label"] == "Download representative paths (parquet)"
    assert hasattr(parquet_buffer, "getvalue")
    parquet_bytes = parquet_buffer.getvalue()
    assert parquet_bytes[:4] == b"PAR1"
    loaded = pd.read_parquet(BytesIO(parquet_bytes))
    assert not loaded.empty
    assert set(loaded.columns) >= {"strategy", "path", "fold", "terminal_wealth"}


def test_build_download_payloads_zip_bundle_content(monkeypatch: pytest.MonkeyPatch) -> None:
    page, _stub = _load_page(monkeypatch)

    summary_table = pd.DataFrame({"Strategy": ["A"], "Sharpe (median)": [1.1]})
    filtered_results = _sample_results().results_frame

    payloads = page._build_download_payloads(summary_table, filtered_results)
    zip_payload = next(payload for payload in payloads if payload["mime"] == "application/zip")
    zip_buffer = zip_payload["data"]

    assert zip_payload["label"] == "Download ZIP bundle"
    assert hasattr(zip_buffer, "getvalue")
    with zipfile.ZipFile(BytesIO(zip_buffer.getvalue())) as bundle:
        assert set(bundle.namelist()) == {"summary.csv", "representative_paths.parquet"}
        summary_text = bundle.read("summary.csv").decode("utf-8")
        assert "Strategy,Sharpe (median)" in summary_text
        assert bundle.read("representative_paths.parquet")[:4] == b"PAR1"


def test_build_download_payloads_binary_buffers_are_rewound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page, _stub = _load_page(monkeypatch)

    summary_table = pd.DataFrame({"Strategy": ["A"], "Sharpe (median)": [1.1]})
    filtered_results = _sample_results().results_frame

    payloads = page._build_download_payloads(summary_table, filtered_results)
    parquet_payload = next(
        payload for payload in payloads if payload["mime"] == "application/x-parquet"
    )
    zip_payload = next(payload for payload in payloads if payload["mime"] == "application/zip")

    parquet_buffer = parquet_payload["data"]
    zip_buffer = zip_payload["data"]

    assert hasattr(parquet_buffer, "tell")
    assert hasattr(zip_buffer, "tell")
    assert parquet_buffer.tell() == 0
    assert zip_buffer.tell() == 0


def test_empty_filtered_results_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)

    empty_results = DummyResults(results_frame=pd.DataFrame())

    page._render_results(empty_results, fold_selection="Fold 2")

    assert any("No results available" in message for message in stub.warning_messages)
    assert not stub.dataframes
    assert not stub.downloads
