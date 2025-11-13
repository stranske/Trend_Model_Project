from __future__ import annotations

import gc
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

import trend_portfolio_app.app as app


def test_pipeline_proxy_prefers_gc_patched_module(monkeypatch):
    base_module = SimpleNamespace(run=lambda cfg: "base")
    monkeypatch.setitem(sys.modules, "trend_analysis.pipeline", base_module)
    pkg_module = SimpleNamespace(run=lambda cfg: "pkg")
    monkeypatch.setattr(app, "_trend_pkg", SimpleNamespace(pipeline=pkg_module))

    patched_module = ModuleType("trend_analysis.pipeline")
    patched_module.run = lambda cfg: "patched"
    monkeypatch.setattr(gc, "get_objects", lambda: [patched_module])

    result = app.pipeline.run(object())
    assert result == "patched"
    assert app._PIPELINE_DEBUG[-1][0] == "run"


def test_pipeline_proxy_falls_back_to_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "trend_analysis.pipeline", None)
    pkg_module = SimpleNamespace(run=lambda cfg: "pkg")
    monkeypatch.setattr(app, "_trend_pkg", SimpleNamespace(pipeline=pkg_module))
    monkeypatch.setattr(app, "_resolve_pipeline", lambda: pkg_module)
    monkeypatch.setattr(gc, "get_objects", lambda: [])

    result = app.pipeline.run(object())
    assert result == "pkg"


def test_pipeline_proxy_uses_package_attribute(monkeypatch):
    module = ModuleType("trend_analysis.pipeline")
    monkeypatch.setitem(sys.modules, "trend_analysis.pipeline", module)
    pkg_module = SimpleNamespace(run=lambda cfg: "package")
    monkeypatch.setattr(app, "_trend_pkg", SimpleNamespace(pipeline=pkg_module))
    monkeypatch.setattr(app, "_resolve_pipeline", lambda: module)
    monkeypatch.setattr(gc, "get_objects", lambda: [])

    result = app.pipeline.run(object())
    assert result == "package"


def test_pipeline_proxy_handles_gc_errors(monkeypatch):
    base_module = SimpleNamespace(run=lambda cfg: "base")
    monkeypatch.setitem(sys.modules, "trend_analysis.pipeline", base_module)
    monkeypatch.setattr(app, "_trend_pkg", SimpleNamespace(pipeline=base_module))

    def raising_get_objects() -> list[object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(gc, "get_objects", raising_get_objects)

    result = app.pipeline.run(object())
    assert result == "base"


def test_columns_normalisation_uses_placeholders(monkeypatch):
    class StubColumns:
        def __call__(self, spec):
            return []

    monkeypatch.setattr(app.st, "columns", StubColumns())
    placeholder = object()
    monkeypatch.setattr(app.st, "empty", lambda: placeholder)

    cols = app._columns(2)
    assert len(cols) == 2
    assert cols[0] is placeholder


def test_summary_helpers_and_multi_rounding():
    df = pd.DataFrame({"Sharpe": [0.12345, 0.98765]})
    summary = app._summarise_run_df(df)
    assert summary.iloc[0, 0] == pytest.approx(0.1234)

    result = {
        "out_sample_stats": {
            "user": SimpleNamespace(cagr=0.1, sharpe=0.8),
            "equal": {"cagr": 0.05, "sharpe": 0.6},
        },
        "benchmark_ir": {"user": {"FundA": 0.3, "equal_weight": 0.1}},
    }
    summary_table = app._build_summary_from_result(result)
    assert "ir_user" in summary_table.columns

    multi_results = [
        {
            "period": ("2020-01", "2020-12", "2021-01", "2021-12"),
            "out_ew_stats": {"sharpe": 0.8, "cagr": 0.1},
            "out_user_stats": SimpleNamespace(sharpe=0.9, cagr=0.12),
        },
    ]
    multi_summary = app._summarise_multi(multi_results)
    assert list(multi_summary.columns) == [
        "in_start",
        "in_end",
        "out_start",
        "out_end",
        "ew_sharpe",
        "user_sharpe",
        "ew_cagr",
        "user_cagr",
    ]


def test_apply_session_state_merges_nested(monkeypatch):
    state = {
        "data.csv_path": "data.csv",
        "portfolio.mode": "rank",
        "sample_split.window._months": "3",
    }
    monkeypatch.setattr(app.st, "session_state", state)
    cfg: dict[str, object] = {}

    app._apply_session_state(cfg)
    assert cfg["data"]["csv_path"] == "data.csv"
    assert cfg["portfolio"]["mode"] == "rank"
    assert cfg["sample_split"]["window"]["length"] == 63


class StubStreamlit:
    def __init__(self):
        self.session_state = {
            "data.csv_path": "input.csv",
            "portfolio.mode": "rank",
        }
        self.warnings: list[str] = []
        self.successes: list[str] = []
        self.dataframes: list[pd.DataFrame] = []
        self.downloads: list[tuple[str, bytes]] = []
        self.captions: list[str] = []
        self.line = []
        self.bar = []
        self.infos: list[str] = []

    def header(self, text: str) -> None:
        pass

    def subheader(self, text: str) -> None:
        pass

    def button(self, label: str, **_: object) -> bool:
        if label == "Run Single Period":
            return True
        if label == "Run Multi-Period":
            return True
        return False

    def text_input(self, label: str, **_: object) -> str:
        return "input.csv"

    def download_button(self, label: str, *, data: bytes, **_: object) -> None:
        self.downloads.append((label, data))

    class _Spinner:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def spinner(self, message: str) -> "StubStreamlit._Spinner":
        return StubStreamlit._Spinner()

    def dataframe(self, df: pd.DataFrame, **_: object) -> None:
        self.dataframes.append(df.copy())

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def success(self, message: str) -> None:
        self.successes.append(message)

    def caption(self, text: str) -> None:
        self.captions.append(text)

    def line_chart(self, df: pd.DataFrame) -> None:
        self.line.append(df.copy())

    def bar_chart(self, df: pd.DataFrame) -> None:
        self.bar.append(df.copy())

    def info(self, message: str) -> None:
        self.infos.append(message)

    def empty(self):
        return app._NullContext()


def test_render_run_section_success(monkeypatch):
    stub = StubStreamlit()
    monkeypatch.setattr(app, "st", stub)
    monkeypatch.setattr(
        app, "_columns", lambda spec: [app._NullContext(), app._NullContext()]
    )
    monkeypatch.setattr(app, "_build_cfg", lambda cfg: cfg)

    summary_df = pd.DataFrame({"Sharpe": [0.8], "CAGR": [0.12]})
    full_result = {
        "out_sample_stats": {"user": SimpleNamespace(cagr=0.1, sharpe=0.8)},
        "benchmark_ir": {"user": {"FundA": 0.3, "equal_weight": 0.1}},
        "risk_diagnostics": {
            "asset_volatility": pd.DataFrame({"FundA": [0.2]}),
            "portfolio_volatility": pd.Series([0.1], index=pd.Index(["2024-01"])),
            "turnover": pd.Series([0.05], index=pd.Index(["2024-01"])),
            "turnover_value": 0.1234,
        },
    }
    monkeypatch.setattr(app.pipeline, "run", lambda cfg: summary_df)
    monkeypatch.setattr(app.pipeline, "run_full", lambda cfg: full_result)
    monkeypatch.setattr(
        app,
        "run_multi",
        lambda cfg: [
            {
                "period": ("2020-01", "2020-12", "2021-01", "2021-12"),
                "out_ew_stats": {"sharpe": 0.5, "cagr": 0.07},
                "out_user_stats": {"sharpe": 0.6, "cagr": 0.08},
            }
        ],
    )

    cfg_dict = {"data": {}, "portfolio": {}}
    app._render_run_section(cfg_dict)

    assert stub.successes
    assert stub.downloads
    assert any("Realised asset volatility" in text for text in stub.captions)
    assert stub.line and stub.bar


def test_render_run_section_failure_message(monkeypatch):
    stub = StubStreamlit()
    monkeypatch.setattr(app, "st", stub)
    monkeypatch.setattr(
        app, "_columns", lambda spec: [app._NullContext(), app._NullContext()]
    )
    monkeypatch.setattr(app, "_build_cfg", lambda cfg: cfg)
    monkeypatch.setattr(app, "_summarise_run_df", lambda *_: None)
    monkeypatch.setattr(app.pipeline, "run", lambda cfg: None)

    def raise_full(cfg):
        raise FileNotFoundError("missing.csv")

    monkeypatch.setattr(app.pipeline, "run_full", raise_full)
    monkeypatch.setattr(app, "run_multi", lambda cfg: [])

    cfg_dict = {"data": {}, "portfolio": {}}
    app._render_run_section(cfg_dict)

    assert stub.warnings and "missing.csv" in stub.warnings[0]


def test_render_run_section_uses_dataframe_fallback(monkeypatch):
    stub = StubStreamlit()
    stub.line_chart = "not-callable"
    stub.bar_chart = "not-callable"
    monkeypatch.setattr(app, "st", stub)
    monkeypatch.setattr(
        app, "_columns", lambda spec: [app._NullContext(), app._NullContext()]
    )
    monkeypatch.setattr(app, "_build_cfg", lambda cfg: cfg)

    summary_df = pd.DataFrame({"Sharpe": [0.8]})
    full_result = {
        "risk_diagnostics": {
            "asset_volatility": pd.DataFrame({"FundA": [0.2]}),
            "portfolio_volatility": pd.Series([0.1], index=pd.Index(["2024-01"])),
            "turnover": pd.Series([0.05], index=pd.Index(["2024-01"])),
        }
    }
    monkeypatch.setattr(app.pipeline, "run", lambda cfg: summary_df)
    monkeypatch.setattr(app.pipeline, "run_full", lambda cfg: full_result)
    monkeypatch.setattr(app, "run_multi", lambda cfg: [])

    cfg_dict = {"data": {}, "portfolio": {}}
    app._render_run_section(cfg_dict)

    # Fallback should record dataframes for each diagnostic
    assert len(stub.dataframes) >= 3


def test_render_run_section_logs_info_on_partial_results(monkeypatch):
    stub = StubStreamlit()
    monkeypatch.setattr(app, "st", stub)
    monkeypatch.setattr(
        app, "_columns", lambda spec: [app._NullContext(), app._NullContext()]
    )
    monkeypatch.setattr(app, "_build_cfg", lambda cfg: cfg)

    summary_df = pd.DataFrame({"Sharpe": [0.8]})
    monkeypatch.setattr(app.pipeline, "run", lambda cfg: summary_df)

    def raise_error(cfg: object) -> None:
        raise ValueError("failed run_full")

    monkeypatch.setattr(app.pipeline, "run_full", raise_error)
    monkeypatch.setattr(app, "_summarise_run_df", lambda df: summary_df)
    monkeypatch.setattr(app, "run_multi", lambda cfg: [])

    cfg_dict = {"data": {}, "portfolio": {}}
    app._render_run_section(cfg_dict)

    assert any("failed run_full" in message for message in stub.infos)
