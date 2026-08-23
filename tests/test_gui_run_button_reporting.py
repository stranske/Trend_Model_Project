"""Regression coverage for observable GUI Run-button outcomes."""

from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis.gui import app


def _launch_run_button(monkeypatch, tmp_path, formats):  # noqa: ANN001
    store = app.ParamStore(
        cfg={
            "export": {
                "formats": formats,
                "directory": str(tmp_path),
                "filename": "report",
            }
        }
    )
    monkeypatch.setattr(app, "load_state", lambda: store)
    monkeypatch.setattr(app, "discover_plugins", lambda: None)
    monkeypatch.setattr(
        app,
        "build_config_from_store",
        lambda _: SimpleNamespace(export=store.cfg["export"], sample_split={}),
    )
    return store, app.launch().children[-1]


def test_run_button_signals_an_unsupported_export_format(monkeypatch, tmp_path):
    store, run_button = _launch_run_button(monkeypatch, tmp_path, ["csv"])
    store.cfg["export"]["formats"] = ["parquet"]
    store.dirty = True
    monkeypatch.setattr(app.pipeline, "run", lambda _: pytest.fail("pipeline must not run"))
    saved: list[app.ParamStore] = []
    monkeypatch.setattr(app, "save_state", saved.append)

    with pytest.warns(UserWarning, match="Unsupported export format"):
        run_button.click()

    assert not list(tmp_path.iterdir())
    assert saved == []
    assert store.dirty is not False


def test_run_button_signals_an_empty_result(monkeypatch, tmp_path):
    store, run_button = _launch_run_button(monkeypatch, tmp_path, ["csv"])
    store.dirty = True
    monkeypatch.setattr(app.pipeline, "run", lambda _: pd.DataFrame())
    saved: list[app.ParamStore] = []
    monkeypatch.setattr(app, "save_state", saved.append)

    with pytest.warns(UserWarning, match="produced no metrics"):
        run_button.click()

    assert not list(tmp_path.iterdir())
    assert saved == []
    assert store.dirty is not False


def test_run_button_exports_every_requested_format(monkeypatch, tmp_path):
    store, run_button = _launch_run_button(monkeypatch, tmp_path, ["csv", "json"])
    store.dirty = True
    monkeypatch.setattr(app.pipeline, "run", lambda _: pd.DataFrame({"value": [1.0]}))
    monkeypatch.setattr(app, "save_state", lambda _: None)

    run_button.click()

    assert (tmp_path / "report_metrics.csv").is_file()
    assert (tmp_path / "report_metrics.json").is_file()
    assert store.dirty is False


def test_run_button_reports_pipeline_failure(monkeypatch, tmp_path):
    store, run_button = _launch_run_button(monkeypatch, tmp_path, ["csv"])
    store.dirty = True
    monkeypatch.setattr(
        app.pipeline,
        "run",
        lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    saved: list[app.ParamStore] = []
    monkeypatch.setattr(app, "save_state", saved.append)

    with pytest.warns(UserWarning, match="Run failed: boom"):
        run_button.click()

    assert not list(tmp_path.iterdir())
    assert saved == []
    assert store.dirty is True


def test_run_button_skips_empty_summary_for_non_excel_exports(monkeypatch, tmp_path):
    store, run_button = _launch_run_button(monkeypatch, tmp_path, ["csv", "xlsx"])
    monkeypatch.setattr(app.pipeline, "run", lambda _: pd.DataFrame({"value": [1.0]}))
    monkeypatch.setattr(
        app.pipeline,
        "run_full",
        lambda _: {"metrics": pd.DataFrame({"value": [1.0]})},
    )
    monkeypatch.setattr(app, "save_state", lambda _: None)

    run_button.click()

    assert (tmp_path / "report_metrics.csv").is_file()
    assert (tmp_path / "report.xlsx").is_file()
    assert not (tmp_path / "report_summary.csv").exists()
