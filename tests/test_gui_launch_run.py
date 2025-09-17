from types import SimpleNamespace

import pandas as pd

import trend_analysis.gui.app as app


def _stub_store(output_format: str, path: str) -> app.ParamStore:
    store = app.ParamStore()
    store.cfg = {"output": {"format": output_format, "path": path}}
    return store


def test_launch_run_button_uses_exporter(monkeypatch, tmp_path):
    store = _stub_store("csv", str(tmp_path / "report.csv"))

    monkeypatch.setattr(app, "load_state", lambda: store)
    monkeypatch.setattr(app, "discover_plugins", lambda: None)

    cfg = SimpleNamespace(output=store.cfg["output"], sample_split={})
    monkeypatch.setattr(app, "build_config_from_store", lambda s: cfg)

    metrics = pd.DataFrame({"value": [1.0]})
    monkeypatch.setattr(app.pipeline, "run", lambda cfg: metrics)
    monkeypatch.setattr(app.pipeline, "run_full", lambda cfg: {"result": 1})

    captured: dict[str, object] = {}

    def fake_export(data, path, *_):  # noqa: ANN001 - signature mimics exporter
        captured["data"] = data
        captured["path"] = path

    monkeypatch.setitem(app.export.EXPORTERS, "csv", fake_export)
    monkeypatch.setattr(app, "save_state", lambda st: captured.setdefault("saved", st))

    root = app.launch()
    run_btn = root.children[-1]
    run_btn.click()

    assert "data" in captured
    assert captured["data"]["metrics"].equals(metrics)
    assert captured["path"] == store.cfg["output"]["path"]
    assert store.dirty is False


def test_launch_run_button_exports_excel(monkeypatch, tmp_path):
    store = _stub_store("excel", str(tmp_path / "analysis"))

    monkeypatch.setattr(app, "load_state", lambda: store)
    monkeypatch.setattr(app, "discover_plugins", lambda: None)

    sample_split = {
        "in_start": "2020-01",
        "in_end": "2020-06",
        "out_start": "2020-07",
        "out_end": "2020-12",
    }
    cfg = SimpleNamespace(output=store.cfg["output"], sample_split=sample_split)
    monkeypatch.setattr(app, "build_config_from_store", lambda s: cfg)

    metrics = pd.DataFrame({"value": [1.0]})
    monkeypatch.setattr(app.pipeline, "run", lambda cfg: metrics)
    monkeypatch.setattr(app.pipeline, "run_full", lambda cfg: {"full": "results"})

    captured: dict[str, object] = {}

    def fake_summary_formatter(res, *args):  # noqa: ANN001 - matches usage
        captured["summary_args"] = (res, *args)
        return lambda *_: None

    monkeypatch.setattr(app.export, "make_summary_formatter", fake_summary_formatter)

    def fake_export_to_excel(data, path, **kwargs):  # noqa: ANN001
        captured["excel"] = (data, path, kwargs)

    monkeypatch.setattr(app.export, "export_to_excel", fake_export_to_excel)
    monkeypatch.setattr(app, "save_state", lambda st: captured.setdefault("saved", st))

    root = app.launch()
    run_btn = root.children[-1]
    run_btn.click()

    assert "excel" in captured
    data, path, kwargs = captured["excel"]
    assert set(data) >= {"metrics", "summary"}
    assert path.endswith(".xlsx")
    assert "default_sheet_formatter" in kwargs
    assert captured["summary_args"][0] == {"full": "results"}
