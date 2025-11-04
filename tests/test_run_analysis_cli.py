from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis import run_analysis


class DummyResult(SimpleNamespace):
    """Helper namespace mirroring the API result interface."""

    metrics: pd.DataFrame
    details: dict


@pytest.fixture
def config_factory():
    def factory(**data_overrides):
        data = {"csv_path": "sample.csv"}
        data.update(data_overrides)
        return SimpleNamespace(
            data=data,
            sample_split={
                "in_start": "2020-01-01",
                "in_end": "2020-12-31",
                "out_start": "2021-01-01",
                "out_end": "2021-12-31",
            },
            export={"directory": "output", "formats": ["json"], "filename": "report"},
        )

    return factory


def test_main_requires_csv_path(monkeypatch, config_factory):
    cfg = config_factory()
    cfg.data.pop("csv_path")

    loaded_path = {}

    def fake_load(path):
        loaded_path["value"] = path
        return cfg

    monkeypatch.setattr(run_analysis, "load", fake_load)

    with pytest.raises(KeyError):
        run_analysis.main(["--config", "config.yml"])

    assert loaded_path["value"] == "config.yml"


def test_main_errors_when_load_csv_returns_none(monkeypatch, config_factory):
    cfg = config_factory()

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)

    captured_kwargs = {}

    def fake_load_csv(path, *, errors="raise", missing_policy=None, missing_limit=None):
        captured_kwargs.update(
            {
                "path": path,
                "errors": errors,
                "missing_policy": missing_policy,
                "missing_limit": missing_limit,
            }
        )
        return None

    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis.api, "run_simulation", pytest.fail)

    with pytest.raises(FileNotFoundError):
        run_analysis.main(["--config", "config.yml"])

    assert captured_kwargs == {
        "path": "sample.csv",
        "errors": "raise",
        "missing_policy": None,
        "missing_limit": None,
    }


def test_main_passes_missing_policy_when_supported(monkeypatch, config_factory, capsys):
    cfg = config_factory(missing_policy="forward", missing_limit=2)

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)

    received = {}

    def fake_load_csv(
        path,
        *,
        errors="raise",
        missing_policy=None,
        missing_limit=None,
    ):
        received["args"] = (path, errors, missing_policy, missing_limit)
        return pd.DataFrame({"ret": [0.1, 0.2]})

    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)

    def fake_run_simulation(config, df):
        assert config is cfg
        assert not df.empty
        return DummyResult(metrics=pd.DataFrame(), details={})

    monkeypatch.setattr(run_analysis.api, "run_simulation", fake_run_simulation)

    assert run_analysis.main(["--config", "config.yml", "--detailed"]) == 0

    assert received["args"] == ("sample.csv", "raise", "forward", 2)

    out = capsys.readouterr()
    assert out.out == "No results\n"


def test_main_maps_missing_policy_to_nan_alias(monkeypatch, config_factory):
    cfg = config_factory(missing_policy="drop", missing_limit=3)

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)

    captured = {}

    def fake_load_csv(path, *, errors="raise", nan_policy=None, nan_limit=None):
        captured["kwargs"] = {
            "path": path,
            "errors": errors,
            "nan_policy": nan_policy,
            "nan_limit": nan_limit,
        }
        return pd.DataFrame({"ret": [0.1]})

    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)

    def fake_run_simulation(config, df):
        return DummyResult(metrics=pd.DataFrame(), details={})

    monkeypatch.setattr(run_analysis.api, "run_simulation", fake_run_simulation)

    run_analysis.main(["--config", "config.yml", "--detailed"])

    assert captured["kwargs"] == {
        "path": "sample.csv",
        "errors": "raise",
        "nan_policy": "drop",
        "nan_limit": 3,
    }


def test_main_handles_empty_details(monkeypatch, config_factory, capsys):
    cfg = config_factory()

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)

    monkeypatch.setattr(
        run_analysis,
        "load_csv",
        lambda path, *, errors="raise": pd.DataFrame({"ret": [0.0]}),
    )

    result = DummyResult(metrics=pd.DataFrame({"metric": [1.0]}), details={})
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: result)

    assert run_analysis.main(["--config", "config.yml"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "No results\n"


def test_main_formats_summary_and_exports_without_excel(monkeypatch, config_factory, capsys):
    cfg = config_factory()
    cfg.export["directory"] = "out"
    cfg.export["formats"] = ["json"]

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)

    monkeypatch.setattr(
        run_analysis,
        "load_csv",
        lambda path, *, errors="raise": pd.DataFrame({"ret": [0.5, 0.6]}),
    )

    details = {"key": "value"}
    metrics = pd.DataFrame({"metric": [1.0]})
    monkeypatch.setattr(
        run_analysis.api,
        "run_simulation",
        lambda *_: DummyResult(metrics=metrics, details=details),
    )

    formatted_summary = "summary text"
    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *a: formatted_summary)

    export_calls = []
    monkeypatch.setattr(run_analysis.export, "export_data", lambda *a, **k: export_calls.append((a, k)))

    assert run_analysis.main(["--config", "config.yml"]) == 0

    captured = capsys.readouterr()
    assert "summary text" in captured.out

    assert len(export_calls) == 1
    data_arg, path_arg = export_calls[0][0]
    kwargs = export_calls[0][1]
    assert path_arg == "out/report"
    assert kwargs == {"formats": ["json"]}
    assert data_arg["metrics"].equals(metrics)


def test_main_skips_export_when_directory_missing(monkeypatch, config_factory, capsys):
    cfg = config_factory()
    cfg.export["directory"] = ""
    cfg.export["formats"] = ["json"]

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)

    monkeypatch.setattr(
        run_analysis,
        "load_csv",
        lambda path, *, errors="raise": pd.DataFrame({"ret": [0.4, 0.5]}),
    )

    details = {"value": 1}
    metrics = pd.DataFrame({"metric": [2.0]})
    monkeypatch.setattr(
        run_analysis.api,
        "run_simulation",
        lambda *_: DummyResult(metrics=metrics, details=details),
    )

    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *a: "summary text")

    export_calls = []
    monkeypatch.setattr(run_analysis.export, "export_data", lambda *a, **k: export_calls.append((a, k)))

    assert run_analysis.main(["--config", "config.yml"]) == 0

    captured = capsys.readouterr()
    assert "summary text" in captured.out
    assert export_calls == []


def test_main_exports_excel_and_other_formats(monkeypatch, config_factory, capsys):
    cfg = config_factory()
    cfg.export.update({"directory": "out", "formats": ["xlsx", "json"], "filename": "analysis"})

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)

    monkeypatch.setattr(
        run_analysis,
        "load_csv",
        lambda path, *, errors="raise": pd.DataFrame({"ret": [0.7, 0.8]}),
    )

    regime_df = pd.DataFrame({"regime": ["bull"], "value": [1]})
    details = {
        "performance_by_regime": regime_df,
        "regime_notes": ["note1", "note2"],
    }
    metrics = pd.DataFrame({"metric": [2.0]})

    monkeypatch.setattr(
        run_analysis.api,
        "run_simulation",
        lambda *_: DummyResult(metrics=metrics, details=details),
    )

    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *a: "summary")

    summary_frame = pd.DataFrame({"summary": [1]})
    monkeypatch.setattr(run_analysis.export, "summary_frame_from_result", lambda details: summary_frame)

    formatter_called = {}

    def fake_make_formatter(*args, **kwargs):
        formatter_called["called"] = True
        return "formatter"

    monkeypatch.setattr(run_analysis.export, "make_summary_formatter", fake_make_formatter)

    excel_calls = []
    monkeypatch.setattr(run_analysis.export, "export_to_excel", lambda *a, **k: excel_calls.append((a, k)))

    data_calls = []
    monkeypatch.setattr(run_analysis.export, "export_data", lambda *a, **k: data_calls.append((a, k)))

    assert run_analysis.main(["--config", "config.yml"]) == 0

    assert formatter_called["called"]
    assert excel_calls
    excel_args, excel_kwargs = excel_calls[0]
    assert excel_args[1] == "out/analysis.xlsx"
    assert excel_kwargs == {"default_sheet_formatter": "formatter"}
    assert "summary" in excel_args[0]["summary"].columns[0] or "summary" in excel_args[0]

    assert data_calls
    data_args, data_kwargs = data_calls[0]
    assert data_args[1] == "out/analysis"
    assert data_kwargs == {"formats": ["json"]}

    captured = capsys.readouterr()
    assert "summary" in captured.out


def test_main_supports_loader_without_errors(monkeypatch, config_factory, capsys):
    cfg = config_factory()

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)

    def loader_without_errors(path):
        assert path == "sample.csv"
        return pd.DataFrame({"ret": [0.1, 0.2]})

    monkeypatch.setattr(run_analysis, "load_csv", loader_without_errors)

    monkeypatch.setattr(
        run_analysis.api,
        "run_simulation",
        lambda *_: DummyResult(metrics=pd.DataFrame(), details={}),
    )

    assert run_analysis.main(["--config", "config.yml", "--detailed"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "No results\n"


def test_main_ignores_unknown_missing_policy(monkeypatch, config_factory):
    cfg = config_factory(missing_policy="forward", missing_limit=5)

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)

    captured = {}

    def loader_without_policy(path):
        captured["path"] = path
        captured["kwargs"] = {}
        return pd.DataFrame({"ret": [0.3]})

    monkeypatch.setattr(run_analysis, "load_csv", loader_without_policy)

    monkeypatch.setattr(
        run_analysis.api,
        "run_simulation",
        lambda *_: DummyResult(metrics=pd.DataFrame(), details={}),
    )

    assert run_analysis.main(["--config", "config.yml", "--detailed"]) == 0

    assert captured["path"] == "sample.csv"
    assert captured["kwargs"] == {}


def test_main_applies_default_export_targets(monkeypatch, config_factory, capsys):
    cfg = config_factory()
    cfg.export.clear()

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)

    monkeypatch.setattr(
        run_analysis,
        "load_csv",
        lambda path, *, errors="raise": pd.DataFrame({"ret": [0.9]}),
    )

    details = {
        "performance_by_regime": pd.DataFrame({"regime": ["bull"], "value": [1]})
    }
    metrics = pd.DataFrame({"metric": [3.0]})

    monkeypatch.setattr(
        run_analysis.api,
        "run_simulation",
        lambda *_: DummyResult(metrics=metrics, details=details),
    )

    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *a: "summary")
    monkeypatch.setattr(run_analysis.export, "summary_frame_from_result", lambda details: pd.DataFrame({"summary": [2]}))

    formatter_called = {}

    def fake_formatter(*args, **kwargs):
        formatter_called["called"] = True
        return "formatter"

    monkeypatch.setattr(run_analysis.export, "make_summary_formatter", fake_formatter)

    excel_calls = []
    monkeypatch.setattr(run_analysis.export, "export_to_excel", lambda *a, **k: excel_calls.append((a, k)))

    data_calls = []
    monkeypatch.setattr(run_analysis.export, "export_data", lambda *a, **k: data_calls.append((a, k)))

    assert run_analysis.main(["--config", "config.yml"]) == 0

    captured = capsys.readouterr()
    assert "summary" in captured.out
    assert formatter_called.get("called")
    assert excel_calls
    excel_args, excel_kwargs = excel_calls[0]
    assert excel_args[1] == str(Path(run_analysis.DEFAULT_OUTPUT_DIRECTORY) / "analysis.xlsx")
    assert excel_kwargs == {"default_sheet_formatter": "formatter"}
    assert data_calls == []
