"""Tests for the :mod:`trend_analysis.run_analysis` CLI helper."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis import run_analysis


def _make_config(**overrides: object) -> SimpleNamespace:
    """Create a minimal configuration object for ``run_analysis.main``."""

    defaults = {
        "data": {"csv_path": "data.csv"},
        "sample_split": {
            "in_start": "2020-01-01",
            "in_end": "2020-06-30",
            "out_start": "2020-07-01",
            "out_end": "2020-12-31",
        },
        "export": {"directory": "output", "formats": ["json"], "filename": "result"},
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_main_requires_csv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _make_config(data={})
    monkeypatch.setattr(run_analysis, "load", lambda _path: config)

    with pytest.raises(KeyError):
        run_analysis.main(["--config", "config.yml"])


def test_main_raises_when_loader_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _make_config()
    monkeypatch.setattr(run_analysis, "load", lambda _path: config)

    load_calls: dict[str, object] = {}

    def fake_load_csv(
        path: str,
        *,
        errors: str = "log",
        missing_policy: object | None = None,
        missing_limit: object | None = None,
        **kwargs: object,
    ) -> None:
        load_calls["path"] = path
        load_calls["kwargs"] = {
            "errors": errors,
            "missing_policy": missing_policy,
            "missing_limit": missing_limit,
            "extra": kwargs,
        }
        return None

    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: pytest.fail("run_simulation should not run"))

    with pytest.raises(FileNotFoundError):
        run_analysis.main(["--config", "config.yml"])

    assert load_calls["path"] == "data.csv"
    kwargs = load_calls["kwargs"]
    assert kwargs["errors"] == "raise"
    assert kwargs["missing_policy"] is None
    assert kwargs["missing_limit"] is None
    assert kwargs["extra"] == {}


def test_main_runs_pipeline_and_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _make_config(
        data={"csv_path": "input.csv", "nan_policy": "ffill", "nan_limit": 2},
    )
    monkeypatch.setattr(run_analysis, "load", lambda _path: config)

    captured_load: dict[str, object] = {}

    def fake_load_csv(
        path: str,
        *,
        errors: str = "log",
        missing_policy: object | None = None,
        missing_limit: object | None = None,
        **kwargs: object,
    ) -> pd.DataFrame:
        captured_load["path"] = path
        captured_load["kwargs"] = {
            "errors": errors,
            "missing_policy": missing_policy,
            "missing_limit": missing_limit,
            "extra": kwargs,
        }
        return pd.DataFrame({"metric": [1.0]})

    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)

    metrics = pd.DataFrame({"return": [0.1]})
    regime_table = pd.DataFrame({"regime": ["Bull"], "value": [1.2]})
    result = SimpleNamespace(
        metrics=metrics,
        details={
            "performance_by_regime": regime_table,
            "regime_notes": ["Note one", "Note two"],
        },
    )

    run_args: dict[str, object] = {}

    def fake_run_simulation(cfg: object, df: pd.DataFrame) -> SimpleNamespace:
        run_args["config"] = cfg
        run_args["dataframe"] = df.copy()
        return result

    monkeypatch.setattr(run_analysis.api, "run_simulation", fake_run_simulation)

    format_args: dict[str, object] = {}

    def fake_format_summary_text(details: dict[str, object], *split: str) -> str:
        format_args["details"] = details
        format_args["split"] = split
        return "Rendered summary"

    monkeypatch.setattr(run_analysis.export, "format_summary_text", fake_format_summary_text)

    export_calls: dict[str, object] = {}

    def fake_export_data(data: dict[str, object], path: str, *, formats: list[str]) -> None:
        export_calls["data"] = data
        export_calls["path"] = path
        export_calls["formats"] = formats

    monkeypatch.setattr(run_analysis.export, "export_data", fake_export_data)

    status = run_analysis.main(["--config", "config.yml"])
    assert status == 0

    assert captured_load["path"] == "input.csv"
    kwargs = captured_load["kwargs"]
    assert kwargs["errors"] == "raise"
    assert kwargs["missing_policy"] == "ffill"
    assert kwargs["missing_limit"] == 2
    assert kwargs["extra"] == {}

    assert run_args["config"] is config
    pd.testing.assert_frame_equal(run_args["dataframe"], pd.DataFrame({"metric": [1.0]}))

    assert format_args["split"] == (
        "2020-01-01",
        "2020-06-30",
        "2020-07-01",
        "2020-12-31",
    )
    assert export_calls["path"].endswith("output/result")
    assert export_calls["formats"] == ["json"]

    exported = export_calls["data"]
    assert list(exported.keys()) == ["metrics", "performance_by_regime", "regime_notes"]
    pd.testing.assert_frame_equal(exported["metrics"], metrics)
    pd.testing.assert_frame_equal(exported["performance_by_regime"], regime_table)


def test_main_uses_existing_missing_policy(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    config = _make_config(
        data={
            "csv_path": "input.csv",
            "missing_policy": {"value": "drop"},
            "missing_limit": {"value": 3},
        },
    )
    monkeypatch.setattr(run_analysis, "load", lambda _path: config)

    captured: dict[str, object] = {}

    def loader(
        path: str,
        *,
        errors: str = "log",
        missing_policy: object | None = None,
        missing_limit: object | None = None,
    ) -> pd.DataFrame:
        captured["path"] = path
        captured["errors"] = errors
        captured["missing_policy"] = missing_policy
        captured["missing_limit"] = missing_limit
        return pd.DataFrame({"metric": [5.0]})

    monkeypatch.setattr(run_analysis, "load_csv", loader)

    result = SimpleNamespace(
        metrics=pd.DataFrame({"metric": [5.0]}),
        details={},
    )
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: result)
    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *_: pytest.fail("no summary"))
    monkeypatch.setattr(run_analysis.export, "export_data", lambda *_args, **_kwargs: pytest.fail("no export expected"))

    status = run_analysis.main(["--config", "config.yml", "--detailed"])
    assert status == 0
    out = capsys.readouterr().out
    assert "metric" in out

    assert captured["path"] == "input.csv"
    assert captured["errors"] == "raise"
    assert captured["missing_policy"] == {"value": "drop"}
    assert captured["missing_limit"] == {"value": 3}


def test_main_supports_legacy_nan_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _make_config(data={"csv_path": "legacy.csv", "nan_policy": "bfill", "nan_limit": 1})
    monkeypatch.setattr(run_analysis, "load", lambda _path: config)

    captured: dict[str, object] = {}

    def legacy_loader(
        path: str,
        *,
        nan_policy: object | None = None,
        nan_limit: object | None = None,
        **_kwargs: object,
    ) -> pd.DataFrame:
        captured["path"] = path
        captured["nan_policy"] = nan_policy
        captured["nan_limit"] = nan_limit
        return pd.DataFrame({"metric": [7.0]})

    monkeypatch.setattr(run_analysis, "load_csv", legacy_loader)

    metrics = pd.DataFrame({"return": [0.2]})
    result = SimpleNamespace(
        metrics=metrics,
        details={"performance_by_regime": pd.DataFrame({"regime": [], "value": []}), "regime_notes": []},
    )
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: result)

    export_calls: dict[str, object] = {}
    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *_args: "Legacy summary")

    def record_export(data: dict[str, object], path: str, *, formats: list[str]) -> None:
        export_calls["data"] = data
        export_calls["path"] = path
        export_calls["formats"] = formats

    monkeypatch.setattr(run_analysis.export, "export_data", record_export)

    status = run_analysis.main(["--config", "config.yml"])
    assert status == 0

    assert captured["path"] == "legacy.csv"
    assert captured["nan_policy"] == "bfill"
    assert captured["nan_limit"] == 1
    assert export_calls["formats"] == ["json"]


def test_main_falls_back_to_default_export(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _make_config(export={})
    monkeypatch.setattr(run_analysis, "load", lambda _path: config)

    def loader(
        path: str,
        *,
        errors: str = "log",
        missing_policy: object | None = None,
        missing_limit: object | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame({"metric": [3.0]})

    monkeypatch.setattr(run_analysis, "load_csv", loader)

    result = SimpleNamespace(
        metrics=pd.DataFrame({"metric": [3.0]}),
        details={"performance_by_regime": pd.DataFrame({"regime": ["A"], "value": [1.0]}), "regime_notes": ["X"]},
    )
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: result)

    monkeypatch.setattr(run_analysis, "DEFAULT_OUTPUT_DIRECTORY", "generated")
    monkeypatch.setattr(run_analysis, "DEFAULT_OUTPUT_FORMATS", ["csv"])
    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *_args: "Default summary")

    export_calls: dict[str, object] = {}

    def record_export(data: dict[str, object], path: str, *, formats: list[str]) -> None:
        export_calls["data"] = data
        export_calls["path"] = path
        export_calls["formats"] = formats

    monkeypatch.setattr(run_analysis.export, "export_data", record_export)

    status = run_analysis.main(["--config", "config.yml"])
    assert status == 0

    assert export_calls["path"].endswith("generated/analysis")
    assert export_calls["formats"] == ["csv"]
    assert "regime_notes" in export_calls["data"]


def test_main_detailed_mode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    config = _make_config(export={})
    monkeypatch.setattr(run_analysis, "load", lambda _path: config)

    def fake_load_csv(
        path: str,
        *,
        errors: str = "log",
        **kwargs: object,
    ) -> pd.DataFrame:
        assert errors == "raise"
        assert kwargs == {}
        return pd.DataFrame({"metric": [2.0]})

    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)

    result = SimpleNamespace(
        metrics=pd.DataFrame({"metric": [2.0]}),
        details={"performance_by_regime": pd.DataFrame(), "regime_notes": []},
    )
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: result)
    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *_: pytest.fail("summary not expected"))
    monkeypatch.setattr(run_analysis.export, "export_data", lambda *_args, **_kwargs: pytest.fail("no export"))

    status = run_analysis.main(["--config", "config.yml", "--detailed"])
    assert status == 0

    out = capsys.readouterr().out
    assert "metric" in out
