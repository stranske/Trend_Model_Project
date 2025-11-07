"""Targeted tests for :mod:`trend_analysis.run_analysis`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from trend_analysis import run_analysis as run_analysis_mod


class DummyResult(SimpleNamespace):
    metrics: pd.DataFrame
    details: dict[str, Any]


@pytest.fixture()
def sample_config(tmp_path: Path) -> SimpleNamespace:
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    return SimpleNamespace(
        data={
            "csv_path": "data.csv",
            "missing_policy": {"*": "ffill"},
            "missing_limit": {"Asset": 5},
        },
        sample_split={
            "in_start": "2020-01-01",
            "in_end": "2020-06-30",
            "out_start": "2020-07-01",
            "out_end": "2020-12-31",
        },
        export={
            "directory": str(export_dir),
            "formats": ["excel", "json"],
            "filename": "analysis",
        },
    )


def test_main_requires_csv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimpleNamespace(data={}, sample_split={}, export={})

    monkeypatch.setattr(run_analysis_mod, "load", lambda path: cfg)

    with pytest.raises(KeyError):
        run_analysis_mod.main(["-c", "config.yml"])


def test_main_detailed_mode_handles_empty_metrics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], sample_config: SimpleNamespace
) -> None:
    sample_config.export["directory"] = None
    sample_config.export["formats"] = []

    df = pd.DataFrame({"Date": pd.date_range("2020-01-31", periods=3, freq="ME"), "Asset": [1.0, 2.0, 3.0]})

    def fake_load_csv(
        path: str,
        *,
        errors: str = "log",
        missing_policy: Any | None = None,
        missing_limit: Any | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        assert errors == "raise"
        return df

    empty_result = DummyResult(metrics=pd.DataFrame(), details={})

    monkeypatch.setattr(run_analysis_mod, "load", lambda path: sample_config)
    monkeypatch.setattr(run_analysis_mod, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis_mod.api, "run_simulation", lambda cfg, frame: empty_result)

    rc = run_analysis_mod.main(["-c", "config.yml", "--detailed"])
    assert rc == 0

    captured = capsys.readouterr()
    assert "No results" in captured.out


def test_main_exports_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], sample_config: SimpleNamespace) -> None:
    csv_calls: list[dict[str, Any]] = []

    def fake_load_csv(
        path: str,
        *,
        errors: str = "log",
        missing_policy: Any | None = None,
        missing_limit: Any | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        csv_calls.append(
            {
                "errors": errors,
                "missing_policy": missing_policy,
                "missing_limit": missing_limit,
                **kwargs,
            }
        )
        return pd.DataFrame({"Date": pd.date_range("2020-01-31", periods=4, freq="ME"), "Asset": [1.0, 2.0, 3.0, 4.0]})

    regime_table = pd.DataFrame({"regime": ["Growth"], "return": [0.1]})
    metrics = pd.DataFrame({"metric": ["Sharpe"], "value": [1.2]})
    details: dict[str, Any] = {
        "performance_by_regime": regime_table,
        "regime_notes": ["note one", "note two"],
    }
    result = DummyResult(metrics=metrics, details=details)

    summary_frame = pd.DataFrame({"summary": ["ok"]})
    formatter_sentinel = object()
    excel_calls: list[tuple[dict[str, Any], str, object]] = []
    data_calls: list[tuple[dict[str, Any], str, list[str]]] = []

    monkeypatch.setattr(run_analysis_mod, "load", lambda path: sample_config)
    monkeypatch.setattr(run_analysis_mod, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis_mod.api, "run_simulation", lambda cfg, frame: result)
    monkeypatch.setattr(run_analysis_mod.export, "format_summary_text", lambda *a, **k: "summary text")
    monkeypatch.setattr(run_analysis_mod.export, "make_summary_formatter", lambda *a, **k: formatter_sentinel)
    monkeypatch.setattr(run_analysis_mod.export, "summary_frame_from_result", lambda payload: summary_frame)
    monkeypatch.setattr(
        run_analysis_mod.export,
        "export_to_excel",
        lambda data, path, default_sheet_formatter: excel_calls.append((data, path, default_sheet_formatter)),
    )

    def fake_export_data(data: dict[str, Any], path: str, formats: list[str]) -> None:
        data_calls.append((data, path, formats))

    monkeypatch.setattr(run_analysis_mod.export, "export_data", fake_export_data)

    rc = run_analysis_mod.main(["-c", "config.yml"])
    assert rc == 0

    # load_csv should be invoked with coercion parameters derived from config
    assert csv_calls
    kwargs = csv_calls.pop()
    assert kwargs["missing_policy"] == sample_config.data["missing_policy"]
    assert kwargs["missing_limit"] == sample_config.data["missing_limit"]

    captured = capsys.readouterr()
    assert "summary text" in captured.out

    assert excel_calls and data_calls
    excel_payload, excel_path, formatter = excel_calls[0]
    assert "summary" in excel_payload
    assert "performance_by_regime" in excel_payload
    assert "regime_notes" in excel_payload
    assert formatter is formatter_sentinel
    assert excel_path.endswith("analysis.xlsx")

    data_payload, data_path, formats = data_calls[0]
    assert data_path.endswith("analysis")
    assert formats == ["json"]
    assert data_payload is excel_payload


def test_main_supports_legacy_nan_keys(monkeypatch: pytest.MonkeyPatch, sample_config: SimpleNamespace) -> None:
    sample_config.data.pop("missing_policy", None)
    sample_config.data.pop("missing_limit", None)
    sample_config.data["nan_policy"] = {"*": "zero"}
    sample_config.data["nan_limit"] = {"Asset": 7}
    sample_config.export["directory"] = None
    sample_config.export["formats"] = []

    csv_calls: list[dict[str, Any]] = []

    def fake_load_csv(
        path: str,
        *,
        errors: str = "log",
        nan_policy: Any | None = None,
        nan_limit: Any | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        csv_calls.append(
            {
                "errors": errors,
                "nan_policy": nan_policy,
                "nan_limit": nan_limit,
                **kwargs,
            }
        )
        return pd.DataFrame({"Date": pd.date_range("2020-01-31", periods=2, freq="ME"), "Asset": [1.0, 2.0]})

    result = DummyResult(metrics=pd.DataFrame({"metric": [1]}), details={"foo": "bar"})

    monkeypatch.setattr(run_analysis_mod, "load", lambda path: sample_config)
    monkeypatch.setattr(run_analysis_mod, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis_mod.api, "run_simulation", lambda cfg, frame: result)

    rc = run_analysis_mod.main(["-c", "config.yml", "--detailed"])
    assert rc == 0

    assert csv_calls
    kwargs = csv_calls.pop()
    assert kwargs["nan_policy"] == sample_config.data["nan_policy"]
    assert kwargs["nan_limit"] == sample_config.data["nan_limit"]


def test_main_raises_when_loader_returns_none(monkeypatch: pytest.MonkeyPatch, sample_config: SimpleNamespace) -> None:
    def fake_load_csv(
        path: str,
        *,
        errors: str = "log",
        missing_policy: Any | None = None,
        missing_limit: Any | None = None,
        **kwargs: Any,
    ) -> None:
        return None

    monkeypatch.setattr(run_analysis_mod, "load", lambda path: sample_config)
    monkeypatch.setattr(run_analysis_mod, "load_csv", fake_load_csv)

    with pytest.raises(FileNotFoundError):
        run_analysis_mod.main(["-c", "config.yml"])


def test_main_handles_loader_without_errors_parameter(
    monkeypatch: pytest.MonkeyPatch, sample_config: SimpleNamespace
) -> None:
    csv_calls: list[dict[str, Any]] = []

    def fake_load_csv(
        path: str,
        *,
        missing_policy: Any | None = None,
        missing_limit: Any | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        csv_calls.append(
            {
                "missing_policy": missing_policy,
                "missing_limit": missing_limit,
                **kwargs,
            }
        )
        return pd.DataFrame({"Date": pd.date_range("2020-01-31", periods=2, freq="ME"), "Asset": [0.0, 1.0]})

    monkeypatch.setattr(run_analysis_mod, "load", lambda path: sample_config)
    monkeypatch.setattr(run_analysis_mod, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis_mod.api, "run_simulation", lambda cfg, frame: DummyResult(metrics=frame[:0], details={}))

    rc = run_analysis_mod.main(["-c", "config.yml", "--detailed"])
    assert rc == 0

    kwargs = csv_calls.pop()
    assert "errors" not in kwargs
    assert kwargs["missing_policy"] == sample_config.data["missing_policy"]
    assert kwargs["missing_limit"] == sample_config.data["missing_limit"]


def test_main_without_missing_policy_settings(monkeypatch: pytest.MonkeyPatch, sample_config: SimpleNamespace) -> None:
    sample_config.data = {"csv_path": "data.csv"}

    observed: list[dict[str, Any]] = []

    def fake_load_csv(
        path: str,
        *,
        errors: str = "log",
        missing_policy: Any | None = None,
        missing_limit: Any | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        observed.append(
            {
                "errors": errors,
                "missing_policy": missing_policy,
                "missing_limit": missing_limit,
                **kwargs,
            }
        )
        return pd.DataFrame({"Date": pd.date_range("2020-01-31", periods=2, freq="ME"), "Asset": [1.0, 2.0]})

    monkeypatch.setattr(run_analysis_mod, "load", lambda path: sample_config)
    monkeypatch.setattr(run_analysis_mod, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis_mod.api, "run_simulation", lambda cfg, frame: DummyResult(metrics=frame, details={}))

    rc = run_analysis_mod.main(["-c", "config.yml", "--detailed"])
    assert rc == 0

    kwargs = observed.pop()
    assert kwargs["missing_policy"] is None
    assert kwargs["missing_limit"] is None


def test_main_defaults_output_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_config: SimpleNamespace, capsys: pytest.CaptureFixture[str]) -> None:
    sample_config.export["directory"] = None
    sample_config.export["formats"] = []

    csv_frame = pd.DataFrame({"Date": pd.date_range("2020-01-31", periods=3, freq="ME"), "Asset": [1.0, 1.1, 1.2]})
    result = DummyResult(
        metrics=pd.DataFrame({"metric": ["Sharpe"], "value": [1.0]}),
        details={"performance_by_regime": pd.DataFrame({"regime": ["base"], "value": [0.5]})},
    )

    export_calls: dict[str, Any] = {}

    monkeypatch.setattr(run_analysis_mod, "DEFAULT_OUTPUT_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(run_analysis_mod, "DEFAULT_OUTPUT_FORMATS", ["json"])
    monkeypatch.setattr(run_analysis_mod, "load", lambda path: sample_config)
    monkeypatch.setattr(
        run_analysis_mod,
        "load_csv",
        lambda path, **kwargs: csv_frame,
    )
    monkeypatch.setattr(run_analysis_mod.api, "run_simulation", lambda cfg, frame: result)
    monkeypatch.setattr(run_analysis_mod.export, "format_summary_text", lambda *a, **k: "summary")
    monkeypatch.setattr(run_analysis_mod.export, "make_summary_formatter", lambda *a, **k: object())
    monkeypatch.setattr(
        run_analysis_mod.export, "summary_frame_from_result", lambda details: pd.DataFrame({"value": [1]})
    )
    monkeypatch.setattr(
        run_analysis_mod.export,
        "export_data",
        lambda data, path, formats: export_calls.setdefault("export", (data, path, formats)),
    )
    rc = run_analysis_mod.main(["-c", "config.yml"])
    assert rc == 0

    captured = capsys.readouterr()
    assert "summary" in captured.out

    assert "export" in export_calls
    data_path = export_calls["export"][1]
    assert data_path.startswith(str(tmp_path))
