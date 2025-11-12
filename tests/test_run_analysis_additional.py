"""Additional coverage for ``trend_analysis.run_analysis`` CLI entry point."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis import run_analysis


class DummyResult:
    def __init__(self) -> None:
        dates = pd.date_range("2024-01-31", periods=2, freq="M")
        self.metrics = pd.DataFrame({"metric": [1.0, 2.0]}, index=dates)
        self.details = {"summary": "ok"}


class DetailedResult:
    def __init__(self) -> None:
        index = pd.date_range("2024-01-31", periods=3, freq="M")
        self.metrics = pd.DataFrame({"metric": [1.0, 2.0, 3.0]}, index=index)
        self.details = {
            "performance_by_regime": pd.DataFrame(
                {"return": [0.1, 0.2]}, index=["in_sample", "out_sample"]
            ),
            "regime_notes": ("bull", "bear"),
        }


class EmptyResult:
    def __init__(self) -> None:
        self.metrics = pd.DataFrame()
        self.details: dict[str, object] = {}


def _base_config() -> SimpleNamespace:
    return SimpleNamespace(
        data={"csv_path": "data.csv"},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-03",
            "out_start": "2020-04",
            "out_end": "2020-06",
        },
        export={"directory": None, "formats": ["csv"], "filename": "analysis"},
    )


def test_main_passes_missing_policy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _base_config()
    cfg.data["missing_policy"] = "zeros"
    cfg.data["missing_limit"] = 7

    captured: dict[str, object] = {}

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)

    def fake_load_csv(
        path: str,
        *,
        errors: str = "log",
        missing_policy: object | None = None,
        missing_limit: object | None = None,
    ) -> pd.DataFrame:
        captured["kwargs"] = {
            "errors": errors,
            "missing_policy": missing_policy,
            "missing_limit": missing_limit,
        }
        return pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-31", periods=2, freq="M"),
                "Fund": [0.01, 0.02],
            }
        )

    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: DummyResult())
    monkeypatch.setattr(
        run_analysis.export, "format_summary_text", lambda *args, **kwargs: "Summary"
    )

    result = run_analysis.main(["-c", "config.yml"])
    assert result == 0
    out = capsys.readouterr().out
    assert "Summary" in out
    assert captured["kwargs"]["missing_policy"] == "zeros"
    assert captured["kwargs"]["missing_limit"] == 7
    assert captured["kwargs"]["errors"] == "raise"


def test_main_maps_nan_policy_when_signature_uses_nan(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _base_config()
    cfg.data.pop("missing_policy", None)
    cfg.data.pop("missing_limit", None)
    cfg.data["nan_policy"] = "ffill"
    cfg.data["nan_limit"] = 3

    captured: dict[str, object] = {}

    def fake_load(_path: str) -> SimpleNamespace:
        return cfg

    def fake_load_csv(
        path: str,
        *,
        errors: str = "log",
        nan_policy: object = None,
        nan_limit: object = None,
    ) -> pd.DataFrame:
        captured["errors"] = errors
        captured["nan_policy"] = nan_policy
        captured["nan_limit"] = nan_limit
        return pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-31", periods=2, freq="M"),
                "Fund": [0.01, 0.02],
            }
        )

    monkeypatch.setattr(run_analysis, "load", fake_load)
    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: DummyResult())
    monkeypatch.setattr(
        run_analysis.export, "format_summary_text", lambda *args, **kwargs: "Summary"
    )

    assert run_analysis.main(["-c", "cfg.yml"]) == 0
    out = capsys.readouterr().out
    assert "Summary" in out
    assert captured["errors"] == "raise"
    assert captured["nan_policy"] == "ffill"
    assert captured["nan_limit"] == 3


def test_main_uses_nan_fallback_and_default_exports(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _base_config()
    cfg.data["missing_policy"] = "zeros"
    cfg.data["missing_limit"] = 2
    cfg.export = {"filename": "custom"}

    captured: dict[str, object] = {}
    export_calls: list[tuple[dict[str, pd.DataFrame], str]] = []
    export_data_calls: list[tuple[dict[str, pd.DataFrame], str, list[str]]] = []

    def fake_load(_path: str) -> SimpleNamespace:
        return cfg

    def fake_load_csv(
        path: str,
        *,
        errors: str = "log",
        nan_policy: object | None = None,
        nan_limit: object | None = None,
    ) -> pd.DataFrame:
        captured["kwargs"] = {
            "path": path,
            "errors": errors,
            "nan_policy": nan_policy,
            "nan_limit": nan_limit,
        }
        return pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-31", periods=3, freq="M"),
                "Fund": [0.01, 0.02, 0.03],
            }
        )

    def fake_format_summary(*_args: object, **_kwargs: object) -> str:
        return "Summary"

    def fake_formatter(*_args: object, **_kwargs: object) -> str:
        return "formatter"

    def fake_summary_frame(_details: dict[str, object]) -> pd.DataFrame:
        return pd.DataFrame({"value": [1]})

    def fake_export_to_excel(
        data: dict[str, pd.DataFrame], path: str, **_: object
    ) -> None:
        export_calls.append((data, path))

    def fake_export_data(
        data: dict[str, pd.DataFrame], path: str, *, formats: list[str]
    ) -> None:
        export_data_calls.append((data, path, formats))

    monkeypatch.setattr(run_analysis, "load", fake_load)
    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: DetailedResult())
    monkeypatch.setattr(run_analysis.export, "format_summary_text", fake_format_summary)
    monkeypatch.setattr(run_analysis.export, "make_summary_formatter", fake_formatter)
    monkeypatch.setattr(
        run_analysis.export, "summary_frame_from_result", fake_summary_frame
    )
    monkeypatch.setattr(run_analysis.export, "export_to_excel", fake_export_to_excel)
    monkeypatch.setattr(run_analysis.export, "export_data", fake_export_data)

    result = run_analysis.main(["-c", "config.yml"])
    assert result == 0
    out = capsys.readouterr().out
    assert "Summary" in out

    kwargs = captured["kwargs"]
    assert kwargs["errors"] == "raise"
    assert kwargs["nan_policy"] == "zeros"
    assert kwargs["nan_limit"] == 2

    assert export_calls, "Expected export_to_excel to be invoked with defaults"
    exported_data, exported_path = export_calls.pop()
    assert exported_path.endswith("outputs/custom.xlsx")
    assert "performance_by_regime" in exported_data
    assert "regime_notes" in exported_data

    assert not export_data_calls


def test_main_requires_csv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimpleNamespace(data={}, sample_split={}, export={})
    monkeypatch.setattr(run_analysis, "load", lambda _path: cfg)
    with pytest.raises(KeyError):
        run_analysis.main(["-c", "cfg.yml"])


def test_main_raises_when_load_csv_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_config()
    monkeypatch.setattr(run_analysis, "load", lambda _path: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError):
        run_analysis.main(["-c", "cfg.yml"])


def test_main_detailed_no_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _base_config()

    def fake_load_csv(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-31", periods=2, freq="M"),
                "Fund": [0.01, 0.02],
            }
        )

    monkeypatch.setattr(run_analysis, "load", lambda _path: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: EmptyResult())

    exit_code = run_analysis.main(["-c", "cfg.yml", "--detailed"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No results" in out

def test_main_passes_nan_policy_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_load_csv(
        path: str,
        *,
        errors: str | None = None,
        nan_policy: str | None = None,
        nan_limit: int | None = None,
    ) -> pd.DataFrame:
        captured["path"] = path
        captured["errors"] = errors
        captured["nan_policy"] = nan_policy
        captured["nan_limit"] = nan_limit
        return pd.DataFrame({"value": [1.0]})

    cfg = SimpleNamespace(
        data={"csv_path": Path("input.csv"), "nan_policy": "ffill", "nan_limit": 7},
        sample_split={},
        export={},
    )

    monkeypatch.setattr(run_analysis, "load", lambda _path: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: EmptyResult())

    exit_code = run_analysis.main(["-c", "cfg.yml", "--detailed"])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "No results" in output
    assert captured["path"] == "input.csv"
    assert captured["errors"] == "raise"
    assert captured["nan_policy"] == "ffill"
    assert captured["nan_limit"] == 7


def test_main_prefers_missing_policy_signature(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_load_csv(
        path: str,
        *,
        errors: str | None = None,
        missing_policy: str | None = None,
        missing_limit: int | None = None,
    ) -> pd.DataFrame:
        captured["path"] = path
        captured["errors"] = errors
        captured["missing_policy"] = missing_policy
        captured["missing_limit"] = missing_limit
        return pd.DataFrame({"value": [1.0]})

    cfg = SimpleNamespace(
        data={
            "csv_path": "input.csv",
            "missing_policy": "drop",
            "missing_limit": 3,
            "nan_policy": "ignored",
            "nan_limit": 5,
        },
        sample_split={},
        export={},
    )

    monkeypatch.setattr(run_analysis, "load", lambda _path: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: EmptyResult())

    exit_code = run_analysis.main(["-c", "cfg.yml", "--detailed"])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "No results" in output
    assert captured["path"] == "input.csv"
    assert captured["errors"] == "raise"
    assert captured["missing_policy"] == "drop"
    assert captured["missing_limit"] == 3
