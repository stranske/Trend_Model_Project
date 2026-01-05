"""Coverage-focused tests for ``trend_analysis.run_analysis`` CLI entry point."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from trend_analysis import run_analysis


class _DummyResult:
    def __init__(self, metrics: pd.DataFrame, details: dict[str, Any]):
        self.metrics = metrics
        self.details = details


def _build_config(csv_path: str, export_dir: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        data={
            "csv_path": csv_path,
            "missing_policy": "drop",
            "missing_limit": 0.1,
        },
        sample_split={
            "in_start": "2020-01-01",
            "in_end": "2020-06-30",
            "out_start": "2020-07-01",
            "out_end": "2020-12-31",
        },
        export={
            "directory": export_dir or "outputs",
            "formats": ["json"],
            "filename": "report",
        },
    )


def _stub_load_csv(
    path: str,
    *,
    errors=None,
    missing_policy=None,
    missing_limit=None,
    nan_policy=None,
    nan_limit=None,
):
    return pd.DataFrame({"return": [0.01, -0.02]})


def _stub_run_simulation(cfg: SimpleNamespace, df: pd.DataFrame) -> _DummyResult:
    metrics = pd.DataFrame({"metric": [1.0]})
    details = {
        "performance_by_regime": pd.DataFrame({"regime": ["in"], "sharpe": [1.2]}),
        "regime_notes": ["note"],
    }
    return _DummyResult(metrics=metrics, details=details)


def test_main_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    export_calls: dict[str, Any] = {}

    cfg = _build_config(csv_path=str(tmp_path / "data.csv"), export_dir=str(tmp_path / "out"))

    monkeypatch.setattr(run_analysis, "load", lambda _: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", _stub_load_csv)
    monkeypatch.setattr(run_analysis, "api", SimpleNamespace(run_simulation=_stub_run_simulation))

    def _format_summary_text(*args: Any, **kwargs: Any) -> str:
        export_calls["summary_args"] = (args, kwargs)
        return "summary"

    def _export_data(payload: dict[str, Any], path: str, *, formats: list[str]):
        export_calls["payload_keys"] = set(payload)
        export_calls["path"] = path
        export_calls["formats"] = formats

    monkeypatch.setattr(run_analysis.export, "format_summary_text", _format_summary_text)
    monkeypatch.setattr(run_analysis.export, "export_data", _export_data)
    monkeypatch.setattr(
        run_analysis.export, "make_summary_formatter", lambda *_, **__: lambda df: df
    )

    exit_code = run_analysis.main(["-c", "config.yml"])

    assert exit_code == 0
    assert export_calls["formats"] == ["json"]
    assert export_calls["payload_keys"] == {
        "metrics",
        "performance_by_regime",
        "regime_notes",
    }


def test_main_raises_for_missing_csv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_analysis,
        "load",
        lambda _: SimpleNamespace(data={}, export={}, sample_split={}),
    )
    with pytest.raises(KeyError):
        run_analysis.main(["--config", "config.yml"])


def test_main_raises_when_load_csv_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    cfg = _build_config(csv_path=str(tmp_path / "missing.csv"))
    monkeypatch.setattr(run_analysis, "load", lambda _: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", lambda *_args, **_kwargs: None)

    with pytest.raises(FileNotFoundError):
        run_analysis.main(["--config", "config.yml"])


def test_main_uses_nan_policy_fallbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    cfg = SimpleNamespace(
        data={"csv_path": tmp_path / "data.csv", "nan_policy": "ffill", "nan_limit": 2},
        sample_split={"in_start": "a", "in_end": "b", "out_start": "c", "out_end": "d"},
        export={
            "directory": str(tmp_path / "out"),
            "formats": ["json"],
            "filename": "report",
        },
    )

    captured_kwargs: dict[str, Any] = {}

    def _load_csv(path: str, *, errors=None, nan_policy=None, nan_limit=None):
        captured_kwargs.update(
            {
                "path": path,
                "errors": errors,
                "nan_policy": nan_policy,
                "nan_limit": nan_limit,
            }
        )
        return pd.DataFrame({"return": [0.03]})

    monkeypatch.setattr(run_analysis, "load", lambda _: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", _load_csv)
    monkeypatch.setattr(run_analysis, "api", SimpleNamespace(run_simulation=_stub_run_simulation))
    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *_, **__: "summary")
    monkeypatch.setattr(run_analysis.export, "export_data", lambda *_, **__: None)
    monkeypatch.setattr(
        run_analysis.export, "make_summary_formatter", lambda *_, **__: lambda df: df
    )

    exit_code = run_analysis.main(["--config", "config.yml"])

    assert exit_code == 0
    assert captured_kwargs == {
        "path": str(tmp_path / "data.csv"),
        "errors": "raise",
        "nan_policy": "ffill",
        "nan_limit": 2,
    }


def test_main_detailed_mode_handles_empty_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    cfg = _build_config(csv_path=str(tmp_path / "data.csv"))

    def _load_csv(path: str, *, errors=None, missing_policy=None, missing_limit=None):
        return pd.DataFrame({"return": []})

    empty_result = _DummyResult(metrics=pd.DataFrame(), details={})

    monkeypatch.setattr(run_analysis, "load", lambda _: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", _load_csv)
    monkeypatch.setattr(
        run_analysis, "api", SimpleNamespace(run_simulation=lambda *_: empty_result)
    )

    exit_code = run_analysis.main(["--detailed", "--config", "config.yml"])

    assert exit_code == 0


def test_main_applies_default_export_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    cfg = SimpleNamespace(
        data={
            "csv_path": tmp_path / "data.csv",
            "missing_policy": "drop",
            "missing_limit": 0.1,
        },
        sample_split={
            "in_start": "start",
            "in_end": "mid",
            "out_start": "mid2",
            "out_end": "end",
        },
        export={},
    )

    monkeypatch.setattr(run_analysis, "load", lambda _: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", _stub_load_csv)

    export_calls: dict[str, Any] = {}

    def _export_to_excel(payload: dict[str, Any], path: str, *, default_sheet_formatter=None):
        export_calls["path"] = path
        export_calls["keys"] = set(payload)
        export_calls["sheet"] = default_sheet_formatter

    monkeypatch.setattr(run_analysis, "api", SimpleNamespace(run_simulation=_stub_run_simulation))
    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *_, **__: "summary")
    monkeypatch.setattr(run_analysis.export, "summary_frame_from_result", lambda _: pd.DataFrame())
    monkeypatch.setattr(
        run_analysis.export, "make_summary_formatter", lambda *_, **__: lambda df: df
    )
    monkeypatch.setattr(run_analysis.export, "export_to_excel", _export_to_excel)
    monkeypatch.setattr(run_analysis.export, "export_data", lambda *_, **__: None)

    exit_code = run_analysis.main(["--config", "config.yml"])

    assert exit_code == 0
    assert {"metrics", "summary"}.issubset(export_calls["keys"])
    assert export_calls["path"].endswith("analysis.xlsx")
