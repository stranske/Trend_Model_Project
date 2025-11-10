"""Focused regression coverage for ``trend_analysis.run_analysis.main``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd
import pytest

from trend_analysis import run_analysis as run_analysis_mod


class DummyConfig(SimpleNamespace):
    """Container matching the attributes ``run_analysis.main`` expects."""

    data: dict[str, Any]
    sample_split: dict[str, str]
    export: dict[str, Any]


class DummyResult(SimpleNamespace):
    """API result stub with pandas payloads."""

    metrics: pd.DataFrame
    details: dict[str, Any]


@pytest.fixture()
def config_template() -> DummyConfig:
    return DummyConfig(
        data={
            "csv_path": "prices.csv",
            "missing_policy": "forward_fill",
            "missing_limit": 4,
        },
        sample_split={
            "in_start": "2023-01-01",
            "in_end": "2023-06-30",
            "out_start": "2023-07-01",
            "out_end": "2023-12-31",
        },
        export={"directory": "", "formats": [], "filename": "analysis"},
    )


def _detailed_result() -> DummyResult:
    metrics = pd.DataFrame({"Sharpe": [1.23]}, index=["Portfolio"])
    return DummyResult(metrics=metrics, details={"regime_notes": []})


def _summary_result() -> DummyResult:
    metrics = pd.DataFrame({"Sharpe": [1.11]}, index=["Portfolio"])
    details = {
        "performance_by_regime": pd.DataFrame(
            {"Return": [0.05]}, index=pd.Index(["Bull"], name="Regime")
        ),
        "regime_notes": ["No drawdowns"]
    }
    return DummyResult(metrics=metrics, details=details)


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(_detailed_result, id="detailed"),
        pytest.param(_summary_result, id="summary"),
    ],
)
def test_main_translates_missing_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config_template: DummyConfig,
    factory: Callable[[], DummyResult],
) -> None:
    """Ensure old and new missing-data keys are mapped onto ``load_csv``."""

    calls: list[dict[str, Any]] = []

    def fake_load(path: str) -> DummyConfig:
        if factory is _summary_result:
            config_template.data = {
                "csv_path": Path("prices.csv"),
                "nan_policy": "keep",
                "nan_limit": 2,
            }
        return config_template

    def load_csv_new(
        path: str,
        *,
        errors: str = "raise",
        missing_policy: str | None = None,
        missing_limit: int | None = None,
    ) -> pd.DataFrame:
        calls.append(
            {
                "path": path,
                "errors": errors,
                "missing_policy": missing_policy,
                "missing_limit": missing_limit,
            }
        )
        return pd.DataFrame({"returns": [0.01, 0.02]})

    def load_csv_legacy(
        path: str,
        *,
        errors: str = "raise",
        nan_policy: str | None = None,
        nan_limit: int | None = None,
    ) -> pd.DataFrame:
        calls.append(
            {
                "path": path,
                "errors": errors,
                "nan_policy": nan_policy,
                "nan_limit": nan_limit,
            }
        )
        return pd.DataFrame({"returns": [0.03, 0.04]})

    monkeypatch.setattr(run_analysis_mod, "load", fake_load)
    monkeypatch.setattr(
        run_analysis_mod,
        "load_csv",
        load_csv_new if factory is _detailed_result else load_csv_legacy,
    )
    monkeypatch.setattr(
        run_analysis_mod.api, "run_simulation", lambda cfg, df: factory()
    )

    summary_calls: list[tuple[dict[str, Any], str, str, str, str]] = []
    export_calls: list[tuple[dict[str, pd.DataFrame], str, list[str]]] = []

    def fake_format_summary(
        details: dict[str, Any],
        in_start: str,
        in_end: str,
        out_start: str,
        out_end: str,
    ) -> str:
        summary_calls.append((details, in_start, in_end, out_start, out_end))
        return "SUMMARY"

    def fake_export_data(
        data: dict[str, pd.DataFrame],
        path: str,
        *,
        formats: list[str],
    ) -> None:
        export_calls.append((data, path, formats))

    if factory is _summary_result:
        monkeypatch.setattr(
            run_analysis_mod.export, "format_summary_text", fake_format_summary
        )
        monkeypatch.setattr(run_analysis_mod.export, "export_data", fake_export_data)
        config_template.export = {
            "directory": "output",
            "formats": ["json"],
            "filename": "analysis",
        }

    argv = ["-c", "config.yml"]
    if factory is _detailed_result:
        argv.append("--detailed")

    rc = run_analysis_mod.main(argv)
    assert rc == 0

    assert calls, "load_csv should be invoked"
    recorded = calls[0]
    assert recorded["path"] == "prices.csv"
    assert recorded["errors"] == "raise"
    if factory is _detailed_result:
        assert recorded["missing_policy"] == "forward_fill"
        assert recorded["missing_limit"] == 4
    else:
        assert recorded["nan_policy"] == "keep"
        assert recorded["nan_limit"] == 2

    captured = capsys.readouterr().out
    if factory is _detailed_result:
        assert "Sharpe" in captured
    else:
        assert summary_calls
        details, *_ = summary_calls[0]
        assert "performance_by_regime" in details
        assert export_calls
        data, path, formats = export_calls[0]
        assert path.endswith("output/analysis")
        assert formats == ["json"]
        assert "performance_by_regime" in data
        notes = data.get("regime_notes")
        assert isinstance(notes, pd.DataFrame)


def test_main_requires_csv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing CSV path should raise a descriptive error."""

    monkeypatch.setattr(
        run_analysis_mod,
        "load",
        lambda *_: DummyConfig(data={}, sample_split={}, export={}),
    )

    with pytest.raises(KeyError):
        run_analysis_mod.main(["-c", "config.yml"])


def test_main_raises_when_load_csv_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``None`` from ``load_csv`` should surface as ``FileNotFoundError``."""

    cfg = DummyConfig(
        data={"csv_path": "missing.csv"},
        sample_split={"in_start": "", "in_end": "", "out_start": "", "out_end": ""},
        export={"directory": "", "formats": [], "filename": "analysis"},
    )
    monkeypatch.setattr(run_analysis_mod, "load", lambda *_: cfg)
    monkeypatch.setattr(run_analysis_mod, "load_csv", lambda *_, **__: None)

    with pytest.raises(FileNotFoundError):
        run_analysis_mod.main(["-c", "config.yml"])
