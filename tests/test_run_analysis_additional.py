"""Additional coverage for ``trend_analysis.run_analysis`` CLI entry point."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis import run_analysis


class DummyResult:
    def __init__(self) -> None:
        dates = pd.date_range("2024-01-31", periods=2, freq="M")
        self.metrics = pd.DataFrame({"metric": [1.0, 2.0]}, index=dates)
        self.details = {"summary": "ok"}


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


def test_main_passes_missing_policy(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
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
        return pd.DataFrame({"Date": pd.date_range("2024-01-31", periods=2, freq="M"), "Fund": [0.01, 0.02]})

    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: DummyResult())
    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *args, **kwargs: "Summary")

    result = run_analysis.main(["-c", "config.yml"])
    assert result == 0
    out = capsys.readouterr().out
    assert "Summary" in out
    assert captured["kwargs"]["missing_policy"] == "zeros"
    assert captured["kwargs"]["missing_limit"] == 7
    assert captured["kwargs"]["errors"] == "raise"


def test_main_maps_nan_policy_when_signature_uses_nan(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _base_config()
    cfg.data.pop("missing_policy", None)
    cfg.data.pop("missing_limit", None)
    cfg.data["nan_policy"] = "ffill"
    cfg.data["nan_limit"] = 3

    captured: dict[str, object] = {}

    def fake_load(_path: str) -> SimpleNamespace:
        return cfg

    def fake_load_csv(path: str, *, errors: str = "log", nan_policy: object = None, nan_limit: object = None) -> pd.DataFrame:
        captured["errors"] = errors
        captured["nan_policy"] = nan_policy
        captured["nan_limit"] = nan_limit
        return pd.DataFrame({"Date": pd.date_range("2024-01-31", periods=2, freq="M"), "Fund": [0.01, 0.02]})

    monkeypatch.setattr(run_analysis, "load", fake_load)
    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: DummyResult())
    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *args, **kwargs: "Summary")

    assert run_analysis.main(["-c", "cfg.yml"]) == 0
    out = capsys.readouterr().out
    assert "Summary" in out
    assert captured["errors"] == "raise"
    assert captured["nan_policy"] == "ffill"
    assert captured["nan_limit"] == 3
