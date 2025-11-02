"""Regression tests covering argument adaptation in ``run_analysis.main``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from trend_analysis import run_analysis as run_analysis_mod


class DummyResult:
    def __init__(self, metrics: pd.DataFrame, details: dict[str, Any]):
        self.metrics = metrics
        self.details = details


@pytest.fixture(autouse=True)
def _no_export_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as mp:
        mp.setattr(
            run_analysis_mod.export,
            "format_summary_text",
            lambda *args, **kwargs: "summary",
        )
        mp.setattr(
            run_analysis_mod.export,
            "make_summary_formatter",
            lambda *args, **kwargs: lambda df: None,
        )
        mp.setattr(
            run_analysis_mod.export,
            "summary_frame_from_result",
            lambda details: pd.DataFrame(),
        )
        mp.setattr(run_analysis_mod.export, "export_data", lambda *args, **kwargs: None)
        mp.setattr(
            run_analysis_mod.export, "export_to_excel", lambda *args, **kwargs: None
        )
        yield


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        data={
            "csv_path": "input.csv",
            "missing_policy": {"A": "drop"},
            "missing_limit": {"A": 2},
        },
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-06",
            "out_start": "2020-07",
            "out_end": "2020-12",
        },
        export={"directory": "out", "formats": [], "filename": "analysis"},
    )


def _make_result() -> DummyResult:
    metrics = pd.DataFrame({"metric": [1.0]})
    details = {
        "summary": "ok",
        "performance_by_regime": pd.DataFrame(),
        "regime_notes": [],
    }
    return DummyResult(metrics, details)


def test_main_passes_missing_policy_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config()
    captured: dict[str, Any] = {}

    def fake_load_csv(
        path: str,
        *,
        errors: str | None = None,
        missing_policy=None,
        missing_limit=None,
    ) -> pd.DataFrame:
        captured["errors"] = errors
        captured["missing_policy"] = missing_policy
        captured["missing_limit"] = missing_limit
        return pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-31", periods=3, freq="ME"),
                "A": [0.01, 0.02, 0.03],
            }
        )

    with monkeypatch.context() as mp:
        mp.setattr(run_analysis_mod, "load", lambda path: cfg)
        mp.setattr(run_analysis_mod, "load_csv", fake_load_csv)
        mp.setattr(
            run_analysis_mod.api, "run_simulation", lambda cfg, df: _make_result()
        )

        rc = run_analysis_mod.main(["-c", "config.yml"])

    assert rc == 0
    assert captured["errors"] == "raise"
    assert captured["missing_policy"] is cfg.data["missing_policy"]
    assert captured["missing_limit"] is cfg.data["missing_limit"]


def test_main_falls_back_to_nan_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config()
    captured: dict[str, Any] = {}

    def fake_load_csv(
        path: str, *, errors: str | None = None, nan_policy=None, nan_limit=None
    ) -> pd.DataFrame:
        captured["errors"] = errors
        captured["nan_policy"] = nan_policy
        captured["nan_limit"] = nan_limit
        return pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-31", periods=3, freq="ME"),
                "A": [0.01, 0.02, 0.03],
            }
        )

    with monkeypatch.context() as mp:
        mp.setattr(run_analysis_mod, "load", lambda path: cfg)
        mp.setattr(run_analysis_mod, "load_csv", fake_load_csv)
        mp.setattr(
            run_analysis_mod.api, "run_simulation", lambda cfg, df: _make_result()
        )

        rc = run_analysis_mod.main(["-c", "config.yml"])

    assert rc == 0
    assert captured["errors"] == "raise"
    assert captured["nan_policy"] is cfg.data["missing_policy"]
    assert captured["nan_limit"] is cfg.data["missing_limit"]
