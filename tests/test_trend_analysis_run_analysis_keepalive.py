"""Focused coverage tests for :mod:`trend_analysis.run_analysis`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis import run_analysis


@pytest.fixture
def sample_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data={"csv_path": "sample.csv", "missing_policy": {"*": "zero"}, "missing_limit": 3},
        sample_split={
            "in_start": "2020-01-01",
            "in_end": "2020-06-30",
            "out_start": "2020-07-01",
            "out_end": "2020-12-31",
        },
        export={"directory": str(tmp_path), "formats": ["json"], "filename": "analysis"},
    )


class DummyResult(SimpleNamespace):
    pass


def _make_result(
    metrics: pd.DataFrame | None = None, details: dict | None = None
) -> DummyResult:
    if metrics is None:
        metrics = pd.DataFrame([[1.0]], columns=["value"])
    if details is None:
        details = {}
    return DummyResult(metrics=metrics, details=details)


def test_main_detailed_branch(monkeypatch: pytest.MonkeyPatch, sample_config: SimpleNamespace, capsys: pytest.CaptureFixture[str]) -> None:
    metrics = pd.DataFrame([[1.0], [2.0]], columns=["value"])
    result = _make_result(metrics=metrics)

    def detailed_load(
        path: str,
        *,
        errors: str = "log",
        missing_policy: object | None = None,
        missing_limit: object | None = None,
    ) -> pd.DataFrame:
        assert errors == "raise"
        assert missing_policy == sample_config.data["missing_policy"]
        assert missing_limit == sample_config.data["missing_limit"]
        return pd.DataFrame({"value": [1.0]})

    monkeypatch.setattr(run_analysis, "load", lambda path: sample_config)
    monkeypatch.setattr(run_analysis, "load_csv", detailed_load)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda cfg, df: result)

    rc = run_analysis.main(["-c", "config.yml", "--detailed"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "value" in out


def test_main_summary_branch(monkeypatch: pytest.MonkeyPatch, sample_config: SimpleNamespace, capsys: pytest.CaptureFixture[str]) -> None:
    details = {
        "performance_by_regime": pd.DataFrame({"regime": ["expansion"], "return": [0.5]}),
        "regime_notes": ["note"],
    }
    result = _make_result(details=details)

    exported: dict[str, tuple] = {}

    def fake_format(details: dict, *_: object) -> str:
        return f"summary:{len(details)}"

    def fake_export_data(data: dict, path: str, *, formats: list[str]) -> None:
        exported["data"] = (tuple(sorted(data)), path, tuple(formats))

    monkeypatch.setattr(run_analysis, "load", lambda path: sample_config)
    monkeypatch.setattr(run_analysis, "load_csv", lambda path, **kwargs: pd.DataFrame({"value": [1.0]}))
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda cfg, df: result)
    monkeypatch.setattr(run_analysis.export, "format_summary_text", fake_format)
    monkeypatch.setattr(run_analysis.export, "make_summary_formatter", lambda *a, **k: "formatter")
    monkeypatch.setattr(run_analysis.export, "summary_frame_from_result", lambda *a, **k: pd.DataFrame({"summary": [1]}))
    monkeypatch.setattr(run_analysis.export, "export_to_excel", lambda *a, **k: None)
    monkeypatch.setattr(run_analysis.export, "export_data", fake_export_data)

    rc = run_analysis.main(["-c", "config.yml"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip().startswith("summary:")
    assert "data" in exported


def test_main_missing_csv_path_raises(monkeypatch: pytest.MonkeyPatch, sample_config: SimpleNamespace) -> None:
    sample_config.data = {}
    monkeypatch.setattr(run_analysis, "load", lambda path: sample_config)
    with pytest.raises(KeyError):
        run_analysis.main(["-c", "config.yml"])


def test_main_handles_load_csv_none(monkeypatch: pytest.MonkeyPatch, sample_config: SimpleNamespace) -> None:
    monkeypatch.setattr(run_analysis, "load", lambda path: sample_config)
    monkeypatch.setattr(run_analysis, "load_csv", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError):
        run_analysis.main(["-c", "config.yml"])


def test_main_respects_missing_policy_aliases(monkeypatch: pytest.MonkeyPatch, sample_config: SimpleNamespace) -> None:
    # Simulate legacy nan_* keys in the config and ensure they are forwarded.
    sample_config.data = {"csv_path": "sample.csv", "nan_policy": {"Asset": "ffill"}, "nan_limit": 5}
    observed: dict[str, object] = {}

    def fake_load(
        path: str,
        *,
        nan_policy: object | None = None,
        nan_limit: object | None = None,
    ) -> pd.DataFrame:
        observed.update(
            {
                "nan_policy": nan_policy,
                "nan_limit": nan_limit,
            }
        )
        return pd.DataFrame({"value": [1.0]})

    monkeypatch.setattr(run_analysis, "load", lambda path: sample_config)
    monkeypatch.setattr(run_analysis, "load_csv", fake_load)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda cfg, df: _make_result())
    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *a, **k: "summary")
    monkeypatch.setattr(run_analysis.export, "export_data", lambda *a, **k: None)
    monkeypatch.setattr(run_analysis.export, "make_summary_formatter", lambda *a, **k: None)
    monkeypatch.setattr(run_analysis.export, "summary_frame_from_result", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(run_analysis.export, "export_to_excel", lambda *a, **k: None)

    run_analysis.main(["-c", "config.yml"])
    assert observed["nan_policy"] == {"Asset": "ffill"}
    assert observed["nan_limit"] == 5
