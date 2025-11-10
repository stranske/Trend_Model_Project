"""Focused coverage for ``trend_analysis.run_analysis`` CLI helper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pandas.testing as tm
import pytest

from trend_analysis import run_analysis


class DummyConfig(SimpleNamespace):
    """Simple config object matching the attributes ``main`` expects."""

    def __init__(
        self,
        *,
        data: dict[str, Any],
        export: dict[str, Any] | None = None,
        sample_split: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            data=data,
            export=export or {},
            sample_split=sample_split
            or {
                "in_start": "2020-01-01",
                "in_end": "2020-12-31",
                "out_start": "2021-01-01",
                "out_end": "2021-12-31",
            },
        )


class DummyResult(SimpleNamespace):
    def __init__(self, metrics: pd.DataFrame, details: dict[str, Any]) -> None:
        super().__init__(metrics=metrics, details=details)


@pytest.fixture()
def default_result() -> DummyResult:
    metrics = pd.DataFrame({"Sharpe": [1.2]})
    details = {
        "performance_by_regime": pd.DataFrame({"Sharpe": [1.2]}),
        "regime_notes": ["note"],
        "summary": "payload",
    }
    return DummyResult(metrics=metrics, details=details)


@pytest.fixture()
def export_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(
        run_analysis.export,
        "export_data",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    return calls


def test_main_requires_csv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_analysis, "load", lambda _: DummyConfig(data={}))
    with pytest.raises(KeyError):
        run_analysis.main(["-c", "config.yml"])


def test_main_raises_when_loader_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DummyConfig(data={"csv_path": "missing.csv"})
    monkeypatch.setattr(run_analysis, "load", lambda _: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", lambda *_, **__: None)
    with pytest.raises(FileNotFoundError):
        run_analysis.main(["-c", "config.yml"])


def test_main_normalises_missing_policy_aliases(
    monkeypatch: pytest.MonkeyPatch,
    default_result: DummyResult,
    export_spy: list[tuple[tuple[Any, ...], dict[str, Any]]],
) -> None:
    captured: dict[str, Any] = {}

    def fake_load_csv(
        path: str,
        *,
        errors: str | None = None,
        missing_policy: str | None = None,
        missing_limit: int | None = None,
    ) -> pd.DataFrame:
        captured["path"] = path
        captured["kwargs"] = {
            "errors": errors,
            "missing_policy": missing_policy,
            "missing_limit": missing_limit,
        }
        return pd.DataFrame({"a": [1.0]})

    cfg = DummyConfig(
        data={
            "csv_path": "input.csv",
            "missing_policy": " both ",
            "missing_limit": "7",
        },
        export={"directory": ".", "formats": ["json"], "filename": "analysis"},
    )

    monkeypatch.setattr(run_analysis, "load", lambda _: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: default_result)
    monkeypatch.setattr(
        run_analysis.export, "format_summary_text", lambda *_, **__: "summary"
    )

    exit_code = run_analysis.main(["-c", "config.yml"])

    assert exit_code == 0
    assert captured["path"].endswith("input.csv")
    assert captured["kwargs"]["errors"] == "raise"
    assert captured["kwargs"]["missing_policy"].strip().lower() == "both"
    assert int(captured["kwargs"]["missing_limit"]) == 7

    assert len(export_spy) == 1
    (data_arg, path_arg), kwargs = export_spy[0]
    assert path_arg == str(Path(".") / "analysis")
    assert kwargs == {"formats": ["json"]}
    assert set(data_arg.keys()) == {"metrics", "performance_by_regime", "regime_notes"}
    tm.assert_frame_equal(data_arg["metrics"], default_result.metrics)
    tm.assert_frame_equal(
        data_arg["performance_by_regime"],
        default_result.details["performance_by_regime"],
    )
    tm.assert_frame_equal(
        data_arg["regime_notes"],
        pd.DataFrame({"note": default_result.details["regime_notes"]}),
        check_dtype=False,
    )


def test_main_supports_nan_policy_aliases(
    monkeypatch: pytest.MonkeyPatch,
    default_result: DummyResult,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, Any]] = []

    def loader_with_nan(
        path: str,
        *,
        errors: str | None = None,
        nan_policy: str | None = None,
        nan_limit: int | None = None,
    ) -> pd.DataFrame:
        calls.append(
            {
                "path": path,
                "kwargs": {
                    "errors": errors,
                    "nan_policy": nan_policy,
                    "nan_limit": nan_limit,
                },
            }
        )
        return pd.DataFrame({"value": [1.0]})

    cfg = DummyConfig(
        data={
            "csv_path": "input.csv",
            "nan_policy": "zero_fill",
            "nan_limit": 3,
        },
        export={},
    )

    monkeypatch.setattr(run_analysis, "load", lambda _: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", loader_with_nan)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: default_result)
    monkeypatch.setattr(
        run_analysis.export, "format_summary_text", lambda *_, **__: "summary"
    )
    monkeypatch.setattr(run_analysis.export, "export_data", lambda *_, **__: None)

    exit_code = run_analysis.main(["-c", "config.yml", "--detailed"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Sharpe" in captured.out
    assert calls[0]["kwargs"]["errors"] == "raise"
    assert calls[0]["kwargs"]["nan_policy"].strip().lower() == "zero_fill"
    assert calls[0]["kwargs"]["nan_limit"] == 3
