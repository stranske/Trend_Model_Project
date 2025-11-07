from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from trend_analysis import run_analysis


def _build_config(**overrides: Any) -> SimpleNamespace:
    data = overrides.pop("data", {"csv_path": "data.csv"})
    sample_split = overrides.pop(
        "sample_split",
        {
            "in_start": "2020-01-01",
            "in_end": "2020-12-31",
            "out_start": "2021-01-01",
            "out_end": "2021-12-31",
        },
    )
    export_cfg = overrides.pop(
        "export",
        {"directory": "out", "formats": ["xlsx", "json"], "filename": "analysis"},
    )
    if overrides:
        raise AssertionError(f"Unused overrides: {sorted(overrides)}")
    return SimpleNamespace(data=data, sample_split=sample_split, export=export_cfg)


def _make_result(details: dict[str, Any] | None = None) -> SimpleNamespace:
    metrics = pd.DataFrame({"metric": [1.0]})
    details = details or {"performance": "ok"}
    return SimpleNamespace(metrics=metrics, details=details)


def test_main_requires_csv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _build_config(data={})
    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)

    with pytest.raises(KeyError):
        run_analysis.main(["--config", "config.yml"])


def test_main_passes_missing_policy_and_exports_excel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_load_csv(
        path: str,
        *,
        errors: str = "raise",
        missing_policy: str | None = None,
        missing_limit: int | None = None,
    ) -> pd.DataFrame:
        calls["path"] = path
        calls["errors"] = errors
        calls["missing_policy"] = missing_policy
        calls["missing_limit"] = missing_limit
        return pd.DataFrame({"value": [1]})

    cfg = _build_config(
        data={
            "csv_path": "trend.csv",
            "missing_policy": "strict",
            "missing_limit": 5,
        }
    )

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)

    regime_table = pd.DataFrame({"regime": ["bull"], "value": [0.4]})
    regime_notes = ["note-a", "note-b"]

    export_calls: dict[str, Any] = {"excel": [], "data": []}

    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda cfg, df: _make_result({
        "performance_by_regime": regime_table,
        "regime_notes": regime_notes,
    }))

    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *a, **k: "summary")

    def fake_formatter(*args: Any, **kwargs: Any) -> str:
        return "formatter"

    monkeypatch.setattr(run_analysis.export, "make_summary_formatter", fake_formatter)
    monkeypatch.setattr(
        run_analysis.export,
        "summary_frame_from_result",
        lambda details: pd.DataFrame({"summary": ["row"]}),
    )

    def fake_export_to_excel(data: dict[str, Any], path: str, **kwargs: Any) -> None:
        export_calls["excel"].append((data, path, kwargs))

    def fake_export_data(data: dict[str, Any], path: str, *, formats: list[str]) -> None:
        export_calls["data"].append((data, path, formats))

    monkeypatch.setattr(run_analysis.export, "export_to_excel", fake_export_to_excel)
    monkeypatch.setattr(run_analysis.export, "export_data", fake_export_data)

    exit_code = run_analysis.main(["--config", "trend.yml"])

    assert exit_code == 0
    assert calls == {
        "path": "trend.csv",
        "errors": "raise",
        "missing_policy": "strict",
        "missing_limit": 5,
    }
    assert export_calls["excel"], "export_to_excel should be invoked"
    assert export_calls["data"], "export_data should be invoked for non-excel formats"
    excel_data, excel_path, _ = export_calls["excel"][0]
    assert "performance_by_regime" in excel_data
    assert "regime_notes" in excel_data
    assert excel_path.endswith("analysis.xlsx")
    other_data, other_path, formats = export_calls["data"][0]
    assert other_path.endswith("analysis")
    assert formats == ["json"]
    # The supplementary export data should match the excel export payload
    assert other_data.keys() >= {"metrics", "performance_by_regime", "regime_notes"}


def test_main_uses_nan_aliases_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_load_csv(
        path: str,
        *,
        errors: str = "raise",
        nan_policy: str | None = None,
        nan_limit: int | None = None,
    ) -> pd.DataFrame:
        calls["path"] = path
        calls["errors"] = errors
        calls["nan_policy"] = nan_policy
        calls["nan_limit"] = nan_limit
        return pd.DataFrame({"value": [1]})

    cfg = _build_config(
        data={"csv_path": "alias.csv", "nan_policy": "lenient", "nan_limit": 7},
        export={},
    )

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda cfg, df: _make_result())
    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *a, **k: "summary")
    monkeypatch.setattr(run_analysis.export, "make_summary_formatter", lambda *a, **k: "formatter")
    monkeypatch.setattr(
        run_analysis.export,
        "summary_frame_from_result",
        lambda details: pd.DataFrame({"summary": ["row"]}),
    )

    export_targets: list[str] = []

    def fake_export_to_excel(data: dict[str, Any], path: str, **kwargs: Any) -> None:
        export_targets.append(path)

    monkeypatch.setattr(run_analysis.export, "export_to_excel", fake_export_to_excel)
    monkeypatch.setattr(run_analysis.export, "export_data", lambda *a, **k: None)

    exit_code = run_analysis.main(["--config", "alias.yml"])

    assert exit_code == 0
    assert calls == {
        "path": "alias.csv",
        "errors": "raise",
        "nan_policy": "lenient",
        "nan_limit": 7,
    }
    # Defaults should be applied when export config is empty.
    assert export_targets == ["outputs/analysis.xlsx"]


def test_main_exports_without_excel_formats(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _build_config(
        export={"directory": "custom", "formats": ["json"], "filename": "report"}
    )

    def fake_load_csv(path: str, *, errors: str = "raise") -> pd.DataFrame:
        return pd.DataFrame({"value": [1]})

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda cfg, df: _make_result())
    monkeypatch.setattr(run_analysis.export, "format_summary_text", lambda *a, **k: "summary")
    monkeypatch.setattr(run_analysis.export, "make_summary_formatter", lambda *a, **k: "formatter")
    monkeypatch.setattr(run_analysis.export, "summary_frame_from_result", lambda *a, **k: pd.DataFrame())

    export_calls: dict[str, int] = {"excel": 0, "data": 0}

    monkeypatch.setattr(run_analysis.export, "export_to_excel", lambda *a, **k: export_calls.__setitem__("excel", export_calls["excel"] + 1))

    def fake_export_data(data: dict[str, Any], path: str, *, formats: list[str]) -> None:
        export_calls["data"] += 1
        assert path.endswith("custom/report")
        assert formats == ["json"]

    monkeypatch.setattr(run_analysis.export, "export_data", fake_export_data)

    exit_code = run_analysis.main(["--config", "custom.yml"])

    assert exit_code == 0
    assert export_calls == {"excel": 0, "data": 1}


def test_main_detailed_handles_empty_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _build_config()

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", lambda *a, **k: pd.DataFrame({"value": [1]}))

    empty_result = SimpleNamespace(metrics=pd.DataFrame(), details={})
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda cfg, df: empty_result)

    captured: list[str] = []
    monkeypatch.setattr("builtins.print", lambda message: captured.append(message))

    exit_code = run_analysis.main(["--config", "trend.yml", "--detailed"])

    assert exit_code == 0
    assert captured == ["No results"]


def test_main_raises_when_loader_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _build_config()

    monkeypatch.setattr(run_analysis, "load", lambda path: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", lambda *a, **k: None)

    with pytest.raises(FileNotFoundError):
        run_analysis.main(["--config", "trend.yml"])
