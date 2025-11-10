from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import trend_analysis.run_analysis as run_analysis


def _make_config(
    tmp_path: Path,
    *,
    data_overrides: dict[str, Any] | None = None,
    export_overrides: dict[str, Any] | None = None,
) -> SimpleNamespace:
    data = {"csv_path": str(tmp_path / "inputs" / "market.csv")}
    if data_overrides:
        data.update(data_overrides)

    export = {"directory": str(tmp_path / "outputs"), "formats": ["excel", "json"], "filename": "summary"}
    if export_overrides:
        export.update(export_overrides)

    return SimpleNamespace(
        data=data,
        sample_split={
            "in_start": "2020-01-01",
            "in_end": "2020-06-30",
            "out_start": "2020-07-01",
            "out_end": "2020-12-31",
        },
        export=export,
    )


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    (tmp_path / "inputs").mkdir()
    (tmp_path / "outputs").mkdir()
    return tmp_path


def test_main_exports_all_formats_and_respects_missing_policy(monkeypatch: pytest.MonkeyPatch, config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _make_config(
        config_dir,
        data_overrides={"missing_policy": {"Value": "BackFill"}, "missing_limit": {"Value": 3}},
    )

    monkeypatch.setattr(run_analysis, "load", lambda _: cfg)

    load_calls: list[tuple[Any, ...]] = []

    def fake_load_csv(
        path: str,
        *,
        errors: str = "log",
        include_date_column: bool = True,
        missing_policy: dict[str, str] | None = None,
        missing_limit: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        load_calls.append((path, errors, include_date_column, missing_policy, missing_limit, kwargs))
        return pd.DataFrame({"Value": [1.0, 2.0]})

    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)

    def fake_run_simulation(config: SimpleNamespace, frame: pd.DataFrame) -> SimpleNamespace:
        assert config is cfg
        assert list(frame.columns) == ["Value"]
        return SimpleNamespace(
            metrics=pd.DataFrame({"Sharpe": [1.23]}),
            details={
                "performance_by_regime": pd.DataFrame({"regime": ["bull"], "value": [1.0]}),
                "regime_notes": ("outperformed",),
            },
        )

    monkeypatch.setattr(run_analysis.api, "run_simulation", fake_run_simulation)

    summary_args: dict[str, tuple[Any, ...]] = {}
    def record_summary(details: dict[str, Any], *dates: str) -> str:
        summary_args["call"] = (details, *dates)
        return "summary text"

    monkeypatch.setattr(run_analysis.export, "format_summary_text", record_summary)
    monkeypatch.setattr(run_analysis.export, "make_summary_formatter", lambda *args, **kwargs: "formatter")
    monkeypatch.setattr(run_analysis.export, "summary_frame_from_result", lambda details: pd.DataFrame({"metric": [1]}))

    export_calls: list[tuple[str, Path | tuple[str, ...]]] = []

    def record_export_to_excel(data: dict[str, Any], target: str, **kwargs: Any) -> None:
        export_calls.append(("excel", Path(target)))
        assert "summary" in data
        assert "performance_by_regime" in data
        assert "regime_notes" in data

    def record_export_data(data: dict[str, Any], target: str, formats: list[str] | None = None) -> None:
        export_calls.append(("data", Path(target)))
        assert formats == ["json"]
        assert "summary" in data

    monkeypatch.setattr(run_analysis.export, "export_to_excel", record_export_to_excel)
    monkeypatch.setattr(run_analysis.export, "export_data", record_export_data)

    rc = run_analysis.main(["--config", "config.yml"])
    assert rc == 0

    output = capsys.readouterr().out
    assert "summary text" in output

    assert load_calls == [
        (
            str(config_dir / "inputs" / "market.csv"),
            "raise",
            True,
            {"Value": "BackFill"},
            {"Value": 3},
            {},
        )
    ]
    assert summary_args["call"][1:] == ("2020-01-01", "2020-06-30", "2020-07-01", "2020-12-31")
    assert [kind for kind, _ in export_calls] == ["excel", "data"]


def test_main_falls_back_to_nan_policy(monkeypatch: pytest.MonkeyPatch, config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _make_config(
        config_dir,
        data_overrides={"nan_policy": {"Rate": "Zero"}, "nan_limit": {"Rate": 4}},
        export_overrides={"directory": "", "formats": []},
    )

    monkeypatch.setattr(run_analysis, "load", lambda _: cfg)

    captured: dict[str, tuple[Any, ...]] = {}

    def fake_loader(
        path: str,
        *,
        errors: str = "log",
        nan_policy: dict[str, str] | None = None,
        nan_limit: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        captured["call"] = (path, errors, nan_policy, nan_limit, kwargs)
        return pd.DataFrame({"Value": [1.0]})

    monkeypatch.setattr(run_analysis, "load_csv", fake_loader)
    monkeypatch.setattr(run_analysis.api, "run_simulation", lambda *_: SimpleNamespace(metrics=pd.DataFrame(), details={}))

    rc = run_analysis.main(["--config", "fallback.yml"])
    assert rc == 0

    assert capsys.readouterr().out.strip() == "No results"
    assert captured["call"] == (
        str(config_dir / "inputs" / "market.csv"),
        "raise",
        {"Rate": "Zero"},
        {"Rate": 4},
        {},
    )


def test_main_detailed_flag_prints_metrics(monkeypatch: pytest.MonkeyPatch, config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _make_config(config_dir)
    monkeypatch.setattr(run_analysis, "load", lambda _: cfg)
    monkeypatch.setattr(
        run_analysis,
        "load_csv",
        lambda *args, **kwargs: pd.DataFrame({"Value": [2.0]}),
    )
    metrics = pd.DataFrame({"Return": [0.5]})
    monkeypatch.setattr(
        run_analysis.api,
        "run_simulation",
        lambda *_: SimpleNamespace(metrics=metrics, details={}),
    )

    rc = run_analysis.main(["--config", "config.yml", "--detailed"])
    assert rc == 0
    output = capsys.readouterr().out
    assert "Return" in output and "0.5" in output


def test_main_raises_when_csv_missing(monkeypatch: pytest.MonkeyPatch, config_dir: Path) -> None:
    cfg = _make_config(config_dir, data_overrides={})
    cfg.data.pop("csv_path")
    monkeypatch.setattr(run_analysis, "load", lambda _: cfg)
    with pytest.raises(KeyError):
        run_analysis.main(["--config", "config.yml"])


def test_main_raises_when_loader_returns_none(monkeypatch: pytest.MonkeyPatch, config_dir: Path) -> None:
    cfg = _make_config(config_dir)
    monkeypatch.setattr(run_analysis, "load", lambda _: cfg)
    monkeypatch.setattr(run_analysis, "load_csv", lambda *args, **kwargs: None)
    with pytest.raises(FileNotFoundError):
        run_analysis.main(["--config", "config.yml"])
