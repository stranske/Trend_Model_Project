"""Focused regression tests for the run_analysis CLI helper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis import run_analysis


def _make_config(data: dict[str, object]) -> SimpleNamespace:
    """Construct a lightweight config stub matching the real object."""

    return SimpleNamespace(
        data=data,
        export={"directory": None, "formats": ["json"], "filename": "analysis"},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-02",
            "out_start": "2020-03",
            "out_end": "2020-04",
        },
    )


def test_main_requires_csv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configs without a csv_path should raise a KeyError early."""

    monkeypatch.setattr(run_analysis, "load", lambda _: _make_config({}))

    with pytest.raises(KeyError, match="csv_path"):
        run_analysis.main(["-c", "config.yml"])


def test_main_populates_missing_policy_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit missing-policy configuration should be forwarded verbatim."""

    captured: dict[str, object] = {}

    def fake_load(_: str) -> SimpleNamespace:
        return _make_config(
            {
                "csv_path": "data.csv",
                "missing_policy": "zero",
                "missing_limit": 12,
            }
        )

    def fake_load_csv(
        path: str,
        *,
        errors: str = "raise",
        missing_policy: str | None = None,
        missing_limit: int | None = None,
    ) -> pd.DataFrame:
        captured["args"] = {
            "path": path,
            "errors": errors,
            "missing_policy": missing_policy,
            "missing_limit": missing_limit,
        }
        return pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-31", periods=3, freq="ME"),
                "FundA": [0.01, 0.02, 0.03],
            }
        )

    def fake_run_simulation(cfg: SimpleNamespace, df: pd.DataFrame) -> SimpleNamespace:
        assert not df.empty
        return SimpleNamespace(
            metrics=pd.DataFrame({"metric": [1.0]}),
            details={"summary": "ok", "performance_by_regime": pd.DataFrame()},
        )

    monkeypatch.setattr(run_analysis, "load", fake_load)
    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(run_analysis.api, "run_simulation", fake_run_simulation)
    monkeypatch.setattr(
        run_analysis.export, "format_summary_text", lambda *_, **__: "summary"
    )

    exit_code = run_analysis.main(["-c", "config.yml"])

    assert exit_code == 0
    assert captured["args"] == {
        "path": "data.csv",
        "errors": "raise",
        "missing_policy": "zero",
        "missing_limit": 12,
    }


def test_main_translates_nan_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy nan_* keys should still be accepted when load_csv lacks the modern names."""

    captured: dict[str, object] = {}

    def fake_load(_: str) -> SimpleNamespace:
        return _make_config(
            {
                "csv_path": "data.csv",
                "nan_policy": "ffill",
                "nan_limit": 7,
            }
        )

    def fake_load_csv(
        path: str,
        *,
        errors: str = "raise",
        nan_policy: str | None = None,
        nan_limit: int | None = None,
    ) -> pd.DataFrame:
        captured["kwargs"] = {
            "path": path,
            "errors": errors,
            "nan_policy": nan_policy,
            "nan_limit": nan_limit,
        }
        return pd.DataFrame(
            {
                "Date": pd.date_range("2024-02-29", periods=2, freq="ME"),
                "FundA": [0.05, 0.01],
            }
        )

    monkeypatch.setattr(run_analysis, "load", fake_load)
    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(
        run_analysis.api,
        "run_simulation",
        lambda *_: SimpleNamespace(
            metrics=pd.DataFrame({"metric": [2.0]}), details={"summary": "ok"}
        ),
    )
    monkeypatch.setattr(
        run_analysis.export, "format_summary_text", lambda *_, **__: "summary"
    )

    exit_code = run_analysis.main(["-c", "config.yml"])

    assert exit_code == 0
    assert captured["kwargs"] == {
        "path": "data.csv",
        "errors": "raise",
        "nan_policy": "ffill",
        "nan_limit": 7,
    }


def test_main_skips_unsupported_missing_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """No kwargs should be forwarded when load_csv lacks both legacy and new names."""

    def fake_load(_: str) -> SimpleNamespace:
        return _make_config(
            {
                "csv_path": "data.csv",
                "missing_policy": "drop",
                "missing_limit": 5,
            }
        )

    def fake_load_csv(path: str, *, errors: str = "raise") -> pd.DataFrame:
        assert path == "data.csv"
        assert errors == "raise"
        return pd.DataFrame(
            {
                "Date": pd.date_range("2024-03-31", periods=2, freq="ME"),
                "FundA": [0.01, 0.015],
            }
        )

    monkeypatch.setattr(run_analysis, "load", fake_load)
    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(
        run_analysis.api,
        "run_simulation",
        lambda *_: SimpleNamespace(
            metrics=pd.DataFrame({"metric": [3.0]}), details={"summary": "ok"}
        ),
    )
    monkeypatch.setattr(
        run_analysis.export, "format_summary_text", lambda *_, **__: "summary"
    )

    exit_code = run_analysis.main(["-c", "config.yml"])

    assert exit_code == 0


def test_main_raises_when_load_csv_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing CSV should bubble up as FileNotFoundError."""

    monkeypatch.setattr(
        run_analysis,
        "load",
        lambda _: _make_config({"csv_path": "missing.csv"}),
    )
    monkeypatch.setattr(run_analysis, "load_csv", lambda *_, **__: None)

    with pytest.raises(FileNotFoundError):
        run_analysis.main(["-c", "config.yml"])


def test_main_handles_detailed_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When --detailed is provided the metrics-only branch should trigger."""

    monkeypatch.setattr(
        run_analysis,
        "load",
        lambda _: _make_config({"csv_path": "data.csv"}),
    )

    def fake_load_csv(
        path: str,
        *,
        errors: str = "raise",
    ) -> pd.DataFrame:
        assert path == "data.csv"
        return pd.DataFrame({"Date": pd.date_range("2024-01-31", periods=1, freq="ME")})

    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(
        run_analysis.api,
        "run_simulation",
        lambda *_: SimpleNamespace(metrics=pd.DataFrame(), details={}),
    )

    exit_code = run_analysis.main(["-c", "config.yml", "--detailed"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "No results" in captured.out


def test_main_applies_default_export_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When no export settings are provided, defaults should be used."""

    sentinel_dir = tmp_path / "export"
    sentinel_formats = ["json"]
    export_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    cfg = _make_config({"csv_path": "data.csv"})
    cfg.export = {"directory": None, "formats": [], "filename": "analysis"}

    monkeypatch.setattr(run_analysis, "DEFAULT_OUTPUT_DIRECTORY", str(sentinel_dir))
    monkeypatch.setattr(run_analysis, "DEFAULT_OUTPUT_FORMATS", sentinel_formats)
    monkeypatch.setattr(run_analysis, "load", lambda _: cfg)

    def fake_load_csv(path: str, *, errors: str = "raise") -> pd.DataFrame:
        assert path == "data.csv"
        assert errors == "raise"
        return pd.DataFrame({"Date": pd.date_range("2024-01-31", periods=1, freq="ME")})

    monkeypatch.setattr(run_analysis, "load_csv", fake_load_csv)
    monkeypatch.setattr(
        run_analysis.export,
        "format_summary_text",
        lambda *_: "summary text",
    )
    monkeypatch.setattr(
        run_analysis.api,
        "run_simulation",
        lambda *_: SimpleNamespace(
            metrics=pd.DataFrame({"metric": [1.23]}),
            details={
                "summary": "ok",
                "performance_by_regime": pd.DataFrame(),
                "regime_notes": ["note"],
            },
        ),
    )
    monkeypatch.setattr(
        run_analysis.export,
        "export_data",
        lambda *args, **kwargs: export_calls.append((args, kwargs)),
    )

    exit_code = run_analysis.main(["-c", "config.yml"])

    assert exit_code == 0
    assert export_calls, "export_data should be invoked with default settings"
    (data_args, data_kwargs) = export_calls[0]
    assert Path(data_args[1]).parent == sentinel_dir
    assert "formats" in data_kwargs
    assert data_kwargs["formats"] == sentinel_formats
