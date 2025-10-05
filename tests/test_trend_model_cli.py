"""Unit tests covering the ``trend-run`` helper module."""

from __future__ import annotations

import types
from pathlib import Path

import pandas as pd
import pytest

import trend_model.cli as trend_model_cli


def test_load_configuration_supports_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A TOML configuration should be parsed and validated via ``load_config``."""

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
version = "1.0"

[data]
csv_path = "data.csv"

[preprocessing]

[vol_adjust]

[sample_split]

[portfolio]

[metrics]

[export]

[run]
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_load_config(payload: dict[str, object]) -> object:
        captured["payload"] = payload
        return types.SimpleNamespace()

    monkeypatch.setattr(trend_model_cli, "load_config", fake_load_config)

    resolved_path, cfg = trend_model_cli._load_configuration(config_path)

    assert resolved_path == config_path.resolve()
    assert isinstance(cfg, types.SimpleNamespace)
    assert captured["payload"]["data"]["csv_path"] == "data.csv"


def test_run_executes_pipeline_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run`` should drive the legacy helpers and emit artefacts."""

    config_path = tmp_path / "config.yml"
    config_path.write_text("stub", encoding="utf-8")

    config = types.SimpleNamespace(
        data={"csv_path": "returns.csv"},
        export={},
        sample_split={},
        portfolio={},
    )

    result = types.SimpleNamespace(metrics=pd.DataFrame({"metric": [1]}), details={})

    monkeypatch.setattr(
        trend_model_cli,
        "_load_configuration",
        lambda path: (config_path.resolve(), config),
    )
    monkeypatch.setattr(
        trend_model_cli.trend_cli,
        "_resolve_returns_path",
        lambda cfg_path, cfg, override: tmp_path / "returns.csv",
    )
    monkeypatch.setattr(
        trend_model_cli.trend_cli,
        "_ensure_dataframe",
        lambda path: pd.DataFrame({"Date": ["2020-01-31"], "Fund": [0.1]}),
    )
    monkeypatch.setattr(
        trend_model_cli.trend_cli,
        "_determine_seed",
        lambda cfg, override: 17,
    )

    pipeline_calls: list[tuple] = []

    def fake_run_pipeline(cfg, df, **kwargs):
        pipeline_calls.append((cfg, df, kwargs))
        return result, "run123", None

    monkeypatch.setattr(trend_model_cli.trend_cli, "_run_pipeline", fake_run_pipeline)

    summary_calls: list[tuple] = []
    monkeypatch.setattr(
        trend_model_cli.trend_cli,
        "_print_summary",
        lambda cfg, res: summary_calls.append((cfg, res)),
    )

    export_calls: list[tuple] = []
    monkeypatch.setattr(
        trend_model_cli.trend_cli,
        "_prepare_export_config",
        lambda cfg, directory, formats: export_calls.append((directory, formats)),
    )

    write_calls: list[tuple] = []
    monkeypatch.setattr(
        trend_model_cli.trend_cli,
        "_write_report_files",
        lambda out_dir, cfg, res, run_id: write_calls.append((out_dir, run_id)),
    )

    monkeypatch.setattr(
        trend_model_cli,
        "generate_unified_report",
        lambda result, cfg, run_id, include_pdf: types.SimpleNamespace(
            html="<html></html>", pdf_bytes=b"PDF" if include_pdf else None
        ),
    )

    output_path = tmp_path / "report.html"
    out_dir = tmp_path / "exports"

    rc = trend_model_cli.run(
        [
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--out",
            str(out_dir),
            "--pdf",
        ]
    )

    assert rc == 0
    assert pipeline_calls and summary_calls
    assert export_calls == [(out_dir.resolve(), trend_model_cli.DEFAULT_REPORT_FORMATS)]
    assert write_calls == [(out_dir.resolve(), "run123")]
    assert output_path.exists()
    assert output_path.with_suffix(".pdf").exists()


def test_run_handles_report_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Runtime failures should surface as a clean exit code and message."""

    config_path = tmp_path / "config.yml"
    config_path.write_text("stub", encoding="utf-8")
    config = types.SimpleNamespace(data={"csv_path": "returns.csv"}, export={}, sample_split={})

    monkeypatch.setattr(
        trend_model_cli,
        "_load_configuration",
        lambda path: (config_path.resolve(), config),
    )
    monkeypatch.setattr(
        trend_model_cli.trend_cli,
        "_resolve_returns_path",
        lambda cfg_path, cfg, override: tmp_path / "returns.csv",
    )
    monkeypatch.setattr(
        trend_model_cli.trend_cli,
        "_ensure_dataframe",
        lambda path: pd.DataFrame({}),
    )
    monkeypatch.setattr(
        trend_model_cli.trend_cli,
        "_determine_seed",
        lambda cfg, override: 1,
    )
    monkeypatch.setattr(
        trend_model_cli.trend_cli,
        "_run_pipeline",
        lambda *args, **kwargs: (types.SimpleNamespace(metrics=pd.DataFrame(), details={}), "r1", None),
    )
    monkeypatch.setattr(
        trend_model_cli.trend_cli,
        "_print_summary",
        lambda cfg, res: None,
    )
    monkeypatch.setattr(
        trend_model_cli,
        "generate_unified_report",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    rc = trend_model_cli.run(
        ["--config", str(config_path), "--output", str(tmp_path / "report.html")]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "Error: boom" in captured.err

