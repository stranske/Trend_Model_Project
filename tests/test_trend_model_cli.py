from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trend_model import cli


def test_load_configuration_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cli._load_configuration(str(tmp_path / "absent.yml"))


def test_load_configuration_with_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[section]\nvalue = 1\n", encoding="utf-8")

    payload = {"test": "ok"}
    load_calls: list[Path] = []

    monkeypatch.setattr(cli, "_load_toml_payload", lambda path: payload)

    def fake_load_config(data: dict[str, object]) -> dict[str, object]:
        load_calls.append(Path.cwd())
        assert data == payload
        return {"loaded": True}

    monkeypatch.setattr(cli, "load_config", fake_load_config)

    original_cwd = Path.cwd()
    resolved_path, config = cli._load_configuration(str(cfg_path))

    assert resolved_path == cfg_path
    assert config == {"loaded": True}
    assert load_calls == [cfg_path.parent]
    assert Path.cwd() == original_cwd


def test_run_requires_artefacts_when_formats_supplied(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("config", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        cli.run(["--config", str(cfg_path), "--formats", "csv"])

    assert exc.value.code == 2


def test_run_success_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("config", encoding="utf-8")
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text("date,value\n2024-01,1\n", encoding="utf-8")

    export_dir = tmp_path / "artefacts"
    report_path = tmp_path / "output" / "report.html"

    cfg_obj = SimpleNamespace()

    monkeypatch.setattr(cli, "_load_configuration", lambda path: (Path(path), cfg_obj))
    monkeypatch.setattr(cli, "_resolve_returns_path", lambda *_: returns_path)
    monkeypatch.setattr(cli, "_ensure_dataframe", lambda *_: pd.DataFrame({"v": [1]}))
    monkeypatch.setattr(cli, "_determine_seed", lambda *_: 123)

    result_obj = SimpleNamespace()
    monkeypatch.setattr(
        cli,
        "_run_pipeline",
        lambda *_args, **_kwargs: (result_obj, "run123", tmp_path / "log.jsonl"),
    )
    written = []
    monkeypatch.setattr(cli, "_print_summary", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_write_report_files", lambda *a, **k: written.append(a))
    monkeypatch.setattr(
        cli,
        "_resolve_report_output_path",
        lambda *_: report_path,
    )

    class Artefacts:
        def __init__(self) -> None:
            self.html = "<html></html>"
            self.pdf_bytes = b"pdf"

    monkeypatch.setattr(cli, "generate_unified_report", lambda *a, **k: Artefacts())

    exit_code = cli.run(
        [
            "--config",
            str(cfg_path),
            "--returns",
            str(returns_path),
            "--output",
            str(report_path),
            "--artefacts",
            str(export_dir),
            "--formats",
            "csv",
            "json",
            "--pdf",
        ]
    )

    assert exit_code == 0
    assert (report_path).read_text(encoding="utf-8") == "<html></html>"
    assert (report_path.with_suffix(".pdf")).read_bytes() == b"pdf"
    assert written and written[0][0] == export_dir
