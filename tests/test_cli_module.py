from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# Import from the root-level cli module, not trend_analysis.cli
import cli


def _write_csv(tmp_path: Path, rows: list[dict[str, Any]], *, name: str = "returns.csv") -> Path:
    csv_path = tmp_path / name
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_load_returns_validates_csv_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="data.csv_path must be provided"):
        cli._load_returns({}, base_dir=tmp_path)


def test_load_returns_parses_dates_and_columns(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        [
            {"Date": "2020-01-02", "a": 1, "b": 2},
            {"Date": "2020-01-01", "a": 3, "b": 4},
        ],
    )

    result = cli._load_returns(
        {"csv_path": csv_path.name, "columns": ["b", "a"]}, base_dir=tmp_path
    )

    assert list(result.columns) == ["b", "a"]
    assert result.index[0].strftime("%Y-%m-%d") == "2020-01-01"
    assert result.loc["2020-01-02", "a"] == 1.0


def test_load_returns_errors_on_missing_date_or_numeric_columns(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, [{"Date": "2020-01-01", "a": "x"}])

    with pytest.raises(ValueError, match="Date column 'Missing'"):
        cli._load_returns({"csv_path": csv_path, "date_column": "Missing"}, base_dir=tmp_path)

    with pytest.raises(ValueError, match="No numeric columns"):
        cli._load_returns({"csv_path": csv_path}, base_dir=tmp_path)


def test_load_cv_spec_reads_config_and_resolves_paths(tmp_path: Path) -> None:
    _write_csv(tmp_path, [{"Date": "2020-01-01", "r1": 1}])
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
data:
  csv_path: returns.csv
  date_column: Date
params:
  alpha: [0.1, 0.2]
cv:
  folds: 4
  expand: false
output:
  dir: results/output
""",
        encoding="utf-8",
    )

    data, params, folds, expand, output_path = cli._load_cv_spec(cfg_path)

    assert folds == 4
    assert expand is False
    assert params == {"alpha": [0.1, 0.2]}
    assert output_path.is_absolute()
    assert output_path.name == "output"
    assert set(data.columns) == {"r1"}


def test_build_parser_sets_expected_expand_defaults() -> None:
    parser = cli._build_parser()

    args_default = parser.parse_args(["cv", "--config", "c.yaml"])
    assert args_default.expand is None

    args_expand = parser.parse_args(["cv", "--config", "c.yaml", "--expand"])
    assert args_expand.expand is True

    args_roll = parser.parse_args(["cv", "--config", "c.yaml", "--rolling"])
    assert args_roll.expand is False


def test_handle_cv_overrides_config_and_exports(monkeypatch, tmp_path: Path, capsys) -> None:
    _write_csv(tmp_path, [{"Date": "2020-01-01", "r": 1}])
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
data:
  csv_path: returns.csv
cv:
  folds: 2
  expand: true
output:
  dir: relative/out
""",
        encoding="utf-8",
    )

    called = {}

    def fake_walk_forward(data, *, folds, expand, params):
        called["walk_forward"] = {
            "folds": folds,
            "expand": expand,
            "params": params,
            "rows": len(data),
        }
        return "report"

    def fake_export(report, output_dir):
        called["export"] = output_dir
        return {
            "folds": output_dir / "f.csv",
            "summary": output_dir / "s.csv",
            "markdown": output_dir / "m.md",
        }

    monkeypatch.setitem(
        sys.modules,
        "analysis.cv",
        type("m", (), {"walk_forward": fake_walk_forward, "export_report": fake_export}),
    )

    args = argparse.Namespace(
        config=str(cfg_path),
        folds=5,
        expand=False,
        output_dir=str(tmp_path / "custom/out"),
    )

    exit_code = cli._handle_cv(args)

    captured = capsys.readouterr().out
    assert "Folds written" in captured
    assert exit_code == 0
    assert called["walk_forward"] == {
        "folds": 5,
        "expand": False,
        "params": {},
        "rows": 1,
    }
    assert Path(called["export"]).resolve() == (tmp_path / "custom/out").resolve()


def test_handle_cv_errors_on_missing_config(tmp_path: Path) -> None:
    args = argparse.Namespace(
        config=str(tmp_path / "missing.yaml"),
        folds=None,
        expand=None,
        output_dir=None,
    )

    with pytest.raises(SystemExit, match="Config not found"):
        cli._handle_cv(args)


def test_handle_report_invokes_renderer(monkeypatch, tmp_path: Path, capsys) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("{}", encoding="utf-8")

    called = {}

    def fake_load(path):
        called["load"] = Path(path)
        return {"results": True}

    def fake_render(results, out):
        called["render"] = {"results": results, "out": out}
        return Path("md"), Path("png")

    monkeypatch.setattr(cli, "load_results_payload", fake_load)
    monkeypatch.setattr(cli, "render", fake_render)

    args = argparse.Namespace(last_run=str(payload_path), output=str(tmp_path / "out.md"))

    exit_code = cli._handle_report(args)

    captured = capsys.readouterr().out
    assert "Tearsheet written" in captured
    assert exit_code == 0
    assert called == {
        "load": payload_path,
        "render": {"results": {"results": True}, "out": (tmp_path / "out.md")},
    }


def test_main_dispatches_to_handlers(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(cli, "_handle_cv", lambda args: calls.append("cv") or 0)
    monkeypatch.setattr(cli, "_handle_report", lambda args: calls.append("report") or 0)

    assert cli.main(["cv", "--config", "dummy.yaml"]) == 0
    assert cli.main(["report", "--last-run", "r.json"]) == 0
    assert calls == ["cv", "report"]
