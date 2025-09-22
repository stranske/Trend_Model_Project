from __future__ import annotations

from pathlib import Path

import pandas as pd

from trend.cli import (
    SCENARIO_WINDOWS,
    _adjust_for_scenario,
    _determine_seed,
    _resolve_returns_path,
    build_parser,
    main,
)
from trend_analysis.api import RunResult


def test_build_parser_contains_expected_subcommands() -> None:
    parser = build_parser()
    expected_subcommands = {"run", "report", "stress", "app"}
    for subcommand in expected_subcommands:
        # Should not raise SystemExit
        try:
            args = parser.parse_args([subcommand])
        except SystemExit:
            assert False, f"Subcommand '{subcommand}' not recognized by parser"
        # The subcommand should be set in the namespace
        assert getattr(args, "subcommand", None) == subcommand


def test_resolve_returns_path_uses_config_directory(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("", encoding="utf-8")

    class DummyCfg:
        data = {"csv_path": "data/returns.csv"}

    resolved = _resolve_returns_path(cfg_path, DummyCfg(), None)
    assert resolved == (tmp_path / "data" / "returns.csv").resolve()


def test_determine_seed_precedence(monkeypatch) -> None:
    class DummyCfg:
        seed = 7

    cfg = DummyCfg()
    assert _determine_seed(cfg, 21) == 21
    monkeypatch.setenv("TREND_SEED", "33")
    cfg_env = DummyCfg()
    assert _determine_seed(cfg_env, None) == 33
    monkeypatch.delenv("TREND_SEED")
    cfg_default = DummyCfg()
    assert _determine_seed(cfg_default, None) == 7


def test_adjust_for_scenario_updates_sample_split() -> None:
    class DummyCfg:
        sample_split = {}

    cfg = DummyCfg()
    _adjust_for_scenario(cfg, "2008")
    assert cfg.sample_split["in_start"] == SCENARIO_WINDOWS["2008"][0][0]
    assert cfg.sample_split["out_end"] == SCENARIO_WINDOWS["2008"][1][1]


def test_main_run_invokes_pipeline(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Date,Mgr_01\n2020-01-31,0.01\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_ensure_dataframe(path: Path) -> pd.DataFrame:
        captured["returns_path"] = path
        return pd.DataFrame({"Date": ["2020-01-31"], "Mgr_01": [0.01]})

    def fake_run_pipeline(*args, **kwargs):  # type: ignore[override]
        captured["pipeline_args"] = kwargs
        return RunResult(pd.DataFrame(), {}, 42, {}), "run123", Path("log.json")

    monkeypatch.setattr("trend.cli._ensure_dataframe", fake_ensure_dataframe)
    monkeypatch.setattr("trend.cli._run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("trend.cli._print_summary", lambda *args, **kwargs: None)

    exit_code = main(["run", "--config", "config/demo.yml", "--returns", str(csv_path)])

    assert exit_code == 0
    assert captured["returns_path"] == csv_path.resolve()
    pipeline_kwargs = captured["pipeline_args"]
    assert pipeline_kwargs["source_path"] == csv_path.resolve()
    assert pipeline_kwargs["structured_log"] is True


def test_main_report_uses_requested_directory(monkeypatch, tmp_path: Path) -> None:
    dummy_result = RunResult(pd.DataFrame(), {}, 42, {})

    def fake_run_pipeline(*args, **kwargs):  # type: ignore[override]
        return dummy_result, "runABC", None

    monkeypatch.setattr("trend.cli._ensure_dataframe", lambda _p: pd.DataFrame())
    monkeypatch.setattr("trend.cli._run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("trend.cli._print_summary", lambda *args, **kwargs: None)
    recorded: dict[str, Path] = {}

    def fake_write(out_dir: Path, *args, **kwargs) -> None:
        recorded["dir"] = out_dir

    monkeypatch.setattr("trend.cli._write_report_files", fake_write)

    out_dir = tmp_path / "reports"
    exit_code = main(
        [
            "report",
            "--config",
            "config/demo.yml",
            "--out",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert recorded["dir"] == out_dir


def test_main_stress_passes_scenario(monkeypatch, tmp_path: Path) -> None:
    dummy_result = RunResult(pd.DataFrame(), {}, 42, {})

    captured: dict[str, object] = {}

    def fake_run_pipeline(*args, **kwargs):  # type: ignore[override]
        captured.update(kwargs)
        return dummy_result, "runXYZ", None

    monkeypatch.setattr("trend.cli._ensure_dataframe", lambda _p: pd.DataFrame())
    monkeypatch.setattr("trend.cli._run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("trend.cli._print_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr("trend.cli._write_report_files", lambda *args, **kwargs: None)

    exit_code = main(
        [
            "stress",
            "--config",
            "config/demo.yml",
            "--scenario",
            "2008",
        ]
    )

    assert exit_code == 0
    assert captured["structured_log"] is False
