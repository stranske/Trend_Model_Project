from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from trend_analysis import run_multi_analysis


def test_main_prints_message_when_no_results(monkeypatch, capsys):
    stub_cfg = SimpleNamespace(export={})

    monkeypatch.setattr(run_multi_analysis, "load", lambda path: stub_cfg)
    monkeypatch.setattr(run_multi_analysis, "run_mp", lambda cfg: [])

    exit_code = run_multi_analysis.main(["-c", "demo.yml"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "No results" in captured.out


def test_main_uses_default_export_targets(monkeypatch):
    stub_cfg = SimpleNamespace(export={})
    results = [
        {
            "period": ("2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"),
            "metrics": {"Sharpe": 1.23},
        }
    ]

    calls: dict[str, object] = {}

    def fake_export(results_arg, path, *, formats, include_metrics):  # noqa: D401
        calls["results"] = results_arg
        calls["path"] = Path(path)
        calls["formats"] = formats
        calls["include_metrics"] = include_metrics

    monkeypatch.setattr(run_multi_analysis, "load", lambda path: stub_cfg)
    monkeypatch.setattr(run_multi_analysis, "run_mp", lambda cfg: results)
    monkeypatch.setattr(run_multi_analysis.export, "export_phase1_multi_metrics", fake_export)

    exit_code = run_multi_analysis.main(["-c", "demo.yml"])

    assert exit_code == 0
    assert calls["results"] == results
    assert calls["path"] == Path("outputs") / "analysis"
    assert calls["formats"] == ["excel"]
    assert calls["include_metrics"] is True
