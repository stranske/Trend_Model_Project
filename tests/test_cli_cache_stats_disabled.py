from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from trend_analysis import cli
from trend_analysis.api import RunResult


def test_cli_suppresses_cache_stats_when_absent(monkeypatch, capsys, tmp_path):
    # Prepare tiny dataset
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=3, freq="ME"),
            "A": [0.01, 0.02, 0.03],
            "B": [0.02, 0.01, 0.00],
        }
    )
    df.to_csv(csv_path, index=False)

    # Config with no cache / performance flags set (simulate disabled cache)
    cfg = SimpleNamespace(
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-02",
            "out_start": "2020-03",
            "out_end": "2020-03",
        },
        export={"directory": "ignored", "formats": []},
        vol_adjust={},
        portfolio={},
        benchmarks={},
        metrics={},
        run={},
        performance={"enable_cache": False},
        seed=11,
    )

    # Monkeypatch loaders
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "load_csv", lambda path, **_: df.copy())

    # Fake structured logging to capture events
    log_calls = []

    def fake_log_step(run_id, step, message, level="INFO", **extra):
        payload = dict(extra)
        payload.setdefault("event", step)
        log_calls.append((run_id, step, message, payload))

    from trend_analysis import logging as run_logging

    monkeypatch.setattr(run_logging, "log_step", fake_log_step)
    monkeypatch.setattr(cli, "_log_step", fake_log_step)
    monkeypatch.setattr(run_logging, "init_run_logger", lambda run_id, path: None)
    monkeypatch.setattr(
        run_logging, "get_default_log_path", lambda run_id: tmp_path / "log.jsonl"
    )

    # Stub formatting and export side effects
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a, **k: "summary")
    monkeypatch.setattr(cli.export, "export_to_excel", lambda *a, **k: None)
    monkeypatch.setattr(cli.export, "export_data", lambda *a, **k: None)
    monkeypatch.setattr(
        cli.export, "make_summary_formatter", lambda *a, **k: lambda *_a, **_k: None
    )

    # RunResult without any embedded cache_stats structures
    run_result = RunResult(
        metrics=pd.DataFrame({"metric": [1.0]}),
        details={"periods": [{"note": "no cache"}]},
        seed=11,
        environment={"python": "3.11", "numpy": "1.26", "pandas": "2.2"},
    )
    monkeypatch.setattr(cli, "run_simulation", lambda *a, **k: run_result)

    rc = cli.main(
        [
            "run",
            "-c",
            str(tmp_path / "cfg.yml"),
            "-i",
            str(csv_path),
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    # Assert that cache statistics header is NOT printed
    assert "Cache statistics:" not in out

    # Ensure no cache_stats structured event emitted
    assert not any(
        evt for evt in log_calls if evt[1] == "cache_stats"
    ), "No cache_stats events should be emitted when caching is disabled"
