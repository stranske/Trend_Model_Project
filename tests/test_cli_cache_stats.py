from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from trend_analysis.api import RunResult


def test_cli_emits_cache_stats(monkeypatch, capsys, tmp_path):
    from trend_analysis import cli

    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=4, freq="ME"),
            "A": 0.01,
            "B": 0.02,
        }
    )
    df.to_csv(csv_path, index=False)

    cfg = SimpleNamespace(
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-02",
            "out_start": "2020-03",
            "out_end": "2020-04",
        },
        export={"directory": "ignored", "formats": []},
        vol_adjust={},
        portfolio={},
        benchmarks={},
        metrics={},
        run={},
        seed=7,
    )

    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "load_csv", lambda path, **_: df.copy())

    log_calls: list[tuple[str, str, str, dict[str, object]]] = []

    def fake_log_step(run_id, step, message, level="INFO", **extra):
        payload = dict(extra)
        payload.setdefault("event", step)
        log_calls.append((run_id, step, message, payload))

    from trend_analysis import logging as run_logging

    monkeypatch.setattr(run_logging, "log_step", fake_log_step)
    monkeypatch.setattr(cli, "_log_step", fake_log_step)
    monkeypatch.setattr(run_logging, "init_run_logger", lambda run_id, path: None)
    monkeypatch.setattr(run_logging, "get_default_log_path", lambda run_id: tmp_path / "log.jsonl")
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *args, **kwargs: "summary")
    monkeypatch.setattr(cli.export, "export_to_excel", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.export, "export_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli.export,
        "make_summary_formatter",
        lambda *args, **kwargs: lambda *a, **k: None,
    )

    run_result = RunResult(
        metrics=pd.DataFrame({"metric": [1.0]}),
        details={
            "something": [
                {
                    "cache_stats": {
                        "entries": 2,
                        "hits": 5,
                        "misses": 1,
                        "incremental_updates": 3,
                    }
                }
            ]
        },
        seed=7,
        environment={"python": "3.11", "numpy": "1.26", "pandas": "2.2"},
    )

    monkeypatch.setattr(cli, "run_simulation", lambda *args, **kwargs: run_result)

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

    captured = capsys.readouterr()
    assert "Cache statistics:" in captured.out
    assert "Entries: 2" in captured.out
    assert "Incremental updates: 3" in captured.out

    cache_events = [call for call in log_calls if call[1] == "cache_stats"]
    assert cache_events, "expected cache_stats log_step call"
    event_payload = cache_events[-1][3]
    assert event_payload["event"] == "cache_stats"
    assert event_payload["entries"] == 2
    assert event_payload["hits"] == 5
    assert event_payload["misses"] == 1
    assert event_payload["incremental_updates"] == 3
