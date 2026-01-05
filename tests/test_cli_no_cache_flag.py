from types import SimpleNamespace

import pandas as pd

from trend_analysis import cli
from trend_analysis.api import RunResult


def test_cli_respects_no_cache_flag(monkeypatch, tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=3, freq="ME"),
            "Fund": [0.01, 0.02, 0.03],
        }
    )
    df.to_csv(csv_path, index=False)

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
    )

    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "load_csv", lambda path, **_: df.copy())

    toggles: list[bool] = []
    monkeypatch.setattr(cli, "set_cache_enabled", lambda enabled: toggles.append(enabled))

    run_result = RunResult(
        metrics=pd.DataFrame({"metric": [1.0]}),
        details={"periods": []},
        seed=7,
        environment={"python": "3.11"},
    )
    monkeypatch.setattr(cli, "run_simulation", lambda *a, **k: run_result)
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a, **k: "summary")
    monkeypatch.setattr(cli.export, "export_to_excel", lambda *a, **k: None)
    monkeypatch.setattr(cli.export, "export_data", lambda *a, **k: None)

    rc = cli.main(
        [
            "run",
            "-c",
            str(tmp_path / "cfg.yml"),
            "-i",
            str(csv_path),
            "--no-cache",
        ]
    )

    assert rc == 0
    assert toggles and toggles[0] is False
