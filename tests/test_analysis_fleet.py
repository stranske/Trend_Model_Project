from __future__ import annotations

import json
import socket

import pandas as pd

from trend_analysis import api
from trend_analysis.config import Config
from trend_analysis.llm.tracing import load_fleet_records


def _returns_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=12, freq="ME"),
            "RF": [0.0] * 12,
            "Manager_A": [0.02 + 0.001 * i for i in range(12)],
            "Manager_B": [0.015 + 0.002 * i for i in range(12)],
            "SPX": [0.01 + 0.001 * i for i in range(12)],
        }
    )


def _config() -> Config:
    return Config(
        version="1",
        data={
            "date_column": "Date",
            "frequency": "M",
            "risk_free_column": "RF",
        },
        preprocessing={},
        vol_adjust={"target_vol": 1.0},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-06",
            "out_start": "2020-07",
            "out_end": "2020-12",
        },
        portfolio={},
        benchmarks={"spx": "SPX"},
        metrics={},
        export={},
        run={},
        seed=42,
    )


def test_run_simulation_emits_no_secret_fleet_record_without_egress(monkeypatch, tmp_path) -> None:
    fleet_path = tmp_path / "langsmith-fleet.ndjson"
    monkeypatch.setenv("TREND_LANGSMITH_FLEET_PATH", str(fleet_path))
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    def _no_socket(*_args, **_kwargs):  # pragma: no cover - failure helper
        raise AssertionError("deterministic fleet emission must not open sockets")

    monkeypatch.setattr(socket, "socket", _no_socket)

    result = api.run_simulation(_config(), _returns_frame())

    assert not result.metrics.empty
    records = load_fleet_records(path=fleet_path)
    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == "langsmith-fleet/v1"
    assert record["source_repo"] == "stranske/Trend_Model_Project"
    assert record["operation"] == "analysis-run"
    assert record["status"] == "no_secret"
    assert record["provider"] == "deterministic"
    assert record["model"] == "trend-analysis"
    assert record["latency_ms"] is not None

    domain = record["domain"]
    assert domain["analysis_status"] == "success"
    assert domain["validation_status"] == "deterministic"
    assert domain["dataset_id"].startswith("sha256:")
    assert domain["config_fingerprint"].startswith("sha256:")
    assert domain["artifact_refs"]["analysis_summary"].startswith("sha256:")

    serialized = json.dumps(record, sort_keys=True)
    assert "Manager_A" not in serialized
    assert "Manager_B" not in serialized
    assert "SPX" not in serialized
    assert "0.021" not in serialized


def test_run_simulation_fleet_record_is_best_effort(monkeypatch, tmp_path) -> None:
    directory_path = tmp_path / "fleet-dir"
    directory_path.mkdir()
    monkeypatch.setenv("TREND_LANGSMITH_FLEET_PATH", str(directory_path))
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    result = api.run_simulation(_config(), _returns_frame())

    assert not result.metrics.empty
