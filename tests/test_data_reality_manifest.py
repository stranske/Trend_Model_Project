from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trend_analysis.data import load_csv
from trend_analysis.export.run_envelope import to_run_envelope
from trend_analysis.io.market_data import MissingPolicyFillDetails
from trend_analysis.reporting.run_artifacts import write_run_artifacts


class _Result:
    seed = 42
    timings = {"wall_ms": 1.0}
    warnings: list[dict[str, object]] = []
    environment: dict[str, object] = {}
    diagnostic = None


def _write_manifest(tmp_path: Path, frame: pd.DataFrame) -> dict[str, object]:
    input_path = tmp_path / "returns.csv"
    input_path.write_text("Date,A\n2021-01-31,0.01\n", encoding="utf-8")
    run_dir = write_run_artifacts(
        output_dir=tmp_path / "out",
        run_id="data-reality",
        config={"data": {"missing_policy": "ffill"}},
        config_path="config.yml",
        input_path=input_path,
        data_frame=frame,
        metrics_frame=pd.DataFrame(),
        run_details={},
        exported_files=[],
        summary_text="",
    )
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def test_partial_missing_instrument_recorded(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-31", periods=3, freq="ME"),
            "Mgr_A": [0.01, None, 0.03],
            "Mgr_B": [0.02, 0.02, 0.01],
        }
    )
    frame.attrs["market_data_missing_policy"] = "ffill"
    frame.attrs["market_data_missing_policy_limit"] = 2
    frame.attrs["market_data_missing_policy_summary"] = "ffill applied to Mgr_A"
    frame.attrs["market_data_frequency_missing_periods"] = 1
    frame.attrs["market_data_frequency_max_gap_periods"] = 1
    frame.attrs["market_data"] = {
        "metadata": type(
            "Metadata",
            (),
            {
                "missing_policy_filled": {
                    "Mgr_A": MissingPolicyFillDetails(method="ffill", count=1)
                },
                "missing_policy_dropped": [],
            },
        )()
    }

    manifest = _write_manifest(tmp_path, frame)

    reality = manifest["data_reality"]
    assert reality["policy"] == "ffill"
    assert reality["policy_limit"] == 2
    assert reality["frequency_missing_periods"] == 1
    mgr_a = next(item for item in reality["instruments"] if item["label"] == "Mgr_A")
    assert mgr_a["missing_count"] == 1
    assert mgr_a["disposition"] == "filled"
    assert reality["filled"] == [
        {
            "label": "Mgr_A",
            "disposition": "filled",
            "reason": "ffill",
            "method": "ffill",
            "count": 1,
        }
    ]


def test_all_good_run_emits_empty_lists(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-31", periods=2, freq="ME"),
            "Mgr_A": [0.01, 0.03],
        }
    )
    frame.attrs["market_data_missing_policy"] = "drop"
    frame.attrs["market_data_missing_policy_summary"] = "no missing values"

    manifest = _write_manifest(tmp_path, frame)

    reality = manifest["data_reality"]
    assert reality["policy"] == "drop"
    assert reality["filled"] == []
    assert reality["dropped"] == []
    assert reality["unknown"] == []
    assert reality["instruments"][0]["disposition"] == "kept"


def test_demo_fixture_manifest_records_configured_missing_policy(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    frame = load_csv(
        str(repo_root / "demo" / "demo_returns.csv"),
        missing_policy="drop",
        missing_limit=None,
    )

    manifest = _write_manifest(tmp_path, frame)

    reality = manifest["data_reality"]
    assert reality["policy"] == "drop"
    assert "instruments" in reality
    assert len(reality["instruments"]) > 1


def test_run_envelope_projects_manifest_data_reality(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-31", periods=2, freq="ME"),
            "Mgr_A": [0.01, 0.03],
        }
    )
    frame.attrs["market_data_missing_policy"] = "zero"
    manifest = _write_manifest(tmp_path, frame)
    manifest_path = Path(manifest["run_directory"]) / "manifest.json"

    envelope = to_run_envelope(_Result(), config={}, manifest_path=manifest_path)

    assert envelope["data_reality"]["policy"] == "zero"
