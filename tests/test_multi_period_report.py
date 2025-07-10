from pathlib import Path

import pandas as pd

from trend_analysis.config import Config
from trend_analysis.multi_period.engine import run
from trend_analysis.multi_period.report import build_frames, export_multi_period


def make_cfg(tmp_path: Path) -> Config:
    csv = tmp_path / "data.csv"
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    pd.DataFrame({"Date": dates, "RF": 0.0, "A": 0.01, "B": 0.02}).to_csv(
        csv, index=False
    )
    return Config(
        version="1",
        data={"csv_path": str(csv)},
        preprocessing={},
        vol_adjust={"target_vol": 1.0},
        sample_split={},
        portfolio={},
        metrics={},
        export={},
        run={},
        multi_period={
            "frequency": "M",
            "in_sample_len": 2,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-06",
        },
    )


def test_build_frames_has_summary(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    res = run(cfg)
    frames = build_frames(res)
    assert "summary" in frames
    assert frames["summary"].shape[0] == len(res["summary"]["stats"])  # type: ignore[index]
    assert any(k.startswith("period_") for k in frames)


def test_export_multi_period_outputs(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    res = run(cfg)
    out = tmp_path / "out"
    export_multi_period(res, str(out), formats=["excel", "csv"])
    xlsx = out / "analysis.xlsx"
    csv = out / "analysis.csv"
    assert xlsx.exists()
    assert csv.exists()
    expected = {f"period_{i}" for i in range(1, len(res["periods"]) + 1)}  # type: ignore[arg-type]
    expected.add("summary")
    with pd.ExcelFile(xlsx) as xf:
        assert set(xf.sheet_names) == expected
    data = pd.read_csv(csv)
    assert set(data["sheet"]) == expected
