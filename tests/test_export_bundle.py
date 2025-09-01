import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from trend_analysis.export.bundle import export_bundle


def _write_input(tmp_path: Path) -> Path:
    p = tmp_path / "input.csv"
    p.write_text("x\n1\n")
    return p


@dataclass
class DummyRun:
    portfolio: pd.Series
    config: dict
    seed: int
    input_path: Path

    def summary(self):
        return {"total_return": float(self.portfolio.sum())}


def test_export_bundle(tmp_path):
    input_path = _write_input(tmp_path)
    run = DummyRun(
        portfolio=pd.Series(
            [0.01, -0.02], index=pd.date_range("2020-01", periods=2, freq="ME")
        ),
        config={"foo": 1},
        seed=42,
        input_path=input_path,
    )
    out = tmp_path / "bundle.zip"
    export_bundle(run, out)
    assert out.exists()

    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert "results/portfolio.csv" in names
        assert "charts/equity_curve.png" in names
        assert "summary.xlsx" in names
        assert "run_meta.json" in names
        assert "README.txt" in names
        with z.open("run_meta.json") as f:
            meta = json.load(f)
    assert meta["config"] == {"foo": 1}
    assert meta["seed"] == 42
    assert "python" in meta["versions"]
    assert len(meta.get("git_hash", "")) >= 7
    assert meta["input_sha256"] is not None
    assert "created" in meta["receipt"]
