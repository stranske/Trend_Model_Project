import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trend_analysis import export
from trend_analysis.export.bundle import export_bundle
from trend_analysis.util.hash import sha256_config, sha256_file, sha256_text


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
        assert "receipt.txt" in names
        with z.open("run_meta.json") as f:
            meta = json.load(f)
        receipt = z.read("receipt.txt").decode("utf-8")

    input_sha = sha256_file(input_path)
    cfg_sha = sha256_config({"foo": 1})
    expected_run_id = sha256_text("|".join([input_sha, cfg_sha, str(42)]))

    assert meta["config"] == {"foo": 1}
    assert meta["seed"] == 42
    assert meta["config_sha256"] == cfg_sha
    assert meta["run_id"] == expected_run_id
    assert "python" in meta["environment"]
    assert len(meta.get("git_hash", "")) >= 7
    assert meta["input_sha256"] == input_sha
    assert "created" in meta["receipt"]
    assert f"run_id: {expected_run_id}" in receipt

    # New: outputs sha256 map
    outputs = meta.get("outputs", {})
    assert isinstance(outputs, dict) and outputs
    # Must include at least these files
    for required in [
        "results/portfolio.csv",
        "charts/equity_curve.png",
        "charts/drawdown.png",
        "summary.xlsx",
        "README.txt",
        "receipt.txt",
    ]:
        assert required in outputs
        assert isinstance(outputs[required], str)
        assert len(outputs[required]) == 64
        int(outputs[required], 16)  # valid hex


def test_receipt_deterministic(tmp_path):
    input_path = _write_input(tmp_path)
    run = DummyRun(
        portfolio=pd.Series(
            [0.01, -0.02], index=pd.date_range("2020-01", periods=2, freq="ME")
        ),
        config={"foo": 1},
        seed=42,
        input_path=input_path,
    )
    out1 = tmp_path / "bundle1.zip"
    out2 = tmp_path / "bundle2.zip"
    export_bundle(run, out1)
    export_bundle(run, out2)
    with zipfile.ZipFile(out1) as z1, zipfile.ZipFile(out2) as z2:
        r1 = z1.read("receipt.txt")
        r2 = z2.read("receipt.txt")
    assert r1 == r2


def test_export_bundle_empty_portfolio(tmp_path):
    """export_bundle should handle empty portfolio without crashing."""
    input_path = _write_input(tmp_path)
    run = DummyRun(
        portfolio=pd.Series(dtype=float),
        config={},
        seed=1,
        input_path=input_path,
    )
    out = tmp_path / "empty_bundle.zip"
    export_bundle(run, out)
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        # Placeholder charts should still be created
        assert "charts/equity_curve.png" in names
        assert "charts/drawdown.png" in names


def test_export_data_all_formats_content(tmp_path):
    df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    data = {"sheet": df}
    out = tmp_path / "report"
    export.export_data(data, str(out), formats=["csv", "xlsx", "json", "txt"])

    csv_path = tmp_path / "report_sheet.csv"
    xlsx_path = tmp_path / "report.xlsx"
    json_path = tmp_path / "report_sheet.json"
    txt_path = tmp_path / "report_sheet.txt"

    assert csv_path.exists()
    assert xlsx_path.exists()
    assert json_path.exists()
    assert txt_path.exists()

    pd.testing.assert_frame_equal(pd.read_csv(csv_path), df, check_dtype=False)
    pd.testing.assert_frame_equal(
        pd.read_excel(xlsx_path, sheet_name="sheet"), df, check_dtype=False
    )
    with open(json_path) as f:
        json_data = json.load(f)
    pd.testing.assert_frame_equal(pd.DataFrame(json_data), df, check_dtype=False)
    assert txt_path.read_text() == df.to_string(index=False)
