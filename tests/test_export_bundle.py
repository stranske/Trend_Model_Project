import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from trend_analysis import export
from trend_analysis.export import bundle as bundle_mod
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
    seed: int | None = None
    input_path: object | None = None
    benchmark: pd.Series | None = None
    weights: pd.DataFrame | None = None
    summary_override: dict | None = None

    def summary(self):
        if self.summary_override is not None:
            return self.summary_override
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
        assert "results/equity_bootstrap.csv" in names
        assert "charts/equity_curve.png" in names
        assert "summary.xlsx" in names
        assert "run_meta.json" in names
        assert "README.txt" in names
        assert "receipt.txt" in names
        with z.open("run_meta.json") as f:
            meta = json.load(f)
        receipt = z.read("receipt.txt").decode("utf-8")
        bootstrap_df = pd.read_csv(
            z.open("results/equity_bootstrap.csv"), comment="#", index_col=0
        )

    assert list(bootstrap_df.columns) == ["p05", "median", "p95"]
    assert not bootstrap_df.empty

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
        "results/equity_bootstrap.csv",
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


def test_export_bundle_optional_outputs(tmp_path):
    """Optional run attributes should produce additional bundle artifacts."""

    portfolio = pd.Series(
        [0.1, 0.2], index=pd.date_range("2024-01-31", periods=2, freq="ME")
    )
    benchmark = pd.Series(
        [0.05, 0.07], index=pd.date_range("2024-01-31", periods=2, freq="ME")
    )
    weights = pd.DataFrame({"fund_a": [0.6, 0.4], "fund_b": [0.4, 0.6]})

    run = DummyRun(
        portfolio=portfolio,
        config={"foo": "bar"},
        seed=None,
        input_path=object(),  # Triggers TypeError branch for Path conversion
        benchmark=benchmark,
        weights=weights,
    )

    out = tmp_path / "bundle_optional.zip"
    export_bundle(run, out)

    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert {"results/benchmark.csv", "results/weights.csv"}.issubset(names)
        assert "results/equity_bootstrap.csv" in names
        meta = json.load(z.open("run_meta.json"))

    assert meta["seed"] is None
    assert meta["input_sha256"] is None
    outputs = meta["outputs"]
    assert outputs["results/benchmark.csv"]
    assert outputs["results/weights.csv"]
    assert outputs["results/equity_bootstrap.csv"]


def test_export_bundle_summary_default_when_not_callable(tmp_path):
    """Fallback summary should use total return when attribute is not
    callable."""

    portfolio = pd.Series(
        [0.1, -0.05, 0.03],
        index=pd.date_range("2023-01-31", periods=3, freq="ME"),
    )
    run = SimpleNamespace(
        portfolio=portfolio,
        config={"foo": 1},
        seed=7,
        input_path=_write_input(tmp_path),
        summary={"should": "ignore"},
    )

    original_to_excel = pd.DataFrame.to_excel
    written: list[pd.DataFrame] = []

    def spy(self: pd.DataFrame, excel_writer, *args, **kwargs):
        written.append(self.copy())
        return original_to_excel(self, excel_writer, *args, **kwargs)

    with patch("pandas.DataFrame.to_excel", new=spy):
        export_bundle(run, tmp_path / "bundle_fallback.zip")

    assert written, "Expected summary DataFrame to be written"
    df_written = written[-1]
    assert pytest.approx(df_written.loc[0, "total_return"]) == float(portfolio.sum())


def test_export_bundle_requires_portfolio(tmp_path):
    """A portfolio attribute is mandatory for bundle creation."""

    run = SimpleNamespace(config={}, seed=0, input_path=None)
    with pytest.raises(ValueError, match="portfolio"):
        export_bundle(run, tmp_path / "bundle_missing.zip")


def test_git_hash_fallback():
    """_git_hash should return a sentinel when git metadata is unavailable."""

    with patch(
        "trend_analysis.export.bundle.subprocess.check_output",
        side_effect=FileNotFoundError,
    ):
        assert bundle_mod._git_hash() == "unknown"


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


def test_export_bundle_env_version_fallback(monkeypatch, tmp_path):
    """If importlib metadata lookup fails, fallback version should be used."""

    fake_meta = ModuleType("importlib.metadata")

    def boom(_: str) -> str:
        raise RuntimeError("no version info")

    fake_meta.version = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "importlib.metadata", fake_meta)

    run = DummyRun(
        portfolio=pd.Series(
            [0.02, 0.01], index=pd.date_range("2021-01", periods=2, freq="ME")
        ),
        config={"demo": True},
        seed=5,
        input_path=_write_input(tmp_path),
    )

    out = tmp_path / "env_fallback.zip"
    export_bundle(run, out)

    with zipfile.ZipFile(out) as z:
        meta = json.load(z.open("run_meta.json"))

    assert meta["environment"].get("trend_analysis") == "0"


def test_export_bundle_accepts_dict_portfolio(tmp_path):
    portfolio = {"2020-01-31": 0.01, "2020-02-29": -0.02}
    run = DummyRun(
        portfolio=portfolio,  # type: ignore[arg-type]
        config={},
        seed=None,
        input_path=None,
        summary_override={"total_return": 0.0},
    )
    out = tmp_path / "dict_bundle.zip"
    export_bundle(run, out)
    assert out.exists()


def test_export_bundle_rejects_list_portfolio(tmp_path):
    run = DummyRun(
        portfolio=[0.1, 0.2],  # type: ignore[arg-type]
        config={},
        seed=None,
        input_path=None,
    )
    out = tmp_path / "list_bundle.zip"
    with pytest.raises(ValueError, match="Cannot convert portfolio"):
        export_bundle(run, out)


def test_export_bundle_warns_on_generic_portfolio(tmp_path):
    class WeirdPortfolio:
        def __iter__(self):
            return iter([0.1, 0.2])

    run = DummyRun(
        portfolio=WeirdPortfolio(),  # type: ignore[arg-type]
        config={},
        seed=None,
        input_path=None,
        summary_override={"total_return": 0.0},
    )
    out = tmp_path / "warn_bundle.zip"
    with pytest.warns(UserWarning):
        export_bundle(run, out)


def test_export_bundle_records_log_file(tmp_path):
    input_path = _write_input(tmp_path)
    run = DummyRun(
        portfolio=pd.Series(
            [0.01], index=pd.date_range("2020-01-31", periods=1, freq="ME")
        ),
        config={},
        seed=1,
        input_path=input_path,
    )
    run.log_file = tmp_path / "run.log"
    run.log_file.write_text("log", encoding="utf-8")
    out = tmp_path / "log_bundle.zip"
    export_bundle(run, out)
    with zipfile.ZipFile(out) as z:
        meta = json.loads(z.read("run_meta.json").decode("utf-8"))
    assert meta["log_file"].endswith("run.log")
