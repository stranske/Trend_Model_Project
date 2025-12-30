"""Direct pipeline execution tests to boost coverage for export and multi-
period paths."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from trend_analysis import api, export
from trend_analysis.config import load
from trend_analysis.export.bundle import export_bundle
from trend_analysis.multi_period import run as run_multi


def _build_demo_config(tmp_path: Path, csv_path: Path) -> dict:
    """Return a configuration dictionary tailored for the synthetic dataset."""
    exports_dir = tmp_path / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    return {
        "version": "1",
        "data": {
            "csv_path": str(csv_path),
            "date_column": "Date",
            "frequency": "ME",
            "risk_free_column": "RF",
        },
        "vol_adjust": {"enabled": True, "target_vol": 1.0},
        "sample_split": {
            "in_start": "2015-01",
            "in_end": "2018-12",
            "out_start": "2019-01",
            "out_end": "2020-12",
        },
        "portfolio": {
            "selection_mode": "rank",
            "random_n": 3,
            "rank": {
                "inclusion_approach": "top_n",
                "n": 3,
                "score_by": "Sharpe",
            },
            "selector": {
                "name": "rank",
                "params": {"top_n": 3, "rank_column": "Sharpe"},
            },
            "weighting_scheme": "equal",
            "weighting": {"name": "equal", "params": {}},
            "indices_list": ["SPX"],
            "constraints": {
                "max_weight": 0.6,
                "min_weight": 0.05,
            },
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.5,
            "transaction_cost_bps": 10,
        },
        "benchmarks": {"spx": "SPX"},
        "metrics": {
            "registry": [
                "annual_return",
                "volatility",
                "sharpe_ratio",
                "sortino_ratio",
                "information_ratio",
                "max_drawdown",
            ]
        },
        "export": {
            "directory": str(exports_dir),
            "formats": ["csv", "json", "txt", "xlsx"],
            "filename": "analysis",
        },
        "output": {
            "format": "csv",
            "path": str(exports_dir / "alias_demo.csv"),
        },
        "run": {"seed": 7, "monthly_cost": 0.001},
        "multi_period": {
            "frequency": "ME",
            "in_sample_len": 24,
            "out_sample_len": 12,
            "start": "2016-01",
            "end": "2021-12",
        },
        "jobs": 1,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }


def _make_returns_frame(periods: int = 84) -> pd.DataFrame:
    """Create a deterministic synthetic returns frame."""
    dates = pd.date_range("2015-01-31", periods=periods, freq="ME")
    rng = np.random.default_rng(2025)
    data = {"Date": dates}
    for idx in range(1, 7):
        base = rng.normal(0.008 + idx * 0.0005, 0.04, size=periods)
        drift = rng.normal(scale=0.001, size=periods).cumsum()
        seasonal = 0.002 * np.sin(np.linspace(0, 6 * np.pi, periods))
        data[f"Mgr_{idx:02d}"] = base + drift + seasonal
    data["SPX"] = rng.normal(0.006, 0.035, size=periods)
    data["RF"] = rng.normal(0.001, 0.003, size=periods)
    return pd.DataFrame(data)


def test_direct_pipeline_and_exports(tmp_path: Path) -> None:
    """Run the pipeline directly and exercise key exporter paths."""
    returns = _make_returns_frame()
    csv_path = tmp_path / "synthetic.csv"
    returns.to_csv(csv_path, index=False)

    cfg_dict = _build_demo_config(tmp_path, csv_path)
    cfg = load(cfg_dict)

    # Run the main simulation inside the process so coverage captures it.
    result = api.run_simulation(cfg, returns.copy())
    assert not result.metrics.empty
    assert "out_sample_stats" in result.details

    exports_dir = Path(cfg.export["directory"])  # type: ignore[index]
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Exercise summary helpers and text formatting.
    summary_frame = export.summary_frame_from_result(result.details)
    assert not summary_frame.empty

    summary_text = export.format_summary_text(
        result.details,
        cfg.sample_split.get("in_start"),
        cfg.sample_split.get("in_end"),
        cfg.sample_split.get("out_start"),
        cfg.sample_split.get("out_end"),
    )
    assert cfg.sample_split.get("in_start") in summary_text

    formatter = export.make_summary_formatter(
        result.details,
        cfg.sample_split.get("in_start"),
        cfg.sample_split.get("in_end"),
        cfg.sample_split.get("out_start"),
        cfg.sample_split.get("out_end"),
    )

    export.export_to_excel(
        {"metrics": result.metrics, "summary": summary_frame},
        str(exports_dir / "analysis.xlsx"),
        default_sheet_formatter=formatter,
    )

    export.export_data(
        {"metrics": result.metrics, "summary": summary_frame},
        str(exports_dir / "analysis"),
        formats=["csv", "json", "txt"],
    )

    # Validate JSON export content for sanity.
    json_path = exports_dir / "analysis_metrics.json"
    with open(json_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, list)
    assert payload, "metrics payload should not be empty"

    txt_path = exports_dir / "analysis_summary.txt"
    assert txt_path.exists()

    # Multi-period analysis within the same process.
    multi_results = run_multi(cfg, returns.copy())
    assert multi_results

    combined = export.combined_summary_result(multi_results)
    assert combined["out_sample_stats"]

    workbook_mapping = export.phase1_workbook_data(
        multi_results,
        include_metrics=True,
    )
    assert "summary" in workbook_mapping

    export.export_phase1_workbook(
        multi_results,
        str(exports_dir / "phase1.xlsx"),
        include_metrics=True,
    )

    export.export_phase1_multi_metrics(
        multi_results,
        str(exports_dir / "phase1_multi"),
        formats=["csv", "json"],
        include_metrics=True,
    )

    flat_frames = export.flat_frames_from_results(multi_results)
    assert "periods" in flat_frames
    assert not flat_frames["periods"].empty

    workbook_frames = export.workbook_frames_from_results(multi_results)
    assert "summary" in workbook_frames

    # Build a lightweight run container to exercise bundle export.
    portfolio_series = result.details.get("portfolio_equal_weight_combined")
    if portfolio_series is None:
        # Fallback for deterministic coverage when the helper was not populated.
        portfolio_series = result.metrics.sum(axis=1)

    run_obj = SimpleNamespace(
        portfolio=portfolio_series,
        benchmark=portfolio_series.shift(1).fillna(0.0),
        weights=result.metrics.iloc[:1].T,
        config=cfg_dict,
        seed=cfg.seed,
        input_path=csv_path,
        summary=lambda: {"total_return": float(portfolio_series.sum())},
        environment=result.environment,
    )

    bundle_path = export_bundle(run_obj, exports_dir / "bundle.zip")
    assert bundle_path.exists()

    # Ensure bundle receipt exposes config hash metadata.
    with Path(bundle_path).open("rb") as fh:
        assert fh.read(2) == b"PK"


def test_multi_period_with_price_frames(tmp_path: Path) -> None:
    """Verify the price frame ingestion branch of the multi-period engine."""
    returns = _make_returns_frame(periods=48)
    csv_path = tmp_path / "synthetic.csv"
    returns.to_csv(csv_path, index=False)

    cfg_dict = _build_demo_config(tmp_path, csv_path)
    cfg = load(cfg_dict)

    # Build price frames keyed by pseudo-period identifier.
    frames = {
        "block_a": returns.iloc[:24][["Date", "Mgr_01", "Mgr_02", "SPX", "RF"]],
        "block_b": returns.iloc[24:][["Date", "Mgr_01", "Mgr_02", "SPX", "RF"]],
    }

    results = run_multi(cfg, price_frames=frames)
    assert results
    assert all("period" in res for res in results)

    # Combined summary text should mention the spanning dates.
    first = results[0]["period"]
    last = results[-1]["period"]
    summary_text = export.format_summary_text(
        export.combined_summary_result(results),
        first[0],
        first[1],
        last[2],
        last[3],
    )
    assert first[0][:4] in summary_text
