import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, List

import matplotlib
import pandas as pd

from trend_analysis.backtesting import BacktestResult, bootstrap_equity
from trend_analysis.costs import CostModel
from trend_analysis.util.hash import (
    normalise_for_json,
    sha256_config,
    sha256_file,
    sha256_text,
)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], encoding="utf-8", shell=False
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Ensure a non-empty fallback so callers always receive at least a
        # short hash.  This prevents downstream checks from failing when the
        # repository metadata isn't available (e.g. in a zipped release
        # environment where the ``.git`` directory is absent).
        return "unknown"


def export_bundle(run: Any, path: Path) -> Path:
    """Write a zipped analysis bundle.

    Parameters
    ----------
    run : Any
        Object carrying at least a ``portfolio`` Series and optional
        ``benchmark``, ``weights``, ``config``, ``seed`` and ``input_path``
        attributes.
    path : Path
        Location of the resulting ``.zip`` file.
    """
    path = Path(path)

    # Use temporary directory for bundle assembly
    with tempfile.TemporaryDirectory(prefix="trend_bundle_") as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        results_dir = bundle_dir / "results"
        charts_dir = bundle_dir / "charts"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(exist_ok=True)
        charts_dir.mkdir(exist_ok=True)

        # Pre-compute hashes and run identifier --------------------------------
        config = getattr(run, "config", {})
        seed = getattr(run, "seed", None)

        # input_path may be missing or explicitly None; handle safely
        _inp = getattr(run, "input_path", None)
        try:
            input_path = Path(_inp) if _inp else None
        except TypeError:
            # Guard against non-pathlike types
            input_path = None
        input_sha256 = (
            sha256_file(input_path)
            if input_path is not None and input_path.exists()
            else None
        )
        config_sha256 = sha256_config(config)
        run_id_src = "|".join(
            filter(
                None,
                [input_sha256, config_sha256, str(seed) if seed is not None else ""],
            )
        )
        run_id = sha256_text(run_id_src)

        # ------------------------------------------------------------------
        # Results CSVs
        # ------------------------------------------------------------------
        try:
            portfolio = getattr(run, "portfolio")
        except AttributeError:
            raise ValueError(
                "The 'portfolio' attribute is required for bundle creation "
                "but was not found in the provided 'run' object."
            )
        if not isinstance(portfolio, pd.Series):
            # Attempt to preserve temporal structure if possible
            if isinstance(portfolio, dict):
                # Use dict keys as index
                portfolio = pd.Series(
                    list(portfolio.values()), index=list(portfolio.keys())
                )
            elif isinstance(portfolio, (list, tuple)):
                raise ValueError(
                    "Cannot convert portfolio of type list/tuple to pandas Series without an index. "
                    "Please provide a portfolio with a temporal index (e.g., dict or pandas Series)."
                )
            else:
                # Fallback: try to convert, but warn user
                import warnings

                warnings.warn(
                    f"Converting portfolio of type {type(portfolio)} to pandas Series without specifying an index. "
                    "This may result in loss of temporal structure.",
                    UserWarning,
                )
                portfolio = pd.Series(portfolio)
        portfolio = portfolio.astype(float)
        equity_curve = (1 + portfolio.fillna(0)).cumprod()

        bootstrap_band: pd.DataFrame | None = None
        bootstrap_fn = getattr(run, "bootstrap_band", None)
        if callable(bootstrap_fn):
            try:
                bootstrap_band = bootstrap_fn()
            except Exception:  # pragma: no cover - defensive fallback
                bootstrap_band = None
        if bootstrap_band is None:
            try:
                dates_attr = getattr(run, "dates", None)
                if dates_attr is not None:
                    try:
                        calendar = pd.DatetimeIndex(dates_attr)
                    except Exception:
                        calendar = pd.DatetimeIndex([])
                else:
                    calendar = pd.DatetimeIndex([])
                drawdown = (
                    equity_curve / equity_curve.cummax() - 1
                    if not equity_curve.empty
                    else equity_curve
                )
                backtest = BacktestResult(
                    returns=portfolio,
                    equity_curve=equity_curve,
                    weights=pd.DataFrame(dtype=float),
                    turnover=pd.Series(dtype=float),
                    per_period_turnover=pd.Series(dtype=float),
                    transaction_costs=pd.Series(dtype=float),
                    rolling_sharpe=pd.Series(dtype=float),
                    drawdown=drawdown,
                    metrics={},
                    calendar=calendar,
                    window_mode="rolling",
                    window_size=max(len(calendar), 1) if len(calendar) else 1,
                    training_windows={},
                    cost_model=CostModel(),
                )
                bootstrap_band = bootstrap_equity(backtest)
            except Exception:  # pragma: no cover - defensive fallback
                bootstrap_band = None
        if bootstrap_band is not None:
            bootstrap_band = bootstrap_band.reindex(equity_curve.index).copy()

        with open(results_dir / "portfolio.csv", "w", encoding="utf-8") as f:
            f.write(f"# run_id: {run_id}\n")
            portfolio.to_csv(f, header=["return"])
        benchmark = getattr(run, "benchmark", None)
        if benchmark is not None:
            with open(results_dir / "benchmark.csv", "w", encoding="utf-8") as f:
                f.write(f"# run_id: {run_id}\n")
                pd.Series(benchmark).to_csv(f, header=["return"])
        weights = getattr(run, "weights", None)
        if weights is not None:
            with open(results_dir / "weights.csv", "w", encoding="utf-8") as f:
                f.write(f"# run_id: {run_id}\n")
                pd.DataFrame(weights).to_csv(f)
        if bootstrap_band is not None and not bootstrap_band.dropna(how="all").empty:
            with open(results_dir / "equity_bootstrap.csv", "w", encoding="utf-8") as f:
                f.write(f"# run_id: {run_id}\n")
                bootstrap_band.to_csv(f)

        # ------------------------------------------------------------------
        # Charts PNGs
        # ------------------------------------------------------------------
        def _to_list(values: Iterable[Any]) -> List[Any]:
            return list(values)

        def _plot_x(index: pd.Index) -> list[Any]:
            if isinstance(index, pd.PeriodIndex):
                return _to_list(index.to_timestamp().to_pydatetime())
            if isinstance(index, pd.DatetimeIndex):
                return _to_list(index.to_pydatetime())
            return _to_list(index.to_list())

        def _write_charts(eq: pd.Series, band: pd.DataFrame | None) -> None:
            # Configure non-interactive backend and import pyplot lazily
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt  # locally scoped import

            # ------------------------------------------------------------------
            # Equity curve
            # ------------------------------------------------------------------
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            if not eq.empty:
                x = _plot_x(eq.index)
                ax.plot(x, eq.values, label="Realised")
                if band is not None:
                    band_aligned = band.reindex(eq.index)
                    valid = band_aligned[["p05", "p95"]].notna().all(axis=1)
                    if valid.any():
                        x_band = [x_i for x_i, ok in zip(x, valid) if ok]
                        p05 = band_aligned.loc[valid, "p05"].to_numpy()
                        p95 = band_aligned.loc[valid, "p95"].to_numpy()
                        median = band_aligned.loc[valid, "median"].to_numpy()
                        ax.fill_between(
                            x_band,
                            p05,
                            p95,
                            alpha=0.2,
                            label="Bootstrap 5–95%",
                        )
                        ax.plot(
                            x_band,
                            median,
                            linestyle="--",
                            label="Bootstrap median",
                        )
                if ax.has_data():
                    ax.legend(loc="best")
            else:  # pragma: no cover - visual placeholder for empty data
                ax.set_axis_off()
            ax.set_title("Equity Curve")
            fig.savefig(charts_dir / "equity_curve.png", metadata={"run_id": run_id})
            plt.close(fig)

            # ------------------------------------------------------------------
            # Drawdown chart
            # ------------------------------------------------------------------
            if not eq.empty:
                dd = eq / eq.cummax() - 1
            else:
                dd = pd.Series(dtype=float)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            if not dd.empty:
                ax.plot(_plot_x(dd.index), dd.values)
            else:  # pragma: no cover - visual placeholder for empty data
                ax.set_axis_off()
            ax.set_title("Drawdown")
            fig.savefig(charts_dir / "drawdown.png", metadata={"run_id": run_id})
            plt.close(fig)

        _write_charts(equity_curve, bootstrap_band)

        # ------------------------------------------------------------------
        # Summary workbook
        # ------------------------------------------------------------------
        summary: dict[str, Any]
        summ_fn = getattr(run, "summary", None)
        if callable(summ_fn):
            s = summ_fn()
            summary = dict(s) if isinstance(s, dict) else {}
        else:
            summary = {"total_return": float(portfolio.sum())}
        row = {"run_id": run_id}
        row.update(summary)
        pd.DataFrame([row]).to_excel(bundle_dir / "summary.xlsx", index=False)

        # ------------------------------------------------------------------
        # Compute outputs hashes (relative file paths -> sha256)
        # ------------------------------------------------------------------
        def _rel(p: Path) -> str:
            return str(p.relative_to(bundle_dir).as_posix())

        outputs: dict[str, str] = {}

        files_to_hash: list[Path] = [
            results_dir / "portfolio.csv",
            charts_dir / "equity_curve.png",
            charts_dir / "drawdown.png",
            bundle_dir / "summary.xlsx",
            bundle_dir / "README.txt",  # written below, but path reserved
            bundle_dir / "receipt.txt",  # written below, but path reserved
        ]
        # Optionals if present
        opt_files = [
            results_dir / "benchmark.csv",
            results_dir / "weights.csv",
            results_dir / "equity_bootstrap.csv",
        ]
        for fp in opt_files:
            if fp.exists():
                files_to_hash.append(fp)

        # NOTE: README.txt and receipt.txt are written just below; we'll fill
        # their hashes after writing those files.

        # ------------------------------------------------------------------
        # Metadata manifest
        # ------------------------------------------------------------------
        env = getattr(run, "environment", {"python": sys.version.split()[0]})
        try:
            import importlib.metadata as _ilmd

            env.setdefault("trend_analysis", _ilmd.version("trend-analysis"))
        except Exception:
            env.setdefault("trend_analysis", "0")

        # Attempt to propagate a user-visible run_id if one was attached upstream
        provided_run_id = getattr(run, "run_id", None)
        meta: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": provided_run_id or run_id,
            "config": config,
            "config_sha256": config_sha256,
            "seed": seed,
            "environment": env,
            "git_hash": _git_hash(),
            "receipt": {
                "created": _dt.datetime.now(_dt.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            },
            "input_sha256": input_sha256,
        }
        # Pass through structured log reference if present on the run object
        log_file = getattr(run, "log_file", None)
        if log_file:
            meta["log_file"] = str(log_file)

        # We'll set meta["outputs"] after we have all files written and hashed

        # ------------------------------------------------------------------
        # Receipt
        # ------------------------------------------------------------------
        receipt_lines = [
            f"run_id: {run_id}",
            f"input_sha256: {input_sha256}",
            f"config_sha256: {config_sha256}",
        ]
        if seed is not None:
            receipt_lines.append(f"seed: {seed}")
        receipt_lines.append(f"git_hash: {meta['git_hash']}")
        receipt_path = bundle_dir / "receipt.txt"
        receipt_path.write_text("\n".join(receipt_lines) + "\n", encoding="utf-8")

        # ------------------------------------------------------------------
        # README
        # ------------------------------------------------------------------
        commit_hash = meta.get("git_hash", "unavailable") or "unavailable"
        commit_short = str(commit_hash)[:8]
        readme_path = bundle_dir / "README.txt"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                f"""Trend Analysis Bundle
====================

This bundle contains the complete results of a trend analysis run, including data,
charts, and metadata necessary for reproducibility and sharing.

Generated: {_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC

Contents:
---------
results/
  portfolio.csv      - Portfolio returns time series
  benchmark.csv      - Benchmark returns (if available)
  weights.csv        - Portfolio weights over time (if available)
  equity_bootstrap.csv - Bootstrap equity quantiles (5th percentile/median/95th percentile) if computed

charts/
  equity_curve.png   - Cumulative performance with optional bootstrap band
  drawdown.png       - Drawdown analysis chart

summary.xlsx         - Summary metrics and performance statistics
run_meta.json        - Configuration, environment, and reproducibility metadata
README.txt           - This file

Reproducibility:
---------------
To reproduce these results:
1. Use the same input data (SHA256 hash in run_meta.json)
2. Apply the configuration from run_meta.json
3. Use the same software versions listed in environment section
4. Set the same random seed if specified

For more information about the Trend Analysis Project, visit:
https://github.com/stranske/Trend_Model_Project

Git commit: {commit_short}
"""
            )

        # Now that README.txt and receipt.txt exist, compute output hashes
        for fp in [*files_to_hash, readme_path, receipt_path]:
            if fp.exists():
                outputs[_rel(fp)] = sha256_file(fp)

        # Attach outputs map and write the manifest
        meta["outputs"] = outputs
        with open(bundle_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(normalise_for_json(meta), f, indent=2)

        # ------------------------------------------------------------------
        # Zip everything
        # ------------------------------------------------------------------
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(bundle_dir):
                for name in files:
                    fp = Path(root) / name
                    z.write(fp, fp.relative_to(bundle_dir))

    return path
