import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, List

import matplotlib
import pandas as pd

from trend_analysis.util.hash import sha256_config, sha256_file, sha256_text

_MANIFEST_SCHEMA_REF = "trend_analysis/export/manifest_schema.json"


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
        raw_seed = getattr(run, "seed", None)
        if raw_seed is None:
            seed = None
        else:
            try:
                seed = int(raw_seed)
            except (TypeError, ValueError):
                seed = raw_seed

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

        # ------------------------------------------------------------------
        # Charts PNGs
        # ------------------------------------------------------------------
        def _write_charts() -> None:
            # Configure non-interactive backend and import pyplot lazily
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt  # locally scoped import

            eq = (1 + portfolio.fillna(0)).cumprod()

            # ------------------------------------------------------------------
            # Equity curve
            # ------------------------------------------------------------------
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            if not eq.empty:
                eq.plot(ax=ax)
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
                dd.plot(ax=ax)
            else:  # pragma: no cover - visual placeholder for empty data
                ax.set_axis_off()
            ax.set_title("Drawdown")
            fig.savefig(charts_dir / "drawdown.png", metadata={"run_id": run_id})
            plt.close(fig)

        _write_charts()

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
        opt_files = [results_dir / "benchmark.csv", results_dir / "weights.csv"]
        for fp in opt_files:
            if fp.exists():
                files_to_hash.append(fp)

        # NOTE: README.txt and receipt.txt are written just below; we'll fill
        # their hashes after writing those files.

        # ------------------------------------------------------------------
        # Metadata manifest
        # ------------------------------------------------------------------
        env_attr = getattr(run, "environment", None)
        environment = dict(env_attr) if isinstance(env_attr, dict) else {}
        environment.setdefault("python", sys.version.split()[0])
        try:
            import importlib.metadata as _ilmd

            environment.setdefault("trend_analysis", _ilmd.version("trend-analysis"))
        except Exception:
            environment.setdefault("trend_analysis", "0")
        environment = {k: environment[k] for k in sorted(environment)}
        versions = dict(environment)
        git_hash = _git_hash()
        seeds: List[Any] = [seed] if seed is not None else []

        meta: dict[str, Any] = {
            "$schema": _MANIFEST_SCHEMA_REF,
            "schema_version": "1.0",
            "run_id": run_id,
            "config": config,
            "config_sha256": config_sha256,
            "seed": seed,
            "seeds": seeds,
            "environment": environment,
            "versions": versions,
            "git_hash": git_hash,
            "input_sha256": input_sha256,
        }

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
        receipt_lines.append(f"git_hash: {git_hash}")
        receipt_path = bundle_dir / "receipt.txt"
        receipt_path.write_text("\n".join(receipt_lines) + "\n", encoding="utf-8")

        # ------------------------------------------------------------------
        # README
        # ------------------------------------------------------------------
        commit_hash = git_hash or "unavailable"
        commit_short = str(commit_hash)[:8]
        readme_path = bundle_dir / "README.txt"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                f"""Trend Analysis Bundle
====================

This bundle contains the complete results of a trend analysis run, including data,
charts, and metadata necessary for reproducibility and sharing.

Generated: {_dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

Contents:
---------
results/
  portfolio.csv      - Portfolio returns time series
  benchmark.csv      - Benchmark returns (if available)
  weights.csv        - Portfolio weights over time (if available)

charts/
  equity_curve.png   - Cumulative portfolio performance visualization
  drawdown.png       - Drawdown analysis chart

summary.xlsx         - Summary metrics and performance statistics
run_meta.json        - Configuration manifest with hashes and versions
README.txt           - This file

Reproducibility:
---------------
To reproduce these results:
1. Use the same input data (SHA256 hash in run_meta.json)
2. Apply the configuration from run_meta.json
3. Use the same software versions listed in the versions section
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
            json.dump(meta, f, indent=2, sort_keys=True)

        # ------------------------------------------------------------------
        # Zip everything
        # ------------------------------------------------------------------
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(bundle_dir):
                for name in files:
                    fp = Path(root) / name
                    z.write(fp, fp.relative_to(bundle_dir))

    return path
