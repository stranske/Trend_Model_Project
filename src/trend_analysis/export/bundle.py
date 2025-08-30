from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import datetime as _dt
import hashlib
import os
import zipfile
import subprocess
import sys

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], encoding="utf-8")
            .strip()
        )
    except Exception:
        return ""


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
    bundle_dir = path.with_suffix("")
    results_dir = bundle_dir / "results"
    charts_dir = bundle_dir / "charts"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    charts_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Results CSVs
    # ------------------------------------------------------------------
    try:
        portfolio = getattr(run, "portfolio")
    except AttributeError:
        raise ValueError("The 'portfolio' attribute is required for bundle creation but was not found in the provided 'run' object.")
    portfolio.to_csv(results_dir / "portfolio.csv", header=["return"])
    benchmark = getattr(run, "benchmark", None)
    if benchmark is not None:
        pd.Series(benchmark).to_csv(results_dir / "benchmark.csv", header=["return"])
    weights = getattr(run, "weights", None)
    if weights is not None:
        pd.DataFrame(weights).to_csv(results_dir / "weights.csv")

    # ------------------------------------------------------------------
    # Charts PNGs
    # ------------------------------------------------------------------
    eq = (1 + portfolio.fillna(0)).cumprod()
    fig, ax = plt.subplots()
    eq.plot(ax=ax)
    ax.set_title("Equity Curve")
    fig.savefig(charts_dir / "equity_curve.png")
    plt.close(fig)

    dd = eq / eq.cummax() - 1
    fig, ax = plt.subplots()
    dd.plot(ax=ax)
    ax.set_title("Drawdown")
    fig.savefig(charts_dir / "drawdown.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Summary workbook
    # ------------------------------------------------------------------
    summary = {}
    summ_fn = getattr(run, "summary", None)
    if callable(summ_fn):
        summary = summ_fn()
    else:
        summary = {"total_return": float(portfolio.sum())}
    pd.DataFrame([summary]).to_excel(bundle_dir / "summary.xlsx", index=False)

    # ------------------------------------------------------------------
    # Metadata manifest
    # ------------------------------------------------------------------
    meta = {
        "config": getattr(run, "config", {}),
        "seed": getattr(run, "seed", None),
        "versions": {
            "python": sys.version.split()[0],
        },
        "git_hash": _git_hash(),
        "receipt": {"created": _dt.datetime.utcnow().isoformat() + "Z"},
    }
    try:
        import importlib.metadata as _ilmd

        meta["versions"]["trend_analysis"] = _ilmd.version("trend-analysis")
    except Exception:
        meta["versions"]["trend_analysis"] = "0"

    input_path = Path(getattr(run, "input_path", ""))
    if input_path and input_path.exists():
        meta["input_sha256"] = _sha256_file(input_path)
    else:
        meta["input_sha256"] = None

    with open(bundle_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    with open(bundle_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("Trend analysis bundle\n")

    # ------------------------------------------------------------------
    # Zip everything
    # ------------------------------------------------------------------
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(bundle_dir):
            for name in files:
                fp = Path(root) / name
                z.write(fp, fp.relative_to(bundle_dir))
    return path
