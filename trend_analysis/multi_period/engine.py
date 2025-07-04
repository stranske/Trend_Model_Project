"""Vectorised multi-period back‑testing engine."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from ..config import Config
from ..data import load_csv
from ..pipeline import _run_analysis
from .scheduler import generate_periods


def run(cfg: Config, df: pd.DataFrame | None = None) -> List[Dict[str, object]]:
    """Run ``_run_analysis`` across multiple periods.

    Parameters
    ----------
    cfg : Config
        Loaded configuration object. ``cfg.multi_period`` drives the
        scheduling logic.
    df : pd.DataFrame, optional
        Pre-loaded returns data.  If ``None`` the CSV pointed to by
        ``cfg.data['csv_path']`` is loaded via :func:`load_csv`.

    Returns
    -------
    list[dict[str, object]]
        One result dictionary per generated period.  Each result is the
        full output of ``_run_analysis`` augmented with a ``period`` key
        for reference.
    """

    if df is None:
        csv_path = cfg.data.get("csv_path")
        if not csv_path:
            raise KeyError("cfg.data['csv_path'] must be provided")
        df = load_csv(csv_path)
        if df is None:
            raise FileNotFoundError(csv_path)

    periods = generate_periods(cfg.model_dump())
    results: List[Dict[str, object]] = []
    for pt in periods:
        res = _run_analysis(
            df,
            pt.in_start[:7],
            pt.in_end[:7],
            pt.out_start[:7],
            pt.out_end[:7],
            cfg.vol_adjust.get("target_vol", 1.0),
            cfg.run.get("monthly_cost", 0.0),
            selection_mode=cfg.portfolio.get("selection_mode", "all"),
            random_n=cfg.portfolio.get("random_n", 8),
            custom_weights=cfg.portfolio.get("custom_weights"),
            rank_kwargs=cfg.portfolio.get("rank"),
            manual_funds=cfg.portfolio.get("manual_list"),
            indices_list=cfg.portfolio.get("indices_list"),
            benchmarks=cfg.benchmarks,
            seed=cfg.portfolio.get("random_seed", 42),
        )
        if res is None:
            continue
        res = dict(res)
        res["period"] = (
            pt.in_start,
            pt.in_end,
            pt.out_start,
            pt.out_end,
        )
        results.append(res)
    return results
