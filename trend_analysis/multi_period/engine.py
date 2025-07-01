"""Multi‑period engine stub (Phase‑2 scaffolding)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ..data import load_csv
from ..pipeline import single_period_run
from .scheduler import generate_periods
from .replacer import Rebalancer


def run(cfg: Dict[str, Any]) -> Dict[str, object]:
    """Execute multi-period backtest based on ``cfg``."""

    schedule = generate_periods(cfg)

    df = cfg.get("dataframe")
    if df is None:
        path = cfg["data"]["csv_path"]
        df = load_csv(path)
        if df is None:
            raise FileNotFoundError(path)

    reb = Rebalancer(cfg)
    weights = pd.Series(dtype=float)
    results: Dict[str, Any] = {}
    summary_rows: list[pd.Series] = []

    for period in schedule:
        res = single_period_run(
            df,
            period.in_start,
            period.in_end,
            period.out_start,
            period.out_end,
            cfg["vol_adjust"].get("target_vol", 1.0),
            cfg.get("run", {}).get("monthly_cost", 0.0),
            selection_mode=cfg.get("portfolio", {}).get("selection_mode", "all"),
            random_n=cfg.get("portfolio", {}).get("random_n", 8),
            rank_kwargs=cfg.get("portfolio", {}).get("rank"),
            manual_funds=cfg.get("portfolio", {}).get("manual_list"),
            indices_list=cfg.get("portfolio", {}).get("indices_list"),
            seed=cfg.get("random_seed", 42),
        )
        if res is None:
            continue

        score = res.get("score_frame", pd.DataFrame())
        weights = reb.apply_triggers(weights, score)
        res["weights"] = weights
        results[period.out_end] = res
        summary_rows.append(weights)

        chk_dir = cfg.get("checkpoint_dir")
        if chk_dir:
            Path(chk_dir).mkdir(parents=True, exist_ok=True)
            out_file = Path(chk_dir) / f"{period.out_end}.parquet"
            weights.to_frame("weight").to_parquet(out_file)

    summary = (
        pd.DataFrame(summary_rows, index=[p.out_end for p in schedule])
        if summary_rows
        else pd.DataFrame()
    )

    return {"periods": results, "summary": summary}
