from __future__ import annotations

import pandas as pd

from .config import Config
from .data import load_csv
from . import analyze


def run(cfg: Config) -> pd.DataFrame:
    """Run the trend analysis pipeline and return results as a DataFrame."""
    data_path = cfg.data.get("returns_path")
    if data_path is None:
        raise ValueError("Config.data must contain 'returns_path'")
    df = load_csv(str(data_path))
    if df is None:
        raise FileNotFoundError(data_path)
    res = analyze.run_analysis(
        df,
        selected=cfg.portfolio.get("manual_list", []),
        w_vec=None,
        w_dict=None,
        rf_col="RF",
        in_start=cfg.sample_split.get("in_start", "2020-01"),
        in_end=cfg.sample_split.get("in_end", "2020-12"),
        out_start=cfg.sample_split.get("out_start", "2021-01"),
        out_end=cfg.sample_split.get("out_end", "2021-12"),
        target_vol=cfg.vol_adjust.get("target_vol", 0.1),
        monthly_cost=cfg.portfolio.get("monthly_cost", 0.0),
    )
    return pd.json_normalize(res)


__all__ = ["run"]
