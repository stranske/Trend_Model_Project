from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def run_analysis(
    df: pd.DataFrame,
    selected: List[str],
    w_vec: Optional[np.ndarray],
    w_dict: Optional[Dict[str, float]],
    rf_col: str,
    in_start: str,
    in_end: str,
    out_start: str,
    out_end: str,
    target_vol: float = 1.0,
    monthly_cost: float = 0.0,
    indices_list: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Simple analysis on a return DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a ``'Date'`` column and fund return columns.
    selected : list[str]
        Columns to include in the analysis.
    w_vec : numpy.ndarray or None
        Optional weight vector (unused).
    w_dict : dict[str, float] or None
        Optional dictionary of weights. If ``None`` equal weights are used.
    rf_col : str
        Name of the risk-free column (unused).
    in_start, in_end, out_start, out_end : str
        Date boundaries for the in-sample and out-of-sample windows.
    target_vol : float, default 1.0
        Unused placeholder for compatibility.
    monthly_cost : float, default 0.0
        Unused placeholder for compatibility.
    indices_list : list[str] or None
        Optional index columns (unused).

    Returns
    -------
    dict
        Dictionary with selected funds, weights, mean returns and dropped funds.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    in_mask = df["Date"].between(pd.to_datetime(in_start), pd.to_datetime(in_end))
    out_mask = df["Date"].between(pd.to_datetime(out_start), pd.to_datetime(out_end))

    in_sample = df.loc[in_mask, selected]
    out_sample = df.loc[out_mask, selected]

    good = [
        c
        for c in selected
        if not in_sample[c].isna().any() and not out_sample[c].isna().any()
    ]
    dropped = list(set(selected) - set(good))
    selected = good

    if w_dict is None:
        w_dict = {f: 1 / len(selected) for f in selected} if selected else {}
    result: Dict[str, object] = {
        "selected_funds": selected,
        "fund_weights": w_dict,
        "in_sample_mean": {f: float(in_sample[f].mean()) for f in selected},
        "out_sample_mean": {f: float(out_sample[f].mean()) for f in selected},
        "dropped": dropped,
    }
    return result
