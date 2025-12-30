# Vol_Adj_Trend_Analysis_Cleanup

# --- 1. SETUP CELL ---
import logging
import random
import sys

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import VBox

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s")


# --- 2. Identify Risk-Free Fund ---
def identify_risk_free_fund(df):
    """
    Identify the risk-free column (smallest stddev among numeric columns).
    Returns the column name.
    """
    cols = df.columns.drop("Date", errors="ignore")
    stdevs = {col: df[col].dropna().std() if not df[col].dropna().empty else np.inf for col in cols}
    rf = min(stdevs, key=stdevs.get)
    logging.info(f"Risk-free column: {rf}")
    return rf


# --- 3. Robust CSV Reader ---
def robust_read_csv(path):
    """
    Load CSV with fallback strategies:
    – Default engine
    – BOM-stripped with Python engine
    – Skip bad lines with Python engine
    """
    try:
        return pd.read_csv(path)
    except Exception:
        pass
    try:
        return pd.read_csv(path, sep=",", encoding="utf-8-sig", engine="python")
    except Exception:
        pass
    return pd.read_csv(
        path,
        sep=",",
        engine="python",
        encoding="utf-8-sig",
        skip_blank_lines=True,
        on_bad_lines="skip",
    )


# --- 4. Utility Functions ---
def consecutive_gaps(series, threshold=3):
    """
    Return True if there are ≥ threshold consecutive NaNs in the series.
    """
    count = 0
    for v in series:
        count = count + 1 if pd.isna(v) else 0
        if count >= threshold:
            return True
    return False


def fill_short_gaps_with_zero(series):
    """
    Replace runs of 1–2 NaNs with 0.0; leave longer gaps intact.
    """
    mask = series.isna().astype(int)
    runs = mask.groupby((mask == 0).cumsum()).transform("sum")
    out = series.copy()
    out[(mask == 1) & (runs <= 2)] = 0.0
    return out


# --- 5. Annualized Metrics ---

# Shared column specs
COLUMN_SPECS = [
    (0, 28, None),
    (1, 12, "int"),
    (2, 15, "pct1"),
    (3, 15, "pct1"),
    (4, 15, "dec2"),
    (5, 15, "dec2"),
    (6, 15, "pct1"),
]


# --- 6. Select Funds ---
def select_funds(
    df,
    rf_col,
    fund_columns,
    in_sdate,
    in_edate,
    out_sdate,
    out_edate,
    selection_mode="all",
    random_n=8,
):
    candidates = [f for f in fund_columns if "index" not in f.lower()]
    valid = []
    for f in candidates:
        in_sub = df.loc[in_sdate:in_edate, f]
        out_sub = df.loc[out_sdate:out_edate, f]
        if in_sub.notna().all() and out_sub.notna().all():
            if not consecutive_gaps(in_sub) and not consecutive_gaps(out_sub):
                valid.append(f)
    if selection_mode == "random" and len(valid) > random_n:
        return random.sample(valid, random_n)
    return valid


# --- 7. Custom Weights UI ---
def get_custom_weights(selected_funds):
    weight_widgets = {
        f: widgets.BoundedIntText(
            value=0, min=0, max=100, description=f, layout=widgets.Layout(width="250px")
        )
        for f in selected_funds
    }
    confirm = widgets.Button(description="Confirm", button_style="success")
    error_lbl = widgets.Label(layout=widgets.Layout(color="red"))
    box = VBox(list(weight_widgets.values()) + [confirm, error_lbl])
    display(box)
    weights = {}

    def on_confirm(_):
        total = sum(w.value for w in weight_widgets.values())
        if total != 100:
            error_lbl.value = f"Weights sum to {total}, must be 100."
            weights.clear()
        else:
            for fund, w in weight_widgets.items():
                weights[fund] = w.value / 100.0
            error_lbl.value = "Weights confirmed"

    confirm.on_click(on_confirm)
    return weights


# --- 8. run_analysis ---
# The canonical implementation now lives in ``trend_analysis.pipeline``.
def run_analysis(*args, **kwargs):
    raise NotImplementedError("Deprecated. Use trend_analysis.pipeline.run_analysis instead.")


# --- 9. Export Helpers ---
def build_formats(wb):
    return {
        "int": wb.add_format({"num_format": "0"}),
        "pct1": wb.add_format({"num_format": "0.0%"}),
        "dec2": wb.add_format({"num_format": "0.00"}),
        "bold": wb.add_format({"bold": True}),
    }


def make_portfolio_dfs(results, sample="in"):
    # (full implementation)
    pass


def make_indices_df(df, indices, start, end):
    # (full implementation)
    pass


def write_portfolio_sheet(writer, sheet_name, eq_df, user_df, fmt):
    # (full implementation)
    pass


def write_indices_block(writer, sheet_name, df_idx, start_row, fmt):
    # (full implementation)
    pass


def export_to_excel(results, df, fname, in_start, in_end, out_start, out_end):
    writer = pd.ExcelWriter(fname, engine="xlsxwriter")
    fmt = build_formats(writer.book)
    in_eq, in_usr = make_portfolio_dfs(results, "in")
    out_eq, out_usr = make_portfolio_dfs(results, "out")
    idx_in = make_indices_df(df, results["indices_list"], in_start, in_end)
    idx_out = make_indices_df(df, results["indices_list"], out_start, out_end)
    start_in = len(in_eq) + len(in_usr) + 6
    start_out = len(out_eq) + len(out_usr) + 6
    write_portfolio_sheet(writer, f"IS {in_start}-{in_end}", in_eq, in_usr, fmt)
    write_portfolio_sheet(writer, f"OS {out_start}-{out_end}", out_eq, out_usr, fmt)
    write_indices_block(writer, f"IS {in_start}-{in_end}", idx_in, start_in, fmt)
    write_indices_block(writer, f"OS {out_start}-{out_end}", idx_out, start_out, fmt)
    writer.close()
    logging.info(f"Exported analysis to {fname}")
