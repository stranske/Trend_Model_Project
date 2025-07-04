from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Any

from .data import load_csv
from .metrics import (
    annual_return,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    information_ratio,
)

if TYPE_CHECKING:  # pragma: no cover - for static type checking only
    from .config import Config
    from .core.rank_selection import RiskStatsConfig

del TYPE_CHECKING


@dataclass
class _Stats:
    """Container for performance metrics."""

    cagr: float
    vol: float
    sharpe: float
    sortino: float
    information_ratio: float
    max_drawdown: float


def calc_portfolio_returns(
    weights: NDArray[Any], returns_df: pd.DataFrame
) -> pd.Series:
    """Calculate weighted portfolio returns."""
    return returns_df.mul(weights, axis=1).sum(axis=1)


def single_period_run(
    df: pd.DataFrame,
    start: str,
    end: str,
    *,
    stats_cfg: RiskStatsConfig | None = None,
) -> pd.DataFrame:
    """Return a score frame of metrics for a single period.

    Parameters
    ----------
    df : pd.DataFrame
        Input returns data with a ``Date`` column.
    start, end : str
        Inclusive period in ``YYYY-MM`` format.
    stats_cfg : RiskStatsConfig | None
        Metric configuration; defaults to ``RiskStatsConfig()``.

    Returns
    -------
    pd.DataFrame
        Table of metric values (index = fund code).  The frame is pure
        and carries ``insample_len`` and ``period`` metadata so callers
        can reason about the analysed window.
    """
    from .core.rank_selection import RiskStatsConfig, _compute_metric_series

    if stats_cfg is None:
        stats_cfg = RiskStatsConfig()

    if "Date" not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column")

    df = df.copy()
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])

    def _parse_month(s: str) -> pd.Timestamp:
        return pd.to_datetime(f"{s}-01") + pd.offsets.MonthEnd(0)

    sdate, edate = _parse_month(start), _parse_month(end)
    window = df[(df["Date"] >= sdate) & (df["Date"] <= edate)].set_index("Date")

    metrics = stats_cfg.metrics_to_run
    if not metrics:
        raise ValueError("stats_cfg.metrics_to_run must not be empty")

    parts = [
        _compute_metric_series(window.dropna(axis=1, how="all"), m, stats_cfg)
        for m in metrics
    ]
    score_frame = pd.concat(parts, axis=1)
    score_frame.columns = metrics
    score_frame.attrs["insample_len"] = len(window)
    score_frame.attrs["period"] = (start, end)
    return score_frame.astype(float)


def _compute_stats(df: pd.DataFrame, rf: pd.Series) -> dict[str, _Stats]:
    # Metrics expect 1D Series; iterating keeps the logic simple for a handful
    # of columns and avoids reshaping into higher-dimensional arrays.
    stats = {}
    for col in df:
        stats[col] = _Stats(
            cagr=float(annual_return(df[col])),
            vol=float(volatility(df[col])),
            sharpe=float(sharpe_ratio(df[col], rf)),
            sortino=float(sortino_ratio(df[col], rf)),
            max_drawdown=float(max_drawdown(df[col])),
            information_ratio=float(information_ratio(df[col], rf)),
        )
    return stats


def _run_analysis(
    df: pd.DataFrame,
    in_start: str,
    in_end: str,
    out_start: str,
    out_end: str,
    target_vol: float,
    monthly_cost: float,
    selection_mode: str = "all",
    random_n: int = 8,
    custom_weights: dict[str, float] | None = None,
    rank_kwargs: dict[str, object] | None = None,
    manual_funds: list[str] | None = None,
    indices_list: list[str] | None = None,
    benchmarks: dict[str, str] | None = None,
    seed: int = 42,
) -> dict[str, object] | None:
    from .core.rank_selection import RiskStatsConfig, rank_select_funds

    if df is None:
        return None

    date_col = "Date"
    if date_col not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column")

    df = df.copy()
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)

    def _parse_month(s: str) -> pd.Timestamp:
        return pd.to_datetime(f"{s}-01") + pd.offsets.MonthEnd(0)

    in_sdate, in_edate = _parse_month(in_start), _parse_month(in_end)
    out_sdate, out_edate = _parse_month(out_start), _parse_month(out_end)

    in_df = df[(df[date_col] >= in_sdate) & (df[date_col] <= in_edate)].set_index(
        date_col
    )
    out_df = df[(df[date_col] >= out_sdate) & (df[date_col] <= out_edate)].set_index(
        date_col
    )

    if in_df.empty or out_df.empty:
        return None

    ret_cols = [c for c in df.columns if c != date_col]
    if indices_list:
        idx_set = set(indices_list)  # pragma: no cover - seldom used
        ret_cols = [c for c in ret_cols if c not in idx_set]  # pragma: no cover
    else:
        indices_list = []
    rf_col = min(ret_cols, key=lambda c: df[c].std())
    fund_cols = [c for c in ret_cols if c != rf_col]

    # determine which index columns have complete data
    valid_indices: list[str] = []
    if indices_list:
        idx_in_ok = ~in_df[indices_list].isna().any()  # pragma: no cover
        idx_out_ok = ~out_df[indices_list].isna().any()  # pragma: no cover
        valid_indices = [
            c for c in indices_list if idx_in_ok[c] and idx_out_ok[c]
        ]  # pragma: no cover

    # keep only funds with complete data in both windows
    in_ok = ~in_df[fund_cols].isna().any()
    out_ok = ~out_df[fund_cols].isna().any()
    fund_cols = [c for c in fund_cols if in_ok[c] and out_ok[c]]

    if selection_mode == "random" and len(fund_cols) > random_n:
        rng = np.random.default_rng(seed)
        fund_cols = rng.choice(fund_cols, size=random_n, replace=False).tolist()
    elif selection_mode == "rank":
        mask = (df[date_col] >= in_sdate) & (df[date_col] <= in_edate)
        sub = df.loc[mask, fund_cols]
        stats_cfg = RiskStatsConfig(risk_free=0.0)
        fund_cols = rank_select_funds(sub, stats_cfg, **(rank_kwargs or {}))  # type: ignore[arg-type]
    elif selection_mode == "manual":
        if manual_funds:  # pragma: no cover - rarely hit
            fund_cols = [c for c in fund_cols if c in manual_funds]
        else:
            fund_cols = []  # pragma: no cover

    if not fund_cols:
        return None

    stats_cfg = RiskStatsConfig(risk_free=0.0)
    score_frame = single_period_run(
        df[[date_col] + fund_cols], in_start, in_end, stats_cfg=stats_cfg
    )

    vols = in_df[fund_cols].std() * np.sqrt(12)
    scale_factors = (
        pd.Series(target_vol / vols, index=fund_cols)
        .replace([np.inf, -np.inf], 1.0)
        .fillna(1.0)
    )

    in_scaled = in_df[fund_cols].mul(scale_factors, axis=1) - monthly_cost
    out_scaled = out_df[fund_cols].mul(scale_factors, axis=1) - monthly_cost
    in_scaled = in_scaled.clip(lower=-1.0)
    out_scaled = out_scaled.clip(lower=-1.0)

    rf_in = in_df[rf_col]
    rf_out = out_df[rf_col]

    in_stats = _compute_stats(in_scaled, rf_in)
    out_stats = _compute_stats(out_scaled, rf_out)
    out_stats_raw = _compute_stats(out_df[fund_cols], rf_out)

    ew_weights = np.repeat(1.0 / len(fund_cols), len(fund_cols))
    ew_w_dict = {c: w for c, w in zip(fund_cols, ew_weights)}
    in_ew = calc_portfolio_returns(ew_weights, in_scaled)
    out_ew = calc_portfolio_returns(ew_weights, out_scaled)
    out_ew_raw = calc_portfolio_returns(ew_weights, out_df[fund_cols])

    in_ew_stats = _compute_stats(pd.DataFrame({"ew": in_ew}), rf_in)["ew"]
    out_ew_stats = _compute_stats(pd.DataFrame({"ew": out_ew}), rf_out)["ew"]
    out_ew_stats_raw = _compute_stats(pd.DataFrame({"ew": out_ew_raw}), rf_out)["ew"]

    if custom_weights is None:
        custom_weights = {c: 100 / len(fund_cols) for c in fund_cols}
    user_w = np.array([custom_weights.get(c, 0) / 100 for c in fund_cols])
    user_w_dict = {c: w for c, w in zip(fund_cols, user_w)}

    in_user = calc_portfolio_returns(user_w, in_scaled)
    out_user = calc_portfolio_returns(user_w, out_scaled)
    out_user_raw = calc_portfolio_returns(user_w, out_df[fund_cols])

    in_user_stats = _compute_stats(pd.DataFrame({"user": in_user}), rf_in)["user"]
    out_user_stats = _compute_stats(pd.DataFrame({"user": out_user}), rf_out)["user"]
    out_user_stats_raw = _compute_stats(pd.DataFrame({"user": out_user_raw}), rf_out)[
        "user"
    ]

    benchmark_stats: dict[str, dict[str, _Stats]] = {}
    benchmark_ir: dict[str, dict[str, float]] = {}
    all_benchmarks: dict[str, str] = {}
    if benchmarks:
        all_benchmarks.update(benchmarks)
    for idx in valid_indices:
        if idx not in all_benchmarks:
            all_benchmarks[idx] = idx

    for label, col in all_benchmarks.items():
        if col not in in_df.columns or col not in out_df.columns:
            continue
        benchmark_stats[label] = {
            "in_sample": _compute_stats(pd.DataFrame({label: in_df[col]}), rf_in)[
                label
            ],
            "out_sample": _compute_stats(pd.DataFrame({label: out_df[col]}), rf_out)[
                label
            ],
        }
        ir_series = information_ratio(out_scaled[fund_cols], out_df[col])
        ir_dict = (
            ir_series.to_dict()
            if isinstance(ir_series, pd.Series)
            else {fund_cols[0]: float(ir_series)}
        )
        ir_dict["equal_weight"] = float(information_ratio(out_ew_raw, out_df[col]))
        ir_dict["user_weight"] = float(information_ratio(out_user_raw, out_df[col]))
        benchmark_ir[label] = ir_dict

    return {
        "selected_funds": fund_cols,
        "in_sample_scaled": in_scaled,
        "out_sample_scaled": out_scaled,
        "in_sample_stats": in_stats,
        "out_sample_stats": out_stats,
        "out_sample_stats_raw": out_stats_raw,
        "in_ew_stats": in_ew_stats,
        "out_ew_stats": out_ew_stats,
        "out_ew_stats_raw": out_ew_stats_raw,
        "in_user_stats": in_user_stats,
        "out_user_stats": out_user_stats,
        "out_user_stats_raw": out_user_stats_raw,
        "ew_weights": ew_w_dict,
        "fund_weights": user_w_dict,
        "benchmark_stats": benchmark_stats,
        "benchmark_ir": benchmark_ir,
        "score_frame": score_frame,
    }


def run_analysis(
    df: pd.DataFrame,
    in_start: str,
    in_end: str,
    out_start: str,
    out_end: str,
    target_vol: float,
    monthly_cost: float,
    selection_mode: str = "all",
    random_n: int = 8,
    custom_weights: dict[str, float] | None = None,
    rank_kwargs: dict[str, object] | None = None,
    manual_funds: list[str] | None = None,
    indices_list: list[str] | None = None,
    benchmarks: dict[str, str] | None = None,
    seed: int = 42,
) -> dict[str, object] | None:
    """Backward-compatible wrapper around ``_run_analysis``."""
    return _run_analysis(
        df,
        in_start,
        in_end,
        out_start,
        out_end,
        target_vol,
        monthly_cost,
        selection_mode,
        random_n,
        custom_weights,
        rank_kwargs,
        manual_funds,
        indices_list,
        benchmarks,
        seed,
    )


def run(cfg: Config) -> pd.DataFrame:
    """Execute the analysis pipeline based on ``cfg``."""
    csv_path = cfg.data.get("csv_path")
    if csv_path is None:
        raise KeyError("cfg.data['csv_path'] must be provided")

    df = load_csv(csv_path)
    if df is None:
        raise FileNotFoundError(csv_path)

    split = cfg.sample_split
    res = _run_analysis(
        df,
        cast(str, split.get("in_start")),
        cast(str, split.get("in_end")),
        cast(str, split.get("out_start")),
        cast(str, split.get("out_end")),
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
        return pd.DataFrame()
    stats = cast(dict[str, _Stats], res["out_sample_stats"])
    df = pd.DataFrame({k: vars(v) for k, v in stats.items()}).T
    for label, ir_map in cast(
        dict[str, dict[str, float]], res.get("benchmark_ir", {})
    ).items():
        col = f"ir_{label}"
        df[col] = pd.Series(
            {
                k: v
                for k, v in ir_map.items()
                if k not in {"equal_weight", "user_weight"}
            }
        )
    return df


def run_full(cfg: Config) -> dict[str, object]:
    """Return the full analysis results based on ``cfg``."""
    csv_path = cfg.data.get("csv_path")
    if csv_path is None:
        raise KeyError("cfg.data['csv_path'] must be provided")

    df = load_csv(csv_path)
    if df is None:
        raise FileNotFoundError(csv_path)

    split = cfg.sample_split
    res = _run_analysis(
        df,
        cast(str, split.get("in_start")),
        cast(str, split.get("in_end")),
        cast(str, split.get("out_start")),
        cast(str, split.get("out_end")),
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
    return {} if res is None else res


Stats = _Stats

__all__ = [
    "Stats",
    "calc_portfolio_returns",
    "single_period_run",
    "run_analysis",
    "run",
    "run_full",
]


def __getattr__(name: str) -> object:
    if name == "Stats":
        return _Stats
    raise AttributeError(name)


del Stats
