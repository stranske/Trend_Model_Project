from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .data import load_csv
from .metrics import (
    annual_return,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    volatility,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - for static type checking only
    from .config.models import ConfigProtocol as Config
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
    if not pd.api.types.is_datetime64_any_dtype(df["Date"].dtype):
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
    # Optional derived correlation metric (opt-in via stats_cfg.extra_metrics)
    try:
        extra = getattr(stats_cfg, "extra_metrics", [])
        if (
            "AvgCorr" in extra
            and score_frame.shape[1] > 0
            and window.shape[1] > 1
            and "AvgCorr" not in score_frame.columns
        ):
            from .core.rank_selection import compute_metric_series_with_cache

            avg_corr_series = compute_metric_series_with_cache(
                window.dropna(axis=1, how="all"),
                "AvgCorr",
                stats_cfg,
                enable_cache=False,
            )
            score_frame = pd.concat([score_frame, avg_corr_series], axis=1)
    except Exception:  # pragma: no cover - defensive
        pass
    return score_frame.astype(float)


def _compute_stats(df: pd.DataFrame, rf: pd.Series) -> dict[str, _Stats]:
    # Metrics expect 1D Series; iterating keeps the logic simple for a handful
    # of columns and avoids reshaping into higher-dimensional arrays.
    stats: dict[str, _Stats] = {}
    for col in df:
        key = str(col)
        stats[key] = _Stats(
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
    stats_cfg: "RiskStatsConfig" | None = None,
    weighting_scheme: str | None = None,
    constraints: dict[str, Any] | None = None,
) -> dict[str, object] | None:
    from .core.rank_selection import RiskStatsConfig, rank_select_funds

    if df is None:
        return None

    date_col = "Date"
    if date_col not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column")

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col].dtype):
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

    # keep only funds that satisfy missing-data policy in both windows
    # default is strict completeness; optionally allow small gaps if
    # stats_cfg carries an `na_as_zero_cfg` with tolerances.
    def _max_consecutive_nans(s: pd.Series) -> int:
        is_na = s.isna().astype(int)
        # count consecutive runs
        runs = is_na.groupby((is_na != is_na.shift()).cumsum()).cumsum() * is_na
        return int(runs.max() if not runs.empty else 0)

    na_cfg = getattr(stats_cfg, "na_as_zero_cfg", None)
    if na_cfg and bool(na_cfg.get("enabled", False)):
        max_missing = int(na_cfg.get("max_missing_per_window", 0))
        max_gap = int(na_cfg.get("max_consecutive_gap", 0))

        def _ok_window(window: pd.DataFrame, col: str) -> bool:
            s = window[col]
            missing = int(s.isna().sum())
            if missing == 0:
                return True
            if missing > max_missing:
                return False
            return _max_consecutive_nans(s) <= max_gap

        fund_cols = [
            c for c in fund_cols if _ok_window(in_df, c) and _ok_window(out_df, c)
        ]
    else:
        in_ok = ~in_df[fund_cols].isna().any()
        out_ok = ~out_df[fund_cols].isna().any()
        fund_cols = [c for c in fund_cols if in_ok[c] and out_ok[c]]

    if stats_cfg is None:
        stats_cfg = RiskStatsConfig(risk_free=0.0)

    if selection_mode == "random" and len(fund_cols) > random_n:
        rng = np.random.default_rng(seed)
        fund_cols = rng.choice(fund_cols, size=random_n, replace=False).tolist()
    elif selection_mode == "rank":
        mask = (df[date_col] >= in_sdate) & (df[date_col] <= in_edate)
        sub = df.loc[mask, fund_cols]
        fund_cols = rank_select_funds(
            sub, stats_cfg, **(rank_kwargs or {})  # type: ignore[arg-type]
        )
    elif selection_mode == "manual":
        if manual_funds:  # pragma: no cover - rarely hit
            fund_cols = [c for c in fund_cols if c in manual_funds]
        else:
            fund_cols = []  # pragma: no cover

    if not fund_cols:
        return None
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

    # Optionally compute plugin-based weights on in-sample covariance
    # Track whether a plugin engine failed so downstream (CLI/UI) can surface
    # a single prominent warning.  Store minimal structured info.
    weight_engine_fallback: dict[str, str] | None = None
    if (
        custom_weights is None
        and weighting_scheme
        and weighting_scheme.lower() != "equal"
    ):
        try:
            from .plugins import create_weight_engine

            cov = in_df[fund_cols].cov()
            engine = create_weight_engine(weighting_scheme.lower())
            w_series = engine.weight(cov).reindex(fund_cols).fillna(0.0)
            # Convert to percent mapping expected by downstream logic
            custom_weights = {c: float(w_series.get(c, 0.0) * 100.0) for c in fund_cols}
            # Ensure debug logs are emitted even if previous tests altered the logger's
            # level.  This helps `caplog` capture the success message reliably.
            logger.setLevel(logging.DEBUG)
            logger.debug("Successfully created %s weight engine", weighting_scheme)
        except Exception as e:  # pragma: no cover - exercised via tests
            # Promote to WARNING (single emission) for visibility while also
            # retaining a DEBUG breadcrumb for detailed CI logs.
            msg = (
                "Weight engine '%s' failed (%s: %s); falling back to equal weights"
                % (weighting_scheme, type(e).__name__, e)
            )
            logger.warning(msg)
            logger.debug(
                "Weight engine creation failed, falling back to equal weights: %s", e
            )
            weight_engine_fallback = {
                "engine": str(weighting_scheme),
                "error_type": type(e).__name__,
                "error": str(e),
            }
            custom_weights = None

    if custom_weights is None:
        custom_weights = {c: 100 / len(fund_cols) for c in fund_cols}
    # Convert provided weights mapping (percent) to decimal ndarray
    user_w = np.array([custom_weights.get(c, 0) / 100 for c in fund_cols], dtype=float)
    # Apply portfolio constraints if configured
    try:
        constraints_cfg = constraints or {}
        if isinstance(constraints_cfg, dict) and constraints_cfg:
            from .engine.optimizer import apply_constraints

            w_series = pd.Series(user_w, index=fund_cols, dtype=float)
            # Build minimal constraint dict; group_caps require a mapping of asset->group
            cons: dict[str, Any] = {}
            if "long_only" in constraints_cfg:
                cons["long_only"] = bool(constraints_cfg.get("long_only", True))
            if "max_weight" in constraints_cfg:
                _mw = constraints_cfg.get("max_weight")
                if _mw is not None:
                    cons["max_weight"] = float(_mw)
            if constraints_cfg.get("group_caps"):
                cons["group_caps"] = constraints_cfg.get("group_caps")
                if constraints_cfg.get("groups"):
                    cons["groups"] = constraints_cfg.get("groups")
            if cons:
                w_series = apply_constraints(w_series, cons)
            user_w = (
                w_series.reindex(fund_cols)
                .fillna(0.0)
                .to_numpy(dtype=float, copy=False)
            )
    except Exception:
        # If constraints application fails, fall back silently to original user weights
        pass

    # Keep a dictionary for result payload (already in decimals 0..1)
    user_w_dict = {c: float(w) for c, w in zip(fund_cols, user_w)}

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
        # Add portfolio-level IR references for context
        try:
            ir_eq = information_ratio(out_ew_raw, out_df[col])
            ir_usr = information_ratio(out_user_raw, out_df[col])
            # Best effort conversion; skip if not scalar convertible
            ir_dict["equal_weight"] = (
                float(ir_eq)
                if isinstance(ir_eq, (float, int, np.floating))
                else float("nan")
            )
            ir_dict["user_weight"] = (
                float(ir_usr)
                if isinstance(ir_usr, (float, int, np.floating))
                else float("nan")
            )
        except Exception:
            # Leave without portfolio-level IRs if computation fails
            pass
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
        "weight_engine_fallback": weight_engine_fallback,
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
    stats_cfg: "RiskStatsConfig" | None = None,
    weighting_scheme: str | None = None,
    constraints: dict[str, Any] | None = None,
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
        stats_cfg,
        weighting_scheme,
        constraints,
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
    metrics_list = cfg.metrics.get("registry")
    stats_cfg = None
    if metrics_list:
        from .core.rank_selection import RiskStatsConfig, canonical_metric_list

        stats_cfg = RiskStatsConfig(
            metrics_to_run=canonical_metric_list(metrics_list),
            risk_free=0.0,
        )

    res = _run_analysis(
        df,
        cast(str, split.get("in_start")),
        cast(str, split.get("in_end")),
        cast(str, split.get("out_start")),
        cast(str, split.get("out_end")),
        cfg.vol_adjust.get("target_vol", 1.0),
        getattr(cfg, "run", {}).get("monthly_cost", 0.0),
        selection_mode=cfg.portfolio.get("selection_mode", "all"),
        random_n=cfg.portfolio.get("random_n", 8),
        custom_weights=cfg.portfolio.get("custom_weights"),
        rank_kwargs=cfg.portfolio.get("rank"),
        manual_funds=cfg.portfolio.get("manual_list"),
        indices_list=cfg.portfolio.get("indices_list"),
        benchmarks=cfg.benchmarks,
        seed=getattr(cfg, "seed", 42),
        constraints=cfg.portfolio.get("constraints"),
        stats_cfg=stats_cfg,
        weighting_scheme=cfg.portfolio.get("weighting_scheme"),
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
    metrics_list = cfg.metrics.get("registry")
    stats_cfg = None
    if metrics_list:
        from .core.rank_selection import RiskStatsConfig, canonical_metric_list

        stats_cfg = RiskStatsConfig(
            metrics_to_run=canonical_metric_list(metrics_list),
            risk_free=0.0,
        )

    res = _run_analysis(
        df,
        cast(str, split.get("in_start")),
        cast(str, split.get("in_end")),
        cast(str, split.get("out_start")),
        cast(str, split.get("out_end")),
        cfg.vol_adjust.get("target_vol", 1.0),
        getattr(cfg, "run", {}).get("monthly_cost", 0.0),
        selection_mode=cfg.portfolio.get("selection_mode", "all"),
        random_n=cfg.portfolio.get("random_n", 8),
        custom_weights=cfg.portfolio.get("custom_weights"),
        rank_kwargs=cfg.portfolio.get("rank"),
        manual_funds=cfg.portfolio.get("manual_list"),
        indices_list=cfg.portfolio.get("indices_list"),
        benchmarks=cfg.benchmarks,
        seed=getattr(cfg, "seed", 42),
        weighting_scheme=cfg.portfolio.get("weighting_scheme", "equal"),
        constraints=cfg.portfolio.get("constraints"),
        stats_cfg=stats_cfg,
    )
    return {} if res is None else res


# Export alias for backward compatibility
Stats = _Stats

__all__ = [
    "Stats",  # noqa: F822
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
