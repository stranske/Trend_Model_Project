"""Vectorised multi-period back‑testing engine.

Adds support for a threshold-hold policy with Bayesian weighting in the
multi-period run path. When ``cfg.portfolio.policy == 'threshold_hold'`` we:

1) Build in-sample score frames for each period (covering the full candidate
    universe that has complete data in both IS/OOS windows).
2) Seed holdings using a simple rank selector (top-N) on the configured
    metric for the first period.
3) For subsequent periods, update holdings via :class:`Rebalancer` triggers
    using the score-frame z-scores (keep until soft/hard exits, add on entries).
4) Apply the configured Bayesian weighting scheme to the surviving holdings.
5) Delegate to ``_run_analysis`` in manual mode with the holdings and weights
    to compute scaled returns and all summary statistics, preserving the
    existing result schema expected by exporters/tests.
"""

from __future__ import annotations  # mypy: ignore-errors

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Protocol, cast

import numpy as np
import pandas as pd

from .._typing import FloatArray
from ..constants import NUMERICAL_TOLERANCE_HIGH
from ..core.rank_selection import ASCENDING_METRICS
from ..pipeline import _run_analysis
from ..rebalancing import apply_rebalancing_strategies
from ..universe import (
    MembershipTable,
    MembershipWindow,
    apply_membership_windows,
)
from ..util.missing import apply_missing_policy
from ..weighting import (
    AdaptiveBayesWeighting,
    BaseWeighting,
    EqualWeight,
    ScorePropBayesian,
)
from .loaders import load_benchmarks, load_membership, load_prices
from .replacer import Rebalancer
from .scheduler import generate_periods

# ``trend_analysis.typing`` does not exist in this project; keep the structural
# intent of ``MultiPeriodPeriodResult`` using a simple mapping alias so the
# engine remains importable without introducing a new module dependency.
MultiPeriodPeriodResult = Dict[str, Any]

SHIFT_DETECTION_MAX_STEPS_DEFAULT = 10


def _prepare_returns_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a forward-filled/zero-filled float copy of returns."""

    prepared = df.astype(float, copy=True)
    prepared = prepared.ffill().fillna(0.0)
    return prepared


def _membership_table_from_frame(frame: pd.DataFrame) -> MembershipTable:
    """Convert a membership ledger DataFrame into ``MembershipTable``."""

    if frame is None or frame.empty:
        return {}

    lookup = {str(col).strip().lower(): col for col in frame.columns}
    fund_col = lookup.get("fund") or lookup.get("symbol")
    eff_col = lookup.get("effective_date")
    end_col = lookup.get("end_date")
    if not fund_col or not eff_col or not end_col:
        raise ValueError(
            "membership data must include fund, effective_date, and end_date columns"
        )

    normalised = frame.rename(
        columns={fund_col: "fund", eff_col: "effective_date", end_col: "end_date"}
    ).copy()
    normalised["fund"] = normalised["fund"].astype(str).str.strip()
    normalised["effective_date"] = pd.to_datetime(normalised["effective_date"])
    normalised["end_date"] = pd.to_datetime(normalised["end_date"])

    table: dict[str, tuple[MembershipWindow, ...]] = {}
    grouped = normalised.sort_values(["fund", "effective_date"]).groupby("fund")
    for fund, rows in grouped:
        windows: list[MembershipWindow] = []
        for row in rows.itertuples(index=False):
            eff = pd.Timestamp(getattr(row, "effective_date"))
            end_val = getattr(row, "end_date")
            end = None if pd.isna(end_val) else pd.Timestamp(end_val)
            windows.append(MembershipWindow(eff, end))
        table[fund] = tuple(windows)
    return table


def _compute_turnover_state(
    prev_idx: FloatArray | None,
    prev_vals: FloatArray | None,
    new_series: pd.Series,
) -> tuple[float, FloatArray, FloatArray]:
    """Vectorised turnover computation used by ``run_schedule``.

    Parameters
    ----------
    prev_idx : np.ndarray | None
        Previous weight index values or ``None`` on the first iteration.
    prev_vals : np.ndarray | None
        Previous weight values aligned with ``prev_idx``.
    new_series : pd.Series
        Latest weights indexed by asset identifier.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        Total turnover together with the index/value arrays to persist for the
        next iteration.
    """

    new_series = new_series.astype(float, copy=False)
    nidx = new_series.index.to_numpy()
    nvals = new_series.to_numpy(dtype=float, copy=True)

    if prev_idx is None or prev_vals is None:
        return float(np.abs(nvals).sum()), nidx, nvals

    prev_index = pd.Index(prev_idx)
    prev_series = pd.Series(prev_vals, index=prev_index, dtype=float, copy=False)
    union_index = new_series.index.union(prev_index, sort=False)

    new_aligned: FloatArray = new_series.reindex(union_index, fill_value=0.0).to_numpy(
        dtype=float, copy=False
    )
    prev_aligned: FloatArray = prev_series.reindex(
        union_index, fill_value=0.0
    ).to_numpy(dtype=float, copy=False)

    turnover = float(np.abs(new_aligned - prev_aligned).sum())
    return turnover, nidx, nvals


def _apply_weight_bounds(
    weights: pd.Series,
    min_w_bound: float,
    max_w_bound: float,
) -> pd.Series:
    """Clamp weights to the configured bounds while preserving normalisation."""

    if weights.empty:
        return weights

    bounded = weights.astype(float, copy=False)
    bounded = bounded.clip(lower=0.0)
    capped = bounded.clip(upper=max_w_bound)
    floored = capped.copy()
    floored[floored < min_w_bound] = min_w_bound

    total = floored.sum()
    at_min = floored <= (min_w_bound + NUMERICAL_TOLERANCE_HIGH)
    at_max = floored >= (max_w_bound - NUMERICAL_TOLERANCE_HIGH)

    if total > 1.0 + NUMERICAL_TOLERANCE_HIGH:
        excess = total - 1.0
        donors = floored[~at_min]
        if not donors.empty:
            avail = (donors - min_w_bound).clip(lower=0.0)
            avail_sum = avail.sum()
            if avail_sum > 0:
                cut = (avail / avail_sum) * excess
                floored.loc[donors.index] = (donors - cut).clip(lower=min_w_bound)
    elif total < 1.0 - NUMERICAL_TOLERANCE_HIGH:
        deficit = 1.0 - total
        receivers = floored[~at_max]
        if not receivers.empty:
            room = (max_w_bound - receivers).clip(lower=0.0)
            room_sum = room.sum()
            if room_sum > 0:
                add = (room / room_sum) * deficit
                floored.loc[receivers.index] = (receivers + add).clip(upper=max_w_bound)

    floored = floored.clip(lower=min_w_bound, upper=max_w_bound)

    total = floored.sum()
    if abs(total - 1.0) > 1e-9:
        if total > 1.0:
            excess = total - 1.0
            donors = floored[~(floored <= min_w_bound + NUMERICAL_TOLERANCE_HIGH)]
            if not donors.empty:
                share = (donors - min_w_bound).clip(lower=0.0)
                sh = share.sum()
                if sh > 0:
                    floored.loc[donors.index] = (donors - (share / sh) * excess).clip(
                        lower=min_w_bound
                    )
        else:
            deficit = 1.0 - total
            receivers = floored[~(floored >= max_w_bound - NUMERICAL_TOLERANCE_HIGH)]
            if not receivers.empty:
                room = (max_w_bound - receivers).clip(lower=0.0)
                rm = room.sum()
                if rm > 0:
                    floored.loc[receivers.index] = (
                        receivers + (room / rm) * deficit
                    ).clip(upper=max_w_bound)

    return floored


@dataclass
class Portfolio:
    """Minimal container for weight, turnover and cost history."""

    history: Dict[str, pd.Series] = field(default_factory=dict)
    turnover: Dict[str, float] = field(default_factory=dict)
    costs: Dict[str, float] = field(default_factory=dict)
    total_rebalance_costs: float = 0.0

    def rebalance(
        self,
        date: str | pd.Timestamp,
        weights: pd.DataFrame | pd.Series,
        turnover: float = 0.0,
        cost: float = 0.0,
    ) -> None:
        """Store weights and trading activity for the given date."""
        # Normalise to a pandas Series of floats for storage
        if isinstance(weights, pd.DataFrame):
            if "weight" in weights.columns:
                series = weights["weight"]
            else:
                # take the first column as weights
                series = weights.iloc[:, 0]
        elif isinstance(weights, pd.Series):
            series = weights
        else:
            # Attempt to build a Series from a mapping-like object
            series = pd.Series(weights)
        key = str(pd.to_datetime(date).date())
        self.history[key] = series.astype(float)
        self.turnover[key] = float(turnover)
        self.costs[key] = float(cost)
        self.total_rebalance_costs += float(cost)


class SelectorProtocol(Protocol):
    """Minimal interface required of selectors."""

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Protocol method signature only; concrete selectors implement logic.
        ...


def run_schedule(
    score_frames: Mapping[str, pd.DataFrame],
    selector: SelectorProtocol,
    weighting: BaseWeighting,
    *,
    rank_column: str | None = None,
    rebalancer: "Rebalancer | None" = None,
    rebalance_strategies: List[str] | None = None,
    rebalance_params: Dict[str, Dict[str, Any]] | None = None,
) -> Portfolio:
    """Apply selection and weighting across ``score_frames``.

    Parameters
    ----------
    score_frames : Mapping[str, pd.DataFrame]
        Score frames by date
    selector : SelectorProtocol
        Asset selection protocol
    weighting : BaseWeighting
        Weighting scheme
    rank_column : str, optional
        Column to use for ranking
    rebalancer : Rebalancer, optional
        Fund selection rebalancer (legacy threshold-hold system)
    rebalance_strategies : list[str], optional
        List of rebalancing strategy names to apply
    rebalance_params : dict, optional
        Parameters for each rebalancing strategy

    Returns
    -------
    Portfolio
        Portfolio with weight history
    """

    pf = Portfolio()
    prev_date: pd.Timestamp | None = None
    prev_weights: pd.Series | None = None
    # Fast turnover state (index array + values array)
    prev_tidx: FloatArray | None = None
    prev_tvals: FloatArray | None = None

    def _fast_turnover(
        prev_idx: FloatArray | None,
        prev_vals: FloatArray | None,
        new_series: pd.Series,
    ) -> tuple[float, FloatArray, FloatArray]:
        """Compute turnover between previous and new weights using NumPy.

            Parameters
            ----------
        prev_idx : FloatArray | None
                Previous weight index values (object dtype) or None on first call.
        prev_vals : FloatArray | None
                Previous weight values aligned with ``prev_idx``.
            new_series : pd.Series
                New weights (float) indexed by asset identifier.

            Returns
            -------
            turnover : float
                Sum of absolute weight changes.
        next_idx, next_vals : FloatArray, FloatArray
                Stored index/value arrays for next iteration (copy-safe).
        """
        # First period: turnover = sum(abs(new_w))
        nidx = new_series.index.to_numpy()
        nvals = new_series.to_numpy(dtype=float, copy=True)
        if prev_idx is None or prev_vals is None:
            return float(np.abs(nvals).sum()), nidx, nvals
        # Build unified index mapping only once per call
        # Map previous positions
        pmap = {k: i for i, k in enumerate(prev_idx.tolist())}
        # Map new positions
        # Collect union preserving new ordering first then unseen old to keep determinism
        union_list: list[Any] = []
        seen: set[Any] = set()
        for k in nidx.tolist():
            union_list.append(k)
            seen.add(k)
        for k in prev_idx.tolist():
            if k not in seen:
                union_list.append(k)
                seen.add(k)
        union_arr = np.array(union_list, dtype=object)
        # Allocate aligned arrays
        new_aligned: FloatArray = np.zeros(len(union_arr), dtype=float)
        prev_aligned: FloatArray = np.zeros(len(union_arr), dtype=float)
        # Fill new
        nmap = {k: i for i, k in enumerate(nidx.tolist())}
        for i, k in enumerate(union_arr.tolist()):
            if k in nmap:
                new_aligned[i] = nvals[nmap[k]]
            if k in pmap:
                prev_aligned[i] = prev_vals[pmap[k]]
        turnover = float(np.abs(new_aligned - prev_aligned).sum())
        return turnover, nidx, nvals

    col = (
        rank_column
        or getattr(selector, "rank_column", None)
        or getattr(selector, "column", None)
    )

    for date in sorted(score_frames):
        sf = score_frames[date]
        date_ts = pd.to_datetime(date)
        selected, _ = selector.select(sf)
        target_weights = weighting.weight(selected, date_ts)

        # Apply legacy rebalancer (threshold-hold system) if configured
        if rebalancer is not None:
            if prev_weights is None:
                prev_weights = target_weights["weight"].astype(float)
            prev_weights = rebalancer.apply_triggers(cast(pd.Series, prev_weights), sf)
            target_weights = prev_weights.to_frame("weight")

        # Apply rebalancing strategies if configured
        if rebalance_strategies and rebalance_params:
            current_weights = (
                prev_weights if prev_weights is not None else pd.Series(dtype=float)
            )
            target_weight_series = target_weights["weight"].astype(float)

            # Get scores for priority-based strategies
            scores = sf.get(col) if col and col in sf.columns else None

            final_weights, cost = apply_rebalancing_strategies(
                rebalance_strategies,
                rebalance_params,
                current_weights,
                target_weight_series,
                scores=scores,
            )

            # Fast turnover computation
            turnover, prev_tidx, prev_tvals = _fast_turnover(
                prev_tidx, prev_tvals, final_weights.astype(float)
            )
            weights = final_weights.to_frame("weight")
            prev_weights = final_weights
        else:
            cost = 0.0
            weights = target_weights
            tw = target_weights["weight"].astype(float)
            turnover, prev_tidx, prev_tvals = _compute_turnover_state(
                prev_tidx, prev_tvals, tw
            )
            prev_weights = tw

        pf.rebalance(date, weights, turnover, cost)

        if col and col in sf.columns:
            if prev_date is None:
                days = 0
            else:
                days = (pd.to_datetime(date) - prev_date).days
            s = sf.reindex(weights.index)[col].dropna()
            update_fn = getattr(weighting, "update", None)
            if callable(update_fn):
                try:
                    update_fn(s, days)
                except Exception:  # pragma: no cover - defensive
                    pass
        prev_date = pd.to_datetime(date)

    # Optional debug validation: recompute turnover from stored history and compare
    if os.getenv("DEBUG_TURNOVER_VALIDATE"):
        try:
            dates = sorted(pf.history)
            prev = None
            for d in dates:
                w = pf.history[d].astype(float)
                if prev is None:
                    expected = float(np.abs(w.to_numpy()).sum())
                else:
                    # align via union
                    u = prev.index.union(w.index)
                    dv = w.reindex(u, fill_value=0.0).to_numpy()
                    pv = prev.reindex(u, fill_value=0.0).to_numpy()
                    expected = float(np.abs(dv - pv).sum())
                got = pf.turnover[d]
                if not np.isclose(
                    expected, got, rtol=0, atol=1e-12
                ):  # pragma: no cover
                    raise AssertionError(
                        f"Turnover mismatch for {d}: expected {expected} got {got}"
                    )
                prev = w
        except Exception:  # pragma: no cover - defensive
            pass
    return pf


def run(
    cfg: Any,
    df: pd.DataFrame | None = None,
    price_frames: dict[str, pd.DataFrame] | None = None,
    *,
    membership: pd.DataFrame | None = None,
) -> List[MultiPeriodPeriodResult]:
    """Run the multi‑period back‑test.

    Parameters
    ----------
    cfg : Config
        Loaded configuration object. ``cfg.multi_period`` drives the
        scheduling logic.
    df : pd.DataFrame, optional
        Pre-loaded returns data. Provide this or ``price_frames``.
    price_frames : dict[str, pd.DataFrame], optional
        Pre-computed price data frames by date/period. If provided, used instead
        of ``df``. One of ``df`` or ``price_frames`` must be supplied.
    membership : pd.DataFrame, optional
        Membership ledger DataFrame containing ``fund``, ``effective_date`` and
        ``end_date`` columns. Required when the configuration specifies
        ``data.universe_membership_path``.

    Returns
    -------
    list[dict[str, object]]
        One result dictionary per generated period.  Each result is the
        full output of ``_run_analysis`` augmented with a ``period`` key
        for reference.
    """

    # Validate price_frames parameter
    if price_frames is not None:
        if not isinstance(price_frames, dict):
            raise TypeError("price_frames must be a dict[str, pd.DataFrame] or None")
        for date_key, frame in price_frames.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    f"price_frames['{date_key}'] must be a pandas DataFrame"
                )
            required_columns = ["Date"]
            missing_columns = [
                col for col in required_columns if col not in frame.columns
            ]
            if missing_columns:
                raise ValueError(
                    (
                        f"price_frames['{date_key}'] is missing required columns: "
                        f"{missing_columns}. Required columns are: {required_columns}. "
                        f"Available columns are: {list(frame.columns)}"
                    )
                )

    # If price_frames is provided, use it to build df
    if price_frames is not None:
        # Robustly combine all price frames into a single DataFrame by
        # aligning on 'Date'. Use an outer join to ensure all dates and
        # columns are included, handling missing data gracefully.
        combined_frames = [frame.copy() for frame in price_frames.values()]
        if combined_frames:
            df = pd.concat(
                combined_frames, axis=0, join="outer", ignore_index=True, sort=True
            )
            # Sort by Date to ensure proper ordering
            df = df.sort_values("Date").reset_index(drop=True)
            # Remove any duplicates created during concatenation
            df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        else:
            raise ValueError("price_frames is empty - no data to process")

    data_settings = getattr(cfg, "data", {}) or {}
    missing_policy_cfg = data_settings.get("missing_policy")
    if missing_policy_cfg is None:
        missing_policy_cfg = data_settings.get("nan_policy")
    missing_limit_cfg = data_settings.get("missing_limit")
    if missing_limit_cfg is None:
        missing_limit_cfg = data_settings.get("nan_limit")

    if df is None:
        raise ValueError(
            "multi_period.run requires either a pre-loaded DataFrame or "
            "price_frames; provide an in-memory frame via the 'df' or "
            "'price_frames' argument"
        )

    if "Date" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Date' column")

    # Normalise Date column for consistent slicing downstream
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    original_returns = df.set_index("Date")
    skip_missing_policy = (
        price_frames is not None
        and missing_policy_cfg is None
        and missing_limit_cfg is None
    )

    if skip_missing_policy:
        policy_spec: str | Mapping[str, str] | None = None
        cleaned = original_returns
        _missing_summary = None
    else:
        policy_spec = missing_policy_cfg or "ffill"
        cleaned, _missing_summary = apply_missing_policy(
            original_returns,
            policy=policy_spec,
            limit=missing_limit_cfg,
        )

    cleaned = cleaned.dropna(how="all")
    if cleaned.empty:
        raise ValueError("Missing-data policy removed all assets for analysis")

    membership_required = bool(data_settings.get("universe_membership_path"))
    membership_frame = membership if membership is not None else None
    if membership_frame is not None and membership_frame.empty:
        membership_frame = None
    membership_windows: MembershipTable | None = None
    if membership_frame is not None:
        membership_windows = _membership_table_from_frame(membership_frame)
    elif membership_required:
        raise ValueError(
            "cfg.data['universe_membership_path'] was provided; pass a membership "
            "DataFrame via the 'membership' argument or call run_from_config()."
        )

    if membership_windows:
        missing_entries = sorted(
            asset for asset in cleaned.columns if asset not in membership_windows
        )
        if missing_entries:
            preview = ", ".join(missing_entries[:5])
            raise ValueError(
                "Universe membership is missing effective_date entries for: "
                f"{preview}" + ("…" if len(missing_entries) > 5 else "")
            )
        cleaned = apply_membership_windows(cleaned, membership_windows)
        cleaned = cleaned.dropna(how="all")
        if cleaned.empty:
            raise ValueError("Universe membership removed all rows for analysis")

    # Restore Date column for downstream consumers
    df = cleaned.reset_index()
    preprocessing_cfg = getattr(cfg, "preprocessing", {}) or {}
    observed_freq: str | None = None
    if not df.empty and "Date" in df.columns:
        try:
            observed_freq = pd.infer_freq(pd.DatetimeIndex(df["Date"]))
        except Exception:  # pragma: no cover - inference best effort
            observed_freq = None
    df.attrs["calendar_settings"] = {
        "frequency": observed_freq or data_settings.get("frequency"),
        "timezone": data_settings.get("timezone", "UTC"),
        "holiday_calendar": preprocessing_cfg.get("holiday_calendar"),
    }

    # If policy is not threshold-hold, use the Phase‑1 style per-period runs.
    if str(cfg.portfolio.get("policy", "").lower()) != "threshold_hold":
        periods = generate_periods(cfg.model_dump())
        out_results: List[MultiPeriodPeriodResult] = []
        # Performance flags
        perf_flags = getattr(cfg, "performance", {}) or {}
        enable_cache = bool(perf_flags.get("enable_cache", True))
        incremental_cov = bool(perf_flags.get("incremental_cov", False))
        prev_cov_payload = None  # rolling covariance state
        cov_cache_obj = None
        if enable_cache:
            try:  # lazy import to avoid hard dependency if module layout changes
                from ..perf.cache import CovCache

                cov_cache_obj = CovCache()
            except Exception:  # pragma: no cover - defensive
                cov_cache_obj = None
        prev_in_df = None

        for pt in periods:
            res = _run_analysis(
                df,
                pt.in_start[:7],
                pt.in_end[:7],
                pt.out_start[:7],
                pt.out_end[:7],
                cfg.vol_adjust.get("target_vol", 1.0),
                getattr(cfg, "run", {}).get("monthly_cost", 0.0),
                floor_vol=cfg.vol_adjust.get("floor_vol"),
                warmup_periods=int(cfg.vol_adjust.get("warmup_periods", 0) or 0),
                selection_mode=cfg.portfolio.get("selection_mode", "all"),
                random_n=cfg.portfolio.get("random_n", 8),
                custom_weights=cfg.portfolio.get("custom_weights"),
                rank_kwargs=cfg.portfolio.get("rank"),
                manual_funds=cfg.portfolio.get("manual_list"),
                indices_list=cfg.portfolio.get("indices_list"),
                benchmarks=cfg.benchmarks,
                seed=getattr(cfg, "seed", 42),
                missing_policy=policy_spec,
                missing_limit=missing_limit_cfg,
                risk_window=cfg.vol_adjust.get("window"),
                previous_weights=cfg.portfolio.get("previous_weights"),
                max_turnover=cfg.portfolio.get("max_turnover"),
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

            # (Experimental) attach covariance diag using cache/incremental path for diagnostics.
            # Keeps existing outputs stable; adds optional "cov_diag" key.
            if enable_cache:
                from ..perf.cache import compute_cov_payload, incremental_cov_update

                in_start = pt.in_start[:7]
                in_end = pt.in_end[:7]
                # Recreate in-sample frame identical to _run_analysis slice
                date_col = "Date"
                sub = df.copy()
                if not pd.api.types.is_datetime64_any_dtype(sub[date_col]):
                    sub[date_col] = pd.to_datetime(sub[date_col])
                sub.sort_values(date_col, inplace=True)
                sdate = pd.to_datetime(f"{in_start}-01") + pd.offsets.MonthEnd(0)
                edate = pd.to_datetime(f"{in_end}-01") + pd.offsets.MonthEnd(0)
                in_df_full = sub[
                    (sub[date_col] >= sdate) & (sub[date_col] <= edate)
                ].set_index(date_col)
                # Remove benchmark columns if present in result universe
                fund_cols = [
                    c
                    for c in in_df_full.columns
                    if c not in (cfg.benchmarks or {}).values()
                ]
                in_df_full = in_df_full[fund_cols]
                in_df_prepared = _prepare_returns_frame(in_df_full)

                if (
                    incremental_cov
                    and prev_cov_payload is not None
                    and prev_in_df is not None
                ):
                    same_len = prev_in_df.shape[0] == in_df_prepared.shape[0]
                    same_cols = (
                        prev_in_df.columns.tolist() == in_df_prepared.columns.tolist()
                    )
                    n_rows = in_df_prepared.shape[0]
                    if same_cols and n_rows >= 3:
                        # Determine shift distance k (number of rows replaced at head and appended at tail)
                        k = None
                        if same_len:
                            # Compare trailing blocks to find minimal k
                            raw_max_steps = perf_flags.get(
                                "shift_detection_max_steps",
                                SHIFT_DETECTION_MAX_STEPS_DEFAULT,
                            )
                            try:
                                max_shift_steps = int(raw_max_steps)
                            except (TypeError, ValueError):
                                max_shift_steps = SHIFT_DETECTION_MAX_STEPS_DEFAULT
                            max_shift_steps = max(1, max_shift_steps)
                            for step in range(
                                1, min(max_shift_steps, n_rows - 1)
                            ):  # cap search for safety
                                prev_block = prev_in_df.iloc[step:].to_numpy()
                                new_block = in_df_prepared.iloc[:-step].to_numpy()
                                if np.allclose(
                                    prev_block,
                                    new_block,
                                    rtol=0,
                                    atol=1e-12,
                                ) or np.array_equal(prev_block, new_block):
                                    k = step
                                    break
                        if k is None:
                            # Fallback full recompute
                            prev_cov_payload = compute_cov_payload(
                                in_df_prepared, materialise_aggregates=incremental_cov
                            )
                        else:
                            # Apply k incremental updates sequentially
                            try:
                                for step in range(k):
                                    old_row = prev_in_df.iloc[step].to_numpy(
                                        dtype=float
                                    )
                                    new_row = in_df_prepared.iloc[
                                        n_rows - k + step
                                    ].to_numpy(dtype=float)
                                    prev_cov_payload = incremental_cov_update(
                                        prev_cov_payload, old_row, new_row
                                    )
                                    if cov_cache_obj is not None:
                                        cov_cache_obj.incremental_updates += 1
                            except Exception:  # pragma: no cover - fallback safety
                                prev_cov_payload = compute_cov_payload(
                                    in_df_prepared,
                                    materialise_aggregates=incremental_cov,
                                )
                    else:
                        prev_cov_payload = compute_cov_payload(
                            in_df_prepared, materialise_aggregates=incremental_cov
                        )
                else:
                    from ..perf.cache import compute_cov_payload as _ccp

                    prev_cov_payload = _ccp(
                        in_df_prepared, materialise_aggregates=incremental_cov
                    )
                prev_in_df = in_df_prepared
                res["cov_diag"] = prev_cov_payload.cov.diagonal().tolist()
                if cov_cache_obj is not None:
                    # attach cache stats for observability (does not alter existing keys)
                    res.setdefault("cache_stats", cov_cache_obj.stats())
            out_results.append(cast(MultiPeriodPeriodResult, res))
        return out_results

    # Threshold-hold path with Bayesian weighting
    periods = generate_periods(cfg.model_dump())

    # --- helpers --------------------------------------------------------
    def _parse_month(s: str) -> pd.Timestamp:
        return pd.to_datetime(f"{s}-01") + pd.offsets.MonthEnd(0)

    def _valid_universe(
        full: pd.DataFrame, in_start: str, in_end: str, out_start: str, out_end: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
        date_col = "Date"
        sub = full.copy()
        if not pd.api.types.is_datetime64_any_dtype(sub[date_col]):
            sub[date_col] = pd.to_datetime(sub[date_col])
        sub.sort_values(date_col, inplace=True)
        in_sdate, in_edate = _parse_month(in_start), _parse_month(in_end)
        out_sdate, out_edate = _parse_month(out_start), _parse_month(out_end)
        in_df = sub[
            (sub[date_col] >= in_sdate) & (sub[date_col] <= in_edate)
        ].set_index(date_col)
        out_df = sub[
            (sub[date_col] >= out_sdate) & (sub[date_col] <= out_edate)
        ].set_index(date_col)
        if in_df.empty or out_df.empty:
            return in_df, out_df, [], ""
        ret_cols = [c for c in sub.columns if c != date_col]
        # Exclude indices if configured
        indices_list = cast(list[str] | None, cfg.portfolio.get("indices_list")) or []
        if indices_list:
            idx_set = set(indices_list)
            ret_cols = [c for c in ret_cols if c not in idx_set]
        rf_col = min(ret_cols, key=lambda c: sub[c].std())
        fund_cols = [c for c in ret_cols if c != rf_col]
        # Keep only funds with complete data in both windows
        in_ok = ~in_df[fund_cols].isna().any()
        out_ok = ~out_df[fund_cols].isna().any()
        fund_cols = [c for c in fund_cols if in_ok[c] and out_ok[c]]
        return in_df, out_df, fund_cols, rf_col

    def _score_frame(in_df: pd.DataFrame, funds: list[str]) -> pd.DataFrame:
        # Compute metrics frame for the in-sample window (vectorised)
        from ..core.rank_selection import RiskStatsConfig, _compute_metric_series

        stats_cfg = RiskStatsConfig(risk_free=0.0)
        # Canonical metrics as produced by
        # single_period_run/_compute_metric_series
        metrics = [
            "AnnualReturn",
            "Volatility",
            "Sharpe",
            "Sortino",
            "InformationRatio",
            "MaxDrawdown",
        ]
        parts = [_compute_metric_series(in_df[funds], m, stats_cfg) for m in metrics]
        sf = pd.concat(parts, axis=1)
        sf.columns = [
            "CAGR",
            "Volatility",
            "Sharpe",
            "Sortino",
            "InformationRatio",
            "MaxDrawdown",
        ]
        return sf.astype(float)

    def _ensure_zscore(sf: pd.DataFrame, metric: str) -> pd.DataFrame:
        col = metric if metric in sf.columns else "Sharpe"
        s = sf[col].astype(float)
        mu, sigma = float(s.mean()), float(s.std(ddof=0))
        if not pd.notna(sigma) or sigma == 0:
            z = pd.Series(0.0, index=s.index, dtype=float)
        else:
            z = (s - mu) / sigma
        out = sf.copy()
        out["zscore"] = z
        return out

    # Build selector and weighting
    from ..selector import create_selector_by_name

    th_cfg = cast(dict[str, Any], cfg.portfolio.get("threshold_hold", {}))
    target_n = int(th_cfg.get("target_n", cfg.portfolio.get("random_n", 8)))
    seed_metric = cast(
        str,
        (cfg.portfolio.get("selector", {}) or {})
        .get("params", {})
        .get("rank_column", "Sharpe"),
    )
    selector = create_selector_by_name("rank", top_n=target_n, rank_column=seed_metric)

    # Portfolio constraints
    constraints = cast(dict[str, Any], cfg.portfolio.get("constraints", {}))
    max_funds = int(constraints.get("max_funds", 10))
    min_w_bound = float(constraints.get("min_weight", 0.05))  # decimal
    max_w_bound = float(constraints.get("max_weight", 0.18))  # decimal
    # consecutive below-min to replace
    # Prefer constraints for this rule (it’s a weight constraint),
    # but keep backward‑compat by falling back to threshold_hold if present.
    low_min_strikes_req = int(
        constraints.get(
            "min_weight_strikes",
            th_cfg.get("min_weight_strikes", 2),
        )
    )

    w_cfg = cast(dict[str, Any], cfg.portfolio.get("weighting", {}))
    w_name = str(w_cfg.get("name", "equal")).lower()
    w_params = cast(dict[str, Any], w_cfg.get("params", {}))
    # Column for score‑proportional weightings defaults to Sharpe
    w_column = cast(str, w_params.get("column", "Sharpe"))
    if w_name in {"equal", "ew"}:
        weighting: BaseWeighting = EqualWeight()
    elif w_name in {"score_prop_bayes", "bayes", "score_bayes"}:
        weighting = ScorePropBayesian(
            column=w_column, shrink_tau=float(w_params.get("shrink_tau", 0.25))
        )
    elif w_name in {"adaptive_bayes", "adaptive"}:
        weighting = AdaptiveBayesWeighting(
            half_life=int(w_params.get("half_life", 90)),
            obs_sigma=float(w_params.get("obs_sigma", 0.25)),
            max_w=w_params.get("max_w"),
            prior_mean=w_params.get("prior_mean", "equal"),
            prior_tau=float(w_params.get("prior_tau", 1.0)),
        )
    else:
        weighting = EqualWeight()

    rebalancer = Rebalancer(cfg.model_dump())

    # --- main loop ------------------------------------------------------
    results: List[MultiPeriodPeriodResult] = []
    prev_weights: pd.Series | None = None
    prev_final_weights: pd.Series | None = None
    # Transaction cost and turnover-cap controls (Issue #429)
    tc_bps = float(cfg.portfolio.get("transaction_cost_bps", 0.0))
    max_turnover_cap = float(cfg.portfolio.get("max_turnover", 1.0))
    low_weight_strikes: dict[str, int] = {}

    def _firm(name: str) -> str:
        return str(name).split()[0] if isinstance(name, str) and name else str(name)

    def _dedupe_one_per_firm(
        sf: pd.DataFrame, holdings: list[str], metric: str
    ) -> list[str]:
        if not holdings:
            return holdings
        col = metric if metric in sf.columns else "Sharpe"
        ascending = col in ASCENDING_METRICS
        # sort candidates by metric (and zscore as tiebreaker) best-first
        tmp = sf.loc[
            [h for h in holdings if h in sf.index],
            [col, "zscore" if "zscore" in sf.columns else col],
        ].copy()
        if "zscore" not in tmp.columns:
            tmp["zscore"] = 0.0
        tmp["_firm"] = [_firm(ix) for ix in tmp.index]
        tmp.sort_values([col, "zscore"], ascending=[ascending, False], inplace=True)
        seen: set[str] = set()
        deduped: list[str] = []
        for fund in tmp.index:
            f = _firm(fund)
            if f in seen:
                continue
            seen.add(f)
            deduped.append(fund)
        return deduped

    for pt in periods:
        period_ts = pd.to_datetime(pt.out_end)
        in_df, out_df, fund_cols, _rf_col = _valid_universe(
            df, pt.in_start[:7], pt.in_end[:7], pt.out_start[:7], pt.out_end[:7]
        )
        if not fund_cols:
            # Preserve period alignment: produce a minimal placeholder so downstream
            # consumers expecting one entry per generated period retain indexing.
            # (Chosen over 'continue' because some tests assert len(results) == len(periods)).
            # Represent missing metrics explicitly as ``None`` rather than a
            # tuple of zeroes.  Downstream consumers (and tests) expect a
            # ``None`` placeholder so that an empty universe is distinguishable
            # from genuine statistics that just happen to be zero.
            empty_metrics = None
            results.append(
                cast(
                    MultiPeriodPeriodResult,
                    {
                        "period": (
                            pt.in_start,
                            pt.in_end,
                            pt.out_start,
                            pt.out_end,
                        ),
                        "selected_funds": [],
                        "in_sample_scaled": pd.DataFrame(),
                        "out_sample_scaled": pd.DataFrame(),
                        "in_sample_stats": {},
                        "out_sample_stats": {},
                        "out_sample_stats_raw": {},
                        "in_ew_stats": empty_metrics,
                        "out_ew_stats": empty_metrics,
                        "out_ew_stats_raw": empty_metrics,
                        "in_user_stats": empty_metrics,
                        "out_user_stats": empty_metrics,
                        "out_user_stats_raw": empty_metrics,
                        "ew_weights": {},
                        "fund_weights": {},
                        "benchmark_stats": {},
                        "benchmark_ir": {},
                        "score_frame": pd.DataFrame(),
                        "weight_engine_fallback": None,
                        "manager_changes": [],
                    },
                )
            )
            continue
        sf = _score_frame(in_df, fund_cols)
        metric = cast(str, th_cfg.get("metric", "Sharpe"))
        sf = _ensure_zscore(sf, metric)

        # Determine holdings
        # Track manager changes with reasons for this period
        events: list[dict[str, object]] = []

        if prev_weights is None:
            # Seed via rank selector
            selected, _ = selector.select(sf)
            seed_weights = weighting.weight(selected, period_ts)
            holdings = list(seed_weights.index)
            # Enforce one-per-firm on seed
            holdings = _dedupe_one_per_firm(sf, holdings, metric)
            # Enforce max funds on seed by zscore desc (seed only)
            if len(holdings) > max_funds:
                zsorted = (
                    sf.loc[holdings]
                    .sort_values("zscore", ascending=False)
                    .index.tolist()
                )
                keep: list[str] = []
                seen: set[str] = set()
                for f in zsorted:
                    firm = _firm(f)
                    if firm in seen:
                        continue
                    keep.append(f)
                    seen.add(firm)
                    if len(keep) >= max_funds:
                        break
                holdings = keep
            weights_df = weighting.weight(sf.loc[holdings], period_ts)
            prev_weights = weights_df["weight"].astype(float)
            # Log seed additions
            for f in holdings:
                events.append(
                    {
                        "action": "added",
                        "manager": f,
                        "firm": _firm(f),
                        "reason": "seed",
                        "detail": "initial portfolio seed",
                    }
                )
        else:
            # Use rebalancer to update holdings; then apply Bayesian weights
            # Capture holdings prior to rebalancer
            before_reb = set(prev_weights.index)
            rebased = rebalancer.apply_triggers(prev_weights.astype(float), sf)
            # Restrict to funds available in this period's score-frame
            holdings = [h for h in list(rebased.index) if h in sf.index]
            after_reb = set(holdings)
            # Log drops/adds due to rebalancer z-triggers
            z_exit_soft = float(th_cfg.get("z_exit_soft", -1.0))
            z_entry_soft = float(th_cfg.get("z_entry_soft", 1.0))
            dropped_reb = before_reb - after_reb
            for f in sorted(dropped_reb):
                try:
                    val = (
                        pd.to_numeric(sf.loc[f, "zscore"], errors="coerce")
                        if f in sf.index
                        else pd.NA
                    )
                    z = float(val) if pd.notna(val) else float("nan")
                except Exception:
                    z = float("nan")
                reason = "z_exit" if (pd.notna(z) and z < z_exit_soft) else "rebalance"
                events.append(
                    {
                        "action": "dropped",
                        "manager": f,
                        "firm": _firm(f),
                        "reason": reason,
                        "detail": f"zscore={z:.3f}",
                    }
                )
            added_reb = after_reb - before_reb
            for f in sorted(added_reb):
                try:
                    val = (
                        pd.to_numeric(sf.loc[f, "zscore"], errors="coerce")
                        if f in sf.index
                        else pd.NA
                    )
                    z = float(val) if pd.notna(val) else float("nan")
                except Exception:
                    z = float("nan")
                reason = (
                    "z_entry"
                    if (pd.notna(z) and z > z_entry_soft - NUMERICAL_TOLERANCE_HIGH)
                    else "rebalance"
                )
                events.append(
                    {
                        "action": "added",
                        "manager": f,
                        "firm": _firm(f),
                        "reason": reason,
                        "detail": f"zscore={z:.3f}",
                    }
                )
            # Enforce one-per-firm
            before_dedupe = set(holdings)
            holdings = _dedupe_one_per_firm(sf, holdings, metric)
            after_dedupe = set(holdings)
            dropped_dup = before_dedupe - after_dedupe
            for f in sorted(dropped_dup):
                events.append(
                    {
                        "action": "dropped",
                        "manager": f,
                        "firm": _firm(f),
                        "reason": "one_per_firm",
                        "detail": "duplicate firm pruned",
                    }
                )
            # Do not auto-remove just because we're above max_funds.
            # We still respect max_funds when seeding and when adding new funds
            # elsewhere.
            if len(holdings) == 0:  # guard: reseed if empty
                selected, _ = selector.select(sf)
                holdings = list(selected.index)
                holdings = _dedupe_one_per_firm(sf, holdings, metric)
                for f in holdings:
                    events.append(
                        {
                            "action": "added",
                            "manager": f,
                            "firm": _firm(f),
                            "reason": "reseat",
                            "detail": "reseeding empty portfolio",
                        }
                    )
            weights_df = weighting.weight(sf.loc[holdings], period_ts)
            prev_weights = weights_df["weight"].astype(float)

        # Natural weights (pre-bounds) for strikes on min threshold
        nat_w = prev_weights.copy()

        # Low-weight replacement rule: if a fund naturally < min for N consecutive
        # periods, replace it this period before finalising weights.
        to_remove: list[str] = []
        for f, wv in nat_w.items():
            f_str = str(f)
            if float(wv) < min_w_bound:
                low_weight_strikes[f_str] = int(low_weight_strikes.get(f_str, 0)) + 1
            else:
                low_weight_strikes[f_str] = 0
            if int(low_weight_strikes.get(f_str, 0)) >= low_min_strikes_req:
                to_remove.append(f_str)
        if to_remove:
            # drop and try to refill from high zscore sidelined funds
            holdings = [h for h in holdings if h not in to_remove]
            for f in to_remove:
                # Log low-weight drop
                events.append(
                    {
                        "action": "dropped",
                        "manager": f,
                        "firm": _firm(f),
                        "reason": "low_weight_strikes",
                        "detail": (
                            f"below min {min_w_bound:.2%} for "
                            f"{low_min_strikes_req} periods"
                        ),
                    }
                )
                low_weight_strikes.pop(f, None)
            # Fill to target/min(len(sf), max_funds)
            need = max(0, min(max_funds, target_n) - len(holdings))
            if need > 0:
                candidates = [c for c in sf.index if c not in holdings]
                add_from = (
                    sf.loc[candidates]
                    .sort_values("zscore", ascending=False)
                    .index.tolist()
                )
                for f in add_from:
                    if len(holdings) >= min(max_funds, target_n):
                        break
                    if _firm(f) in {_firm(x) for x in holdings}:
                        continue
                    holdings.append(f)
                    events.append(
                        {
                            "action": "added",
                            "manager": f,
                            "firm": _firm(f),
                            "reason": "replacement",
                            "detail": "filled from highest zscore sidelined",
                        }
                    )
            if holdings:
                weights_df = weighting.weight(sf.loc[holdings], period_ts)
                nat_w = weights_df["weight"].astype(float)
                prev_weights = nat_w.copy()

        # Apply weight bounds and renormalise
        bounded_w = _apply_weight_bounds(prev_weights, min_w_bound, max_w_bound)

        # Enforce optional turnover cap by scaling trades towards target
        target_w = bounded_w.copy()
        if prev_final_weights is None:
            last_aligned = pd.Series(0.0, index=target_w.index)
        else:
            union_ix = prev_final_weights.index.union(target_w.index)
            last_aligned = prev_final_weights.reindex(union_ix, fill_value=0.0)
            target_w = target_w.reindex(union_ix, fill_value=0.0)

        desired_trades = target_w - last_aligned
        desired_turnover = float(desired_trades.abs().sum())
        final_w = target_w.copy()
        if (
            max_turnover_cap < 1.0 - NUMERICAL_TOLERANCE_HIGH
            and desired_turnover > max_turnover_cap + NUMERICAL_TOLERANCE_HIGH
        ):
            # Scale trades proportionally towards target to respect cap
            scale = max_turnover_cap / desired_turnover if desired_turnover > 0 else 0.0
            final_w = last_aligned + desired_trades * scale
        # Ensure bounds and normalisation remain satisfied
        final_w = _apply_weight_bounds(final_w, min_w_bound, max_w_bound)

        # Track turnover/cost for this period; persist weights for next period
        period_turnover = float((final_w - last_aligned).abs().sum())
        period_cost = period_turnover * (tc_bps / 10000.0)
        prev_final_weights = final_w.copy()
        prev_weights = final_w.copy()

        # Prepare custom weights mapping in percent for _run_analysis
        custom: dict[str, float] = {
            str(k): float(v) * 100.0 for k, v in prev_weights.items()
        }

        res = _run_analysis(
            df,
            pt.in_start[:7],
            pt.in_end[:7],
            pt.out_start[:7],
            pt.out_end[:7],
            cfg.vol_adjust.get("target_vol", 1.0),
            getattr(cfg, "run", {}).get("monthly_cost", 0.0),
            floor_vol=cfg.vol_adjust.get("floor_vol"),
            warmup_periods=int(cfg.vol_adjust.get("warmup_periods", 0) or 0),
            selection_mode="manual",
            random_n=cfg.portfolio.get("random_n", 8),
            custom_weights=custom,
            rank_kwargs=None,
            manual_funds=holdings,
            indices_list=cfg.portfolio.get("indices_list"),
            benchmarks=cfg.benchmarks,
            seed=getattr(cfg, "seed", 42),
            risk_window=cfg.vol_adjust.get("window"),
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
        # Attach per-period manager change log and execution stats
        res["manager_changes"] = events
        res["turnover"] = period_turnover
        res["transaction_cost"] = float(period_cost)
        # Append this period's result (was incorrectly outside loop causing only last period kept)
        results.append(cast(MultiPeriodPeriodResult, res))
    # Update complete for this period; next loop will use prev_weights

    return results


def run_from_config(cfg: Any) -> List[MultiPeriodPeriodResult]:
    """Load all inputs declared in ``cfg`` and execute :func:`run`."""

    prices = load_prices(cfg)
    membership_df = load_membership(cfg)
    benchmarks = load_benchmarks(cfg, prices)
    if not benchmarks.empty:
        inputs_meta = prices.attrs.setdefault("inputs", {})
        inputs_meta["benchmarks"] = benchmarks
    membership_arg = membership_df if not membership_df.empty else None
    return run(cfg, df=prices, membership=membership_arg)
