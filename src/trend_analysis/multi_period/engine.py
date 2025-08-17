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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Protocol, Any, cast

import pandas as pd

from ..config import Config
from ..data import load_csv
from ..pipeline import _run_analysis
from .scheduler import generate_periods
from ..weighting import (
    BaseWeighting,
    EqualWeight,
    ScorePropBayesian,
    AdaptiveBayesWeighting,
)
from ..core.rank_selection import ASCENDING_METRICS
from .replacer import Rebalancer
from ..rebalancing import apply_rebalancing_strategies


@dataclass
class Portfolio:
    """Minimal container for weight history."""

    history: Dict[str, pd.Series] = field(default_factory=dict)
    total_rebalance_costs: float = 0.0

    def rebalance(
        self, date: str | pd.Timestamp, weights: pd.DataFrame | pd.Series
    ) -> None:
        """Store weights for the given date."""
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
        self.history[str(pd.to_datetime(date).date())] = series.astype(float)


class SelectorProtocol(Protocol):
    """Minimal interface required of selectors."""

    def select(
        self, score_frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def run_schedule(
    score_frames: Mapping[str, pd.DataFrame],
    selector: SelectorProtocol,
    weighting: BaseWeighting,
    *,
    rank_column: str | None = None,
    rebalancer: "Rebalancer | None" = None,
    price_frames: Mapping[str, pd.Series] | None = None,
) -> Portfolio:
    """Apply selection and weighting across ``score_frames``.

    Parameters
    ----------
    score_frames : Mapping[str, pd.DataFrame]
        Score frames for each rebalancing period
    selector : SelectorProtocol
        Asset selection logic
    weighting : BaseWeighting
        Portfolio weighting scheme
    rank_column : str, optional
        Column to use for ranking, if different from selector default
    rebalancer : Rebalancer, optional
        Rebalancing logic for handling portfolio changes
    price_frames : Mapping[str, pd.Series], optional
        Price/return series for calculating portfolio returns between periods

    Returns
    -------
    Portfolio
        Portfolio object with rebalancing history
    """

    pf = Portfolio()
    prev_date: pd.Timestamp | None = None
    prev_weights: pd.Series | None = None
    total_rebalance_costs = 0.0
    col = (
        rank_column
        or getattr(selector, "rank_column", None)
        or getattr(selector, "column", None)
    )

    for date in sorted(score_frames):
        sf = score_frames[date]
        selected, _ = selector.select(sf)
        target_weights = weighting.weight(selected)

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

            total_rebalance_costs += cost
            weights = final_weights.to_frame("weight")
            prev_weights = final_weights
        else:
            weights = target_weights
            prev_weights = target_weights["weight"].astype(float)

        pf.rebalance(date, weights)

        if col and col in sf.columns:
            if prev_date is None:
                days = 0
            else:
                days = (pd.to_datetime(date) - prev_date).days
            s = sf.loc[weights.index, col]
            update_fn = getattr(weighting, "update", None)
            if callable(update_fn):
                try:
                    update_fn(s, days)
                except Exception:  # pragma: no cover - defensive
                    pass
        prev_date = pd.to_datetime(date)

        # Calculate portfolio returns between periods if price_frames provided
        if price_frames is not None:
            if prev_date is not None and date in price_frames:
                # Calculate portfolio return for this period using weights and price returns
                price_series = price_frames[date]
                portfolio_weights = (
                    weights["weight"] if isinstance(weights, pd.DataFrame) else weights
                )
                # Filter to common assets
                common_assets = portfolio_weights.index.intersection(price_series.index)
                if len(common_assets) > 0:
                    period_weights = portfolio_weights.loc[common_assets]
                    period_returns = price_series.loc[common_assets]
                    # Calculate weighted portfolio return (stored for reference but not used further in this minimal implementation)
                    portfolio_return = (period_weights * period_returns).sum()

    return pf


def run(cfg: Config, df: pd.DataFrame | None = None) -> List[Dict[str, object]]:
    """Run the multi‑period back‑test.

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

    # If policy is not threshold-hold, use the Phase‑1 style per-period runs.
    if str(cfg.portfolio.get("policy", "").lower()) != "threshold_hold":
        periods = generate_periods(cfg.model_dump())
        out_results: List[Dict[str, object]] = []
        for pt in periods:
            res = _run_analysis(
                df,
                pt.in_start[:7],
                pt.in_end[:7],
                pt.out_start[:7],
                pt.out_end[:7],
                cfg.vol_adjust.get("target_vol", 1.0),
                getattr(cfg, "run", {}).get("monthly_cost", 0.0),
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
            out_results.append(res)
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
        # Using the canonical metrics names as produced by single_period_run/_compute_metric_series
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
    from ..selector import RankSelector

    th_cfg = cast(dict[str, Any], cfg.portfolio.get("threshold_hold", {}))
    target_n = int(th_cfg.get("target_n", cfg.portfolio.get("random_n", 8)))
    seed_metric = cast(
        str,
        (cfg.portfolio.get("selector", {}) or {})
        .get("params", {})
        .get("rank_column", "Sharpe"),
    )
    selector = RankSelector(top_n=target_n, rank_column=seed_metric)

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
    results: List[Dict[str, object]] = []
    prev_weights: pd.Series | None = None
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

    def _apply_weight_bounds(w: pd.Series) -> pd.Series:
        if w.empty:
            return w
        w = w.clip(lower=0.0)
        # Step 1: cap at max
        capped = w.clip(upper=max_w_bound)
        # Step 2: floor at min
        floored = capped.copy()
        floored[floored < min_w_bound] = min_w_bound
        # Step 3: adjust to sum to 1.0 without violating bounds
        total = floored.sum()
        # Helper masks
        at_min = floored <= (min_w_bound + 1e-12)
        at_max = floored >= (max_w_bound - 1e-12)

        if total > 1.0 + 1e-12:
            # Need to reduce 'excess' from those above min (prefer free > min)
            excess = total - 1.0
            donors = floored[~at_min]
            if not donors.empty:
                # available to shave above min
                avail = (donors - min_w_bound).clip(lower=0.0)
                avail_sum = avail.sum()
                if avail_sum > 0:
                    cut = (avail / avail_sum) * excess
                    floored.loc[donors.index] = (donors - cut).clip(lower=min_w_bound)
        elif total < 1.0 - 1e-12:
            # Need to distribute deficit to those below max (prefer free < max)
            deficit = 1.0 - total
            receivers = floored[~at_max]
            if not receivers.empty:
                room = (max_w_bound - receivers).clip(lower=0.0)
                room_sum = room.sum()
                if room_sum > 0:
                    add = (room / room_sum) * deficit
                    floored.loc[receivers.index] = (receivers + add).clip(
                        upper=max_w_bound
                    )
        # One more clamp to be safe
        floored = floored.clip(lower=min_w_bound, upper=max_w_bound)
        # Final small normalise if tiny drift remains, distribute to those with room
        total = floored.sum()
        if abs(total - 1.0) > 1e-9:
            if total > 1.0:
                excess = total - 1.0
                donors = floored[~(floored <= min_w_bound + 1e-12)]
                if not donors.empty:
                    share = (donors - min_w_bound).clip(lower=0.0)
                    sh = share.sum()
                    if sh > 0:
                        floored.loc[donors.index] = (
                            donors - (share / sh) * excess
                        ).clip(lower=min_w_bound)
            else:
                deficit = 1.0 - total
                receivers = floored[~(floored >= max_w_bound - 1e-12)]
                if not receivers.empty:
                    room = (max_w_bound - receivers).clip(lower=0.0)
                    rm = room.sum()
                    if rm > 0:
                        floored.loc[receivers.index] = (
                            receivers + (room / rm) * deficit
                        ).clip(upper=max_w_bound)
        return floored

    for pt in periods:
        in_df, out_df, fund_cols, _rf_col = _valid_universe(
            df, pt.in_start[:7], pt.in_end[:7], pt.out_start[:7], pt.out_end[:7]
        )
        if not fund_cols:
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
            seed_weights = weighting.weight(selected)
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
            weights_df = weighting.weight(sf.loc[holdings])
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
            rebased = rebalancer.apply_triggers(cast(pd.Series, prev_weights), sf)
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
                    if (pd.notna(z) and z > z_entry_soft - 1e-12)
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
            # We still respect max_funds when seeding and when adding new funds elsewhere.
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
            weights_df = weighting.weight(sf.loc[holdings])
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
                        "detail": f"below min {min_w_bound:.2%} for {low_min_strikes_req} periods",
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
                weights_df = weighting.weight(sf.loc[holdings])
                nat_w = weights_df["weight"].astype(float)
                prev_weights = nat_w.copy()

        # Apply weight bounds and renormalise
        bounded_w = _apply_weight_bounds(prev_weights)
        prev_weights = bounded_w

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
            selection_mode="manual",
            random_n=cfg.portfolio.get("random_n", 8),
            custom_weights=custom,
            rank_kwargs=None,
            manual_funds=holdings,
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
        # Attach per-period manager change log
        res["manager_changes"] = events
        results.append(res)

    # Update complete for this period; next loop will use prev_weights

    return results
