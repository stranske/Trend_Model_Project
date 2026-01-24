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

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Protocol, cast

import numpy as np
import pandas as pd

from trend.diagnostics import DiagnosticResult

from .._typing import FloatArray
from ..constants import NUMERICAL_TOLERANCE_HIGH
from ..core.rank_selection import ASCENDING_METRICS
from ..data import load_csv
from ..diagnostics import PipelineResult, coerce_pipeline_result
from ..pipeline import (
    _build_trend_spec,
    _compute_stats,
    _invoke_analysis_with_diag,
    _resolve_risk_free_column,
    _resolve_target_vol,
)
from ..portfolio import apply_weight_policy
from ..rebalancing import CashPolicy, apply_rebalancing_strategies
from ..schedules import get_rebalance_dates
from ..timefreq import MONTHLY_DATE_FREQ
from ..universe import (
    MembershipTable,
    MembershipWindow,
    apply_membership_windows,
)
from ..util.frequency import detect_frequency
from ..util.missing import apply_missing_policy
from ..util.risk_free import resolve_risk_free_settings
from ..util.weights import normalize_weights
from ..weighting import (
    AdaptiveBayesWeighting,
    BaseWeighting,
    EqualWeight,
    ScorePropBayesian,
)
from ..weights.robust_config import weight_engine_params_from_robustness
from .loaders import load_benchmarks, load_membership, load_prices
from .replacer import Rebalancer
from .scheduler import generate_periods

# ``trend_analysis.typing`` does not exist in this project; keep the structural
# intent of ``MultiPeriodPeriodResult`` using a simple mapping alias so the
# engine remains importable without introducing a new module dependency.
MultiPeriodPeriodResult = Dict[str, Any]

SHIFT_DETECTION_MAX_STEPS_DEFAULT = 10
_DEFAULT_LOAD_CSV = load_csv

logger = logging.getLogger(__name__)


# Back-compat shim so legacy tests can patch ``engine._run_analysis`` while the
# default implementation continues to funnel through the diagnostics-aware
# pipeline entry point.
def _run_analysis(*args: Any, **kwargs: Any) -> PipelineResult:
    return _invoke_analysis_with_diag(*args, **kwargs)


def _call_pipeline_with_diag(
    *args: Any, **kwargs: Any
) -> DiagnosticResult[dict[str, Any] | None]:
    """Execute ``_run_analysis`` and normalise into a ``DiagnosticResult``.

    Tests and legacy callers monkeypatch ``_run_analysis`` to return raw dict
    payloads; keep accepting those for backwards compatibility.
    """

    payload, diag = coerce_pipeline_result(_run_analysis(*args, **kwargs))
    if payload is None:
        return DiagnosticResult(
            value=None,
            diagnostic=diag,
        )
    return DiagnosticResult(
        value=dict(payload),
        diagnostic=diag,
    )


def _coerce_previous_weights(
    weights: Mapping[str, float] | pd.Series | None,
) -> dict[str, float] | None:
    normalized = normalize_weights(weights)
    if not normalized:
        return None
    return normalized


def _get_missing_policy_settings(
    data_settings: Mapping[str, Any] | None,
) -> tuple[Any, Any]:
    """Return missing-data policy/limit configs with legacy fallbacks."""

    if not data_settings:
        return None, None
    missing_policy_cfg = data_settings.get("missing_policy")
    if missing_policy_cfg is None:
        missing_policy_cfg = data_settings.get("nan_policy")
    missing_limit_cfg = data_settings.get("missing_limit")
    if missing_limit_cfg is None:
        missing_limit_cfg = data_settings.get("nan_limit")
    return missing_policy_cfg, missing_limit_cfg


_resolve_risk_free_settings = resolve_risk_free_settings


class MissingPriceDataError(FileNotFoundError, ValueError):
    """Raised when CSV fallback loading fails in ``run``."""


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


def _as_weight_series(obj: pd.DataFrame | pd.Series | Mapping[str, float]) -> pd.Series:
    """Return a float weight series regardless of the original container."""

    if isinstance(obj, pd.DataFrame):
        if "weight" in obj.columns:
            series = obj["weight"]
        else:
            series = obj.iloc[:, 0]
    elif isinstance(obj, pd.Series):
        series = obj
    else:
        series = pd.Series(obj)
    return series.astype(float)


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
    # Apply the minimum weight constraint only to active positions.
    # Dropped/absent managers must be allowed to remain at 0.
    active = floored > NUMERICAL_TOLERANCE_HIGH
    floored[active & (floored < min_w_bound)] = min_w_bound

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


def _enforce_max_active_positions(
    weights: pd.Series,
    max_active_positions: int | None,
    *,
    protected: Iterable[str] | None = None,
) -> pd.Series:
    """Zero out all but the top ``max_active_positions`` weights."""

    if max_active_positions is None:
        return weights
    try:
        max_active = int(max_active_positions)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return weights
    if max_active <= 0 or weights.empty:
        return weights

    active = weights[weights.abs() > NUMERICAL_TOLERANCE_HIGH]
    if len(active) <= max_active:
        return weights

    protected_set = {str(m) for m in (protected or []) if m is not None}
    if protected_set:
        protected_active = [ix for ix in active.index if str(ix) in protected_set]
        unprotected_active = active.drop(protected_active, errors="ignore")
        slots = max(0, max_active - len(protected_active))
        keep = list(protected_active)
        if slots > 0 and not unprotected_active.empty:
            keep += list(
                unprotected_active.abs()
                .sort_values(ascending=False, kind="mergesort")
                .head(slots)
                .index
            )
    else:
        keep = (
            active.abs()
            .sort_values(ascending=False, kind="mergesort")
            .head(max_active)
            .index
        )
    trimmed = weights.copy()
    trimmed.loc[~trimmed.index.isin(keep)] = 0.0

    target_total = float(weights.sum())
    trimmed_total = float(trimmed.sum())
    if target_total > 0 and trimmed_total > 0:
        trimmed = trimmed * (target_total / trimmed_total)

    return trimmed


def _apply_turnover_penalty(
    target_w: pd.Series,
    last_aligned: pd.Series,
    lambda_tc: float,
    min_w_bound: float,
    max_w_bound: float,
) -> pd.Series:
    """Shrink trades toward the previous allocation to damp turnover.

    A convex combination between the previous weights and the proposed target
    reduces total turnover when ``lambda_tc`` is in ``(0, 1]``. Bounds are
    re-applied to keep the adjusted weights feasible.
    """

    if lambda_tc <= NUMERICAL_TOLERANCE_HIGH:
        return _apply_weight_bounds(target_w, min_w_bound, max_w_bound)

    shrunk = last_aligned + (target_w - last_aligned) * (1.0 - lambda_tc)
    return _apply_weight_bounds(shrunk, min_w_bound, max_w_bound)


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
    cash_policy: CashPolicy | None = None,
    weight_policy: Mapping[str, Any] | None = None,
    seed: int | None = None,
    target_n: int | None = None,
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
    cash_policy : CashPolicy, optional
        Policy controlling explicit cash rows and normalization for rebalancers.

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

    policy_cfg = dict(weight_policy or {})
    policy_mode = str(policy_cfg.get("mode", policy_cfg.get("policy", "drop"))).lower()
    min_assets_policy = int(policy_cfg.get("min_assets", 1) or 0)

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
        sf_for_selection = sf
        date_ts = pd.to_datetime(date)
        selected, _ = selector.select(sf_for_selection)
        target_weights = weighting.weight(selected, date_ts)
        signal_slice = sf[col] if col and col in sf.columns else None
        target_series = apply_weight_policy(
            _as_weight_series(target_weights),
            signal_slice,
            mode=policy_mode,
            min_assets=min_assets_policy,
            previous=prev_weights,
        )
        target_weights = target_series.to_frame("weight")

        # Apply legacy rebalancer (threshold-hold system) if configured
        if rebalancer is not None:
            if prev_weights is None:
                prev_weights = target_weights["weight"].astype(float)
            # For random mode, seed varies per period to get different selections
            period_seed = abs((seed or 42) + hash(str(date)) % 10000)
            prev_weights = rebalancer.apply_triggers(
                cast(pd.Series, prev_weights),
                sf,
                random_seed=period_seed,
                target_n=target_n or 10,
            )
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
                cash_policy=cash_policy,
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

    user_supplied_df = df is not None

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
            user_supplied_df = True
        else:
            raise ValueError("price_frames is empty - no data to process")

    data_settings = getattr(cfg, "data", {}) or {}
    missing_policy_cfg, missing_limit_cfg = _get_missing_policy_settings(data_settings)
    (
        risk_free_column_cfg,
        allow_risk_free_fallback_cfg,
    ) = _resolve_risk_free_settings(data_settings)
    regime_cfg = getattr(cfg, "regime", {}) or {}
    trend_spec = _build_trend_spec(cfg, getattr(cfg, "vol_adjust", {}) or {})

    if df is None:
        csv_path = data_settings.get("csv_path")
        if not csv_path:
            raise KeyError("cfg.data['csv_path'] must be provided")
        try:
            df = load_csv(
                csv_path,
                errors="raise",
                missing_policy=missing_policy_cfg,
                missing_limit=missing_limit_cfg,
            )
        except FileNotFoundError as exc:
            raise MissingPriceDataError(
                "multi_period.run requires either a pre-loaded DataFrame or "
                "price_frames; provide a valid 'csv_path' or in-memory frame"
            ) from exc
        if df is None:
            raise ValueError(f"Failed to load CSV data from '{csv_path}'")
        user_supplied_df = False

    if "Date" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Date' column")

    # Normalise Date column for consistent slicing downstream
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    # If the caller provides in-memory data (df/price_frames) and does not
    # request a missing-data policy, preserve the raw gaps (historical
    # behavior). Some selection policies intentionally treat NaNs as
    # disqualifying within specific windows.
    skip_missing_policy = (
        (price_frames is not None or user_supplied_df)
        and missing_policy_cfg is None
        and missing_limit_cfg is None
    )

    if skip_missing_policy:
        message = (
            "Missing-data policy skipped: user supplied returns/price_frames with "
            "no missing_policy or missing_limit configured; raw gaps may remain."
        )
        logger.info(message)
        policy_spec: str | Mapping[str, str] | None = None
        missing_policy_reason = "skipped_user_supplied_input"
        missing_policy_diagnostic = {
            "applied": False,
            "policy": None,
            "limit": None,
            "reason": "user_supplied_data_without_policy",
            "message": message,
        }
    else:
        policy_spec = missing_policy_cfg or "ffill"
        missing_policy_reason = "applied"
        # Legacy behavior: call apply_missing_policy in this module so tests can
        # monkeypatch it for observability. The canonical cleaning is performed
        # by the monthly normalisation helper later in this function.
        try:
            apply_missing_policy(
                df.set_index("Date"),
                policy=policy_spec,
                limit=missing_limit_cfg,
            )
        except Exception:  # pragma: no cover - best-effort only
            pass
        missing_policy_diagnostic = {
            "applied": True,
            "policy": policy_spec,
            "limit": missing_limit_cfg,
            "reason": "configured",
            "message": None,
        }

    if skip_missing_policy:
        # Preserve user-supplied date stamps when no missing policy was
        # configured. This matches historical behavior and keeps price_frames
        # tests stable.
        cleaned = df.set_index("Date").dropna(how="all")
        if cleaned.empty:
            raise ValueError("Missing-data policy removed all assets for analysis")
    else:
        # Canonical multi-period behavior: resample to month-end and then apply
        # the configured missing-data policy. Calling apply_missing_policy via
        # this module keeps tests able to monkeypatch for observability.
        work = df.copy()
        # Reset index if it has a name that matches a column to avoid ambiguity
        if work.index.name and work.index.name in work.columns:
            work = work.reset_index(drop=True)
        work.sort_values("Date", inplace=True)

        freq_summary = detect_frequency(work["Date"])

        value_cols = [c for c in work.columns if c != "Date"]
        if value_cols:
            numeric = work[value_cols].apply(pd.to_numeric, errors="coerce")
        else:
            numeric = work[value_cols]
        # Create index without assigning a name to avoid ambiguity
        numeric.index = pd.DatetimeIndex(work["Date"], name=None)

        if freq_summary.resampled:
            resampled = (1 + numeric).resample(MONTHLY_DATE_FREQ).prod(min_count=1) - 1
        else:
            resampled = numeric.resample(MONTHLY_DATE_FREQ).last()

        preserve_empty_periods = True
        if isinstance(policy_spec, str):
            preserve_empty_periods = policy_spec.strip().lower() != "drop"
        elif isinstance(policy_spec, Mapping):
            values = [str(v or "").lower() for v in policy_spec.values()]
            preserve_empty_periods = any(v and v != "drop" for v in values)

        if not preserve_empty_periods:
            resampled = resampled.dropna(how="all")
        resampled.index.name = "Date"

        try:
            filled, _missing_result = apply_missing_policy(
                resampled,
                policy=policy_spec,
                limit=missing_limit_cfg,
                enforce_completeness=True,
            )
        except TypeError:
            # Some unit tests monkeypatch apply_missing_policy with a simplified
            # signature; fall back to the legacy call shape.
            filled, _missing_result = apply_missing_policy(
                resampled,
                policy=policy_spec,
                limit=missing_limit_cfg,
            )
        cleaned = filled.dropna(how="all")

    if cleaned.empty:
        raise ValueError("Missing-data policy removed all assets for analysis")

    # ------------------------------------------------------------------
    # Inception-date masking
    #
    # Some vendor datasets encode pre-inception history as a flat line of
    # zeros instead of NaN. Even when a UI missing-policy fills NaNs with
    # zeros, we must ensure those pre-inception rows are treated as missing
    # so funds cannot enter the universe before they exist.
    #
    # Compute inception dates once per run and null-out values before the
    # inferred inception date.
    # ------------------------------------------------------------------
    try:
        from ..data import compute_inception_dates

        inception_raw = compute_inception_dates(cleaned)
        # Apply: before inception -> NaN; never-active columns -> NaN
        for col, inc in inception_raw.items():
            if col not in cleaned.columns:
                continue
            if inc is None:
                cleaned[col] = np.nan
                continue
            try:
                cleaned.loc[cleaned.index < inc, col] = np.nan
            except Exception:  # pragma: no cover - best effort
                pass

        # Drop columns that are entirely missing after masking.
        cleaned = cleaned.dropna(axis=1, how="all")
        # Persist for debugging/export consumers.
        cleaned.attrs = dict(cleaned.attrs)
        cleaned.attrs["inception_dates"] = {
            k: (v.strftime("%Y-%m-%d") if v is not None else None)
            for k, v in inception_raw.items()
        }
    except Exception:  # pragma: no cover - defensive
        pass

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
    missing_policy_metadata = {
        "missing_policy_applied": not skip_missing_policy,
        "missing_policy_reason": missing_policy_reason,
        "missing_policy_spec": policy_spec,
        "missing_policy_message": missing_policy_diagnostic.get("message"),
    }

    if str(cfg.portfolio.get("policy", "").lower()) != "threshold_hold":
        cfg_dump: dict[str, Any] = {}
        try:
            cfg_dump = cfg.model_dump()
        except Exception:  # pragma: no cover - defensive
            cfg_dump = {}

        periods = generate_periods(cfg_dump)
        if not periods:
            logger.warning(
                "generate_periods produced no periods; skipping multi-period run"
            )
            return []
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
        prev_weights_for_pipeline = _coerce_previous_weights(
            cfg.portfolio.get("previous_weights")
        )

        for pt in periods:
            analysis_res = _call_pipeline_with_diag(
                df,
                pt.in_start[:7],
                pt.in_end[:7],
                pt.out_start[:7],
                pt.out_end[:7],
                _resolve_target_vol(getattr(cfg, "vol_adjust", {})),
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
                previous_weights=prev_weights_for_pipeline,
                max_turnover=cfg.portfolio.get("max_turnover"),
                constraints=cfg.portfolio.get("constraints"),
                regime_cfg=regime_cfg,
                risk_free_column=risk_free_column_cfg,
                allow_risk_free_fallback=allow_risk_free_fallback_cfg,
                signal_spec=trend_spec,
            )
            payload = analysis_res.value
            diag = analysis_res.diagnostic
            if payload is None:
                if diag is not None:
                    logger.warning(
                        "Multi-period analysis skipped period %s/%s (%s): %s",
                        pt.in_start,
                        pt.out_start,
                        diag.reason_code,
                        diag.message,
                    )
                continue
            res_dict = dict(payload)
            res_dict.update(missing_policy_metadata)
            res_dict["period"] = (
                pt.in_start,
                pt.in_end,
                pt.out_start,
                pt.out_end,
            )
            res_dict["missing_policy_diagnostic"] = dict(missing_policy_diagnostic)
            risk_diag_payload = res_dict.get("risk_diagnostics")
            if isinstance(risk_diag_payload, dict):
                prev_weights_for_pipeline = _coerce_previous_weights(
                    risk_diag_payload.get("final_weights")
                )
            if prev_weights_for_pipeline is None:
                fund_weights = res_dict.get("fund_weights")
                if isinstance(fund_weights, Mapping):
                    prev_weights_for_pipeline = _coerce_previous_weights(fund_weights)

            # (Experimental) attach covariance diag using cache/incremental path for diagnostics.
            # Keeps existing outputs stable; adds optional "cov_diag" key.
            if enable_cache:
                from ..perf.cache import compute_cov_payload, incremental_cov_update

                in_start = pt.in_start[:7]
                in_end = pt.in_end[:7]
                # Recreate in-sample frame identical to _run_analysis slice
                date_col = "Date"
                sub = df.copy()
                sub[date_col] = pd.to_datetime(sub[date_col], utc=True).dt.tz_localize(
                    None
                )
                sub.sort_values(date_col, inplace=True)
                sdate = pd.to_datetime(f"{in_start}-01", utc=True).tz_localize(
                    None
                ) + pd.offsets.MonthEnd(0)
                edate = pd.to_datetime(f"{in_end}-01", utc=True).tz_localize(
                    None
                ) + pd.offsets.MonthEnd(0)
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
                res_dict["cov_diag"] = prev_cov_payload.cov.diagonal().tolist()
                if cov_cache_obj is not None:
                    # attach cache stats for observability (does not alter existing keys)
                    res_dict.setdefault("cache_stats", cov_cache_obj.stats())
            out_results.append(res_dict)
        return out_results

    # Threshold-hold path with Bayesian weighting
    periods = generate_periods(cfg.model_dump())

    # Multi-period engine normalizes data to month-end cadence.
    # Keep metric annualisation consistent with monthly returns.
    periods_per_year = 12

    # Minimum history is user-configurable (expressed in the same units as the
    # multi-period frequency). Enforce at universe build time so funds cannot be
    # scored/selected before they have sufficient return history.
    mp_cfg = cast(dict[str, Any], cfg.model_dump().get("multi_period", {}) or {})
    freq_raw = str(mp_cfg.get("frequency", "A") or "A").upper()
    if freq_raw in ("A", "YE", "ANNUAL", "ANNUALLY"):
        _months_per_period = 12
    elif freq_raw in ("Q", "QE", "QUARTERLY"):
        _months_per_period = 3
    else:
        _months_per_period = 1
    try:
        min_history_periods = int(
            mp_cfg.get("min_history_periods") or mp_cfg.get("in_sample_len") or 1
        )
    except (TypeError, ValueError):
        min_history_periods = int(mp_cfg.get("in_sample_len") or 1)
    min_history_periods = max(1, min_history_periods)
    # Never allow min-history to exceed the configured lookback.
    try:
        in_sample_len_periods = int(mp_cfg.get("in_sample_len") or min_history_periods)
    except (TypeError, ValueError):
        in_sample_len_periods = min_history_periods
    if in_sample_len_periods > 0:
        min_history_periods = min(min_history_periods, in_sample_len_periods)
    min_history_months = min_history_periods * _months_per_period

    indices_list = cast(list[str] | None, cfg.portfolio.get("indices_list")) or []
    # Benchmarks are inputs/diagnostics, not investable holdings.
    # Exclude their columns from the candidate universe even if they are
    # numeric and present in the returns frame.
    benchmarks_cfg = cast(object, getattr(cfg, "benchmarks", None))
    benchmark_cols: list[str] = []
    if benchmarks_cfg:
        # Config models define `benchmarks` as `dict[str, str]` (label -> column).
        # Some legacy configs use the inverse (column -> label). The engine
        # must exclude the *actual selected index/benchmark series* (i.e. the
        # column present in the returns frame), not the mapping label.
        col_lut = {str(c).strip().lower(): str(c) for c in df.columns}
        candidates: list[str] = []
        if isinstance(benchmarks_cfg, dict):
            candidates.extend([str(v) for v in benchmarks_cfg.values()])
            candidates.extend([str(k) for k in benchmarks_cfg.keys()])
        elif isinstance(benchmarks_cfg, (list, tuple, set)):
            candidates.extend([str(x) for x in benchmarks_cfg])

        seen: set[str] = set()
        for raw in candidates:
            key = str(raw).strip().lower()
            resolved = col_lut.get(key)
            if not resolved or resolved in seen:
                continue
            seen.add(resolved)
            benchmark_cols.append(resolved)
    resolved_rf_col, _resolver_fund_cols, resolved_rf_source = (
        _resolve_risk_free_column(
            df,
            date_col="Date",
            indices_list=indices_list,
            risk_free_column=risk_free_column_cfg,
            allow_risk_free_fallback=allow_risk_free_fallback_cfg,
        )
    )

    # Build a stable investable universe list.
    # If the risk-free column was *configured*, it should not be treated as an
    # investable fund. If it was selected via *fallback* heuristics (e.g.,
    # lowest-vol proxy), keep it investable; the fallback column may be a
    # genuine fund return series.
    numeric_cols_all = [c for c in df.select_dtypes("number").columns if c != "Date"]
    idx_set = {str(c) for c in indices_list}
    idx_set |= {str(c) for c in benchmark_cols}
    resolved_fund_candidates = [c for c in numeric_cols_all if c not in idx_set]
    if resolved_rf_source == "configured" and resolved_rf_col:
        resolved_fund_candidates = [
            c for c in resolved_fund_candidates if c != resolved_rf_col
        ]
    elif resolved_rf_source == "fallback" and resolved_rf_col:
        # Fallback RF selection can legitimately pick a true cash proxy (flat
        # zero-return series). Treat such near-constant columns as non-investable
        # so they don't enter selection/score frames.
        if resolved_rf_col in df.columns:
            try:
                vals = pd.to_numeric(df[resolved_rf_col], errors="coerce").dropna()
                if not vals.empty and float(vals.std(ddof=0)) <= 1e-12:
                    resolved_fund_candidates = [
                        c for c in resolved_fund_candidates if c != resolved_rf_col
                    ]
            except Exception:  # pragma: no cover - defensive
                pass

    # --- helpers --------------------------------------------------------
    def _parse_month(s: str) -> pd.Timestamp:
        return pd.to_datetime(f"{s}-01", utc=True).tz_localize(
            None
        ) + pd.offsets.MonthEnd(0)

    def _valid_universe(
        full: pd.DataFrame,
        in_start: str,
        in_end: str,
        out_start: str,
        out_end: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
        date_col = "Date"
        sub = full.copy()
        sub[date_col] = pd.to_datetime(sub[date_col], utc=True).dt.tz_localize(None)
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
        numeric_cols = [c for c in sub.select_dtypes("number").columns if c != date_col]
        idx_set = {str(c) for c in indices_list}
        idx_set |= {str(c) for c in benchmark_cols}
        fund_cols = [
            c
            for c in resolved_fund_candidates
            if c in numeric_cols and c not in idx_set
        ]

        if not fund_cols:
            return in_df, out_df, [], resolved_rf_col

        # Keep only funds with sufficient in-sample history (last N months
        # present) and complete out-of-sample data.
        # Note: risk metrics require at least 2 observations.
        required_in_months = max(2, int(min_history_months))
        required_in_months = min(required_in_months, int(len(in_df)))

        if required_in_months <= 0:
            return in_df, out_df, [], resolved_rf_col

        in_tail = in_df[fund_cols].tail(required_in_months)
        in_ok = ~in_tail.isna().any()
        out_ok = ~out_df[fund_cols].isna().any()
        fund_cols = [
            c
            for c in fund_cols
            if bool(in_ok.get(c, False)) and bool(out_ok.get(c, False))
        ]

        # Guardrail: exclude funds that are effectively inactive/flatlined at
        # ~0.0 (often vendor pre-inception encoding) even if they are non-missing.
        #
        # Note: Do NOT exclude all constant-return series (std == 0) because
        # downstream metrics code already sanitizes degenerate ratios; callers
        # and tests rely on constant-but-nonzero series remaining eligible.
        if fund_cols:
            try:
                in_abs_max = in_df[fund_cols].astype(float).abs().max()
                # Only use in-sample activity for the "inactive" heuristic.
                # Out-of-sample windows can legitimately contain 0.0 returns.
                active = in_abs_max > NUMERICAL_TOLERANCE_HIGH
                fund_cols = [c for c in fund_cols if bool(active.get(c, False))]
            except Exception:  # pragma: no cover - defensive
                pass
        return in_df, out_df, fund_cols, resolved_rf_col

    def _score_frame(
        in_df: pd.DataFrame,
        funds: list[str],
        *,
        risk_free_override: float | pd.Series | None,
        periods_per_year: int,
    ) -> pd.DataFrame:
        # Compute metrics frame for the in-sample window (vectorised)
        from ..core.rank_selection import RiskStatsConfig, _compute_metric_series

        # IMPORTANT: Do not silently compute RF-sensitive metrics (Sharpe,
        # Sortino, IR) against a 0.0 risk-free unless the caller explicitly
        # enabled an override (constant RF). When no override is provided,
        # fail fast so the caller can supply a risk-free series/column.
        stats_cfg = RiskStatsConfig(
            risk_free=float("nan"),
            periods_per_year=periods_per_year,
        )
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
        parts: list[pd.Series] = []
        for m in metrics:
            # Back-compat: some tests monkeypatch `_compute_metric_series` with a
            # simplified signature that doesn't accept `risk_free_override`.
            if risk_free_override is None:
                parts.append(_compute_metric_series(in_df[funds], m, stats_cfg))
                continue
            try:
                parts.append(
                    _compute_metric_series(
                        in_df[funds],
                        m,
                        stats_cfg,
                        risk_free_override=risk_free_override,
                    )
                )
            except TypeError:
                parts.append(_compute_metric_series(in_df[funds], m, stats_cfg))
        sf = pd.concat(parts, axis=1)
        sf.columns = [
            "CAGR",
            "Volatility",
            "Sharpe",
            "Sortino",
            "InformationRatio",
            "MaxDrawdown",
        ]
        sf = sf.astype(float)
        # Ensure degenerate series (e.g., constant returns -> 0 volatility)
        # do not wipe the investable universe by producing NaN/Inf metrics.
        sf = sf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return sf

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

    # Threshold-hold config can live either under portfolio.threshold_hold
    # (current) or at the portfolio root (legacy/UI snapshots).
    portfolio_cfg = cast(dict[str, Any], cfg.portfolio)
    th_cfg = dict(portfolio_cfg.get("threshold_hold", {}) or {})
    for key in (
        "metric",
        "z_exit_soft",
        "z_exit_hard",
        "z_entry_soft",
        "z_entry_hard",
        "soft_strikes",
        "entry_soft_strikes",
        "entry_eligible_strikes",
        "target_n",
        "blended_weights",
    ):
        if key not in th_cfg and key in portfolio_cfg:
            th_cfg[key] = portfolio_cfg[key]

    def _parse_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    z_exit_hard = _parse_optional_float(th_cfg.get("z_exit_hard"))

    target_n = int(
        th_cfg.get(
            "target_n", portfolio_cfg.get("target_n", cfg.portfolio.get("random_n", 8))
        )
    )
    seed_metric = cast(
        str,
        (cfg.portfolio.get("selector", {}) or {})
        .get("params", {})
        .get("rank_column", "Sharpe"),
    )
    selector = create_selector_by_name("rank", top_n=target_n, rank_column=seed_metric)

    # Extract inclusion approach from rank config (top_n, top_pct, threshold)
    rank_cfg = cast(dict[str, Any], portfolio_cfg.get("rank", {}) or {})
    inclusion_approach = str(rank_cfg.get("inclusion_approach", "top_n"))
    rank_pct = float(rank_cfg.get("pct", 0.10))  # For top_pct mode
    rank_threshold = float(rank_cfg.get("threshold", 1.0))  # For threshold mode
    rank_score_by = str(rank_cfg.get("score_by", "blended"))
    rank_blended_weights = rank_cfg.get("blended_weights")
    # Transform: zscore for threshold mode, raw for ranking modes
    rank_transform = str(rank_cfg.get("transform", "raw"))
    try:
        bottom_k = int(rank_cfg.get("bottom_k") or 0)
    except (TypeError, ValueError):
        bottom_k = 0
    bottom_k = max(0, bottom_k)

    # Portfolio constraints
    constraints = cast(dict[str, Any], cfg.portfolio.get("constraints", {}))
    mp_cfg = cast(dict[str, Any], cfg.model_dump().get("multi_period", {}) or {})
    max_funds_raw = constraints.get("max_funds")
    if max_funds_raw is None:
        max_funds_raw = mp_cfg.get("max_funds")
    max_funds = int(max_funds_raw) if max_funds_raw is not None else 10
    min_funds_raw = constraints.get("min_funds")
    if min_funds_raw is None:
        min_funds_raw = mp_cfg.get("min_funds")
    try:
        min_funds = int(min_funds_raw) if min_funds_raw is not None else 0
    except (TypeError, ValueError):
        min_funds = 0
    min_funds = max(0, min_funds)
    if max_funds > 0:
        min_funds = min(min_funds, max_funds)

    cooldown_periods_raw = portfolio_cfg.get("cooldown_periods")
    if cooldown_periods_raw is None:
        cooldown_periods_raw = portfolio_cfg.get("cooldown_months")
    if cooldown_periods_raw is None:
        cooldown_periods_raw = mp_cfg.get("cooldown_periods")
    if cooldown_periods_raw is None:
        cooldown_periods_raw = mp_cfg.get("cooldown_months")
    try:
        cooldown_periods = (
            int(cooldown_periods_raw) if cooldown_periods_raw is not None else 0
        )
    except (TypeError, ValueError):
        cooldown_periods = 0
    cooldown_periods = max(0, cooldown_periods)
    sticky_add_raw = portfolio_cfg.get("sticky_add_x")
    if sticky_add_raw is None:
        sticky_add_raw = portfolio_cfg.get("sticky_add_periods")
    if sticky_add_raw is None:
        sticky_add_raw = th_cfg.get("sticky_add_x")
    sticky_drop_raw = portfolio_cfg.get("sticky_drop_y")
    if sticky_drop_raw is None:
        sticky_drop_raw = portfolio_cfg.get("sticky_drop_periods")
    if sticky_drop_raw is None:
        sticky_drop_raw = th_cfg.get("sticky_drop_y")
    try:
        sticky_add_periods = max(1, int(sticky_add_raw or 1))
    except (TypeError, ValueError):
        sticky_add_periods = 1
    try:
        sticky_drop_periods = max(1, int(sticky_drop_raw or 1))
    except (TypeError, ValueError):
        sticky_drop_periods = 1
    min_w_bound = float(constraints.get("min_weight", 0.05))  # decimal
    max_w_bound = float(constraints.get("max_weight", 0.18))  # decimal
    raw_max_active = constraints.get("max_active_positions")
    if raw_max_active is None:
        raw_max_active = constraints.get("max_active")
    try:
        max_active_positions = (
            int(raw_max_active) if raw_max_active is not None else None
        )
    except (TypeError, ValueError):
        max_active_positions = None
    if max_active_positions is not None and max_active_positions <= 0:
        max_active_positions = None
    # consecutive below-min to replace
    # Prefer constraints for this rule (it’s a weight constraint),
    # but keep backward‑compat by falling back to threshold_hold if present.
    min_weight_strikes_raw = constraints.get("min_weight_strikes")
    if min_weight_strikes_raw is None:
        min_weight_strikes_raw = th_cfg.get("min_weight_strikes")
    # Low-weight replacement triggers after N consecutive periods where the
    # natural (pre-bounds) weight falls below min_weight. Default to 2 for
    # backward-compatible conservatism, but respect explicit configuration.
    low_min_strikes_req = 2
    if min_weight_strikes_raw is not None:
        try:
            low_min_strikes_req = max(1, int(min_weight_strikes_raw))
        except (TypeError, ValueError):
            low_min_strikes_req = 2

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

    # Risk-based weighting scheme (risk_parity, hrp) from weighting_scheme config.
    # This overrides the legacy `portfolio.weighting` dict config for primary weights.
    weighting_scheme = str(
        cfg.portfolio.get("weighting_scheme", "equal") or "equal"
    ).lower()
    if weighting_scheme == "robust":
        weighting_scheme = "robust_mv"
    use_risk_weighting = weighting_scheme in {
        "risk_parity",
        "hrp",
        "erc",
        "robust_mv",
        "robust_mean_variance",
        "robust_risk_parity",
    }
    risk_weight_engine: Any = None
    if use_risk_weighting:
        try:
            from ..plugins import create_weight_engine

            robustness_cfg = cfg.portfolio.get("robustness")
            if not isinstance(robustness_cfg, Mapping):
                robustness_cfg = getattr(cfg, "robustness", None)
            weight_engine_params = weight_engine_params_from_robustness(
                weighting_scheme,
                robustness_cfg if isinstance(robustness_cfg, Mapping) else None,
            )
            risk_weight_engine = create_weight_engine(
                weighting_scheme, **weight_engine_params
            )
        except Exception:  # pragma: no cover - best-effort only
            use_risk_weighting = False
            risk_weight_engine = None

    policy_cfg = cast(dict[str, Any], cfg.portfolio.get("weight_policy", {}))
    policy_mode = str(policy_cfg.get("mode", policy_cfg.get("policy", "drop"))).lower()
    min_assets_policy = int(policy_cfg.get("min_assets", 1) or 0)

    # Check if random selection mode - if so, disable z-score based entry/exit
    selection_mode = cfg.portfolio.get("selection_mode", "rank")
    is_random_mode = selection_mode == "random"

    # Buy-and-hold mode: hold initial selection, only replace when funds disappear
    is_buy_and_hold = selection_mode == "buy_and_hold"
    buy_hold_cfg = cast(dict[str, Any], portfolio_cfg.get("buy_and_hold", {}) or {})
    buy_hold_initial = str(buy_hold_cfg.get("initial_method", "top_n"))
    buy_hold_n = int(buy_hold_cfg.get("n", target_n))
    buy_hold_pct = float(buy_hold_cfg.get("pct", rank_pct))
    buy_hold_threshold = float(buy_hold_cfg.get("threshold", rank_threshold))

    rebalancer = Rebalancer(cfg.model_dump())

    # --- main loop ------------------------------------------------------
    # Pre-index returns once for intra-period rebalance snapshots.
    df_indexed = df.copy()
    df_indexed["Date"] = pd.to_datetime(df_indexed["Date"], utc=True).dt.tz_localize(
        None
    )
    df_indexed.sort_values("Date", inplace=True)
    df_indexed = df_indexed.set_index("Date")

    results: List[MultiPeriodPeriodResult] = []
    prev_weights: pd.Series | None = None
    prev_final_weights: pd.Series | None = None
    # Transaction cost and turnover-cap controls (Issue #429)
    tc_bps = float(cfg.portfolio.get("transaction_cost_bps", 0.0))
    slippage_bps = float(cfg.portfolio.get("slippage_bps", 0.0))
    max_turnover_cap = float(cfg.portfolio.get("max_turnover", 1.0))
    lambda_tc = float(cfg.portfolio.get("lambda_tc", 0.0) or 0.0)
    low_weight_strikes: dict[str, int] = {}
    cooldown_book: dict[str, int] = {}
    add_streaks: dict[str, int] = {}
    drop_streaks: dict[str, int] = {}
    min_tenure_raw = cfg.portfolio.get("min_tenure_n")
    if min_tenure_raw is None:
        min_tenure_raw = cfg.portfolio.get("min_tenure_periods")
    if min_tenure_raw is None:
        min_tenure_raw = th_cfg.get("min_tenure_n")
    if min_tenure_raw is None:
        min_tenure_raw = th_cfg.get("min_tenure_periods")
    try:
        min_tenure_n = int(min_tenure_raw) if min_tenure_raw is not None else 0
    except (TypeError, ValueError):
        min_tenure_n = 0
    if min_tenure_n < 0:
        min_tenure_n = 0
    holdings_tenure: dict[str, int] = {}

    def _firm(name: str) -> str:
        return str(name).split()[0] if isinstance(name, str) and name else str(name)

    def _eligible_sticky_add(manager: str) -> bool:
        if sticky_add_periods <= 1:
            return True
        return int(add_streaks.get(manager, 0)) >= sticky_add_periods

    def _min_tenure_protected(
        holdings: Iterable[str], score_frame: pd.DataFrame
    ) -> set[str]:
        if min_tenure_n <= 0:
            return set()
        protected: set[str] = set()
        for mgr in holdings:
            mgr_str = str(mgr)
            if int(holdings_tenure.get(mgr_str, 0)) < min_tenure_n:
                protected.add(mgr_str)
        return protected

    def _min_tenure_guard(holdings: Iterable[str]) -> set[str]:
        if min_tenure_n <= 0:
            return set()
        protected: set[str] = set()
        for mgr in holdings:
            mgr_str = str(mgr)
            if int(holdings_tenure.get(mgr_str, 0)) < min_tenure_n:
                protected.add(mgr_str)
        return protected

    def _reapply_min_tenure_guard(
        holdings: list[str],
        *,
        before_reb: set[str],
        score_frame: pd.DataFrame,
        blocked: set[str],
        events: list[dict[str, object]],
        logged: set[str],
        stage: str,
    ) -> list[str]:
        if min_tenure_n <= 0 or not blocked:
            return holdings
        existing = {str(h) for h in holdings}
        for mgr in sorted(blocked):
            if mgr in existing or mgr not in before_reb:
                continue
            if mgr not in score_frame.index:
                continue
            holdings.append(mgr)
            existing.add(mgr)
            if mgr in logged:
                continue
            events.append(
                {
                    "action": "skipped",
                    "manager": mgr,
                    "firm": _firm(mgr),
                    "reason": "min_tenure",
                    "detail": (
                        f"tenure={int(holdings_tenure.get(mgr, 0))}/"
                        f"{min_tenure_n}; stage={stage}"
                    ),
                }
            )
            logged.add(mgr)
        return holdings

    def _start_cooldown(exited: Iterable[str]) -> None:
        if cooldown_periods <= 0:
            return
        for mgr in exited:
            mgr_str = str(mgr)
            if not mgr_str:
                continue
            cooldown_book[mgr_str] = max(
                int(cooldown_book.get(mgr_str, 0)),
                int(cooldown_periods) + 1,
            )

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

    def _dedupe_one_per_firm_with_events(
        sf: pd.DataFrame,
        holdings: list[str],
        metric: str,
        events: list[dict[str, object]],
        protected: set[str] | None = None,
    ) -> list[str]:
        before = [str(h) for h in holdings]
        if protected:
            protected_set = {str(h) for h in protected}
            protected_firms = {_firm(h) for h in protected_set}
            col = metric if metric in sf.columns else "Sharpe"
            ascending = col in ASCENDING_METRICS
            tmp = sf.loc[
                [h for h in before if h in sf.index],
                [col, "zscore" if "zscore" in sf.columns else col],
            ].copy()
            if "zscore" not in tmp.columns:
                tmp["zscore"] = 0.0
            tmp["_firm"] = [_firm(ix) for ix in tmp.index]
            tmp.sort_values([col, "zscore"], ascending=[ascending, False], inplace=True)
            seen: set[str] = set()
            after: list[str] = []
            for fund in tmp.index:
                firm = _firm(fund)
                if fund in protected_set:
                    after.append(fund)
                    continue
                if firm in seen or firm in protected_firms:
                    continue
                seen.add(firm)
                after.append(fund)
        else:
            after = _dedupe_one_per_firm(sf, before, metric)
        removed = sorted(set(before) - set(after))
        for mgr in removed:
            events.append(
                {
                    "action": "dropped",
                    "manager": mgr,
                    "firm": _firm(mgr),
                    "reason": "one_per_firm",
                    "detail": "removed duplicate firm holding",
                }
            )
        return after

    def _rank_scores_for_bottom_k(
        score_frame: pd.DataFrame,
    ) -> tuple[pd.Series, bool]:
        if score_frame.empty:
            return pd.Series(dtype=float), False
        use_seed_metric = inclusion_approach == "top_n" and not is_buy_and_hold
        score_by = seed_metric if use_seed_metric else rank_score_by
        blended_weights = None if use_seed_metric else rank_blended_weights
        transform = "raw" if use_seed_metric else rank_transform

        if score_by == "blended" and blended_weights:
            total_w = sum(blended_weights.values())
            if total_w > 0:
                norm_w = {k: v / total_w for k, v in blended_weights.items()}
            else:
                norm_w = {"Sharpe": 1.0}
            combo = pd.Series(0.0, index=score_frame.index, dtype=float)
            for m, w in norm_w.items():
                if m in score_frame.columns:
                    col_series = score_frame[m].astype(float)
                    mu = float(col_series.mean())
                    sigma = float(col_series.std(ddof=0))
                    z = (
                        (col_series - mu) / sigma
                        if sigma > 0
                        else pd.Series(0.0, index=col_series.index)
                    )
                    if m in ASCENDING_METRICS:
                        z = -z
                    combo += w * z
            scores = combo
        else:
            score_col = score_by if score_by in score_frame.columns else "Sharpe"
            if score_col not in score_frame.columns:
                if score_frame.columns.empty:
                    return pd.Series(dtype=float), False
                score_col = str(score_frame.columns[0])
            scores = score_frame[score_col].astype(float)

        if transform == "zscore":
            mu, sigma = scores.mean(), scores.std(ddof=0)
            if sigma > 0:
                scores = (scores - mu) / sigma
            else:
                scores = pd.Series(0.0, index=scores.index)

        ascending = False
        if score_by in ASCENDING_METRICS and transform != "zscore":
            ascending = True

        return scores, ascending

    def _filter_entry_frame(score_frame: pd.DataFrame) -> pd.DataFrame:
        filtered = score_frame
        if bottom_k <= 0 or filtered.empty:
            return filtered
        scores, ascending = _rank_scores_for_bottom_k(filtered)
        if scores.empty:
            return filtered
        ordered = scores.sort_values(ascending=ascending).index
        if bottom_k >= len(ordered):
            return filtered.iloc[0:0]
        keep = ordered[:-bottom_k]
        return filtered.loc[keep]

    def _filter_entry_candidates(
        candidates: list[str], score_frame: pd.DataFrame
    ) -> list[str]:
        if not candidates:
            return candidates
        eligible_frame = _filter_entry_frame(score_frame)
        if eligible_frame.empty:
            return []
        eligible = {str(ix) for ix in eligible_frame.index}
        return [str(ix) for ix in candidates if str(ix) in eligible]

    def _hard_exit_forced(
        holdings: Iterable[str], score_frame: pd.DataFrame
    ) -> set[str]:
        if z_exit_hard is None or "zscore" not in score_frame.columns:
            return set()
        z = pd.to_numeric(score_frame["zscore"], errors="coerce")
        forced: set[str] = set()
        for mgr in holdings:
            mgr_str = str(mgr)
            if mgr_str not in z.index:
                continue
            val = z.get(mgr_str)
            if pd.notna(val) and float(val) <= z_exit_hard + NUMERICAL_TOLERANCE_HIGH:
                forced.add(mgr_str)
        return forced

    def _apply_policy_to_weights(
        weights_obj: pd.DataFrame | pd.Series | Mapping[str, float],
        signals: pd.Series | None,
    ) -> pd.Series:
        cleaned = apply_weight_policy(
            _as_weight_series(weights_obj),
            signals,
            mode=policy_mode,
            min_assets=min_assets_policy,
            previous=prev_weights,
        )
        if not cleaned.empty:
            return cleaned
        base = _as_weight_series(weights_obj)
        base = base.replace([np.inf, -np.inf], np.nan).dropna()
        if base.empty:
            return cleaned
        logger.warning(
            "Weight policy removed all holdings; falling back to equal weights for %d assets.",
            len(base),
        )
        return pd.Series(1.0 / len(base), index=base.index, dtype=float)

    def _enforce_min_funds(
        sf: pd.DataFrame,
        holdings: list[str],
        *,
        before_reb: set[str] | None,
        cooldowns: Mapping[str, int] | None,
        desired_min: int,
        events: list[dict[str, object]],
    ) -> list[str]:
        if desired_min <= 0:
            return holdings
        if len(holdings) >= desired_min:
            return holdings
        seen_firms = {_firm(str(h)) for h in holdings}
        # Prefer fresh additions (exclude managers previously held this period
        # to avoid silently undoing intended drops).
        excluded = set(before_reb or set())
        in_cooldown = set(cooldowns or {})
        candidates = [
            str(c)
            for c in sf.index
            if str(c) not in holdings
            and str(c) not in excluded
            and str(c) not in in_cooldown
            and _eligible_sticky_add(str(c))
        ]
        candidates = _filter_entry_candidates(candidates, sf)
        if not candidates:
            candidates = [
                str(c)
                for c in sf.index
                if str(c) not in holdings
                and str(c) not in in_cooldown
                and _eligible_sticky_add(str(c))
            ]
            candidates = _filter_entry_candidates(candidates, sf)
        if not candidates:
            return holdings
        # In random mode, shuffle candidates randomly instead of ranking by zscore.
        # Use a period-specific seed so different periods get different shuffles.
        if is_random_mode:
            period_seed_base = getattr(cfg, "seed", 42) or 42
            period_seed = abs(period_seed_base + hash(str(pt)) % 10000)
            rng = np.random.default_rng(period_seed)
            rng.shuffle(candidates)
            ranked = candidates
        else:
            ranked = sf.loc[candidates].sort_values("zscore", ascending=False).index
        for c in ranked:
            if len(holdings) >= desired_min:
                break
            mgr = str(c)
            firm = _firm(mgr)
            if firm in seen_firms:
                continue
            holdings.append(mgr)
            seen_firms.add(firm)
            events.append(
                {
                    "action": "added",
                    "manager": mgr,
                    "firm": firm,
                    "reason": "min_funds",
                    "detail": f"enforced minimum holdings={desired_min}",
                }
            )
        return holdings

    def _compute_weights(
        sf: pd.DataFrame,
        holdings: list[str],
        date: pd.Timestamp,
        returns_window: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute portfolio weights using risk engine or legacy weighting.

        When ``use_risk_weighting`` is True and ``risk_weight_engine`` is set,
        compute weights using the risk-based engine (risk_parity, hrp, etc.).
        Otherwise fall back to the legacy ``weighting`` object.
        """
        if use_risk_weighting and risk_weight_engine is not None:
            # Risk-based weighting requires returns data
            try:
                if returns_window is not None and not returns_window.empty:
                    # Only include holdings that have data in returns_window
                    # to avoid NaN columns that would get 0 weight
                    holdings_in_window = [
                        h for h in holdings if h in returns_window.columns
                    ]
                    subset = returns_window[holdings_in_window].dropna(
                        axis=1, how="all"
                    )
                else:
                    # Fall back to score frame data if no returns window provided
                    subset = sf.loc[holdings]
                if subset.empty or len(subset.columns) < 2:
                    # Not enough data for covariance - fall back to equal weights
                    return weighting.weight(sf.loc[holdings], date)
                cov = subset.cov()
                if cov.isnull().all().all() or cov.empty:
                    return weighting.weight(sf.loc[holdings], date)
                w_series = risk_weight_engine.weight(cov)
                # Ensure all holdings have weights (fill missing with zero)
                # Note: holdings not in returns_window will get 0 weight here
                w_series = w_series.reindex(holdings).fillna(0.0)
                total = w_series.sum()
                if total > 1e-9:
                    w_series = w_series / total
                return pd.DataFrame({"weight": w_series}, index=w_series.index)
            except Exception:  # pragma: no cover - best-effort fallback
                return weighting.weight(sf.loc[holdings], date)
        else:
            return weighting.weight(sf.loc[holdings], date)

    for pt in periods:
        period_ts = pd.to_datetime(pt.out_end)

        if cooldown_periods > 0 and cooldown_book:
            for key in list(cooldown_book.keys()):
                remaining = int(cooldown_book.get(key, 0)) - 1
                if remaining <= 0:
                    cooldown_book.pop(key, None)
                else:
                    cooldown_book[key] = remaining

        in_df, out_df, fund_cols, _rf_col = _valid_universe(
            df,
            pt.in_start[:7],
            pt.in_end[:7],
            pt.out_start[:7],
            pt.out_end[:7],
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
                        "turnover": 0.0,
                        "transaction_cost": 0.0,
                        "missing_policy_diagnostic": dict(missing_policy_diagnostic),
                    },
                )
            )
            continue
        metrics_cfg = cast(dict[str, Any], getattr(cfg, "metrics", {}) or {})
        rf_override_enabled = bool(metrics_cfg.get("rf_override_enabled", False))
        rf_rate_annual = float(metrics_cfg.get("rf_rate_annual", 0.0) or 0.0)
        # Convert annualised RF to a per-period return.
        # Use geometric conversion so 2% annual becomes ~0.165% monthly.
        rf_rate_periodic = (1.0 + rf_rate_annual) ** (
            1.0 / float(periods_per_year)
        ) - 1.0

        # UI semantics:
        # - override disabled: use the selected risk-free column series (if available)
        # - override enabled: ignore the RF column and use a constant RF rate
        rf_override: float | pd.Series | None
        if rf_override_enabled:
            rf_override = float(rf_rate_periodic)
        elif resolved_rf_col and resolved_rf_col in in_df.columns:
            rf_override = in_df[resolved_rf_col]
        else:
            rf_override = None

        if (not rf_override_enabled) and rf_override is None:
            raise ValueError(
                "Risk-free override is disabled but no risk-free series is available. "
                "Select a risk-free column (data.risk_free_column) or enable metrics.rf_override_enabled."
            )

        sf = _score_frame(
            in_df,
            fund_cols,
            risk_free_override=rf_override,
            periods_per_year=int(periods_per_year),
        )
        metric = cast(str, th_cfg.get("metric", "Sharpe"))

        if metric == "blended" and "blended" not in sf.columns:
            from ..core.rank_selection import RiskStatsConfig, blended_score

            blended_weights = cast(
                dict[str, float] | None,
                th_cfg.get("blended_weights")
                or (cfg.portfolio.get("rank", {}) or {}).get("blended_weights"),
            )
            if blended_weights:
                stats_cfg = RiskStatsConfig(
                    risk_free=0.0,
                    periods_per_year=int(periods_per_year),
                )
                sf["blended"] = blended_score(
                    in_df[fund_cols],
                    blended_weights,
                    stats_cfg,
                    risk_free_override=rf_override,
                ).astype(float)
            else:
                # Fall back safely if weights are missing.
                sf["blended"] = sf["Sharpe"].astype(float)
        sf = _ensure_zscore(sf, metric)

        # Determine holdings
        # Track manager changes with reasons for this period
        events: list[dict[str, object]] = []
        forced_exits: set[str] = set()
        min_tenure_blocked: set[str] = set()
        min_tenure_logged: set[str] = set()

        if prev_weights is None:
            # For random mode, do fresh random selection from available universe
            # rather than using the z-score based selector
            if is_random_mode:
                eligible_sf = _filter_entry_frame(sf)
                available = list(eligible_sf.index)
                if not available:
                    # No funds available - skip to placeholder logic below
                    holdings = []
                else:
                    # Seed varies per period to get different selections
                    period_seed = abs(getattr(cfg, "seed", 42) or 42) + abs(
                        hash(str(pt)) % 10000
                    )
                    rng = np.random.default_rng(period_seed)
                    n_select = max(1, min(target_n, len(available)))
                    holdings = list(rng.choice(available, size=n_select, replace=False))
                    # Enforce one-per-firm constraint
                    holdings = _dedupe_one_per_firm_with_events(
                        sf, holdings, metric, events
                    )
                    # If dedupe reduced us, refill with random selection
                    if len(holdings) < n_select:
                        candidates = [c for c in available if c not in holdings]
                        if candidates:
                            rng = np.random.default_rng(period_seed + 1)
                            rng.shuffle(candidates)
                            seen_firms = {_firm(h) for h in holdings}
                            for c in candidates:
                                if len(holdings) >= n_select:
                                    break
                                firm = _firm(str(c))
                                if firm in seen_firms:
                                    continue
                                holdings.append(str(c))
                                seen_firms.add(firm)
            elif is_buy_and_hold:
                # Buy-and-hold mode: select initial holdings using configured method
                # Holdings will be held until data disappears (fund ceases to exist)
                eligible_sf = _filter_entry_frame(sf)
                available = list(eligible_sf.index)
                if not available:
                    holdings = []
                elif buy_hold_initial == "random":
                    # Random initial selection
                    period_seed = abs(
                        (getattr(cfg, "seed", 42) or 42) + hash(str(pt)) % 10000
                    )
                    rng = np.random.default_rng(period_seed)
                    n_select = max(1, min(buy_hold_n, len(available)))
                    holdings = list(rng.choice(available, size=n_select, replace=False))
                    holdings = _dedupe_one_per_firm_with_events(
                        sf, holdings, metric, events
                    )
                    # Refill if dedupe reduced holdings
                    if len(holdings) < n_select:
                        candidates = [c for c in available if c not in holdings]
                        if candidates:
                            rng = np.random.default_rng(period_seed + 1)
                            rng.shuffle(candidates)
                            seen_firms = {_firm(h) for h in holdings}
                            for c in candidates:
                                if len(holdings) >= n_select:
                                    break
                                firm = _firm(str(c))
                                if firm in seen_firms:
                                    continue
                                holdings.append(str(c))
                                seen_firms.add(firm)
                else:
                    # Rank-based initial selection (top_n, top_pct, threshold)
                    # Compute scores for ranking
                    eligible_sf = _filter_entry_frame(sf)
                    if rank_score_by == "blended" and rank_blended_weights:
                        total_w = sum(rank_blended_weights.values())
                        if total_w > 0:
                            norm_w = {
                                k: v / total_w for k, v in rank_blended_weights.items()
                            }
                        else:
                            norm_w = {"Sharpe": 1.0}
                        combo = pd.Series(0.0, index=eligible_sf.index, dtype=float)
                        for m, w in norm_w.items():
                            if m in eligible_sf.columns:
                                col_series = eligible_sf[m].astype(float)
                                mu = float(col_series.mean())
                                sigma = float(col_series.std(ddof=0))
                                z = (
                                    (col_series - mu) / sigma
                                    if sigma > 0
                                    else pd.Series(0.0, index=col_series.index)
                                )
                                if m in ASCENDING_METRICS:
                                    z = -z
                                combo += w * z
                        scores = combo
                    else:
                        score_col = (
                            rank_score_by
                            if rank_score_by in eligible_sf.columns
                            else "Sharpe"
                        )
                        scores = eligible_sf[score_col].astype(float)

                    # Apply zscore transform if threshold mode
                    if buy_hold_initial == "threshold":
                        mu, sigma = scores.mean(), scores.std(ddof=0)
                        if sigma > 0:
                            scores = (scores - mu) / sigma
                        else:
                            scores = pd.Series(0.0, index=scores.index)

                    ascending = False
                    if (
                        rank_score_by in ASCENDING_METRICS
                        and buy_hold_initial != "threshold"
                    ):
                        ascending = True

                    sorted_scores = scores.sort_values(ascending=ascending)
                    all_candidates = list(sorted_scores.index)

                    if buy_hold_initial == "top_n":
                        holdings = all_candidates[:buy_hold_n]
                    elif buy_hold_initial == "top_pct":
                        k = max(1, int(round(len(all_candidates) * buy_hold_pct)))
                        holdings = all_candidates[:k]
                    elif buy_hold_initial == "threshold":
                        mask = (
                            sorted_scores >= buy_hold_threshold
                            if not ascending
                            else sorted_scores <= buy_hold_threshold
                        )
                        holdings = list(sorted_scores[mask].index)
                        # Cap at target size
                        if len(holdings) > buy_hold_n:
                            holdings = holdings[:buy_hold_n]
                    else:
                        holdings = all_candidates[:buy_hold_n]

                    # Enforce one-per-firm constraint
                    holdings = _dedupe_one_per_firm_with_events(
                        sf, holdings, metric, events
                    )
                    # Historical weighting call
                    if holdings:
                        try:
                            weighting.weight(sf.loc[holdings], period_ts)
                        except Exception:  # pragma: no cover
                            pass
            else:
                # Seed via ranking - supports top_n, top_pct, threshold
                # For top_n mode, use the selector to maintain backward compatibility
                # For top_pct and threshold modes, compute directly from score frame
                if inclusion_approach == "top_n":
                    sf_for_selection = _filter_entry_frame(sf)
                    # Use selector for backward compatibility with tests
                    selected, _ = selector.select(sf_for_selection)
                    # Historical behavior: weight() is invoked during seeding even though
                    # holdings may be refined by constraints afterwards. Some weighting
                    # engines (and unit tests) model state across calls.
                    try:
                        weighting.weight(selected, period_ts)
                    except Exception:  # pragma: no cover - best-effort only
                        pass

                    rank_col = getattr(selector, "rank_column", None)
                    if (
                        isinstance(rank_col, str)
                        and rank_col
                        and isinstance(selected, pd.DataFrame)
                        and rank_col in selected.columns
                    ):
                        ascending = rank_col in ASCENDING_METRICS
                        ordered = selected.sort_values(
                            rank_col, ascending=ascending
                        ).index
                        holdings = [str(x) for x in ordered.tolist()]
                    else:
                        holdings = [str(x) for x in selected.index.tolist()]
                else:
                    # For top_pct and threshold modes, compute directly from score frame
                    def _score_and_select_holdings(
                        score_frame: pd.DataFrame,
                        rank_score_by: str,
                        rank_blended_weights: Mapping[str, float] | None,
                        rank_transform: str,
                        inclusion_approach: str,
                        rank_pct: float,
                        rank_threshold: float,
                        target_n: int,
                    ) -> list[str]:
                        # Compute blended score if configured, else use single metric
                        if rank_score_by == "blended" and rank_blended_weights:
                            # Normalize weights
                            total_w = sum(rank_blended_weights.values())
                            if total_w > 0:
                                norm_w = {
                                    k: v / total_w
                                    for k, v in rank_blended_weights.items()
                                }
                            else:
                                norm_w = {"Sharpe": 1.0}
                            # Compute blended score from the score frame
                            combo = pd.Series(0.0, index=score_frame.index, dtype=float)
                            for m, w in norm_w.items():
                                if m in score_frame.columns:
                                    col_series = score_frame[m].astype(float)
                                    # Z-score normalize
                                    mu = float(col_series.mean())
                                    sigma = float(col_series.std(ddof=0))
                                    z = (
                                        (col_series - mu) / sigma
                                        if sigma > 0
                                        else pd.Series(0.0, index=col_series.index)
                                    )
                                    # Invert for ascending metrics (smaller is better)
                                    if m in ASCENDING_METRICS:
                                        z = -z
                                    combo += w * z
                            scores = combo
                        else:
                            # Single metric
                            score_col = (
                                rank_score_by
                                if rank_score_by in score_frame.columns
                                else "Sharpe"
                            )
                            scores = score_frame[score_col].astype(float)

                        # Apply transform if zscore
                        if rank_transform == "zscore":
                            mu, sigma = scores.mean(), scores.std(ddof=0)
                            if sigma > 0:
                                scores = (scores - mu) / sigma
                            else:
                                scores = pd.Series(0.0, index=scores.index)

                        # Determine sort order
                        ascending = False  # Higher score is better for blended/zscore
                        if (
                            rank_score_by in ASCENDING_METRICS
                            and rank_transform != "zscore"
                        ):
                            ascending = True

                        # Sort scores
                        sorted_scores = scores.sort_values(ascending=ascending)
                        all_candidates = list(sorted_scores.index)

                        # Apply inclusion approach
                        if inclusion_approach == "top_pct":
                            # Select top X% of funds
                            k = max(1, int(round(len(all_candidates) * rank_pct)))
                            holdings = all_candidates[:k]
                        elif inclusion_approach == "threshold":
                            # Select funds above threshold
                            if rank_transform == "zscore":
                                # Use z-score threshold
                                mask = (
                                    sorted_scores >= rank_threshold
                                    if not ascending
                                    else sorted_scores <= rank_threshold
                                )
                            else:
                                mask = (
                                    sorted_scores >= rank_threshold
                                    if not ascending
                                    else sorted_scores <= rank_threshold
                                )
                            holdings = list(sorted_scores[mask].index)
                        else:
                            # Fallback to taking target_n funds
                            holdings = all_candidates[:target_n]

                        return holdings

                    sf_for_selection = _filter_entry_frame(sf)
                    holdings = _score_and_select_holdings(
                        sf_for_selection,
                        rank_score_by=rank_score_by,
                        rank_blended_weights=rank_blended_weights,
                        rank_transform=rank_transform,
                        inclusion_approach=inclusion_approach,
                        rank_pct=rank_pct,
                        rank_threshold=rank_threshold,
                        target_n=target_n,
                    )
                    # Historical behavior: weight() is invoked during seeding
                    if holdings:
                        try:
                            weighting.weight(sf.loc[holdings], period_ts)
                        except Exception:  # pragma: no cover - best-effort only
                            pass

                # Cap to the requested target size before applying other constraints.
                if len(holdings) > target_n:
                    holdings = holdings[:target_n]
                # Enforce one-per-firm on seed
                holdings = _dedupe_one_per_firm_with_events(
                    sf, holdings, metric, events
                )
                desired_seed = min(max_funds, target_n)
                # If we're still above the desired size, trim by zscore (best-first).
                if len(holdings) > desired_seed:
                    zsorted = (
                        sf.loc[holdings].sort_values("zscore", ascending=False).index
                    )
                    holdings = list(zsorted[:desired_seed])

                # If dedupe reduced us below the target size, fill from the remaining
                # score-frame candidates, best-first by zscore.
                # Note: For top_pct mode, we respect the percentage selection and don't
                # automatically fill up to target_n, as the intent is to select exactly
                # that percentage of the universe.
                if len(holdings) < desired_seed and inclusion_approach != "top_pct":
                    candidates = [c for c in sf.index if c not in holdings]
                    candidates = _filter_entry_candidates(
                        [str(c) for c in candidates], sf
                    )
                    add_from = (
                        sf.loc[candidates].sort_values("zscore", ascending=False).index
                    )
                    seen_firms = {_firm(h) for h in holdings}
                    for f in add_from:
                        if len(holdings) >= desired_seed:
                            break
                        firm = _firm(str(f))
                        if firm in seen_firms:
                            continue
                        holdings.append(str(f))
                        seen_firms.add(firm)

                # If the risk-free column was inferred via fallback, prefer not to
                # seed it as an investable holding when we can swap it out without
                # reducing the portfolio below the desired size.
                if (
                    resolved_rf_source == "fallback"
                    and resolved_rf_col
                    and resolved_rf_col in holdings
                ):
                    candidates = [c for c in sf.index if c not in holdings]
                    candidates = _filter_entry_candidates(
                        [str(c) for c in candidates], sf
                    )
                    add_from = (
                        sf.loc[candidates].sort_values("zscore", ascending=False).index
                        if candidates
                        else []
                    )
                    seen_firms = {_firm(h) for h in holdings if h != resolved_rf_col}
                    replacement: str | None = None
                    for f in add_from:
                        firm = _firm(str(f))
                        if firm in seen_firms:
                            continue
                        replacement = str(f)
                        break
                    if replacement is not None:
                        holdings = [h for h in holdings if h != resolved_rf_col]
                        holdings.append(replacement)

            # Compute weights using risk engine or fallback to legacy weighting
            # Use holdings (not fund_cols) so newly added funds get proper weight data
            holdings_with_data = [h for h in holdings if h in in_df.columns]
            weights_df = _compute_weights(
                sf, holdings, period_ts, in_df.reindex(columns=holdings_with_data)
            )
            raw_weight_series = _as_weight_series(weights_df)
            signal_slice = sf.loc[holdings, metric] if metric in sf.columns else None
            weight_series = _apply_policy_to_weights(weights_df, signal_slice)
            weights_df = weight_series.to_frame("weight")
            prev_weights = weight_series.astype(float)
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
            # Use rebalancer to update holdings; then apply Bayesian weights.
            # IMPORTANT: cap holdings to the configured portfolio size so the
            # trigger logic cannot accumulate an unbounded number of positions.
            before_reb = set(prev_weights.index)
            hard_exit_forced = _hard_exit_forced(before_reb, sf)
            min_tenure_protected = _min_tenure_protected(before_reb, sf)
            protected_holdings = set(min_tenure_protected or set())

            # Buy-and-hold mode: keep existing holdings, only replace disappeared funds
            if is_buy_and_hold:
                # Check raw out-of-sample data, not the filtered score frame.
                # A fund should only exit when its data actually disappears,
                # not when it fails the completeness filter for new selections.
                current_holdings = []
                exited_funds_set: set[str] = set()
                for h in prev_weights.index:
                    h_str = str(h)
                    # Fund still has data if it has ANY non-null values in out-of-sample
                    if h_str in out_df.columns and out_df[h_str].notna().any():
                        current_holdings.append(h_str)
                    else:
                        exited_funds_set.add(h_str)
                exited_funds = exited_funds_set

                # Log forced exits (data disappeared)
                for mgr in exited_funds:
                    events.append(
                        {
                            "action": "dropped",
                            "manager": mgr,
                            "firm": _firm(mgr),
                            "reason": "data_ceased",
                            "detail": "fund data no longer available",
                        }
                    )
                    forced_exits.add(mgr)
                _start_cooldown(exited_funds)

                # Replace exited funds using the same initial selection method
                n_needed = buy_hold_n - len(current_holdings)
                if n_needed > 0:
                    available = [str(c) for c in sf.index if str(c) not in current_holdings]
                    # Only consider funds that still have out-of-sample data;
                    # prevents re-adding funds whose data has disappeared.
                    if isinstance(out_df, pd.DataFrame) and not out_df.empty:
                        available = [
                            c for c in available if c in out_df.columns and out_df[c].notna().any()
                        ]
                    if cooldown_periods > 0 and cooldown_book:
                        available = [c for c in available if c not in cooldown_book]
                    available = _filter_entry_candidates(available, sf)
                    seen_firms = {_firm(h) for h in current_holdings}

                    if buy_hold_initial == "random":
                        # Random replacement
                        period_seed = abs(getattr(cfg, "seed", 42) or 42) + abs(
                            hash(str(pt)) % 10000
                        )
                        rng = np.random.default_rng(period_seed)
                        rng.shuffle(available)
                        replacements: list[str] = []
                        for c in available:
                            if len(replacements) >= n_needed:
                                break
                            firm = _firm(c)
                            if firm in seen_firms:
                                continue
                            replacements.append(c)
                            seen_firms.add(firm)
                    else:
                        # Rank-based replacement (top_n, top_pct, threshold)
                        if rank_score_by == "blended" and rank_blended_weights:
                            total_w = sum(rank_blended_weights.values())
                            if total_w > 0:
                                norm_w = {
                                    k: v / total_w
                                    for k, v in rank_blended_weights.items()
                                }
                            else:
                                norm_w = {"Sharpe": 1.0}
                            combo = pd.Series(0.0, index=sf.index, dtype=float)
                            for m, w in norm_w.items():
                                if m in sf.columns:
                                    col_series = sf[m].astype(float)
                                    mu = float(col_series.mean())
                                    sigma = float(col_series.std(ddof=0))
                                    z = (
                                        (col_series - mu) / sigma
                                        if sigma > 0
                                        else pd.Series(0.0, index=col_series.index)
                                    )
                                    if m in ASCENDING_METRICS:
                                        z = -z
                                    combo += w * z
                            scores = combo
                        else:
                            score_col = (
                                rank_score_by
                                if rank_score_by in sf.columns
                                else "Sharpe"
                            )
                            scores = sf[score_col].astype(float)

                        # Apply zscore transform if threshold mode
                        if buy_hold_initial == "threshold":
                            mu, sigma = scores.mean(), scores.std(ddof=0)
                            if sigma > 0:
                                scores = (scores - mu) / sigma
                            else:
                                scores = pd.Series(0.0, index=scores.index)

                        ascending = False
                        if (
                            rank_score_by in ASCENDING_METRICS
                            and buy_hold_initial != "threshold"
                        ):
                            ascending = True

                        # Sort scores and filter to available candidates
                        sorted_scores = scores.loc[available].sort_values(
                            ascending=ascending
                        )

                        # Select replacements respecting threshold if applicable
                        if buy_hold_initial == "threshold":
                            mask = (
                                sorted_scores >= buy_hold_threshold
                                if not ascending
                                else sorted_scores <= buy_hold_threshold
                            )
                            candidate_list = list(sorted_scores[mask].index)
                        else:
                            candidate_list = list(sorted_scores.index)

                        replacements = []
                        for c in candidate_list:
                            if len(replacements) >= n_needed:
                                break
                            firm = _firm(str(c))
                            if firm in seen_firms:
                                continue
                            replacements.append(str(c))
                            seen_firms.add(firm)

                    # Log replacements
                    for mgr in replacements:
                        events.append(
                            {
                                "action": "added",
                                "manager": mgr,
                                "firm": _firm(mgr),
                                "reason": "replacement",
                                "detail": f"replaced ceased fund via {buy_hold_initial}",
                            }
                        )
                    current_holdings.extend(replacements)

                # Set proposed holdings (skip normal rebalancer)
                proposed_holdings = current_holdings
            else:
                # For random mode, seed varies per period to get different selections
                # each period. This is essential to avoid survivorship bias - we select
                # from the available universe at each point in time, not funds we know
                # will survive.
                period_seed = abs(getattr(cfg, "seed", 42) or 42) + abs(
                    hash(str(pt)) % 10000
                )
                rebased = rebalancer.apply_triggers(
                    prev_weights.astype(float),
                    sf,
                    random_seed=period_seed,
                    target_n=target_n,
                )

                # Restrict to funds available in this period's score-frame.
                proposed_holdings = [
                    str(h) for h in list(rebased.index) if h in sf.index
                ]

            if hard_exit_forced:
                proposed_holdings = [
                    m for m in proposed_holdings if m not in hard_exit_forced
                ]

            raw_proposed_holdings = [str(h) for h in proposed_holdings]

            # Enforce cooldown: funds recently removed cannot be re-added.
            # Existing holdings are not blocked (cooldown only applies to re-entry).
            if cooldown_periods > 0 and cooldown_book:
                filtered: list[str] = []
                for mgr in proposed_holdings:
                    if mgr in before_reb:
                        filtered.append(mgr)
                        continue
                    if mgr in cooldown_book:
                        events.append(
                            {
                                "action": "skipped",
                                "manager": mgr,
                                "firm": _firm(mgr),
                                "reason": "cooldown",
                                "detail": f"remaining={int(cooldown_book[mgr])}",
                            }
                        )
                        continue
                    filtered.append(mgr)
                proposed_holdings = filtered

            if sticky_add_periods > 1 or sticky_drop_periods > 1:
                before_list = [str(h) for h in before_reb]
                before_set = set(before_list)
                proposed_list = [str(h) for h in proposed_holdings]
                raw_proposed_set = set(raw_proposed_holdings)
                proposed_set = set(proposed_list)
                sf_set = {str(h) for h in sf.index}
                forced_drop = {h for h in before_set if h not in sf_set}

                for mgr in list(add_streaks.keys()):
                    if mgr not in sf_set:
                        add_streaks.pop(mgr, None)
                for mgr in list(drop_streaks.keys()):
                    if mgr not in before_set:
                        drop_streaks.pop(mgr, None)

                for h in sf_set:
                    if h in before_set:
                        add_streaks[h] = 0
                    elif h in raw_proposed_set:
                        add_streaks[h] = int(add_streaks.get(h, 0)) + 1
                    else:
                        add_streaks[h] = 0

                for h in before_set:
                    if h in forced_drop:
                        drop_streaks.pop(h, None)
                        continue
                    if h not in raw_proposed_set:
                        drop_streaks[h] = int(drop_streaks.get(h, 0)) + 1
                    else:
                        drop_streaks[h] = 0

                if sticky_add_periods > 1:
                    blocked_adds: list[str] = []
                    for h in sorted(proposed_set - before_set):
                        if add_streaks.get(h, 0) < sticky_add_periods:
                            proposed_set.discard(h)
                            blocked_adds.append(h)
                    for mgr in blocked_adds:
                        events.append(
                            {
                                "action": "skipped",
                                "manager": mgr,
                                "firm": _firm(mgr),
                                "reason": "sticky_add",
                                "detail": (
                                    f"streak={add_streaks.get(mgr, 0)}/"
                                    f"{sticky_add_periods}"
                                ),
                            }
                        )

                if sticky_drop_periods > 1:
                    blocked_drops: list[str] = []
                    for h in sorted(before_set - proposed_set):
                        if h in forced_drop:
                            continue
                        if drop_streaks.get(h, 0) < sticky_drop_periods:
                            proposed_set.add(h)
                            blocked_drops.append(h)
                    for mgr in blocked_drops:
                        events.append(
                            {
                                "action": "skipped",
                                "manager": mgr,
                                "firm": _firm(mgr),
                                "reason": "sticky_drop",
                                "detail": (
                                    f"streak={drop_streaks.get(mgr, 0)}/"
                                    f"{sticky_drop_periods}"
                                ),
                            }
                        )

                final_holdings: list[str] = []
                for h in proposed_list:
                    if h in proposed_set and h in sf_set and h not in final_holdings:
                        final_holdings.append(h)
                for h in before_list:
                    if h in proposed_set and h in sf_set and h not in final_holdings:
                        final_holdings.append(h)
                proposed_holdings = final_holdings

            if min_tenure_protected:
                min_tenure_blocked = set(min_tenure_protected)
                blocked_drops = [
                    mgr
                    for mgr in [str(h) for h in before_reb]
                    if mgr not in proposed_holdings
                    and mgr in min_tenure_protected
                    and mgr not in hard_exit_forced
                ]
                for mgr in blocked_drops:
                    if mgr in proposed_holdings:
                        continue
                    proposed_holdings.append(mgr)
                    events.append(
                        {
                            "action": "skipped",
                            "manager": mgr,
                            "firm": _firm(mgr),
                            "reason": "min_tenure",
                            "detail": (
                                f"tenure={int(holdings_tenure.get(mgr, 0))}/"
                                f"{min_tenure_n}"
                            ),
                        }
                    )
                    min_tenure_logged.add(mgr)

            # Log attempted adds prior to firm/cap constraints. This preserves
            # the user's intent signal (e.g., a z_entry candidate) even if the
            # candidate is later blocked by one-per-firm.
            z_entry_soft = float(th_cfg.get("z_entry_soft", 1.0))
            attempted_adds = sorted(set(proposed_holdings) - set(before_reb))
            for mgr in attempted_adds:
                try:
                    val = pd.to_numeric(sf.loc[mgr, "zscore"], errors="coerce")
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
                        "manager": mgr,
                        "firm": _firm(mgr),
                        "reason": reason,
                        "detail": f"zscore={z:.3f}",
                    }
                )

            # Preserve the rebalancer's natural intent for portfolio size.
            # After the initial seed, the design is to allow the portfolio to
            # drift within [min_funds, max_funds] based on entry/exit triggers,
            # rather than forcibly refilling to ``target_n`` every period.
            desired_size_natural = len(proposed_holdings)

            # Enforce one-per-firm.
            proposed_holdings = _dedupe_one_per_firm_with_events(
                sf, proposed_holdings, metric, events, protected=protected_holdings
            )

            # If one-per-firm removed holdings, try to refill back to the
            # rebalancer's intended portfolio size (capped by max_funds), using
            # the highest-zscore *new* candidates. Exclude managers already held
            # before the rebalance so we don't silently undo an intended drop.
            # In random mode, use random selection instead of zscore ranking.
            desired_size = desired_size_natural
            if max_funds > 0:
                desired_size = min(desired_size, max_funds)
            if desired_size > 0 and len(proposed_holdings) < desired_size:
                seen_firms = {_firm(str(h)) for h in proposed_holdings}
                candidates = [
                    str(c)
                    for c in sf.index
                    if str(c) not in proposed_holdings
                    and str(c) not in before_reb
                    and str(c) not in cooldown_book
                    and _eligible_sticky_add(str(c))
                ]
                candidates = _filter_entry_candidates(candidates, sf)
                if candidates:
                    if is_random_mode:
                        period_seed = abs(
                            (getattr(cfg, "seed", 42) or 42) + hash(str(pt)) % 10000
                        )
                        rng = np.random.default_rng(period_seed)
                        rng.shuffle(candidates)
                        ranked = candidates
                    else:
                        ranked = (
                            sf.loc[candidates]
                            .sort_values("zscore", ascending=False)
                            .index
                        )
                    for c in ranked:
                        if len(proposed_holdings) >= desired_size:
                            break
                        mgr = str(c)
                        firm = _firm(mgr)
                        if firm in seen_firms:
                            continue
                        proposed_holdings.append(mgr)
                        seen_firms.add(firm)

            # Enforce max holdings size (cap by max_funds only; do not force
            # back to target_n after seeding).
            desired_size = desired_size_natural
            if max_funds > 0:
                desired_size = min(desired_size, max_funds)
            pruned_existing: set[str] = set()
            if desired_size > 0:
                current_set = {str(x) for x in before_reb}
                kept_existing = [
                    str(h) for h in proposed_holdings if str(h) in current_set
                ]
                new_candidates = [
                    str(h) for h in proposed_holdings if str(h) not in current_set
                ]

                def _zscore(mgr: str) -> float:
                    try:
                        val = pd.to_numeric(sf.loc[mgr, "zscore"], errors="coerce")
                        return float(val) if pd.notna(val) else float("nan")
                    except Exception:
                        return float("nan")

                def _random_key(mgr: str) -> float:
                    # For random mode: use a seeded random value for stable ordering
                    # Make the seed period-specific by incorporating the score frame.
                    # Use abs() to ensure non-negative seed (hash can be negative).
                    base_seed = getattr(cfg, "seed", 42) or 42
                    combined_seed = (base_seed, id(sf), mgr)
                    safe_seed = abs(hash(combined_seed))
                    rng = np.random.default_rng(safe_seed)
                    return rng.random()

                # Safety: if current holdings already exceed desired_size (should
                # not happen), prune incumbents by weakest zscore.
                # In random mode, prune randomly instead.
                if len(kept_existing) > desired_size:
                    protected_existing = [
                        mgr for mgr in kept_existing if mgr in protected_holdings
                    ]
                    unprotected_existing = [
                        mgr for mgr in kept_existing if mgr not in protected_holdings
                    ]
                    if is_random_mode:
                        unprotected_sorted = sorted(
                            unprotected_existing, key=_random_key
                        )
                    else:
                        unprotected_sorted = sorted(
                            unprotected_existing, key=_zscore, reverse=True
                        )
                    if protected_existing:
                        slots = max(0, desired_size - len(protected_existing))
                        if slots > 0:
                            keep = protected_existing + unprotected_sorted[:slots]
                        else:
                            keep = protected_existing
                    else:
                        keep = unprotected_sorted[:desired_size]
                    pruned_existing = set(kept_existing) - set(keep)
                    kept_existing = keep
                    for mgr in sorted(pruned_existing):
                        events.append(
                            {
                                "action": "dropped",
                                "manager": mgr,
                                "firm": _firm(mgr),
                                "reason": "cap_max_funds",
                                "detail": f"pruned to {desired_size} holdings",
                            }
                        )

                slots = max(0, desired_size - len(kept_existing))
                admitted: list[str] = []
                if slots > 0 and new_candidates:
                    firms = {_firm(m) for m in kept_existing}
                    if is_random_mode:
                        ranked_new = sorted(new_candidates, key=_random_key)
                    else:
                        ranked_new = sorted(new_candidates, key=_zscore, reverse=True)
                    for mgr in ranked_new:
                        if len(admitted) >= slots:
                            break
                        if _firm(mgr) in firms:
                            continue
                        admitted.append(mgr)
                        firms.add(_firm(mgr))

                rejected = sorted(set(new_candidates) - set(admitted))
                for mgr in rejected:
                    events.append(
                        {
                            "action": "skipped",
                            "manager": mgr,
                            "firm": _firm(mgr),
                            "reason": "cap_max_funds",
                            "detail": f"candidate pruned to {desired_size} holdings",
                        }
                    )

                proposed_holdings = kept_existing + admitted

            if len(proposed_holdings) == 0:  # guard: reseed if empty
                selected, _ = selector.select(_filter_entry_frame(sf))
                proposed_holdings = [str(x) for x in selected.index.tolist()]
                if cooldown_periods > 0 and cooldown_book:
                    filtered = [
                        mgr for mgr in proposed_holdings if mgr not in cooldown_book
                    ]
                    if filtered:
                        for mgr in proposed_holdings:
                            if mgr in cooldown_book:
                                events.append(
                                    {
                                        "action": "skipped",
                                        "manager": mgr,
                                        "firm": _firm(mgr),
                                        "reason": "cooldown",
                                        "detail": "reseed blocked by cooldown",
                                    }
                                )
                        proposed_holdings = filtered
                proposed_holdings = _dedupe_one_per_firm_with_events(
                    sf, proposed_holdings, metric, events
                )
                for f in proposed_holdings:
                    events.append(
                        {
                            "action": "added",
                            "manager": f,
                            "firm": _firm(f),
                            "reason": "reseat",
                            "detail": "reseeding empty portfolio",
                        }
                    )

            # Enforce transaction budget (max add+drop changes per period).
            # This applies to manager add/drop events, not intra-period
            # reweight-only operations.
            max_changes_raw = cfg.portfolio.get("turnover_budget_max_changes")
            try:
                max_changes = int(max_changes_raw) if max_changes_raw is not None else 0
            except (TypeError, ValueError):
                max_changes = 0

            desired_holdings = [str(h) for h in proposed_holdings]
            if max_changes > 0:
                desired_set = set(desired_holdings)
                current_set = set(before_reb)
                desired_add = sorted(desired_set - current_set)
                desired_drop = sorted(current_set - desired_set)
                desired_total = len(desired_add) + len(desired_drop)

                if desired_total > max_changes:

                    def _zscore(mgr: str) -> float:
                        try:
                            val = pd.to_numeric(sf.loc[mgr, "zscore"], errors="coerce")
                            return float(val) if pd.notna(val) else float("nan")
                        except Exception:
                            return float("nan")

                    # Exits/drops are always honoured; turnover budget limits
                    # *additions* (replacements) rather than keeping a fund
                    # that triggered an exit.
                    remaining_set = set(current_set) - set(desired_drop)

                    remaining_budget = max(0, max_changes - len(desired_drop))
                    if len(desired_drop) > max_changes:
                        events.append(
                            {
                                "action": "note",
                                "manager": "",
                                "firm": "",
                                "reason": "turnover_budget",
                                "detail": (
                                    f"drops={len(desired_drop)} exceed max_changes={max_changes}; "
                                    "honouring exits"
                                ),
                            }
                        )

                    add_ranked = sorted(desired_add, key=_zscore, reverse=True)
                    remaining_firms = {_firm(m) for m in remaining_set}
                    selected_adds: list[str] = []
                    for mgr in add_ranked:
                        if len(selected_adds) >= remaining_budget:
                            break
                        if mgr in cooldown_book:
                            continue
                        if _firm(mgr) in remaining_firms:
                            continue
                        selected_adds.append(mgr)
                        remaining_firms.add(_firm(mgr))

                    skipped_adds = sorted(set(desired_add) - set(selected_adds))
                    for mgr in skipped_adds:
                        events.append(
                            {
                                "action": "skipped",
                                "manager": mgr,
                                "firm": _firm(mgr),
                                "reason": "turnover_budget",
                                "detail": f"max_changes={max_changes}",
                            }
                        )

                    desired_holdings = sorted(remaining_set | set(selected_adds))

            holdings = desired_holdings
            after_reb = set(holdings)

            # Enforce minimum holdings after transaction-budget enforcement.
            # This is allowed to exceed the add/drop budget by policy.
            if min_funds > 0 and len(holdings) < min_funds:
                holdings = _enforce_min_funds(
                    sf,
                    holdings,
                    before_reb=before_reb,
                    cooldowns=cooldown_book,
                    desired_min=min_funds,
                    events=events,
                )
                after_reb = set(holdings)

            if min_tenure_blocked:
                holdings = _reapply_min_tenure_guard(
                    holdings,
                    before_reb=before_reb,
                    score_frame=sf,
                    blocked=min_tenure_blocked,
                    events=events,
                    logged=min_tenure_logged,
                    stage="post_budget",
                )
                after_reb = set(holdings)

            # Log drops/adds due to rebalancer z-triggers (post-cap holdings).
            z_exit_soft = float(th_cfg.get("z_exit_soft", -1.0))
            z_entry_soft = float(th_cfg.get("z_entry_soft", 1.0))
            dropped_reb = before_reb - after_reb
            _start_cooldown(dropped_reb)
            for f in sorted(dropped_reb):
                if str(f) in pruned_existing:
                    continue
                try:
                    val = (
                        pd.to_numeric(sf.loc[f, "zscore"], errors="coerce")
                        if f in sf.index
                        else pd.NA
                    )
                    z = float(val) if pd.notna(val) else float("nan")
                except Exception:
                    z = float("nan")
                if pd.notna(z) and z_exit_hard is not None and z <= z_exit_hard:
                    reason = "z_exit_hard"
                    forced_exits.add(str(f))
                elif pd.notna(z) and z < z_exit_soft:
                    reason = "z_exit"
                    forced_exits.add(str(f))
                else:
                    reason = "rebalance"
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
            already_added = {
                str(ev.get("manager"))
                for ev in events
                if ev.get("action") == "added" and ev.get("manager") is not None
            }
            for f in sorted(added_reb):
                if str(f) in already_added:
                    continue
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

            # Compute weights using risk engine or fallback to legacy weighting
            # Use holdings (not fund_cols) so newly added funds get proper weight data
            holdings_with_data = [h for h in holdings if h in in_df.columns]
            weights_df = _compute_weights(
                sf, holdings, period_ts, in_df.reindex(columns=holdings_with_data)
            )
            raw_weight_series = _as_weight_series(weights_df)
            signal_slice = sf.loc[holdings, metric] if metric in sf.columns else None
            weight_series = _apply_policy_to_weights(weights_df, signal_slice)
            weights_df = weight_series.to_frame("weight")
            prev_weights = weight_series.astype(float)

        # Natural weights (pre-bounds) for strikes on min threshold
        nat_w = raw_weight_series.reindex(prev_weights.index).fillna(0.0)

        # Low-weight replacement rule: track consecutive periods where a fund's
        # natural (pre-bounds) weight falls below min_weight, but only enforce
        # replacements starting from the second period (i.e., once a realised
        # prior allocation exists).
        min_tenure_blocked_low_weight = (
            _min_tenure_protected(prev_weights.index, sf) if min_tenure_n > 0 else set()
        )
        to_remove: list[str] = []
        for f, wv in nat_w.items():
            f_str = str(f)
            if float(wv) < min_w_bound:
                low_weight_strikes[f_str] = int(low_weight_strikes.get(f_str, 0)) + 1
            else:
                low_weight_strikes[f_str] = 0
            if (
                prev_final_weights is not None
                and int(low_weight_strikes.get(f_str, 0)) >= low_min_strikes_req
            ):
                if f_str in min_tenure_blocked_low_weight:
                    continue
                to_remove.append(f_str)
        if to_remove:
            size_before_low_weight = len(holdings)
            _start_cooldown(to_remove)
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
            # Replace removed holdings up to the prior portfolio size (capped by
            # max_funds). Do not force-fill to target_n.
            desired_after_low_weight = size_before_low_weight
            if max_funds > 0:
                desired_after_low_weight = min(desired_after_low_weight, max_funds)
            need = max(0, desired_after_low_weight - len(holdings))
            if need > 0:
                candidates = [
                    c
                    for c in sf.index
                    if c not in holdings and _eligible_sticky_add(str(c))
                ]
                if cooldown_periods > 0 and cooldown_book:
                    candidates = [c for c in candidates if str(c) not in cooldown_book]
                add_from = (
                    sf.loc[candidates]
                    .sort_values("zscore", ascending=False)
                    .index.tolist()
                )
                for f in add_from:
                    if len(holdings) >= desired_after_low_weight:
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
                # Compute weights using risk engine or fallback to legacy weighting
                # Use holdings (not fund_cols) so newly added funds get proper weight data
                holdings_with_data = [h for h in holdings if h in in_df.columns]
                weights_df = _compute_weights(
                    sf, holdings, period_ts, in_df.reindex(columns=holdings_with_data)
                )
                raw_weight_series = _as_weight_series(weights_df)
                signal_slice = (
                    sf.loc[holdings, metric] if metric in sf.columns else None
                )
                weight_series = _apply_policy_to_weights(weights_df, signal_slice)
                weights_df = weight_series.to_frame("weight")
                prev_weights = weight_series.astype(float)
                nat_w = raw_weight_series.reindex(prev_weights.index).fillna(0.0)

        # Enforce minimum holdings after low-weight removals/replacements.
        if min_funds > 0 and len(holdings) < min_funds:
            holdings = _enforce_min_funds(
                sf,
                holdings,
                before_reb=(
                    set(prev_weights.index) if prev_weights is not None else None
                ),
                cooldowns=cooldown_book,
                desired_min=min_funds,
                events=events,
            )
            if holdings and prev_weights is not None:
                # Compute weights using risk engine or fallback to legacy weighting
                # Use holdings (not fund_cols) so newly added funds get proper weight data
                holdings_with_data = [h for h in holdings if h in in_df.columns]
                weights_df = _compute_weights(
                    sf, holdings, period_ts, in_df.reindex(columns=holdings_with_data)
                )
                raw_weight_series = _as_weight_series(weights_df)
                signal_slice = (
                    sf.loc[holdings, metric] if metric in sf.columns else None
                )
                weight_series = _apply_policy_to_weights(weights_df, signal_slice)
                weights_df = weight_series.to_frame("weight")
                prev_weights = weight_series.astype(float)
                nat_w = raw_weight_series.reindex(prev_weights.index).fillna(0.0)

        # Apply weight bounds and renormalise
        bounded_w = _apply_weight_bounds(prev_weights, min_w_bound, max_w_bound)
        min_tenure_guard = _min_tenure_guard(bounded_w.index)

        # Preserve the selected holdings set for the pipeline manual selection.
        # Subsequent turnover alignment (union with previous holdings) may
        # introduce additional indices that should not automatically become
        # part of the manual fund list.
        manual_holdings = [str(x) for x in bounded_w.index.tolist()]

        # Enforce optional turnover cap by scaling trades towards target
        target_w = bounded_w.copy()
        if prev_final_weights is None:
            last_aligned = pd.Series(0.0, index=target_w.index)
        else:
            union_ix = prev_final_weights.index.union(target_w.index)
            last_aligned = prev_final_weights.reindex(union_ix, fill_value=0.0)
            target_w = target_w.reindex(union_ix, fill_value=0.0)

        if (
            lambda_tc > NUMERICAL_TOLERANCE_HIGH
            and prev_final_weights is not None
            and float(last_aligned.abs().sum()) > NUMERICAL_TOLERANCE_HIGH
        ):
            target_w = _apply_turnover_penalty(
                target_w, last_aligned, lambda_tc, min_w_bound, max_w_bound
            )

        # Forced exits must not be diluted by turnover-penalty shrinkage.
        # Ensure any holdings flagged for z_exit/z_exit_hard are targeted to 0.
        if forced_exits:
            for mgr in forced_exits:
                if mgr in target_w.index:
                    target_w.loc[mgr] = 0.0
            target_w = _apply_weight_bounds(target_w, min_w_bound, max_w_bound)

        desired_trades = target_w - last_aligned
        desired_turnover = float(desired_trades.abs().sum())
        final_w = target_w.copy()
        if (
            max_turnover_cap < 1.0 - NUMERICAL_TOLERANCE_HIGH
            and desired_turnover > max_turnover_cap + NUMERICAL_TOLERANCE_HIGH
        ):
            # Respect turnover cap, but prioritise forced exits (soft/hard z exits).
            # This prevents below-threshold holdings from lingering indefinitely
            # solely because turnover is capped.
            forced_ix = [ix for ix in desired_trades.index if str(ix) in forced_exits]
            mandatory = desired_trades.copy()
            if forced_ix:
                # Keep only forced exit trades in mandatory bucket
                mandatory.loc[[ix for ix in mandatory.index if ix not in forced_ix]] = (
                    0.0
                )
            else:
                mandatory[:] = 0.0

            mandatory_turnover = float(mandatory.abs().sum())
            optional = desired_trades - mandatory
            optional_turnover = float(optional.abs().sum())

            if mandatory_turnover >= max_turnover_cap - NUMERICAL_TOLERANCE_HIGH:
                # Forced exits alone consume (or exceed) the cap; execute forced exits
                # and skip all other trades.
                final_w = last_aligned + mandatory
            else:
                remaining_turnover = max_turnover_cap - mandatory_turnover
                scale = (
                    remaining_turnover / optional_turnover
                    if optional_turnover > 0
                    else 0.0
                )
                scale = max(0.0, min(1.0, scale))
                final_w = last_aligned + mandatory + optional * scale
        # Ensure bounds and normalisation remain satisfied
        final_w = _apply_weight_bounds(final_w, min_w_bound, max_w_bound)
        final_w = _enforce_max_active_positions(
            final_w, max_active_positions, protected=min_tenure_guard
        )

        # Prepare custom weights mapping in percent for _run_analysis.
        # We keep the internal turnover-cap/bounds logic here, but reconcile the
        # change log against the *actual* weights returned by the pipeline.
        eps = 1e-12
        final_w = final_w[final_w.abs() > eps]
        if not final_w.empty:
            total = float(final_w.sum())
            # Preserve infeasible bound outcomes:
            # - total > 1.0: min_weight floors too large
            # - total < 1.0: max_weight caps too tight
            # Only renormalise when the weights already sum (approximately) to 1.
            if total > eps and abs(total - 1.0) <= 1e-8:
                final_w = final_w / total
        # Only pass the selected holdings (if still present after filtering).
        manual_funds: list[str] = [
            str(h) for h in manual_holdings if h in final_w.index
        ]
        custom: dict[str, float] = {
            str(k): float(v) * 100.0 for k, v in final_w.items()
        }

        # Construct previous weights dict for pipeline (turnover tracking)
        prev_weights_for_pipeline = _coerce_previous_weights(prev_final_weights)

        res = _call_pipeline_with_diag(
            df,
            pt.in_start[:7],
            pt.in_end[:7],
            pt.out_start[:7],
            pt.out_end[:7],
            _resolve_target_vol(getattr(cfg, "vol_adjust", {})),
            getattr(cfg, "run", {}).get("monthly_cost", 0.0),
            floor_vol=cfg.vol_adjust.get("floor_vol"),
            warmup_periods=int(cfg.vol_adjust.get("warmup_periods", 0) or 0),
            selection_mode="manual",
            random_n=cfg.portfolio.get("random_n", 8),
            custom_weights=custom,
            rank_kwargs=None,
            manual_funds=manual_funds,
            indices_list=cfg.portfolio.get("indices_list"),
            benchmarks=cfg.benchmarks,
            seed=getattr(cfg, "seed", 42),
            risk_window=cfg.vol_adjust.get("window"),
            constraints=cfg.portfolio.get("constraints"),
            regime_cfg=regime_cfg,
            risk_free_column=risk_free_column_cfg,
            allow_risk_free_fallback=allow_risk_free_fallback_cfg,
            # Pass turnover parameters for pipeline-level enforcement.
            # The pipeline applies vol-scaling after which turnover may exceed
            # the threshold-hold logic's pre-scaled cap; passing these ensures
            # the final weights respect the turnover constraint.
            previous_weights=prev_weights_for_pipeline,
            max_turnover=max_turnover_cap if max_turnover_cap < 1.0 else None,
            lambda_tc=lambda_tc if lambda_tc > 0 else None,
            signal_spec=trend_spec,
        )
        payload = res.value
        diag = res.diagnostic
        if payload is None:
            if diag is not None:
                logger.warning(
                    "Manual selection period skipped %s/%s (%s): %s",
                    pt.in_start,
                    pt.out_start,
                    diag.reason_code,
                    diag.message,
                )
            continue
        res_dict = dict(payload)
        res_dict.update(missing_policy_metadata)
        # Persist z-scores into the standard score_frame so downstream
        # consumers can audit soft-entry/soft-exit decisions without
        # recomputation.
        score_frame_payload = res_dict.get("score_frame")
        if (
            isinstance(score_frame_payload, pd.DataFrame)
            and not score_frame_payload.empty
        ):
            score_frame_out = score_frame_payload.copy()
            if "zscore" in sf.columns and "zscore" not in score_frame_out.columns:
                score_frame_out = score_frame_out.join(sf[["zscore"]], how="left")
            if (
                metric == "blended"
                and "blended" in sf.columns
                and "blended" not in score_frame_out.columns
            ):
                score_frame_out = score_frame_out.join(sf[["blended"]], how="left")
            res_dict["score_frame"] = score_frame_out
        else:
            res_dict["score_frame"] = sf.copy()

        # Keep a direct copy of the selection frame as well (useful for
        # debugging selection/triggering differences from pipeline metrics).
        res_dict["selection_score_frame"] = sf.copy()
        res_dict["selection_metric"] = metric

        # Determine the realised weights/holdings as used by the pipeline.
        # This is the contract for downstream reporting/export.  In particular,
        # missing-data filters or other pipeline constraints may alter the final
        # investable set.  Manager changes must match these realised holdings.
        pipeline_weights_raw = res_dict.get("fund_weights")
        effective_w: pd.Series
        used_pipeline_weights = False
        if isinstance(pipeline_weights_raw, dict) and pipeline_weights_raw:
            try:
                effective_w = pd.Series(pipeline_weights_raw, dtype=float)
                used_pipeline_weights = True
            except Exception:
                effective_w = final_w.copy()
        else:
            effective_w = final_w.copy()

        effective_w = effective_w[effective_w.abs() > eps]

        # The pipeline may emit weights for non-fund columns (e.g., risk-free,
        # index, or benchmark series). These must not be treated as holdings.
        drop_cols: set[str] = set(str(x) for x in indices_list)
        drop_cols |= {str(x) for x in benchmark_cols}
        if resolved_rf_source == "configured" and resolved_rf_col:
            drop_cols.add(str(resolved_rf_col))
        if drop_cols:
            effective_w = effective_w.drop(labels=list(drop_cols), errors="ignore")

        # Turnover alignment can introduce additional indices (e.g., prior
        # holdings carried at zero then floored by bounds). These should not
        # become realised holdings unless they are part of the manual fund list
        # passed to the pipeline. If we are using pipeline weights directly,
        # respect the pipeline's realised holdings instead of filtering them out.
        if manual_holdings and not used_pipeline_weights:
            manual_set = {str(x) for x in manual_holdings}
            effective_w = effective_w.drop(
                labels=[c for c in effective_w.index if str(c) not in manual_set],
                errors="ignore",
            )

        # Some pipeline fallbacks can yield a populated-but-zero weight mapping.
        # Treat that as unusable and fall back to the intended weights.
        if used_pipeline_weights and effective_w.abs().sum() <= eps:
            effective_w = final_w.copy()
            effective_w = effective_w[effective_w.abs() > eps]
            if drop_cols:
                effective_w = effective_w.drop(labels=list(drop_cols), errors="ignore")
            if manual_holdings:
                manual_set = {str(x) for x in manual_holdings}
                effective_w = effective_w.drop(
                    labels=[c for c in effective_w.index if str(c) not in manual_set],
                    errors="ignore",
                )

        # Enforce max_funds contract on realised holdings.
        # This guards against any upstream components returning extra positions.
        if max_funds > 0 and len(effective_w.index) > max_funds:
            keep = effective_w.abs().sort_values(ascending=False).head(max_funds).index
            effective_w = effective_w.reindex(keep)

        if not effective_w.empty:
            total = float(effective_w.sum())
            if total > eps:
                # Pipeline weights are often expressed in percent (sum≈100).
                # Convert to decimal. Otherwise, preserve non-unit totals that
                # arise from infeasible bounds (sum<1 or sum>1).
                if abs(total - 100.0) <= 1e-6:
                    effective_w = effective_w / 100.0
                elif abs(total - 1.0) <= 1e-8:
                    effective_w = effective_w / total

        # Compute turnover/cost from the realised weights, not the intended ones.
        # Convention: report one-sided turnover (sum of buys or sells). For
        # fully-invested portfolios this is 0.5 * sum(|Δw|). The first rebalance
        # (from cash) is purely buys, so no halving is applied.
        risk_turnover = None
        risk_diag_payload = res_dict.get("risk_diagnostics")
        if isinstance(risk_diag_payload, dict):
            turnover_payload = risk_diag_payload.get("turnover")
            if isinstance(turnover_payload, pd.Series) and not turnover_payload.empty:
                risk_turnover = float(turnover_payload.iloc[-1])
            elif isinstance(turnover_payload, (int, float)):
                risk_turnover = float(turnover_payload)
        if prev_final_weights is None:
            last_effective = pd.Series(0.0, index=effective_w.index)
            if risk_turnover is not None and np.isfinite(risk_turnover):
                period_turnover = risk_turnover
            else:
                abs_diff = float((effective_w - last_effective).abs().sum())
                period_turnover = abs_diff
        else:
            union_ix = prev_final_weights.index.union(effective_w.index)
            last_effective = prev_final_weights.reindex(union_ix, fill_value=0.0)
            effective_w = effective_w.reindex(union_ix, fill_value=0.0)
            if risk_turnover is not None and np.isfinite(risk_turnover):
                period_turnover = 0.5 * risk_turnover
            else:
                abs_diff = float((effective_w - last_effective).abs().sum())
                period_turnover = 0.5 * abs_diff

        period_cost = period_turnover * ((tc_bps + slippage_bps) / 10000.0)

        # Reconcile manager change log to the realised holdings delta.
        actual_before = set(last_effective[last_effective.abs() > eps].index)
        actual_after = set(effective_w[effective_w.abs() > eps].index)

        by_key: dict[tuple[str, str], dict[str, object]] = {}
        for ev in events:
            try:
                action = str(ev.get("action", ""))
                manager = str(ev.get("manager", ""))
            except Exception:
                continue
            if action in {"added", "dropped"} and manager:
                by_key[(manager, action)] = dict(ev)
        delta_added = actual_after - actual_before
        delta_dropped = actual_before - actual_after
        raw_added = {m for (m, a) in by_key if a == "added"}
        raw_dropped = {m for (m, a) in by_key if a == "dropped"}

        # Preserve the original event log (it may contain intra-period churn
        # such as drop+re-add) but ensure we also reflect the realised holdings
        # delta for downstream consumers.
        missing_added = sorted(delta_added - raw_added)
        missing_dropped = sorted(delta_dropped - raw_dropped)
        for manager in missing_added:
            events.append(
                {
                    "action": "added",
                    "manager": manager,
                    "firm": _firm(manager),
                    "reason": "rebalance",
                    "detail": "realised holdings delta",
                }
            )
        for manager in missing_dropped:
            events.append(
                {
                    "action": "dropped",
                    "manager": manager,
                    "firm": _firm(manager),
                    "reason": "rebalance",
                    "detail": "realised holdings delta",
                }
            )

        effective_nonzero = effective_w[effective_w.abs() > eps].copy()
        realised_holdings = [str(x) for x in effective_nonzero.index]
        # Do not emit zero-weight positions: they are not real holdings and
        # confuse downstream audits (e.g., a dropped fund showing up with 0.0).
        res_dict["fund_weights"] = {
            str(k): float(v) for k, v in effective_nonzero.items()
        }

        # Record cooldowns for any managers that exited based on realised holdings.
        if cooldown_periods > 0 and prev_final_weights is not None:
            entered = {str(x) for x in prev_final_weights.index}
            current = set(realised_holdings)
            exited = entered - current
            for mgr in exited:
                cooldown_book[mgr] = int(cooldown_periods) + 1
            # Defensive: if something re-appears in holdings, clear cooldown.
            for mgr in list(cooldown_book.keys()):
                if mgr in current:
                    cooldown_book.pop(mgr, None)

        # Expose intra-period rebalance weight snapshots for UI diagnostics.
        #
        # The threshold-hold engine currently updates holdings at the
        # multi-period cadence (e.g. annually) but users can still configure a
        # rebalance schedule (e.g. quarterly) via ``portfolio.rebalance_freq``.
        # Emit a per-period weights frame keyed by those rebalance dates so the
        # Streamlit UI can render weights by rebalance date.
        try:
            rebalance_freq = str(cfg.portfolio.get("rebalance_freq", "") or "").strip()
        except Exception:  # pragma: no cover - defensive
            rebalance_freq = ""
        rebalance_frame: pd.DataFrame | None = None
        if rebalance_freq and isinstance(out_df, pd.DataFrame) and not out_df.empty:
            try:
                schedule = get_rebalance_dates(out_df.index, rebalance_freq)
                if len(out_df.index) and (out_df.index[0] not in schedule):
                    schedule = schedule.insert(0, out_df.index[0])
                if not schedule.empty:
                    # Recompute weights per rebalance date using a rolling
                    # in-sample window ending at that date. Holdings remain
                    # fixed intra-period; only weights are refreshed.
                    in_len_years = int(mp_cfg.get("in_sample_len", 3) or 3)
                    in_months = max(1, in_len_years * 12)

                    # Prefer configured risk-based weighting for intra-period
                    # rebalances when available.
                    try:
                        from ..plugins import create_weight_engine

                        weighting_scheme = str(
                            cfg.portfolio.get("weighting_scheme", "equal") or "equal"
                        ).lower()
                        risk_engine = create_weight_engine(weighting_scheme)
                        use_risk_engine = weighting_scheme not in {"equal", "ew"}
                    except Exception:  # pragma: no cover - best-effort only
                        risk_engine = None
                        use_risk_engine = False

                    rebalance_rows: list[dict[str, float]] = []
                    prev_reb_w = effective_w.copy()
                    for reb_date in pd.DatetimeIndex(schedule):
                        end_dt = pd.Timestamp(reb_date)
                        start_dt = (
                            end_dt - pd.DateOffset(months=in_months - 1)
                        ) + pd.offsets.MonthEnd(0)

                        window = df_indexed.reindex(columns=realised_holdings).loc[
                            (df_indexed.index >= start_dt)
                            & (df_indexed.index <= end_dt)
                        ]
                        if window.empty:
                            w_row = prev_reb_w
                        else:
                            try:
                                if use_risk_engine and risk_engine is not None:
                                    prepared = _prepare_returns_frame(window)
                                    cov = prepared.cov()
                                    w_series = risk_engine.weight(cov)
                                else:
                                    sf_roll = _score_frame(
                                        window,
                                        realised_holdings,
                                        risk_free_override=rf_override,
                                        periods_per_year=int(periods_per_year),
                                    )
                                    sf_roll = _ensure_zscore(sf_roll, metric)
                                    weights_df_roll = weighting.weight(
                                        sf_roll.loc[realised_holdings], end_dt
                                    )
                                    signal_slice = (
                                        sf_roll.loc[realised_holdings, metric]
                                        if metric in sf_roll.columns
                                        else None
                                    )
                                    w_series = _apply_policy_to_weights(
                                        weights_df_roll, signal_slice
                                    )
                                bounded = _apply_weight_bounds(
                                    w_series.reindex(realised_holdings).fillna(0.0),
                                    min_w_bound,
                                    max_w_bound,
                                )
                                bounded = _enforce_max_active_positions(
                                    bounded,
                                    max_active_positions,
                                    protected=min_tenure_guard,
                                )
                                bounded = bounded[bounded.abs() > eps]
                                total = float(bounded.sum())
                                if total > eps and abs(total - 1.0) <= 1e-8:
                                    bounded = bounded / total
                                w_row = bounded
                                prev_reb_w = w_row
                            except Exception:  # pragma: no cover - best-effort only
                                w_row = prev_reb_w

                        rebalance_rows.append(
                            {str(k): float(v) for k, v in w_row.items()}
                        )

                    rebalance_frame = pd.DataFrame(
                        rebalance_rows,
                        index=pd.DatetimeIndex(schedule),
                    )
                    rebalance_frame.index.name = "rebalance_date"
                    res_dict["rebalance_weights"] = rebalance_frame
            except Exception:  # pragma: no cover - best-effort only
                pass

        if rebalance_frame is not None and not rebalance_frame.empty:
            out_scaled = res_dict.get("out_sample_scaled")
            if isinstance(out_scaled, pd.DataFrame) and not out_scaled.empty:
                weights_by_date = (
                    rebalance_frame.reindex(out_scaled.index).ffill().fillna(0.0)
                )
                weights_by_date = weights_by_date.reindex(
                    columns=out_scaled.columns, fill_value=0.0
                )
                rebalance_returns = (out_scaled * weights_by_date).sum(axis=1)
                res_dict["portfolio_user_weight"] = rebalance_returns
                res_dict["weights_user_weight"] = rebalance_frame

                if rf_override_enabled:
                    rf_out = pd.Series(float(rf_rate_periodic), index=out_scaled.index)
                elif resolved_rf_col and resolved_rf_col in out_df.columns:
                    rf_out = out_df[resolved_rf_col].reindex(out_scaled.index)
                else:
                    rf_out = pd.Series(0.0, index=out_scaled.index)

                res_dict["out_user_stats"] = _compute_stats(
                    pd.DataFrame({"user": rebalance_returns}), rf_out
                )["user"]

                out_raw = out_df.reindex(columns=out_scaled.columns)
                if isinstance(out_raw, pd.DataFrame) and not out_raw.empty:
                    weights_raw = (
                        rebalance_frame.reindex(out_raw.index).ffill().fillna(0.0)
                    )
                    weights_raw = weights_raw.reindex(
                        columns=out_raw.columns, fill_value=0.0
                    )
                    rebalance_raw = (out_raw * weights_raw).sum(axis=1)
                    res_dict["portfolio_user_weight_raw"] = rebalance_raw
                    res_dict["out_user_stats_raw"] = _compute_stats(
                        pd.DataFrame({"user": rebalance_raw}), rf_out
                    )["user"]

        res_dict["selected_funds"] = realised_holdings
        res_dict["period"] = (
            pt.in_start,
            pt.in_end,
            pt.out_start,
            pt.out_end,
        )
        res_dict["missing_policy_diagnostic"] = dict(missing_policy_diagnostic)
        # Attach per-period manager change log and execution stats
        res_dict["manager_changes"] = events
        res_dict["turnover"] = period_turnover
        res_dict["transaction_cost"] = float(period_cost)
        updated_tenure: dict[str, int] = {}
        for mgr in realised_holdings:
            mgr_str = str(mgr)
            updated_tenure[mgr_str] = int(holdings_tenure.get(mgr_str, 0)) + 1
        holdings_tenure = updated_tenure
        res_dict["holding_tenure"] = dict(holdings_tenure)

        # Persist realised weights for next-period turnover logic.
        # Store only non-zero holdings so indices do not accumulate across the
        # union-alignment used for turnover computations.
        prev_final_weights = effective_w[effective_w.abs() > eps].copy()
        prev_weights = prev_final_weights.copy()
        # Append this period's result (was incorrectly outside loop causing only last period kept)
        results.append(res_dict)
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
