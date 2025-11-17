from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from trend_analysis.backtesting import BacktestResult, CostModel, bootstrap_equity

from .event_log import Event, EventLog
from .metrics_extra import AVAILABLE_METRICS
from .policy_engine import CooldownBook, PolicyConfig, decide_hires_fires

logger = logging.getLogger(__name__)

try:
    ta_pipeline: Any = importlib.import_module("trend_analysis.pipeline")
    HAS_TA = True
except Exception:
    HAS_TA = False
    ta_pipeline = None


def compute_score_frame_local(
    panel: pd.DataFrame, rf_annual: float = 0.0
) -> pd.DataFrame:
    idx = panel.index
    out = {}
    for col in panel.columns:
        if col == "Date":
            continue
        r = panel[col].dropna()
        col_metrics = {}
        for name, spec in AVAILABLE_METRICS.items():
            try:
                if name in ("sharpe", "sortino"):
                    val = spec["fn"](r, idx, rf_annual)
                else:
                    val = spec["fn"](r, idx)
                col_metrics[name] = val
            except Exception as e:
                logger.warning(
                    "Failed to compute metric '%s' for column '%s': %s", name, col, e
                )
                col_metrics[name] = np.nan
        out[col] = col_metrics
    df = pd.DataFrame(out).T
    return df


def compute_score_frame(
    panel: pd.DataFrame,
    insample_start: pd.Timestamp,
    insample_end: pd.Timestamp,
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    if "Date" not in panel.columns:
        raise ValueError("DataFrame must contain a 'Date' column")

    # Normalise the Date column so external helpers that expect month-end
    # timestamps at midnight include the period in their filtering logic.
    panel = panel.copy()
    panel["Date"] = pd.to_datetime(panel["Date"]).dt.normalize()

    if HAS_TA:
        fn = getattr(ta_pipeline, "single_period_run", None)
        if callable(fn):
            try:
                sf = fn(
                    panel,
                    insample_start.strftime("%Y-%m"),
                    insample_end.strftime("%Y-%m"),
                )
                # Ensure consistent DataFrame type for downstream
                return cast(pd.DataFrame, sf)
            except (ImportError, AttributeError) as e:
                logger.warning(
                    "External scoring function unavailable or misconfigured"
                    " (ImportError/AttributeError): %s. Falling back to"
                    " local implementation.",
                    e,
                )
            except (ValueError, TypeError) as e:
                logger.warning(
                    "External scoring function failed due to invalid"
                    " parameters or data (ValueError/TypeError): %s."
                    " Falling back to local implementation.",
                    e,
                )
    local_panel = panel.set_index("Date")
    return compute_score_frame_local(
        local_panel.loc[insample_start:insample_end],
        rf_annual=rf_annual,
    )


@dataclass
class SimResult:
    dates: List[pd.Timestamp]
    portfolio: pd.Series
    weights: Dict[pd.Timestamp, pd.Series]
    event_log: EventLog
    benchmark: Optional[pd.Series]
    _bootstrap_cache: Dict[tuple[int, int, int | None], pd.DataFrame] = field(
        default_factory=dict, init=False, repr=False
    )

    def portfolio_curve(self) -> pd.Series:
        return (1 + self.portfolio.fillna(0)).cumprod()

    def drawdown_curve(self) -> pd.Series:
        curve = self.portfolio_curve()
        return curve / curve.cummax() - 1.0

    def bootstrap_band(
        self,
        *,
        n: int = 500,
        block: int = 20,
        random_state: np.random.Generator | int | None = None,
    ) -> pd.DataFrame:
        """Compute (and cache) bootstrap equity quantiles for the portfolio."""

        cache_key: tuple[int, int, int | None] | None
        if random_state is None or isinstance(random_state, (int, np.integer)):
            seed_val = int(random_state) if random_state is not None else None
            cache_key = (n, block, seed_val)
            cached = self._bootstrap_cache.get(cache_key)
            if cached is not None:
                return cached.copy()
        else:
            cache_key = None

        returns = self.portfolio.astype(float)
        equity = (1.0 + returns.fillna(0.0)).cumprod()
        drawdown = equity / equity.cummax() - 1.0 if not equity.empty else equity
        calendar = pd.DatetimeIndex(self.dates) if self.dates else pd.DatetimeIndex([])

        backtest = BacktestResult(
            returns=returns,
            equity_curve=equity,
            weights=pd.DataFrame(dtype=float),
            turnover=pd.Series(dtype=float),
            per_period_turnover=pd.Series(dtype=float),
            transaction_costs=pd.Series(dtype=float),
            rolling_sharpe=pd.Series(dtype=float),
            drawdown=drawdown,
            metrics={},
            calendar=calendar,
            window_mode="rolling",
            window_size=max(len(calendar), 1) if len(calendar) else 1,
            training_windows={},
            cost_model=CostModel(),
        )

        band = bootstrap_equity(backtest, n=n, block=block, random_state=random_state)
        if cache_key is not None:
            self._bootstrap_cache[cache_key] = band.copy()
        return band

    def event_log_df(self) -> pd.DataFrame:
        return self.event_log.to_frame()

    def summary(self) -> Dict[str, Any]:
        curve = self.portfolio_curve()
        dd = self.drawdown_curve().min()
        total = curve.iloc[-1] - 1.0
        ann = (curve.iloc[-1]) ** (12 / max(len(curve), 1)) - 1
        out = {
            "total_return": float(total),
            "max_drawdown": float(dd),
            "ann_return_approx": float(ann),
        }
        if self.benchmark is not None:
            ex = self.portfolio - self.benchmark.reindex_like(self.portfolio).fillna(0)
            ir = ex.mean() * 12 / (ex.std(ddof=0) * np.sqrt(12) + EPS)
            out["information_ratio"] = float(ir)
        return out


class Simulator:
    def __init__(
        self,
        returns_df: pd.DataFrame,
        benchmark_col: Optional[str] = None,
        cash_rate_annual: float = 0.0,
    ):
        self.df = returns_df.copy()
        self.benchmark_col = benchmark_col
        self.cash_rate_annual = cash_rate_annual
        self.benchmark = (
            self.df[benchmark_col]
            if benchmark_col and benchmark_col in self.df.columns
            else None
        )

    def _gen_review_dates(
        self, start: pd.Timestamp, end: pd.Timestamp, freq: str
    ) -> List[pd.Timestamp]:
        dates = pd.period_range(start=start, end=end, freq="M").to_timestamp(how="end")
        if freq.startswith("q"):
            dates = dates[dates.to_period("Q").to_timestamp(how="end") == dates]
        return list(dates)

    def run(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        freq: str,
        lookback_months: int,
        policy: PolicyConfig,
        rebalance: Optional[Dict[str, Any]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> SimResult:
        review_dates = self._gen_review_dates(start, end, freq)
        weights_by_date: Dict[pd.Timestamp, pd.Series] = {}
        event_log = EventLog()
        active: List[str] = []
        cooldowns = CooldownBook()
        eligible_since: Dict[str, int] = {
            m: 0 for m in self.df.columns if m != self.benchmark_col
        }
        # Track how many consecutive periods each active manager has been held
        tenure: Dict[str, int] = {
            m: 0 for m in self.df.columns if m != self.benchmark_col
        }

        portfolio_returns: List[Tuple[Any, float]] = []
        # Rebalance state: track timing and risk stats
        rb_cfg: Dict[str, Any] = dict(rebalance or {})
        rb_cfg.setdefault("bayesian_only", True)
        rb_state: Dict[str, Any] = {
            "since_last_reb": 0,
            "equity_curve": [],  # list of equity values up to current period
        }

        prev_w: pd.Series = pd.Series(dtype=float)

        for i, d in enumerate(review_dates):
            if progress_cb:
                progress_cb(i + 1, len(review_dates))

            for m in list(eligible_since.keys()):
                eligible_since[m] += 1

            insample_end = d
            insample_start = (d - pd.offsets.MonthEnd(lookback_months)).normalize()

            panel = self.df.drop(columns=[self.benchmark_col], errors="ignore")

            score_frame = compute_score_frame(
                panel, insample_start, insample_end, rf_annual=self.cash_rate_annual
            )

            directions = {}
            for m in score_frame.columns:
                name = str(m).lower()
                if any(k in name for k in ["vol", "drawdown", "ulcer", "dd_duration"]):
                    directions[m] = -1
                else:
                    directions[m] = +1

            decisions = decide_hires_fires(
                d,
                score_frame,
                active,
                policy,
                directions,
                cooldowns,
                eligible_since,
                tenure,
            )
            for m, reason in decisions["fire"]:
                if m in active:
                    active.remove(m)
                    cooldowns.set(m, policy.cooldown_months)
                    event_log.append(
                        Event(date=d, action="fire", manager=m, reason=reason)
                    )
            for m, reason in decisions["hire"]:
                if m not in active:
                    active.append(m)
                    tenure[m] = 0  # reset tenure on (re)hire
                    event_log.append(
                        Event(date=d, action="hire", manager=m, reason=reason)
                    )

            # Target weights (from selection/weighting stage). For now, equal-weight.
            if active:
                w_target = pd.Series(1.0 / len(active), index=active, dtype=float)
                if policy.max_weight < 1.0:
                    w_target = w_target.clip(upper=policy.max_weight)
                    w_target = w_target / max(w_target.sum(), EPS)
            else:
                w_target = pd.Series(dtype=float)

            # Realize target via rebalancing pipeline
            w_realized = _apply_rebalance_pipeline(
                prev_weights=prev_w,
                target_weights=w_target,
                date=d,
                rb_cfg=rb_cfg,
                rb_state=rb_state,
                policy=policy,
            )

            # Persist
            weights_by_date[d] = w_realized
            prev_w = w_realized
            # Update tenure counters after deciding holdings
            # Increment for currently active; reset for those not active
            current_set = set(active)
            for m in list(tenure.keys()):
                if m in current_set:
                    tenure[m] = tenure.get(m, 0) + 1
                else:
                    tenure[m] = 0

            next_month = d + pd.offsets.MonthEnd(1)
            if next_month in self.df.index:
                if len(prev_w):
                    row = self.df.loc[next_month]
                    r_next = row.reindex(prev_w.index).astype(float).fillna(0.0)
                    weights_vec = prev_w.astype(float).reindex(r_next.index).fillna(0.0)
                    port_r = float(np.dot(r_next.to_numpy(), weights_vec.to_numpy()))
                else:
                    port_r = 0.0
            else:
                port_r = np.nan
            portfolio_returns.append((next_month, port_r))

            # Update equity curve for drawdown/vol in rb_state
            try:
                if not np.isnan(port_r):
                    ec = rb_state.get("equity_curve", [])
                    last = ec[-1] if ec else 1.0
                    ec.append(last * (1.0 + float(port_r)))
                    rb_state["equity_curve"] = ec
            except (ValueError, TypeError, ArithmeticError, IndexError) as e:
                logger.warning("Failed to update equity curve at %s: %s", d, e)

            cooldowns.tick()

        pr = pd.Series({d: r for d, r in portfolio_returns})
        bench = self.benchmark if self.benchmark is not None else None
        return SimResult(
            dates=review_dates,
            portfolio=pr.dropna(),
            weights=weights_by_date,
            event_log=event_log,
            benchmark=bench,
        )


# Small epsilon to avoid divide-by-zero in IR calculations
EPS = 1e-12


def _apply_rebalance_pipeline(
    *,
    prev_weights: pd.Series,
    target_weights: pd.Series,
    date: pd.Timestamp,
    rb_cfg: Dict[str, Any],
    rb_state: Dict[str, Any],
    policy: PolicyConfig,
) -> pd.Series:
    """Apply rebalancing strategies in order to realize target weights.

    Contract:
        - Inputs: prev_weights (current holdings), target_weights (desired), date,
            rb_cfg (dict), rb_state (mutable), policy.
    - Output: realized weights Series, index is union of prev/target; NaNs treated as 0.
    - Side effects: updates rb_state keys such as since_last_reb and risk stats.
    """
    # Normalize inputs
    pw = prev_weights.astype(float).copy()
    tw = target_weights.astype(float).copy()
    pw = pw[pw != 0.0]
    tw = tw[tw != 0.0]
    all_idx = list(dict.fromkeys(list(pw.index) + list(tw.index)))
    pw = pw.reindex(all_idx).fillna(0.0)
    tw = tw.reindex(all_idx).fillna(0.0)

    # If bayesian_only toggle: pass-through target
    if bool(rb_cfg.get("bayesian_only", True)):
        rb_state["since_last_reb"] = 0
        return tw
    # If no previous holdings, adopt target immediately
    if pw.empty:
        rb_state["since_last_reb"] = 0
        return tw

    # Strategy order and params
    strategies: List[str] = list(rb_cfg.get("strategies", ["drift_band"]))
    params: Dict[str, Any] = dict(rb_cfg.get("params", {}))

    work = pw.copy()

    # Helper to cap weights and normalise
    def _cap_and_norm(s: pd.Series, gross: Optional[float] = None) -> pd.Series:
        out = s.clip(lower=0.0)
        if policy.max_weight < 1.0 and policy.max_weight > 0.0:
            out = out.clip(upper=float(policy.max_weight))
        total = float(out.sum())
        target_sum = float(gross if gross is not None else (1.0 if len(out) else 0.0))
        if total > EPS and target_sum > EPS:
            out = out * (target_sum / total)
        return out

    # Track gross to preserve unless a full rebalance happens
    gross_prev = float(max(pw.sum(), 0.0))
    since_last = int(rb_state.get("since_last_reb", 0))

    for name in strategies:
        if name == "periodic_rebalance":
            cfg = params.get("periodic_rebalance", {})
            interval = int(cfg.get("interval", 1))
            if interval <= 1 or since_last + 1 >= interval:
                work = tw.copy()
                since_last = 0
            else:
                # Skip rebalancing this period
                since_last += 1
        elif name == "drift_band":
            cfg = params.get("drift_band", {})
            band = float(cfg.get("band_pct", 0.03))
            min_trade = float(cfg.get("min_trade", 0.005))
            mode = str(cfg.get("mode", "partial"))
            delta = tw - work
            adjust = pd.Series(0.0, index=delta.index)
            for k, dv in delta.items():
                if abs(dv) <= band:
                    continue
                if mode == "partial":
                    # Move halfway back into band boundary
                    step = np.sign(dv) * max(abs(dv) - band, 0.0)
                else:  # full
                    step = dv
                if abs(step) >= min_trade:
                    adjust[k] = step
            work = _cap_and_norm(work + adjust, gross=gross_prev)
        elif name == "turnover_cap":
            cfg = params.get("turnover_cap", {})
            max_to = float(cfg.get("max_turnover", 0.20))
            # Always aim towards target weights, allocate limited turnover
            candidate = tw.copy()
            d = candidate - work
            total_gap = float(d.abs().sum())
            if total_gap <= max_to + 1e-9:
                work = candidate
            else:
                # Move a fraction alpha towards target so L1 turnover equals max_to
                alpha = float(max_to / total_gap) if total_gap > 0 else 0.0
                work = work + alpha * d
        elif name == "vol_target_rebalance":
            cfg = params.get("vol_target_rebalance", {})
            target_vol = float(cfg.get("target", 0.10))
            lev_min = float(cfg.get("lev_min", 0.5))
            lev_max = float(cfg.get("lev_max", 1.5))
            window = int(cfg.get("window", 6))
            # Estimate realized vol from rb_state equity_curve
            ec: List[float] = list(rb_state.get("equity_curve", []))
            if len(ec) >= window + 1:
                # Compute past window simple returns from equity
                rets = pd.Series(np.diff(ec[-(window + 1) :]) / ec[-(window + 1) : -1])
                vol = float(rets.std(ddof=0)) * np.sqrt(12)
                if vol > 0:
                    lev = float(np.clip(target_vol / vol, lev_min, lev_max))
                    work = work * lev
        elif name == "drawdown_guard":
            cfg = params.get("drawdown_guard", {})
            dd_win = int(cfg.get("dd_window", 12))
            dd_th = float(cfg.get("dd_threshold", 0.10))
            guard_mult = float(cfg.get("guard_multiplier", 0.5))
            recover = float(cfg.get("recover_threshold", 0.05))
            eq_curve: List[float] = list(rb_state.get("equity_curve", []))
            guard_on = bool(rb_state.get("guard_on", False))
            dd = 0.0
            if len(eq_curve) >= 1:
                # Use trailing window if available
                sub = eq_curve[-dd_win:] if len(eq_curve) >= dd_win else eq_curve
                peak = max(sub)
                cur = sub[-1]
                if peak > 0:
                    dd = (cur / peak) - 1.0
            if (not guard_on and dd <= -dd_th) or (guard_on and dd <= -recover):
                guard_on = True
            elif guard_on and dd >= -recover:
                guard_on = False
            rb_state["guard_on"] = guard_on
            if guard_on:
                work = work * guard_mult
        else:
            # Unknown strategy: skip (forward-compat)
            continue

    rb_state["since_last_reb"] = since_last
    # Final sanity: drop tiny weights
    work = work.where(work.abs() > EPS, other=0.0)
    return work
