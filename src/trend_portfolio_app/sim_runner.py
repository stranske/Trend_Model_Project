from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Any, cast
import importlib
import pandas as pd
import numpy as np

from .policy_engine import PolicyConfig, CooldownBook, decide_hires_fires
from .event_log import Event, EventLog
from .metrics_extra import AVAILABLE_METRICS

try:
    ta_pipeline = importlib.import_module("trend_analysis.pipeline")
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
        r = panel[col].dropna()
        col_metrics = {}
        for name, spec in AVAILABLE_METRICS.items():
            try:
                if name in ("sharpe", "sortino"):
                    val = spec["fn"](r, idx, rf_annual)
                else:
                    val = spec["fn"](r, idx)
                col_metrics[name] = val
            except Exception:
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
            except Exception:
                pass
    return compute_score_frame_local(
        panel.loc[insample_start:insample_end], rf_annual=rf_annual
    )


@dataclass
class SimResult:
    dates: List[pd.Timestamp]
    portfolio: pd.Series
    weights: Dict[pd.Timestamp, pd.Series]
    event_log: EventLog
    benchmark: Optional[pd.Series]

    def portfolio_curve(self) -> pd.Series:
        return (1 + self.portfolio.fillna(0)).cumprod()

    def drawdown_curve(self) -> pd.Series:
        curve = self.portfolio_curve()
        return curve / curve.cummax() - 1.0

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
        dates = pd.period_range(start=start, end=end, freq="M").to_timestamp("M")
        if freq.startswith("q"):
            dates = dates[dates.to_period("Q").to_timestamp("Q") == dates]
        return list(dates)

    def run(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        freq: str,
        lookback_months: int,
        policy: PolicyConfig,
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

        portfolio_returns = []

        for i, d in enumerate(review_dates):
            if progress_cb:
                progress_cb(i + 1, len(review_dates))

            for m in list(eligible_since.keys()):
                eligible_since[m] += 1

            insample_end = d
            insample_start = (d - pd.offsets.MonthEnd(lookback_months)).to_pydatetime()
            insample_start = pd.Timestamp(insample_start).normalize()

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

            if active:
                w = pd.Series(1.0 / len(active), index=active)
                if policy.max_weight < 1.0:
                    w = w.clip(upper=policy.max_weight)
                    w = w / w.sum()
            else:
                w = pd.Series(dtype=float)
            weights_by_date[d] = w
            # Update tenure counters after deciding holdings
            # Increment for currently active; reset for those not active
            current_set = set(active)
            for m in tenure:
                if m in current_set:
                    tenure[m] = tenure.get(m, 0) + 1
                else:
                    tenure[m] = 0

            next_month = d + pd.offsets.MonthEnd(1)
            if next_month in self.df.index:
                if len(w):
                    row = self.df.loc[next_month]
                    r_next = row.reindex(w.index).astype(float).fillna(0.0)
                    weights_vec = w.astype(float).reindex(r_next.index).fillna(0.0)
                    port_r = float(np.dot(r_next.to_numpy(), weights_vec.to_numpy()))
                else:
                    port_r = 0.0
            else:
                port_r = np.nan
            portfolio_returns.append((next_month, port_r))

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
