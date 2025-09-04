from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class MetricSpec:
    name: str
    weight: float = 1.0


@dataclass
class PolicyConfig:
    top_k: int = 10
    bottom_k: int = 0
    cooldown_months: int = 3
    min_track_months: int = 24
    max_active: int = 100
    max_weight: float = 0.10
    # Optional guard: once added, must be held at least this many periods
    # before eligible for removal. When 0, disabled.
    min_tenure_n: int = 0
    # Optional: cap total selection changes (hires + fires) per period.
    # When 0, disabled. UI/config default may be >0, engine default keeps
    # backward-compat as no-op unless explicitly set.
    turnover_budget_max_changes: int = 0
    # Optional diversification guard: cap number of holdings per bucket.
    # When 0, disabled. Bucket mapping is manager->bucket label.
    diversification_max_per_bucket: int = 0
    diversification_buckets: Dict[str, str] = field(default_factory=dict)
    # Competing rule sets (ordered). Empty => default behavior (threshold_hold).
    add_rules: List[str] = field(default_factory=list)
    drop_rules: List[str] = field(default_factory=list)
    # Sticky rank window parameters and CI level (simple placeholder gate)
    sticky_add_x: int = 1
    sticky_drop_y: int = 1
    ci_level: float = 0.0
    metrics: List[MetricSpec] = field(default_factory=list)

    def dict(self) -> Dict[str, Any]:
        return {
            "top_k": self.top_k,
            "bottom_k": self.bottom_k,
            "cooldown_months": self.cooldown_months,
            "min_track_months": self.min_track_months,
            "max_active": self.max_active,
            "max_weight": self.max_weight,
            "min_tenure_n": self.min_tenure_n,
            "turnover_budget_max_changes": self.turnover_budget_max_changes,
            "diversification_max_per_bucket": self.diversification_max_per_bucket,
            "diversification_buckets": self.diversification_buckets,
            "add_rules": list(self.add_rules),
            "drop_rules": list(self.drop_rules),
            "sticky_add_x": self.sticky_add_x,
            "sticky_drop_y": self.sticky_drop_y,
            "ci_level": self.ci_level,
            "metrics": [vars(m) for m in self.metrics],
        }


class CooldownBook:
    def __init__(self) -> None:
        self.map: Dict[str, int] = {}

    def tick(self) -> None:
        for k in list(self.map.keys()):
            self.map[k] = max(0, self.map[k] - 1)
            if self.map[k] == 0:
                del self.map[k]

    def set(self, key: str, months: int) -> None:
        self.map[key] = months

    def in_cooldown(self, key: str) -> bool:
        return key in self.map


def zscore(x: pd.Series) -> pd.Series:
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0 or np.isnan(s):
        return x * 0
    return (x - m) / s


def rank_scores(
    score_frame: pd.DataFrame,
    metric_weights: Dict[str, float],
    metric_directions: Dict[str, int],
) -> pd.Series:
    parts: List[pd.Series] = []
    for m, w in metric_weights.items():
        if m not in score_frame.columns:
            continue
        s = score_frame[m].astype(float)
        s = zscore(s) * float(w) * metric_directions.get(m, 1)
        parts.append(s)
    if not parts:
        return pd.Series(index=score_frame.index, dtype=float)
    total = parts[0].copy()
    for s in parts[1:]:
        total = total.add(s, fill_value=0.0)
    return total.astype(float)


def decide_hires_fires(
    asof: pd.Timestamp,
    score_frame: pd.DataFrame,
    current: List[str],
    policy: PolicyConfig,
    directions: Dict[str, int],
    cooldowns: CooldownBook,
    eligible_since: Dict[str, int],
    tenure: Dict[str, int] | None = None,
    rule_state: Dict[str, Any] | None = None,
) -> Dict[str, List[Tuple[str, str]]]:
    eligible = [
        m
        for m in score_frame.index
        if eligible_since.get(m, 0) >= policy.min_track_months
    ]
    sf = score_frame.loc[eligible].copy()
    if sf.empty:
        return {"hire": [], "fire": []}
    rs = rank_scores(sf, {m.name: m.weight for m in policy.metrics}, directions)
    sf["_score"] = rs
    sf = sf.sort_values("_score", ascending=False)
    # Prepare rule gates
    add_rules = policy.add_rules or ["sticky_rank_window", "threshold_hold"]
    drop_rules = policy.drop_rules or ["sticky_rank_window", "threshold_hold"]
    add_streak = (rule_state or {}).get("add_streak", {})
    drop_streak = (rule_state or {}).get("drop_streak", {})

    def allow_add(name: str) -> bool:
        for r in add_rules:
            if r == "sticky_rank_window" and int(policy.sticky_add_x) > 1:
                if int(add_streak.get(name, 0)) < int(policy.sticky_add_x):
                    return False
            if r == "confidence_interval" and float(policy.ci_level) > 0:
                # Placeholder: require non-negative composite score
                score_val = float(sf["_score"].get(name, 0.0))
                if score_val < 0.0:
                    return False
            # threshold_hold imposes no extra gate beyond being a candidate
        return True

    def allow_drop(name: str) -> bool:
        for r in drop_rules:
            if r == "sticky_rank_window" and int(policy.sticky_drop_y) > 1:
                if int(drop_streak.get(name, 0)) < int(policy.sticky_drop_y):
                    return False
            # threshold_hold: handled by bottom_k membership
        return True

    to_fire: List[Tuple[str, str]] = []
    if policy.bottom_k > 0:
        bottom = list(sf.tail(policy.bottom_k).index)
        for m in bottom:
            if m in current and allow_drop(m):
                # Enforce min-tenure guard if configured
                if policy.min_tenure_n > 0 and tenure is not None:
                    if int(tenure.get(m, 0)) < int(policy.min_tenure_n):
                        continue
                to_fire.append((m, "bottom_k"))
    candidates = [
        m for m in list(sf.index) if m not in current and not cooldowns.in_cooldown(m)
    ]
    hires: List[Tuple[str, str]] = []
    next_active = list(set(current) - {x for x, _ in to_fire})
    # Diversification-aware hiring: enforce per-bucket caps if configured
    if (
        policy.diversification_max_per_bucket
        and policy.diversification_max_per_bucket > 0
    ):
        bucket_map = policy.diversification_buckets or {}

        def bucket_of(x: str) -> str:
            # Graceful handling for unknowns: treat as singleton bucket by name
            return bucket_map.get(x, x)

        counts: Dict[str, int] = defaultdict(int)
        for m in next_active:
            counts[bucket_of(m)] += 1
        for m in candidates:
            if len(hires) >= policy.top_k or len(next_active) >= policy.max_active:
                break
            b = bucket_of(m)
            if counts[b] >= policy.diversification_max_per_bucket:
                continue
            if allow_add(m):
                hires.append((m, "top_k"))
                next_active.append(m)
                counts[b] += 1
    else:
        for m in candidates:
            if len(hires) >= policy.top_k or len(next_active) >= policy.max_active:
                break
            if allow_add(m):
                hires.append((m, "top_k"))
                next_active.append(m)
    # Apply turnover budget across hires and fires if enabled
    if policy.turnover_budget_max_changes and (
        len(hires) + len(to_fire) > policy.turnover_budget_max_changes
    ):
        s = sf["_score"].astype(float)
        moves: List[Tuple[float, str, str, str]] = (
            []
        )  # (priority, kind, manager, reason)
        for m, reason in hires:
            # Higher-scored hires have higher priority
            prio = float(s.get(m, np.nan))
            if not np.isnan(prio):
                moves.append((prio, "hire", m, reason))
        for m, reason in to_fire:
            # Lower-scored fires have higher priority ⇒ use negative score
            prio = float(-s.get(m, np.nan))
            if not np.isnan(prio):
                moves.append((prio, "fire", m, reason))
        moves.sort(key=lambda x: x[0], reverse=True)
        kept = moves[: policy.turnover_budget_max_changes]
        kept_hires = [(m, r) for _, k, m, r in kept if k == "hire"]
        kept_fires = [(m, r) for _, k, m, r in kept if k == "fire"]
        hires = kept_hires
        to_fire = kept_fires
    return {"hire": hires, "fire": to_fire}
