from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
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
    metrics: List[MetricSpec] = field(default_factory=list)

    def dict(self):
        return {
            "top_k": self.top_k,
            "bottom_k": self.bottom_k,
            "cooldown_months": self.cooldown_months,
            "min_track_months": self.min_track_months,
            "max_active": self.max_active,
            "max_weight": self.max_weight,
            "metrics": [vars(m) for m in self.metrics],
        }


class CooldownBook:
    def __init__(self):
        self.map: Dict[str, int] = {}

    def tick(self):
        for k in list(self.map.keys()):
            self.map[k] = max(0, self.map[k] - 1)
            if self.map[k] == 0:
                del self.map[k]

    def set(self, key: str, months: int):
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
    parts = []
    for m, w in metric_weights.items():
        if m not in score_frame.columns:
            continue
        s = score_frame[m].astype(float)
        s = zscore(s) * float(w) * metric_directions.get(m, 1)
        parts.append(s)
    if not parts:
        return pd.Series(index=score_frame.index, dtype=float)
    total = sum(parts)
    return total


def decide_hires_fires(
    asof: pd.Timestamp,
    score_frame: pd.DataFrame,
    current: List[str],
    policy: PolicyConfig,
    directions: Dict[str, int],
    cooldowns: CooldownBook,
    eligible_since: Dict[str, int],
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
    to_fire = []
    if policy.bottom_k > 0:
        bottom = list(sf.tail(policy.bottom_k).index)
        for m in bottom:
            if m in current:
                to_fire.append((m, "bottom_k"))
    candidates = [
        m for m in list(sf.index) if m not in current and not cooldowns.in_cooldown(m)
    ]
    hires = []
    for m in candidates[: policy.top_k]:
        hires.append((m, "top_k"))
    next_active = list(set(current) - {x for x, _ in to_fire})
    for m, _ in hires:
        if len(next_active) >= policy.max_active:
            break
        next_active.append(m)
    return {"hire": hires, "fire": to_fire}
