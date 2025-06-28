"""Rebalance portfolio weights based on ranking triggers."""

from __future__ import annotations

from typing import Dict
import warnings

import numpy as np
import pandas as pd


class Rebalancer:  # pylint: disable=too-few-public-methods
    """Apply ranking triggers and adjust portfolio weights."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        mp = cfg["multi_period"]
        self.min_funds = int(mp["min_funds"])
        self.max_funds = int(mp["max_funds"])
        self.triggers: Dict[str, Dict[str, float]] = mp["triggers"]
        self.anchors = mp["weight_curve"]["anchors"]
        self._strike_table: Dict[str, Dict[str, int]] = {}
        self._rng = np.random.default_rng(cfg.get("random_seed", 0))

    def _trigger_sigma(self) -> float:
        """Return the smallest ``sigma`` threshold from the trigger config."""

        return min(float(t["sigma"]) for t in self.triggers.values())

    def apply_triggers(
        self, prev_weights: pd.Series, score_frame: pd.DataFrame
    ) -> pd.Series:
        """Return next-period weights after applying ranking logic."""

        if prev_weights.empty:
            top = score_frame.sort_values("rank").head(self.min_funds)
            weights = pd.Series(1 / len(top), index=top.index)
            self._strike_table = {f: {k: 0 for k in self.triggers} for f in top.index}
            return weights

        removals: list[str] = []
        for fund in prev_weights.index:
            zscore = float(score_frame.loc[fund, "zscore"])
            strikes = self._strike_table.setdefault(fund, {k: 0 for k in self.triggers})
            remove = False
            for name, trig in self.triggers.items():
                sigma = float(trig["sigma"])
                periods = int(trig["periods"])
                if zscore < -sigma:
                    strikes[name] = strikes.get(name, 0) + 1
                else:
                    strikes[name] = 0
                if strikes[name] >= periods:
                    remove = True
            if remove:
                removals.append(fund)

        survivors = [f for f in prev_weights.index if f not in removals]

        if len(survivors) < self.min_funds:
            warnings.warn("Min funds reached; skipping removals.")
            survivors = list(prev_weights.index)
        else:
            for fund in removals:
                self._strike_table.pop(fund, None)

        sigma = self._trigger_sigma()
        if len(survivors) < self.max_funds:
            candidates = score_frame.index.difference(prev_weights.index)
            cand_df = score_frame.loc[candidates]
            cand_df = cand_df[cand_df["zscore"] > sigma]
            if not cand_df.empty:
                cand_df = cand_df.copy()
                cand_df["_rnd"] = self._rng.random(len(cand_df))
                cand_df = cand_df.sort_values(["rank", "_rnd"])
                for fund in cand_df.index:
                    if len(survivors) >= self.max_funds:
                        break
                    survivors.append(fund)
                    self._strike_table[fund] = {k: 0 for k in self.triggers}

        weights = prev_weights.reindex(survivors).fillna(0.0)

        n = len(score_frame)
        x = [float(a[0]) for a in self.anchors]
        y = [float(a[1]) for a in self.anchors]
        multipliers = []
        for fund in weights.index:
            rank = float(score_frame.loc[fund, "rank"])
            pct = 100 * (rank - 1) / max(n - 1, 1)
            multipliers.append(float(np.interp(pct, x, y)))
        weights *= pd.Series(multipliers, index=weights.index)

        new_funds = [f for f in survivors if f not in prev_weights.index]
        for fund in new_funds:
            weights.loc[fund] = 1.0

        weights = weights / weights.sum()
        return weights
