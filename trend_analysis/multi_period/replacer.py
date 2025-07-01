"""Rebalance portfolio weights over multiple periods.

The :class:`Rebalancer` applies simple removal and addition rules based on
performance ``z``-scores.  It keeps track of consecutive low ``z`` events for
each fund and removes holdings when configured strike thresholds are met.  New
funds with sufficiently high ``z``-scores may be introduced, and surviving
holdings are reweighted according to a rank based weight curve.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd
import numpy as np


class Rebalancer:  # pylint: disable=too-few-public-methods
    """Apply removal and addition triggers to portfolio weights."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Store configuration and initialise state."""

        self.cfg = cfg
        mp = cfg["multi_period"]
        self.min_funds = int(mp["min_funds"])
        self.max_funds = int(mp["max_funds"])
        self.triggers = mp.get("triggers", {})
        self.anchors = mp["weight_curve"]["anchors"]
        self._strike_table: dict[str, dict[str, int]] = {}
        self._rng = np.random.default_rng(cfg.get("random_seed"))

    # ------------------------------------------------------------------
    def apply_triggers(
        self,
        prev_weights: Dict[str, float] | pd.Series,
        score_frame: pd.DataFrame,
    ) -> pd.Series:
        """Return weights for the next period after applying triggers.

        Parameters
        ----------
        prev_weights : dict or pandas.Series
            Portfolio weights from the previous period indexed by ``fund_id``.
        score_frame : pandas.DataFrame
            Result of ``single_period_run`` with ``zscore`` and ``rank`` columns.

        Returns
        -------
        pandas.Series
            New weights normalised to one.
        """

        prev_w = (
            prev_weights.copy()
            if isinstance(prev_weights, pd.Series)
            else pd.Series(prev_weights, dtype=float)
        )

        zscores = score_frame["zscore"]
        ranks = score_frame["rank"]

        if prev_w.empty:
            top = ranks.nsmallest(self.min_funds).index
            eq = 1.0 / len(top)
            self._strike_table = {f: {k: 0 for k in self.triggers} for f in top}
            return pd.Series(eq, index=top, dtype=float)

        # ensure strike table entries exist
        for f in prev_w.index:
            self._strike_table.setdefault(f, {k: 0 for k in self.triggers})

        to_drop: list[str] = []
        for f in list(prev_w.index):
            z = float(zscores.get(f, 0.0))
            remove = False
            for name, trig in self.triggers.items():
                sigma = float(trig["sigma"])
                periods = int(trig["periods"])
                if z < -sigma:
                    self._strike_table[f][name] += 1
                else:
                    self._strike_table[f][name] = 0
                if self._strike_table[f][name] >= periods:
                    remove = True
            if remove:
                to_drop.append(f)

        # respect minimum fund constraint
        max_drops = len(prev_w) - self.min_funds
        if max_drops < len(to_drop):
            import warnings

            warnings.warn("Minimum fund limit reached; dropping truncated")
            to_drop = to_drop[: max_drops if max_drops > 0 else 0]

        for f in to_drop:
            prev_w.drop(f, inplace=True)
            self._strike_table.pop(f, None)

        # --------------------------------------------------------------
        # Additions
        available = self.max_funds - len(prev_w)
        if available > 0 and not zscores.empty:
            sigma_add = min(float(t["sigma"]) for t in self.triggers.values())
            mask = (~score_frame.index.isin(prev_w.index)) & (zscores > sigma_add)
            if mask.any():
                cand = score_frame.loc[mask, ["rank"]].copy()
                cand["rnd"] = self._rng.random(len(cand))
                cand.sort_values(["rank", "rnd"], inplace=True)
                for f in cand.index[:available]:
                    prev_w[f] = 1.0
                    self._strike_table[f] = {k: 0 for k in self.triggers}

        if prev_w.empty:
            return prev_w

        # --------------------------------------------------------------
        # Apply weight curve to surviving holdings
        x, y = zip(*sorted(self.anchors))
        r_pct = 100 * (ranks.loc[prev_w.index] - 1) / max(len(ranks) - 1, 1)
        multipliers = pd.Series(np.interp(r_pct, x, y), index=prev_w.index)
        prev_w = prev_w.mul(multipliers, axis=0)

        # weights for new entrants are already set to 1.0 -> leave as is

        prev_w = prev_w / prev_w.sum()
        return prev_w
